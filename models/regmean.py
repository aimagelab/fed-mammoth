import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models import register_model
from typing import List
from models.utils import BaseModel
from networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool

from models.lora import Lora, merge_AB, zero_pad
from tqdm import tqdm


@register_model("regmean")
class RegMean(BaseModel):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        regmean_all: str_to_bool = True,
        alpha_regmean: float = 0.5,
        gram_dtype: str = "32",
    ) -> None:
        super(RegMean, self).__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.avg_type = avg_type
        self.regmean_all = regmean_all
        self.alpha_regmean = alpha_regmean
        self.gram_modules = []
        self.middle_names = {}  # conversion from state_dict() names to the names of the modules
        self.gram_dtype = (
            torch.float32
            if gram_dtype == "32"
            else torch.float16 if gram_dtype == "16" else torch.bfloat16 if gram_dtype == "b16" else torch.float64
        )
        if regmean_all:
            for name, module in self.network.named_modules():
                # if ((("qkv" in name or "mlp" in name or ("proj" in name and "attn" in name)) and self.regmean_all) or "head" in name) and not "drop" in name and not "act" in name and not "norm" in name:
                # list(module.parameters())
                if (
                    (
                        (("qkv" in name or "mlp" in name or ("proj" in name and "attn" in name)) and self.regmean_all)
                        or "head" in name
                    )
                    and len(list(module.parameters())) > 0
                    and len(list(module.children())) == 0
                ):
                    self.gram_modules.append(name)
        else:
            for name, module in self.network.named_modules():
                if "head" in name and len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                    self.gram_modules.append(name)
                    self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.features = {key: torch.tensor([], dtype=self.gram_dtype) for key in self.gram_modules}
        self.gram = {key: torch.tensor([], dtype=self.gram_dtype) for key in self.gram_modules}

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels % self.cpt)

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()

        return loss.item()

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        hooks = {name: None for name in self.gram_modules}
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                # module.forward_handle = module.register_forward_hook(self.hook_forward)
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        with torch.no_grad():
            print()
            for id, (x, y) in enumerate(tqdm(dataloader, desc="Computing Gram matrices")):
                x, y = x.to(self.device), y.to(self.device)
                # TODO handle the fact that the network is not updated
                self.forward(x)
                # if id == 2:
                #    break
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                # hooks[name].remove()
                self.features[name] = self.features[name].to("cpu")
                shape = self.features[name].shape[-1]
                self.gram[name] = (
                    self.features[name] * self.alpha_regmean
                    + (1 - self.alpha_regmean) * torch.eye(shape, dtype=self.gram_dtype) * self.features[name]
                )
                self.features[name] = torch.tensor([], dtype=self.gram_dtype)
                hooks[name].remove()

    def hook_handler(self, name):
        def hook_forward(module, inputs, _):
            x = inputs[0].detach().to(self.gram_dtype)
            if len(x.shape) == 3:
                x = x.view(-1, x.size(-1))
            tmp = torch.zeros(x.size(-1), x.size(-1), device=self.device, dtype=self.gram_dtype)
            torch.matmul(x.T, x, out=tmp)
            if len(self.features[name]) == 0:
                self.features[name] = tmp
            else:
                self.features[name] += tmp

        return hook_forward

    def get_client_info(self, dataloader: DataLoader):
        client_info = super().get_client_info(dataloader)
        if client_info is None:
            client_info = {}
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        client_info["grams"] = deepcopy(self.gram)
        client_info["num_train_samples"] = len(dataloader.dataset.data)
        return client_info

    def to(self, device="cpu"):
        self.network.to(device)
        for name in self.gram_modules:
            self.gram[name] = self.gram[name].to(device)
            self.features[name] = self.features[name].to(device)

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        # regmean solution
        keys = list(self.network.state_dict().keys())
        sd = self.network.state_dict()
        for key in keys:
            if (
                "weight" in key and self.middle_names.get(key) is not None
            ):  # it means that we apply regmean to this layer
                name = self.middle_names[key]
                sd[key] = (
                    torch.stack(
                        [
                            client["state_dict"][key].to(self.gram_dtype) @ client["grams"][name]
                            for client in client_info
                        ]
                    ).sum(0)
                    @ torch.inverse(torch.stack([client["grams"][name] for client in client_info]).sum(0))
                ).to(torch.float32)
            else:
                sd[key] = torch.stack(
                    [client["state_dict"][key] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
                ).sum(
                    0
                )  # fedavg for the other layers
        self.network.load_state_dict(sd)
