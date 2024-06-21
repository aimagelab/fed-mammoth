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


@register_model("lora_regmean")
class LoraRegMean(Lora):

    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1.0,
        r: int = 16,
        enable_lora: list = [True, True, True],
        lora_head: str_to_bool = True,
        cl_merge: str = "run_sum",
        regmean_all: str_to_bool = True,
        alpha_regmean: float = 0.5,
        gram_dtype: str = "32",
    ) -> None:
        super(LoraRegMean, self).__init__(
            fabric, network, device, optimizer, lr, wd_reg, avg_type, lora_alpha, r, enable_lora, lora_head, cl_merge
        )
        self.lora_head = False
        self.avg_type = avg_type
        self.regmean_all = regmean_all
        self.alpha_regmean = alpha_regmean
        self.lora_modules = []
        self.gram_modules = []
        self.middle_names = {}  # conversion from state_dict() names to the names of the modules
        self.gram_dtype = (
            torch.float32
            if gram_dtype == "32"
            else torch.float16 if gram_dtype == "16" else torch.bfloat16 if gram_dtype == "b16" else torch.float64
        )
        # layers to be used for regmean: if regmean_all is True, all layers are used, otherwise only the head
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
                self.lora_modules.append(name)
                self.gram_modules.append(name)
                self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.features = {key: torch.tensor([], dtype=self.gram_dtype) for key in self.gram_modules}
        self.gram = {key: torch.tensor([], dtype=self.gram_dtype) for key in self.gram_modules}

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        hooks = {name: None for name in self.gram_modules}
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                # module.forward_handle = module.register_forward_hook(self.hook_forward)
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        self.set_optimization()
        with torch.no_grad():
            print()
            for id, (x, y) in enumerate(tqdm(dataloader, desc="Computing Gram matrices")):
                x, y = x.to(self.device), y.to(self.device)
                self.forward(x)
                if id > 2:
                    break
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
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
        client_info["grams"] = deepcopy(self.gram)
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        return client_info

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
        cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
        # regmean will always be applied to the head, optionally to the other layers
        # lora instead will always be applied to the other layers, optionally to the head
        if not self.regmean_all:
            # fedavg on Lora matrices for all layers except head
            for key in self.lora_keys:
                self.cur_B[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(0)
                )
                self.cur_A[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(0)
                )

        # regmean solution
        else:
            keys = list(self.network.state_dict().keys())
            w_solution = {key: None for key in self.lora_keys}
            for key in self.lora_keys:
                if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                    name = self.middle_names[key]
                    w_solution[key] = (
                        torch.stack(
                            [
                                client["state_dict"][key].to(self.gram_dtype) @ client["grams"][name]
                                for client in client_info
                            ]
                        ).sum(0)
                        @ torch.inverse(torch.stack([client["grams"][name] for client in client_info]).sum(0))
                    ).to(torch.float32)
            lora_opt = torch.optim.SGD(list(self.cur_B.values()) + list(self.cur_A.values()), lr=1e-3, weight_decay=0)
            num_epochs = 100
            criterion = torch.nn.MSELoss()
            losses = []
            for key in self.lora_keys:
                w_solution[key] = w_solution[key].to(self.device)
                self.cur_A[key] = self.cur_A[key].to(self.device)
                self.cur_B[key] = self.cur_B[key].to(self.device)
                self.cur_A[key].requires_grad = True
                self.cur_B[key].requires_grad = True
                for epoch in range(num_epochs):
                    lora_opt.zero_grad()
                    if "qkv" in key:
                        res = merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
                    else:
                        res = self.cur_B[key] @ self.cur_A[key]
                    loss = criterion(res, w_solution[key])
                    loss.backward()
                    lora_opt.step()
                losses.append(loss.detach())
            print(f"Reconstruction loss: {sum(losses) / len(losses)}")
            del lora_opt, criterion, losses, w_solution

        # regmean on head
        keys = list(self.network.state_dict().keys())
        sd = self.network.state_dict()
        for key in keys:
            # head parameters
            if "head" in key:
                if self.middle_names.get(key) is not None and "head" in key:  # regmean on Linear layer
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
                else:  # fedavg bias
                    sd[key] = torch.stack(
                        [
                            client["state_dict"][key] * norm_weight
                            for client, norm_weight in zip(client_info, norm_weights)
                        ]
                    ).sum(0)

        self.network.load_state_dict(sd)

        self.set_optimization()
