import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from _models import register_model
from typing import List
from _models._utils import BaseModel
from _networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool

from _models.lora import Lora, merge_AB, zero_pad
from tqdm import tqdm


@register_model("regmean_cl")
class RegMean(BaseModel):

    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-5,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        regmean_all: str_to_bool = True,
        alpha_regmean_head: float = 0.5,
        alpha_regmean_backbone: float = -1,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        lr_back: float = -1,
        only_square: int = 0,
        train_bias: str = "all",
        clip_grad: str_to_bool = False,
    ) -> None:
        self.reg_dtype_64 = reg_dtype_64
        self.optimizer_str = optimizer
        self.lr = lr
        self.wd_reg = wd_reg
        self.clip_grad = clip_grad
        if alpha_regmean_backbone < 0:
            alpha_regmean_backbone = alpha_regmean_head
        if "all" not in train_bias:
            for n, p in network.named_parameters():
                if "bias" in n:
                    if "head" not in n:
                        p.requires_grad = False
                    elif "head" in n and "head" not in train_bias:
                        p.requires_grad = False
        self.lr_back = lr_back
        if self.lr_back < 0:
            self.lr_back = self.lr
        #backbone_params, head_params = self.split_backbone_head()
        #params = [{"params": backbone_params, "lr": lr_back}, {"params": head_params}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg)#, params=params)
        self.avg_type = avg_type
        self.regmean_all = regmean_all
        self.alpha_regmean = [alpha_regmean_backbone, alpha_regmean_head]
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
                    len(list(module.parameters())) > 0
                    and len(list(module.children())) == 0
                    and (
                        (
                            (("mlp" in name or ("proj" in name and "attn" in name)) and self.regmean_all)
                            and (
                                only_square <= 0
                                or module.state_dict()["weight"].shape[0]
                                == module.state_dict()["weight"].shape[1]
                                == only_square
                            )
                        )
                        or "head" in name
                        or "qkv" in name
                    )
                ):
                    self.gram_modules.append(name)
                    self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        else:
            for name, module in self.network.named_modules():
                if "head" in name and len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                    self.gram_modules.append(name)
                    self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.features = {key: torch.tensor([], dtype=self.gram_dtype) for key in self.gram_modules}

    def split_backbone_head(self):
        backbone_params = []
        head_params = []
        for n, p in self.network.named_parameters():
            if "head" in n:
                head_params.append(p)
            else:
                backbone_params.append(p)
        return backbone_params, head_params

    def forward(self, x):
        if getattr(self, "inference_net", None) is not None:
            return self.inference_net(x)
        return self.network(x)


    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            if self.clip_grad:
                self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
            self.optimizer.step()

        return loss.item()

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        sd = server_info["state_dict"]
        self.network.load_state_dict(sd)
        self.network.train()
        backbone_params, head_params = self.split_backbone_head()
        params = [{"params": backbone_params, "lr": self.lr_back}, {"params": head_params}]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)
    
    def begin_round_server(self):
        if getattr(self, "original_state_dict", None) is None:
            self.original_state_dict = deepcopy(self.network.state_dict())
        super().begin_round_server()

    def end_round_client(self, dataloader: DataLoader):
        self.network.eval()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        hooks = {name: None for name in self.gram_modules}
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                # module.forward_handle = module.register_forward_hook(self.hook_forward)
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        with torch.no_grad():
            print()
            for id, (x, y) in enumerate(tqdm(dataloader, desc="Computing Gram matrices")):
                x, y = x.to(self.device), y.to(self.device)
                #self.forward(x)
                self.network(x)
                # if id == 2:
                #    break
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                if "head" in name:
                    alpha = self.alpha_regmean[1]
                else:
                    alpha = self.alpha_regmean[0]
                self.features[name] = self.features[name].to("cpu")
                shape = self.features[name].shape[-1]
                self.features[name] = (
                    self.features[name] * alpha
                    + (1 - alpha) * torch.eye(shape, dtype=self.gram_dtype) * self.features[name]
                )
                hooks[name].remove()

    def hook_handler(self, name):
        def hook_forward(module, inputs, _):
            x = inputs[0].detach().to(self.gram_dtype)
            if len(x.shape) == 3:
                x = x.view(-1, x.size(-1))
            tmp = torch.zeros(x.size(-1), x.size(-1), device=self.device, dtype=self.gram_dtype)
            torch.matmul(x.T, x, out=tmp)
            self.features[name] = self.features[name].to(self.device)
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
        client_info["grams"] = deepcopy(self.features)
        client_info["num_train_samples"] = len(dataloader.dataset.data)
        return client_info

    def to(self, device="cpu"):
        self.network.to(device)
        for name in self.gram_modules:
            self.features[name] = self.features[name].to(device)
        return self

    def get_server_info(self):
        return {"state_dict": deepcopy(self.original_state_dict)}

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
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
                            client["state_dict"][key].to(dtype) @ client["grams"][name].to(dtype)
                            for client in client_info
                        ]
                    ).sum(0)
                    @ torch.pinverse(torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0))
                ).to(torch.float32)
            else:
                sd[key] = torch.stack(
                    [client["state_dict"][key] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
                ).sum(
                    0
                )  # fedavg for the other layers
        self.network.load_state_dict(sd)
        if getattr(self, "inference_net", None) is None:
            self.inference_net = deepcopy(self.network)
        # set the inference network to the mean of the tasks in order to 
        # test the model after each communication round
        self.set_inference_tetwork()

    def set_inference_tetwork(self,):
        if self.cur_task > 0:
            self.network = self.network.to(self.device)
            self.inference_net = self.inference_net.to(self.device)
            self.mean_tasks = self.mean_tasks.to(self.device)
            cur_params = self.network.get_params()
            mean_params = self.mean_tasks.get_params()
            self.inference_net.set_params((cur_params + mean_params * self.cur_task) / (self.cur_task + 1))
            self.network = self.network.to("cpu")
            self.mean_tasks = self.mean_tasks.to("cpu")
        torch.cuda.empty_cache()

            


    def end_task_server(self, client_info: List[dict] = None):
        if getattr(self, "mean_tasks", None) is None:
            self.mean_tasks = deepcopy(self.network)
            return
        self.network = self.network.to(self.device)
        self.mean_tasks = self.mean_tasks.to(self.device)
        cur_params = self.network.get_params()
        mean_params = self.mean_tasks.get_params()
        self.mean_tasks.set_params((cur_params + mean_params * self.cur_task) / (self.cur_task + 1))
        self.inference_net.load_state_dict(self.mean_tasks.state_dict())
        self.network = self.network.to("cpu")
        self.mean_tasks = self.mean_tasks.to("cpu")
        self.inference_net = self.inference_net.to(self.device)