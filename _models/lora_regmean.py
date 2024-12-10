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
import math


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
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        regmean_all: str_to_bool = True,
        alpha_regmean: float = 0.5,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
    ) -> None:
        super(LoraRegMean, self).__init__(
            fabric, network, device, optimizer, lr, wd_reg, avg_type, lora_alpha, r, lora_head, cl_merge
        )
        self.reg_dtype_64 = reg_dtype_64
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
        self.fed_weights = {}
        for key in self.lora_keys:
            self.fed_weights[key] = torch.zeros_like((self.cur_B[key] @ self.cur_A[key]))

    def get_optimization_dict(self):
        opti_dict = super().get_optimization_dict()
        for key in self.lora_keys:
            opti_dict[key] += self.fed_weights[key]
        return opti_dict

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.cur_task > 0:
            if self.cl_merge == "run_sum":
                for key in self.lora_keys:
                    self.old_delta[key] += self.fed_weights[key]
                    self.fed_weights[key] = torch.zeros_like(self.fed_weights[key])
            elif self.cl_merge == "run_mean" or "individual" in self.cl_merge:
                for key in self.lora_keys:
                    self.old_delta[key] = (
                        self.old_delta[key] * (self.cur_task - 1) + self.fed_weights[key].detach()
                    ) / self.cur_task
                    self.fed_weights[key] = torch.zeros_like(self.fed_weights[key])
            else:
                raise ValueError("Invalid cl_merge type")

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.old_delta = deepcopy(server_info["old_delta"])
        self.fed_weights = deepcopy(server_info["fed_weights"])
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.head.values()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()), lr=self.lr, weight_decay=self.wd_reg
            )
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def begin_round_server(self):
        super().begin_round_server()
        for key in self.lora_keys:
            self.cur_B[key] = nn.Parameter(torch.zeros_like(self.cur_B[key]), requires_grad=True).to(self.device)
            self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
            nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))

    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        if "run_sum" in self.cl_merge:
            for key in self.lora_keys:
                self.old_delta[key].requires_grad = False
                self.cur_B[key].requires_grad = False
                self.cur_A[key].requires_grad = False
                self.fed_weights[key].requires_grad = False
                self.fed_weights[key] = self.fed_weights[key].to(self.device)
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                if self.cur_task > 0:
                    self.optimization_dict[key] += self.old_delta[key]
                self.optimization_dict[key] += self.fed_weights[key]
        elif "run_mean" in self.cl_merge:
            for key in self.lora_keys:
                self.old_delta[key].requires_grad = False
                self.cur_B[key].requires_grad = False
                self.cur_A[key].requires_grad = False
                self.fed_weights[key].requires_grad = False
                self.fed_weights[key] = self.fed_weights[key].to(self.device)
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                if self.cur_task > 0:
                    tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key]
                    self.optimization_dict[key] += tmp / (self.cur_task + 1)
                else:
                    self.optimization_dict[key] += self.fed_weights[key]
        elif "individual" in self.cl_merge:
            if "sum" in self.cl_merge:
                for key in self.lora_keys:
                    self.old_delta[key].requires_grad = False
                    self.cur_B[key].requires_grad = False
                    self.cur_A[key].requires_grad = False
                    self.fed_weights[key].requires_grad = False
                    self.fed_weights[key] = self.fed_weights[key].to(self.device)
                    self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key]
                        self.optimization_dict[key] += tmp
                    else:
                        self.optimization_dict[key] += self.fed_weights[key]

            elif "mean" in self.cl_merge:
                for key in self.lora_keys:
                    self.old_delta[key].requires_grad = False
                    self.cur_B[key].requires_grad = False
                    self.cur_A[key].requires_grad = False
                    self.fed_weights[key].requires_grad = False
                    self.fed_weights[key] = self.fed_weights[key].to(self.device)
                    self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key].detach()
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += self.fed_weights[key].detach()

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

    def get_server_info(self):
        server_info = super().get_server_info()
        server_info["old_delta"] = deepcopy(self.old_delta)
        server_info["fed_weights"] = deepcopy(self.fed_weights)
        server_info["head"] = deepcopy(self.network.model.head.state_dict())
        return server_info

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
        dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
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
            # self.fed_weights = {key: None for key in self.lora_keys}
            for key in self.lora_keys:
                if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                    name = self.middle_names[key]
                    self.fed_weights[key] = self.fed_weights[key].to("cpu")
                    self.fed_weights[key] += (
                        torch.stack(
                            [
                                (client_B[key] @ client_A[key]).to(dtype) @ client["grams"][name].to(dtype)
                                for client, client_A, client_B in zip(client_info, cl_A, cl_B)
                            ]
                        ).sum(0)
                        @ torch.inverse(torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0))
                    ).to(torch.float32)
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
                                client["state_dict"][key].to(dtype) @ client["grams"][name].to(dtype)
                                for client in client_info
                            ]
                        ).sum(0)
                        @ torch.inverse(torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0))
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

    def to(self, device="cpu"):
        super().to(device)
        for key in self.lora_keys:
            self.fed_weights[key] = self.fed_weights[key].to(device)
        return self
