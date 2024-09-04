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
from models.regmean import RegMean
from tqdm import tqdm
import math


@register_model("lora_regmean2")
class LoraRegMean(Lora, RegMean):

    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        clip_grad: str_to_bool = False,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1.0,
        r: int = 16,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        regmean_all: str_to_bool = True,
        alpha_regmean_head: float = 0.5,
        alpha_regmean_backbone: float = -1,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        slca: str_to_bool = False,
        only_square: int = 0,
        train_bias: str = "all",
    ) -> None:
        self.lora_head = False
        self.reg_dtype_64 = reg_dtype_64
        self.middle_names = {}  # conversion from state_dict() names to the names of the modules
        Lora.__init__(
            self,
            fabric,
            network,
            device,
            optimizer,
            lr,
            clip_grad,
            wd_reg,
            avg_type,
            lora_alpha,
            r,
            lora_head,
            cl_merge,
        )
        RegMean.__init__(
            self,
            fabric,
            network,
            device,
            optimizer,
            lr,
            wd_reg,
            avg_type,
            regmean_all,
            alpha_regmean_head,
            alpha_regmean_backbone,
            gram_dtype,
            reg_dtype_64,
            slca,
            only_square,
            train_bias,
        )
        for name in self.gram_modules:
            self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.fed_weights = {}
        for key in self.lora_keys:
            self.fed_weights[key] = torch.zeros_like((self.cur_B[key].detach() @ self.cur_A[key].detach()))

    def get_optimization_dict(self, fabric=True):
        opti_dict = Lora.get_optimization_dict(self, fabric=fabric)
        for key in self.lora_keys:
            opti_dict[key] += self.fed_weights[key]
        return opti_dict

    def begin_task(self, n_classes_per_task: int):
        BaseModel.begin_task(self, n_classes_per_task)
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
            self.init_matrices()

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        Lora.begin_round_client(self, dataloader, server_info)
        self.fed_weights = deepcopy(server_info["fed_weights"])
        # Regmean already does it, but also mounts the state dict, which is already done by LoRA
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)

    def begin_round_server(self):
        Lora.begin_round_server(
            self,
        )
        self.init_matrices()

    def set_optimization_cur_task(self, fabric=True):
        sd = self.network.state_dict()
        opt_dict = self.get_optimization_dict(fabric=fabric)
        if not self.lora_head:
            for key in self.head_keys:
                opt_dict[key] = sd[key]
        self.optimization_dict = opt_dict

    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = False
            self.cur_A[key].requires_grad = False
            self.fed_weights[key].requires_grad = False
            self.fed_weights[key] = self.fed_weights[key].to(self.device)
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        if "run_sum" in self.cl_merge:
            for key in self.lora_keys:
                if self.cur_task > 0:
                    self.optimization_dict[key] += self.old_delta[key]
                self.optimization_dict[key] += self.fed_weights[key]
        elif "run_mean" in self.cl_merge:
            for key in self.lora_keys:
                if self.cur_task > 0:
                    tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key]
                    self.optimization_dict[key] += tmp / (self.cur_task + 1)
                else:
                    self.optimization_dict[key] += self.fed_weights[key]
        elif "individual" in self.cl_merge:
            if "sum" in self.cl_merge:
                for key in self.lora_keys:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key]
                        self.optimization_dict[key] += tmp
                    else:
                        self.optimization_dict[key] += self.fed_weights[key]
            elif "mean" in self.cl_merge:
                for key in self.lora_keys:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key].detach()
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += self.fed_weights[key].detach()

    def end_round_client(self, dataloader: DataLoader):
        Lora.end_round_client(self, dataloader)
        # self.set_optimization_cur_task(fabric=True)  # loading current task parameters only to compute the Gram matrices
        # setting up the parameters to correctly compute the Gram matrices for the next round
        for key in self.lora_keys:
            self.fed_weights[key] = self.cur_B[key] @ self.cur_A[key]
        self.set_optimization()
        for name in self.gram_modules:
            self.features[name] = self.features[name].to(self.device)
        RegMean.end_round_client(self, dataloader)  # retrieves Gram matrices from hooks

    def get_client_info(self, dataloader: DataLoader):
        client_info = Lora.get_client_info(self, dataloader)
        client_info["grams"] = deepcopy(self.features)
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        return client_info

    def get_server_info(self):
        server_info = Lora.get_server_info(
            self,
        )
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
            dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
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
        Lora.to(self, device)
        for key in self.lora_keys:
            self.fed_weights[key] = self.fed_weights[key].to(device)
