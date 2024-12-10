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

from _models.vera2 import Vera
from _models.lora import Lora, merge_AB, zero_pad
from _models.regmean import RegMean
from _models.lora_regmean2 import LoraRegMean
from tqdm import tqdm
import math


@register_model("vera_regmean2")
class VeraRegMean(Vera, RegMean):
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
        r: int = 1024,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        regmean_all: str_to_bool = True,
        alpha_regmean: float = 0.5,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        d_initial: float = 1.0,
        only_square: int = 0,
    ) -> None:
        self.lora_head = False
        self.reg_dtype_64 = reg_dtype_64
        self.middle_names = {}  # conversion from state_dict() names to the names of the modules
        Vera.__init__(
            self,
            fabric,
            network,
            device,
            optimizer,
            lr,
            wd_reg,
            avg_type,
            lora_alpha,
            r,
            lora_head,
            cl_merge,
            d_initial,
            only_square,
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
            alpha_regmean,
            gram_dtype,
            False,
            only_square,
        )
        for name in self.gram_modules:
            self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.fed_weights = {}
        for key in self.lora_keys:
            self.cur_B = self.cur_B.to(self.device)
            self.cur_A = self.cur_A.to(self.device)
            self.fed_weights[key] = torch.zeros_like(self.vera_multiply(key))

    # @staticmethod
    def vera_multiply_explicit(self, key, vec_b, vec_d, cur_B, cur_A, lora_params_B, lora_params_A):
        return (vec_b[key] * cur_B[0 : lora_params_B[key][0], 0 : lora_params_B[key][1]]) @ (
            vec_d[key] * cur_A[0 : lora_params_A[key][0], 0 : lora_params_A[key][1]]
        )

    def get_optimization_dict(self):
        opti_dict = Vera.get_optimization_dict(
            self,
        )
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
        Vera.begin_round_client(self, dataloader, server_info)
        self.fed_weights = deepcopy(server_info["fed_weights"])

    def begin_round_server(self):
        Vera.begin_round_server(
            self,
        )
        for key in self.lora_keys:
            self.vec_b[key] = nn.Parameter(torch.zeros_like(self.vec_b[key]), requires_grad=True).to(self.device)
            self.vec_d[key] = nn.Parameter(torch.ones_like(self.vec_d[key]) * self.d_initial, requires_grad=True).to(
                self.device
            )

    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B.requires_grad = False
            self.cur_A.requires_grad = False
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
        Vera.end_round_client(self, dataloader)
        self.set_optimization()
        RegMean.end_round_client(self, dataloader)  # retrieves Gram matrices from hooks

    def get_client_info(self, dataloader: DataLoader):
        client_info = Vera.get_client_info(self, dataloader)
        client_info["grams"] = deepcopy(self.gram)
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        return client_info

    def get_server_info(self):
        server_info = Vera.get_server_info(
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
        cl_b = [client["vec_b"] for client in client_info]  # list of B matrices for all clients
        cl_d = [client["vec_d"] for client in client_info]  # list of A matrices for all clients
        self.to("cpu")
        # regmean will always be applied to the head, optionally to the other layers
        # lora instead will always be applied to the other layers, optionally to the head
        dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
        if not self.regmean_all:
            # fedavg on Lora matrices for all layers except head
            for key in self.lora_keys:
                self.vec_b[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_b, norm_weights)]).sum(0)
                )
                self.vec_d[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_d, norm_weights)]).sum(0)
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
                                (
                                    self.vera_multiply_explicit(
                                        key,
                                        client_b,
                                        client_d,
                                        self.cur_B,
                                        self.cur_A,
                                        self.lora_params_B,
                                        self.lora_params_A,
                                    )
                                ).to(dtype)
                                @ client["grams"][name].to(dtype)
                                for client, client_d, client_b in zip(client_info, cl_d, cl_b)
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
        self.to(self.device)
        self.set_optimization()

    def to(self, device="cpu"):
        Vera.to(self, device)
        for key in self.lora_keys:
            self.fed_weights[key] = self.fed_weights[key].to(device)
        return self
