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


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import Optional, List


def zero_pad(x, lora_ind):
    result = x.new_zeros((len(lora_ind), *x.shape[1:]))
    result[lora_ind] = x
    return result


def merge_AB(A, B, lora_ind):
    def T(w):
        # return w.transpose(0, 1) if self.fan_in_fan_out else w
        return w

    delta_w = F.conv1d(A.unsqueeze(0), B.unsqueeze(-1), groups=3).squeeze(
        0
    )  # groups = 3 because we are using q,k and v, shape [768, 2304]
    return T(zero_pad(delta_w, lora_ind))


@register_model("lora")
class Lora(BaseModel):
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
        enable_lora=[True, True, True],
    ) -> None:
        # for LoRA, we keep the mean of the LoRA modules of the old tasks
        super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.lora_alpha = lora_alpha
        self.r = r
        self.enable_lora = enable_lora
        self.lora_keys = []
        self.lora_params = {}
        self.lora_ind = {}
        self.optimizer_str = optimizer
        self.lr = lr
        self.wd_reg = wd_reg
        self.old_B = {}
        self.old_A = {}
        self.cur_B = {}
        self.cur_A = {}
        for name, param in network.named_parameters():
            param.requires_grad = False  # freeze all the parameters
            if "qkv" in name and "weight" in name:
                self.lora_keys.append(name)
                self.lora_ind[name] = torch.zeros((param.shape[0],), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[name][enable_lora, :] = True
                self.lora_ind[name] = self.lora_ind[name].view(-1)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_B[name] = nn.Parameter(torch.zeros(param.shape[0], r), requires_grad=False).to(self.device)
                self.old_A[name] = nn.Parameter(torch.zeros(r * 3, param.shape[1]), requires_grad=False).to(self.device)
                self.cur_B[name] = nn.Parameter(torch.zeros_like(self.old_B[name]), requires_grad=True).to(self.device)
                self.cur_A[name] = nn.Parameter(torch.zeros_like(self.old_A[name]), requires_grad=True).to(self.device)
            elif ("mlp" in name and "weight" in name) or ("proj" in name and "weight" in name and "attn" in name):
                self.lora_keys.append(name)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_B[name] = nn.Parameter(torch.zeros(param.shape[0], r), requires_grad=False).to(self.device)
                self.old_A[name] = nn.Parameter(torch.zeros(r, param.shape[1]), requires_grad=False).to(self.device)
                self.cur_B[name] = nn.Parameter(torch.zeros_like(self.old_B[name]), requires_grad=True).to(self.device)
                self.cur_A[name] = nn.Parameter(torch.zeros_like(self.old_A[name]), requires_grad=True).to(self.device)
        self.optimization_dict = {}
        # self.network = network
        # self.set_optimization()
        self.avg_type = avg_type
        self.pre_B = {}
        self.pre_A = {}
        self.pre_network = None

    # used for testing, using a functional_call() to call the network with self.optimization_dict parameters
    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.lora_keys:
            self.old_B[key].requires_grad = False
            self.old_A[key].requires_grad = False
            self.cur_B[key].requires_grad = False
            self.cur_A[key].requires_grad = False
            if "qkv" in key:
                if self.cur_task > 0:
                    self.optimization_dict[key] += merge_AB(self.old_A[key], self.old_B[key], self.lora_ind[key])
                self.optimization_dict[key] += merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
            else:
                if self.cur_task > 0:
                    self.optimization_dict[key] += self.old_B[key] @ self.old_A[key]
                self.optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        # OptimizerClass = getattr(torch.optim, self.optimizer_str)
        ## self.optimizer = OptimizerClass(self.optimization_dict.values(), lr=self.lr, weight_decay=self.wd_reg)
        # self.optimizer = OptimizerClass(
        #    list(self.cur_B.values()) + list(self.cur_A.values()), lr=self.lr, weight_decay=self.wd_reg
        # )

    def debug_matrices_create(self):
        for key in self.lora_keys:
            self.pre_B[key] = self.cur_B[key].detach().clone()
            self.pre_A[key] = self.cur_A[key].detach().clone()

    def debug_matrices_compare(self):
        num_equal_B = 0
        num_equal_A = 0
        for key in self.lora_keys:
            if torch.allclose(self.pre_B[key], self.cur_B[key]):
                num_equal_B += 1
                print(f"B equal with key {key}")
            if torch.allclose(self.pre_A[key], self.cur_A[key]):
                num_equal_A += 1
                print(f"A equal with key {key}")
        print(f"Number of equal B matrices: {num_equal_B} out of {len(self.lora_keys)}")
        print(f"Number of equal A matrices: {num_equal_A} out of {len(self.lora_keys)}")

    def get_optimization_dict(self):
        optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.lora_keys:
            self.old_B[key].requires_grad = False
            self.old_A[key].requires_grad = False
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            if "qkv" in key:
                if self.cur_task > 0:
                    optimization_dict[key] += merge_AB(self.old_A[key], self.old_B[key], self.lora_ind[key])
                optimization_dict[key] += merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
            else:
                if self.cur_task > 0:
                    optimization_dict[key] += self.old_B[key] @ self.old_A[key]
                optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        for key in self.lora_keys:
            self.old_B[key] = (self.cur_task * self.old_B[key] + self.cur_B[key].detach()) / (self.cur_task + 1)
            self.old_A[key] = (self.cur_task * self.old_A[key] + self.cur_A[key].detach()) / (self.cur_task + 1)
            self.cur_B[key] = nn.Parameter(torch.zeros_like(self.cur_B[key]), requires_grad=True).to(self.device)
            self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
            nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        self.old_A = deepcopy(server_info["old_A"])
        self.old_B = deepcopy(server_info["old_B"])

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(
            list(self.cur_B.values()) + list(self.cur_A.values()), lr=self.lr, weight_decay=self.wd_reg
        )

    def begin_round_server(self):
        return super().begin_round_server()

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict()
        with self.fabric.autocast():
            outputs = functional_call(self.network, optimization_dict, inputs)[
                :, self.cur_offset : self.cur_offset + self.cpt
            ]
            loss = self.loss(outputs, labels % self.cpt)

        if update:
            self.fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(list(self.cur_B.values()) + list(self.cur_A.values()), 1.0)
            self.optimizer.step()
        return loss.item()

    def forward(self, x):
        return functional_call(self.network, self.optimization_dict, x)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
            "num_train_samples": len(dataloader.dataset.data),
        }

    def get_server_info(self):
        return {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
            "old_A": deepcopy(self.old_A),
            "old_B": deepcopy(self.old_B),
        }

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
        cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
        if len(client_info) > 0:
            for key in self.lora_keys:
                self.cur_B[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(0)
                )
                self.cur_A[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(0)
                )
        self.set_optimization()
