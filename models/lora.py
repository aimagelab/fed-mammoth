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
        enable_lora: list = [True, True, True],
        lora_head: str_to_bool = True,
        merging: str = "run_sum",
    ) -> None:
        # for LoRA, we keep the mean of the LoRA modules of the old tasks
        super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.lora_alpha = lora_alpha
        self.r = r
        self.enable_lora = enable_lora
        self.merging = merging
        self.lora_keys = []
        self.lora_params = {}
        self.lora_ind = {}
        self.optimizer_str = optimizer
        self.lr = lr
        self.wd_reg = wd_reg
        self.lora_head = lora_head
        self.head_keys = []
        self.old_delta = {}
        self.cur_B = {}
        self.cur_A = {}
        for name, param in network.named_parameters():
            param.requires_grad = False  # freeze all the parameters
            if not self.lora_head and "head" in name:
                self.head_keys.append(name)
            if "qkv" in name and "weight" in name:
                self.lora_keys.append(name)
                self.lora_ind[name] = torch.zeros((param.shape[0],), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[name][enable_lora, :] = True
                self.lora_ind[name] = self.lora_ind[name].view(-1)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_delta[name] = nn.Parameter(
                    torch.zeros(param.shape[0], param.shape[1]), requires_grad=False
                ).to(self.device)
                self.cur_B[name] = nn.Parameter(torch.zeros(param.shape[0], r), requires_grad=True).to(self.device)
                self.cur_A[name] = nn.Parameter(torch.zeros(r * 3, param.shape[1]), requires_grad=True).to(self.device)
                nn.init.kaiming_uniform_(self.cur_A[name], a=math.sqrt(5))
            elif (
                ("mlp" in name and "weight" in name)
                or ("proj" in name and "weight" in name and "attn" in name)
                or (self.lora_head and "head" in name and "weight" in name)
            ):
                self.lora_keys.append(name)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_delta[name] = nn.Parameter(
                    torch.zeros(param.shape[0], param.shape[1]), requires_grad=False
                ).to(self.device)
                self.cur_B[name] = nn.Parameter(torch.zeros(param.shape[0], r), requires_grad=True).to(self.device)
                self.cur_A[name] = nn.Parameter(torch.zeros(r, param.shape[1]), requires_grad=True).to(self.device)
                nn.init.kaiming_uniform_(self.cur_A[name], a=math.sqrt(5))
        self.optimization_dict = {}
        self.old_tasks_A = None
        self.old_tasks_B = None
        if self.merging == "individual":
            self.old_tasks_A = {}
            self.old_tasks_B = {}
        # self.network = network
        # self.set_optimization()
        self.avg_type = avg_type
        self.pre_B = {}
        self.pre_A = {}
        self.pre_network = None

    # used for testing, using a functional_call() to call the network with self.optimization_dict parameters
    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        if "run_sum" in self.merging:
            for key in self.lora_keys:
                self.old_delta[key].requires_grad = False
                self.cur_B[key].requires_grad = False
                self.cur_A[key].requires_grad = False
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                if "qkv" in key:
                    self.lora_ind[key] = self.lora_ind[key].to(self.device)
                    if self.cur_task > 0:
                        self.optimization_dict[key] += self.old_delta[key]
                    self.optimization_dict[key] += merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
                else:
                    if self.cur_task > 0:
                        self.optimization_dict[key] += self.old_delta[key]
                    self.optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        else:
            for key in self.lora_keys:
                self.old_delta[key].requires_grad = False
                self.cur_B[key].requires_grad = False
                self.cur_A[key].requires_grad = False
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                if "qkv" in key:
                    self.lora_ind[key] = self.lora_ind[key].to(self.device)
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
                else:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.cur_B[key] @ self.cur_A[key]
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        

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
        if not self.lora_head:
            for key in self.head_keys:
                optimization_dict[key].requires_grad = True
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            if "qkv" in key:
                if self.cur_task > 0 and not "individual" in self.merging:
                    optimization_dict[key] += self.old_delta[key]
                optimization_dict[key] += merge_AB(self.cur_A[key], self.cur_B[key], self.lora_ind[key])
            else:
                if self.cur_task > 0 and not "individual" in self.merging:
                    optimization_dict[key] += self.old_delta[key]
                optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def get_dummy_optimization_dict(self):
        return deepcopy(dict(self.network.named_parameters()))

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.cur_task > 0:
            if self.merging == "run_sum":
                for key in self.lora_keys:
                    if "qkv" in key:
                        self.old_delta[key] += merge_AB(
                            self.cur_A[key].detach(), self.cur_B[key].detach(), self.lora_ind[key]
                        )
                    else:
                        self.old_delta[key] += self.cur_B[key].detach() @ self.cur_A[key].detach()
                    self.cur_B[key] = nn.Parameter(torch.zeros_like(self.cur_B[key]), requires_grad=True).to(self.device)
                    self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
                    nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))
            elif self.merging == "run_mean" or self.merging == "individual":
                for key in self.lora_keys:
                    if "qkv" in key:
                        self.old_delta[key] = (
                            self.old_delta[key] * (self.cur_task - 1)
                            + merge_AB(self.cur_A[key].detach(), self.cur_B[key].detach(), self.lora_ind[key])
                        ) / self.cur_task
                    else:
                        self.old_delta[key] = (
                            self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                        ) / self.cur_task
                    self.cur_B[key] = nn.Parameter(torch.zeros_like(self.cur_B[key]), requires_grad=True).to(self.device)
                    self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
                    nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))
            
            else:
                raise ValueError("Invalid merging type")

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        self.old_delta = deepcopy(server_info["old_delta"])
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            for p in self.network.model.head.parameters():
                p.requires_grad = True

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.network.model.head.parameters()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()), lr=self.lr, weight_decay=self.wd_reg
            )
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def begin_round_server(self):
        return super().begin_round_server()

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict()
        # optimization_dict = self.get_dummy_optimization_dict()
        with self.fabric.autocast():
            outputs = functional_call(self.network, optimization_dict, inputs)[
                :, self.cur_offset : self.cur_offset + self.cpt
            ]
            loss = self.loss(outputs, labels % self.cpt)

        if update:
            self.fabric.backward(loss)
            # torch.nn.utils.clip_grad_norm_(list(self.cur_B.values()) + list(self.cur_A.values()), 1.0)
            self.optimizer.step()
        return loss.item()

    def forward(self, x):
        return functional_call(self.network, self.optimization_dict, x)

    def get_client_info(self, dataloader: DataLoader):
        client_info = {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            client_info["head"] = deepcopy(self.network.model.head.state_dict())
        return client_info

    def get_server_info(self):
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        server_info = {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
            "old_delta": deepcopy(self.old_delta),
        }
        if not self.lora_head:
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
        if not self.lora_head:
            heads = [client["head"] for client in client_info]  # list of head matrices for all clients
            head_sd = self.network.model.head.state_dict()
            for key in head_sd.keys():
                head_sd[key] = torch.stack(
                    [head[key] * norm_weight for head, norm_weight in zip(heads, norm_weights)]
                ).sum(0)
            self.network.model.head.load_state_dict(head_sd)

        if len(client_info) > 0:
            for key in self.lora_keys:
                self.cur_B[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(0)
                )
                self.cur_A[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(0)
                )

        self.set_optimization()
