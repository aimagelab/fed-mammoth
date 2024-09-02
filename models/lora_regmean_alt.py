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
from utils.tools import str_to_bool, compute_fisher_expectation_fabric

from models.lora import Lora, merge_AB, zero_pad
from models.regmean import RegMean
from models.lora_regmean2 import LoraRegMean
from tqdm import tqdm
import math


@register_model("lora_regmean_alt")
class LoraRegMeanAlt(LoraRegMean):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 0.0003,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1,
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
        train_matrix: str = "alt",
    ) -> None:
        LoraRegMean.__init__(
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
            regmean_all,
            alpha_regmean_head,
            alpha_regmean_backbone,
            gram_dtype,
            reg_dtype_64,
            slca,
            only_square,
            train_bias,
        )
        assert train_matrix.lower() in ["alt", "a", "b"]
        self.train_matrix = train_matrix
        if "alt" not in self.train_matrix:
            self.cur_train_matrix = self.train_matrix
        else:
            self.cur_train_matrix = "B"
        self.cur_round = 0
        self.set_train_matrix()

    def begin_task(self, n_classes_per_task: int):
        LoraRegMean.begin_task(self, n_classes_per_task)
        self.cur_round = 0

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        LoraRegMean.begin_round_client(self, dataloader, server_info)
        self.set_train_matrix()
        self.cur_round += 1

    def begin_round_server(self):
        self.set_train_matrix()
        self.cur_round += 1

    def end_round_server(self, client_info: List[dict]):
        with torch.no_grad():
            if self.avg_type == "weighted":
                total_samples = sum([client["num_train_samples"] for client in client_info])
                norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            else:
                weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
                norm_weights = [w / sum(weights) for w in weights]
            cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
            cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
            self.to("cpu")
            dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
            eps = 5e-7
            keys = list(self.network.state_dict().keys())
            if self.cur_train_matrix == "A":
                # merge As
                cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
                for key in self.lora_keys:
                    if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                        name = self.middle_names[key]
                    B = self.cur_B[key].to("cpu").to(dtype)
                    E = torch.stack(
                        [
                            (B_[key].to(dtype) @ A_[key].to(dtype)) @ client["grams"][name].to(dtype)
                            for B_, A_, client in zip(cl_B, cl_A, client_info)
                        ]
                    ).sum(0)
                    G = torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0)
                    A = torch.pinverse(B.T @ B) @ B.T @ E @ torch.pinverse(G)  # A
                    self.cur_A[key] = A.to(torch.float32)
            else:
                # merge Bs
                cl_B = [client["cur_B"] for client in client_info]  # list of A matrices for all clients
                for key in self.lora_keys:
                    if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                        name = self.middle_names[key]
                    A = self.cur_A[key].to("cpu").to(dtype)
                    E2 = torch.stack(
                        [
                            (B_[key].to(dtype) @ A) @ client["grams"][name].to(dtype)
                            for B_, client in zip(cl_B, client_info)
                        ]
                    ).sum(0)
                    G_inv = torch.pinverse(
                        A @ torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0) @ A.T
                    )
                    B = E2 @ A.T @ G_inv  # B
                    self.cur_B[key] = B.to(torch.float32)
                # bt btb eg
                # eg ata at

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
        for key in self.lora_keys:
            self.fed_weights[key] = self.cur_B[key] @ self.cur_A[key]
        self.set_optimization()
        self.to(self.device)

    def set_optimization(self, fabric=True):
        with torch.no_grad():
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

    def set_train_matrix(self):
        if "alt" in self.train_matrix:
            if "A" in self.cur_train_matrix or self.cur_round == 0:
                # Train B
                for key in self.lora_keys:
                    self.cur_A[key] = self.cur_A[key].detach()
                    if not self.cur_B[key].requires_grad:
                        self.cur_B[key].requires_grad = True
                self.cur_train_matrix = "B"
            else:
                # Train A
                for key in self.lora_keys:
                    if not self.cur_A[key].requires_grad:
                        self.cur_A[key].requires_grad = True
                    self.cur_B[key] = self.cur_B[key].detach()
                self.cur_train_matrix = "A"
        else:
            if "A" in self.train_matrix:
                for key in self.lora_keys:
                    if not self.cur_A[key].requires_grad:
                        self.cur_A[key].requires_grad = True
                    self.cur_B[key] = self.cur_B[key].detach()
            else:
                for key in self.lora_keys:
                    self.cur_A[key] = self.cur_A[key].detach()
                    if not self.cur_B[key].requires_grad:
                        self.cur_B[key].requires_grad = True
