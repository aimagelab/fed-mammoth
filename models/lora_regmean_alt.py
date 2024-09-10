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
        clip_grad: str_to_bool = False,
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
        fisher_maxiter: int = -1,
    ) -> None:
        self.fisher_maxiter = fisher_maxiter
        LoraRegMean.__init__(
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
        # self.old_delta_fisher = deepcopy(self.old_delta)
        # self.old_fisher = deepcopy(self.old_delta_fisher)
        # self.cur_fisher = deepcopy(self.old_delta_fisher)

    def get_optimization_dict(self, fabric=True):
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key].requires_grad = True
                optimization_dict[key] = self.head[key]
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            if self.cur_task > 0 and not "individual" in self.cl_merge and not "fisher" in self.cl_merge:
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def begin_task(self, n_classes_per_task: int):
        BaseModel.begin_task(self, n_classes_per_task)
        if "fisher" in self.cl_merge and getattr(self, "old_delta_fisher", None) is not None:
            self.to("cpu")
            for key in self.lora_keys:
                self.old_delta_fisher[key] = self.old_delta_fisher[key] + (
                    (self.cur_B[key] @ self.cur_A[key]) * self.cur_fisher[key]
                )
                self.old_fisher[key] += self.cur_fisher[key]
                # filler in order to test something meaningful after each comm round (not the last one)
                self.old_delta[key] = (
                    self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                ) / self.cur_task
            self.to(self.device)
            self.init_matrices()
        else:
            if self.cur_task > 0:
                for key in self.lora_keys:
                    self.old_delta[key] = (
                        self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                    ) / self.cur_task
                self.init_matrices()
        self.cur_round = 0

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        LoraRegMean.begin_round_client(self, dataloader, server_info)
        self.fed_weights = {key: torch.zeros_like(self.fed_weights[key]) for key in self.lora_keys}
        self.set_train_matrix()
        self.cur_round += 1

    def begin_round_server(self):
        self.set_train_matrix()
        self.cur_round += 1

    def end_round_client(self, dataloader: DataLoader):
        LoraRegMean.end_round_client(self, dataloader)
        self.to("cpu", only_trainable=False)

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
            if not self.regmean_all:
                # fedavg on Lora matrices for all layers except head
                for key in self.lora_keys:
                    self.cur_B[key] = nn.Parameter(
                        torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(
                            0
                        )
                    )
                    self.cur_A[key] = nn.Parameter(
                        torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(
                            0
                        )
                    )
            else:
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
                            @ torch.inverse(
                                torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0)
                            )
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
            self.detach()
            self.set_optimization()
            self.to(self.device)
    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        client_info = {
            "cur_A": self.cur_A,
            "cur_B": self.cur_B,
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            client_info["head"] = self.network.model.head.state_dict()
        client_info["grams"] = self.features
        client_info["state_dict"] = self.network.state_dict()
        return client_info

    def end_task_client(self, dataloader: DataLoader = None):
        fisher = None
        if "fisher" in self.cl_merge:
            classes = set()
            num_samples = 0
            for images, labels in dataloader:
                num_samples += images.shape[0]
                for label in labels:
                    if label not in classes:
                        classes.add(label.item())
            classes = list(sorted(classes))
            self.optimizer.zero_grad()
            if self.regmean_all:
                precision = torch.get_float32_matmul_precision()
                # torch.set_float32_matmul_precision("high")
                # self.detach()
                # self.set_optimization_cur_task(fabric=False)
                self.detach()
                # for key in self.lora_keys:
                #    self.cur_B[key].requires_grad = True
                #    self.cur_A[key].requires_grad = True
                merged_params = {
                    # key: self.network.state_dict()[key] + (self.cur_B[key] @ self.cur_A[key]) for key in self.lora_keys
                    key: torch.tensor(self.cur_B[key] @ self.cur_A[key], requires_grad=True)
                    for key in self.lora_keys
                }
                # adding lora parameters on the network (using .module to get rid of fabric wrapper)
                self.optimization_dict = deepcopy(dict(self.network.module.state_dict()))
                for key in self.lora_keys:
                    self.optimization_dict[key] += merged_params[key]
                torch.cuda.empty_cache()
                # self.set_optimization_cur_task(fabric=False)
                fisher = compute_fisher_expectation_fabric(
                    network=self,
                    data_loader=dataloader,
                    device=self.device,
                    classes=classes,
                    fabric=None,
                    parameters=list(merged_params.values()),
                    maxiter=self.fisher_maxiter,
                ).reshape(-1)
                # self.set_optimization_cur_task(fabric=True)
                torch.set_float32_matmul_precision(precision)
                fisher = fisher.to("cpu")
            return {"fisher": fisher, "num_samples": num_samples}

    def end_task_server(self, client_info: List[dict] = None):
        with torch.no_grad():
            if "fisher" in self.cl_merge:
                try:
                    getattr(self, "old_delta_fisher")
                except AttributeError:
                    self.old_delta_fisher = {
                        key: torch.zeros_like(self.old_delta[key], requires_grad=False) for key in self.lora_keys
                    }
                try:
                    getattr(self, "old_fisher")
                except AttributeError:
                    self.old_fisher = {
                        key: torch.zeros_like(self.old_delta[key], requires_grad=False) for key in self.lora_keys
                    }
                fishers = torch.stack([client_info[i]["fisher"] for i in range(len(client_info))])
                num_samples = torch.tensor([client_info[i]["num_samples"] for i in range(len(client_info))]).reshape(
                    -1, 1
                )
                eps = 1e-12
                avg_fisher = (fishers * num_samples).sum(0) / num_samples.sum()
                # avg_fisher = avg_fisher.to(self.device)
                del fishers
                self.to("cpu")
                torch.cuda.empty_cache()
                fisher_dict = {}
                counter = 0
                for key in self.lora_keys:
                    merged_params = self.cur_B[key] @ self.cur_A[key]
                    fisher_dict[key] = avg_fisher[counter : merged_params.numel() + counter].reshape(
                        merged_params.shape
                    )
                    counter += merged_params.numel()
                with torch.no_grad():
                    self.optimization_dict = deepcopy(dict(self.network.state_dict()))
                    # for key in self.optimization_dict.keys():
                    #    self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                    for key in self.lora_keys:
                        self.old_delta_fisher[key].requires_grad = False
                        self.cur_B[key].requires_grad = False
                        self.cur_A[key].requires_grad = False
                        self.fed_weights[key].requires_grad = False
                        # self.fed_weights[key] = self.fed_weights[key].to(self.device)
                        # self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                        # self.old_delta_fisher[key] = self.old_delta[key].to(self.device)
                        # self.old_fisher[key] = self.old_fisher[key].to(self.device)
                        if self.cur_task > 0:
                            tmp = self.old_delta_fisher[key] + (self.fed_weights[key].detach() * fisher_dict[key]) + eps
                            self.optimization_dict[key] += tmp / (self.old_fisher[key] + fisher_dict[key] + eps)
                        else:
                            self.optimization_dict[key] += self.fed_weights[key].detach()
                    for key in self.network.state_dict().keys():
                        self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                try:
                    getattr(self, "cur_fisher")
                except AttributeError:
                    self.cur_fisher = {key: fisher_dict[key] for key in self.lora_keys}
                else:
                    for key in self.lora_keys:
                        self.cur_fisher[key] = fisher_dict[key]
                del fisher_dict
            # self.to("cpu")

    def set_optimization_cur_task(self, fabric=True):
        self.detach()
        self.to(self.device)
        sd = self.network.state_dict()
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].to(self.device)
            if self.cur_task > 0 and not "individual" in self.cl_merge and not "fisher" in self.cl_merge:
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        self.optimization_dict = optimization_dict

    def set_optimization(self, fabric=True):
        with torch.no_grad():
            self.optimization_dict = deepcopy(dict(self.network.state_dict()))
            for key in self.optimization_dict.keys():
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            for key in self.lora_keys:
                self.old_delta[key] = self.old_delta[key].detach()
                self.cur_B[key] = self.cur_B[key].detach()
                self.cur_A[key] = self.cur_A[key].detach()
                self.fed_weights[key] = self.fed_weights[key].detach()
                self.fed_weights[key] = self.fed_weights[key].to(self.device)
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                self.old_delta[key] = self.old_delta[key].to(self.device)
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
            elif "individual" in self.cl_merge or "fisher" in self.cl_merge:
                if "sum" in self.cl_merge:
                    for key in self.lora_keys:
                        if self.cur_task > 0:
                            tmp = (self.old_delta[key] * self.cur_task) + self.fed_weights[key]
                            self.optimization_dict[key] += tmp
                        else:
                            self.optimization_dict[key] += self.fed_weights[key]
                elif "mean" in self.cl_merge or "fisher" in self.cl_merge:
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

    def to(self, device="cpu", only_trainable=True):
        if "cpu" in device or not only_trainable:  # we move everything to the cpu
            self.network = self.network.to(device)
            for key in self.lora_keys:
                self.cur_A[key] = self.cur_A[key].to(device)
                self.cur_B[key] = self.cur_B[key].to(device)
                self.fed_weights[key] = self.fed_weights[key].to(device)
                self.old_delta[key] = self.old_delta[key].to(device)
            for key in self.head_keys:
                self.head[key] = self.head[key].to(device)
            for key in self.gram_modules:
                self.features[key] = self.features[key].to(device)
            if getattr(self, "old_delta_fisher", None) is not None:
                for key in self.lora_keys:
                    self.old_delta_fisher[key] = self.old_delta_fisher[key].to(device)
            if getattr(self, "old_fisher", None) is not None:
                for key in self.lora_keys:
                    self.old_fisher[key] = self.old_fisher[key].to(device)
            if getattr(self, "cur_fisher", None) is not None:
                for key in self.lora_keys:
                    self.cur_fisher[key] = self.cur_fisher[key].to(device)
        else:  # we move only the trainable parameters to the device
            self.network = self.network.to(device)
            for key in self.lora_keys:
                self.cur_A[key] = self.cur_A[key].to(device)
                self.cur_B[key] = self.cur_B[key].to(device)
        return self

    def detach(self):
        for key in self.lora_keys:
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.old_delta[key] = self.old_delta[key].detach()
            self.fed_weights[key] = self.fed_weights[key].detach()
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
        if getattr(self, "old_delta_fisher", None) is not None:
            for key in self.lora_keys:
                self.old_delta_fisher[key] = self.old_delta_fisher[key].detach()
        if getattr(self, "old_fisher", None) is not None:
            for key in self.lora_keys:
                self.old_fisher[key] = self.old_fisher[key].detach()
        if getattr(self, "cur_fisher", None) is not None:
            for key in self.lora_keys:
                self.cur_fisher[key] = self.cur_fisher[key].detach()
