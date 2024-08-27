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


@register_model("lora_regmean_fisher")
class LoraRegmeanFisher(LoraRegMean):
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
        alpha_regmean_head: float = 0.5,
        alpha_regmean_backbone: float = 0.5,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        fisher_maxiter: int = -1,
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
        )
        self.fisher_maxiter = fisher_maxiter
        self.classes = []
        self.fed_weights = {}
        self.fisher = {}
        for key in self.lora_keys:
            self.fed_weights[key] = torch.zeros_like(self.cur_B[key] @ self.cur_A[key])

    def begin_task(self, n_classes_per_task: int):
        LoraRegMean.begin_task(self, n_classes_per_task)
        self.init_matrices(reverse=False)

    def get_client_info(self, dataloader: DataLoader):
        client_info = Lora.get_client_info(self, dataloader)
        client_info["grams"] = deepcopy(self.gram)
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        client_info["fisher"] = self.fisher
        return client_info

    def get_server_info(self):
        server_info = LoraRegMean.get_server_info(
            self,
        )
        server_info["old_delta"] = deepcopy(self.old_delta)
        server_info["cur_B"] = deepcopy(self.cur_B)
        server_info["cur_A"] = deepcopy(self.cur_A)
        server_info["head"] = deepcopy(self.network.model.head.state_dict())
        return server_info

    def end_round_client(self, dataloader: DataLoader):
        Lora.end_round_client(self, dataloader)
        self.set_optimization()
        RegMean.end_round_client(self, dataloader)  # retrieves Gram matrices from hooks
        self.optimization_dict = self.get_optimization_dict()
        if len(self.classes) == 0:
            classes = set()
            for images, labels in dataloader:
                for label in labels:
                    if label not in self.classes:
                        classes.add(label.item())
            self.classes = list(sorted(classes))
        self.optimizer.zero_grad()
        for key in self.lora_keys:
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = False
        if self.regmean_all:
            fisher = compute_fisher_expectation_fabric(
                self, dataloader, self.device, self.classes, None, list(self.cur_B.values()), self.fisher_maxiter
            ).reshape(-1)
            self.fisher = {}
            counter = 0
            for key in self.lora_keys:
                self.fisher[key] = (
                    fisher[counter : self.cur_B[key].numel() + counter].reshape(self.cur_B[key].shape).to("cpu")
                )
                counter += self.cur_B[key].numel()
        else:
            self.fisher = None

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
            fishers = [client["fisher"] for client in client_info] if self.regmean_all else None
            # regmean will always be applied to the head, optionally to the other layers
            # lora instead will always be applied to the other layers, optionally to the head
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
            # regmean solution
            else:
                self.network.to("cpu")
                dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
                eps = 1e-8
                keys = list(self.network.state_dict().keys())
                # self.fed_weights = {key: None for key in self.lora_keys}
                for key in self.lora_keys:
                    if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                        name = self.middle_names[key]
                        B = torch.stack(
                            [((fish[key].to(dtype) + eps) * B_[key].to(dtype)) for fish, B_ in zip(fishers, cl_B)]
                        ).sum(0) / (torch.stack([fish[key].to(dtype) + eps for fish in fishers]).sum(0))
                        G = torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0)
                        E = torch.stack(
                            [
                                (B_[key].to(dtype) @ A_[key].to(dtype) + self.network.state_dict()[key].to(dtype))
                                @ client["grams"][name].to(dtype)
                                for B_, A_, client in zip(cl_B, cl_A, client_info)
                            ]
                        ).sum(0)
                        A = torch.pinverse(B.T @ B) @ B.T @ E @ torch.pinverse(G)
                        self.cur_B[key] = nn.Parameter(B.detach().clone().to(torch.float32)).to(self.device)
                        self.cur_A[key] = nn.Parameter(A.detach().clone().to(torch.float32)).to(self.device)
                        del B, G, E, A
                        # self.fed_weights[key] = self.fed_weights[key].to("cpu")
                        # self.fed_weights[key] += (
                        #    torch.stack(
                        #        [
                        #            (client_B[key] @ client_A[key]).to(dtype) @ client["grams"][name].to(dtype)
                        #            for client, client_A, client_B in zip(client_info, cl_A, cl_B)
                        #        ]
                        #    ).sum(0)
                        #    @ torch.inverse(torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0))
                        # ).to(torch.float32)
                self.network.to(self.device)
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
            for key in self.lora_keys:
                self.cur_A[key][self.cur_A[key] < -0.25] = 0
                self.cur_A[key][self.cur_A[key] > 0.25] = 0
            self.network.load_state_dict(sd)

            self.set_optimization()

    def set_optimization(self):
        sd = self.network.state_dict()
        opt_dict = self.get_optimization_dict()
        if not self.lora_head:
            for key in self.head_keys:
                opt_dict[key] = sd[key]
        self.optimization_dict = opt_dict
