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
from utils.tools import str_to_bool, compute_fisher_expectation_fabric

from _models.vera2 import Vera
from _models.lora import Lora, merge_AB, zero_pad
from _models.regmean import RegMean
from _models.lora_regmean2 import LoraRegMean
from _models.vera_regmean2 import VeraRegMean
from tqdm import tqdm
import math


@register_model("vera_regmean_fisher")
class VeraRegMeanFisher(VeraRegMean):
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
        r: int = 768,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        regmean_all: str_to_bool = True,
        alpha_regmean: float = 0.5,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        d_initial: float = 0.1,
        fisher_maxiter: int = -1,
    ) -> None:
        self.fisher_maxiter = fisher_maxiter
        self.lora_head = False
        self.reg_dtype_64 = reg_dtype_64
        self.middle_names = {}  # conversion from state_dict() names to the names of the modules
        VeraRegMean.__init__(
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
            alpha_regmean,
            gram_dtype,
            reg_dtype_64,
            d_initial,
            r,
        )
        # Vera.__init__(
        #    self,
        #    fabric,
        #    network,
        #    device,
        #    optimizer,
        #    lr,
        #    wd_reg,
        #    avg_type,
        #    lora_alpha,
        #    r,
        #    lora_head,
        #    cl_merge,
        #    d_initial,
        #    only_square=r,
        # )
        # RegMean.__init__(
        #    self,
        #    fabric,
        #    network,
        #    device,
        #    optimizer,
        #    lr,
        #    wd_reg,
        #    avg_type,
        #    regmean_all,
        #    alpha_regmean,
        #    gram_dtype,
        #    only_square=r,
        # )
        for name in self.gram_modules:
            self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.fed_weights = {}
        for key in self.lora_keys:
            self.cur_B = self.cur_B.to(self.device)
            self.cur_A = self.cur_A.to(self.device)
            self.fed_weights[key] = torch.zeros_like(self.vera_multiply(key))
        self.classes = []
        self.fisher = {}

    def begin_task(self, n_classes_per_task: int):
        self.classes = []
        super().begin_task(n_classes_per_task)

    def end_round_client(self, dataloader: DataLoader):
        Vera.end_round_client(self, dataloader)
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
            self.vec_d[key].requires_grad = False
        fisher = compute_fisher_expectation_fabric(
            self, dataloader, self.device, self.classes, self.fabric, list(self.vec_b.values()), self.fisher_maxiter
        ).reshape(-1)
        self.fisher = {}
        counter = 0
        for key in self.lora_keys:
            self.fisher[key] = (
                fisher[counter : self.vec_b[key].numel() + counter].reshape(self.vec_b[key].shape).to("cpu")
            )
            counter += self.vec_b[key].numel()
        print(len(self.fisher), self.fisher.keys())

    def get_client_info(self, dataloader: DataLoader):
        client_info = Vera.get_client_info(self, dataloader)
        client_info["grams"] = deepcopy(self.gram)
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        client_info["fisher"] = self.fisher
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
        self.to("cpu")
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        cl_b = [client["vec_b"] for client in client_info]  # list of B matrices for all clients
        cl_d = [client["vec_d"] for client in client_info]  # list of A matrices for all clients
        fishers = [client["fisher"] for client in client_info]
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
                    self.vec_b[key] = self.vec_b[key].to("cpu")
                    self.vec_d[key] = self.vec_d[key].to("cpu")
                    self.vec_b[key] = (
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
                        @ torch.inverse(
                            self.cur_A[0 : self.lora_params_A[key][0], 0 : self.lora_params_A[key][1]].to(dtype)
                        )
                        * (1 / self.vec_d[key].to(dtype)).swapaxes(0, 1)
                        @ torch.inverse(
                            self.cur_B[0 : self.lora_params_B[key][0], 0 : self.lora_params_B[key][1]].to(dtype)
                        )
                    ).to(torch.float32)
                    self.vec_d[key] = (
                        torch.inverse(self.cur_B[0 : self.lora_params_B[key][0], 0 : self.lora_params_B[key][1]]).to(
                            dtype
                        )
                        @ torch.sum(torch.stack([fish[key] for fish in fishers]), dim=0).to(dtype)
                        * (
                            1
                            / torch.sum(
                                torch.stack([fish[key] * vec_b[key] for fish, vec_b in zip(fishers, cl_b)]), dim=0
                            ).squeeze()
                        ).to(dtype)
                        @ torch.sum(
                            torch.stack(
                                [
                                    self.vera_multiply_explicit(
                                        key,
                                        vec_b,
                                        vec_d,
                                        self.cur_B,
                                        self.cur_A,
                                        self.lora_params_B,
                                        self.lora_params_A,
                                    ).to(dtype)
                                    @ client["grams"][name].to(dtype)
                                    for vec_b, vec_d, client in zip(cl_b, cl_d, client_info)
                                ]
                            ),
                            dim=0,
                        ).to(dtype)
                        @ torch.inverse(
                            torch.sum(
                                torch.stack([client_info[i]["grams"][name].to(dtype) for i in range(len(client_info))]),
                                dim=0,
                            )
                        ).to(dtype)
                        @ torch.inverse(
                            self.cur_A[0 : self.lora_params_A[key][0], 0 : self.lora_params_A[key][1]].to(dtype)
                        )
                    ).to(torch.float32)

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
