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

from _models.lora import Lora, merge_AB, zero_pad
from _models.regmean import RegMean
from _models.lora_pre import Lora
from tqdm import tqdm
import math
from transformers import AutoModelForSequenceClassification, T5ForSequenceClassification, T5Model


@register_model("lorm")
class LoRM(Lora, RegMean):
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
        lr_B: float = -1,
        lr_A: float = -1,
        only_square: int = 0,
        train_bias: str = "all",
        train_matrix: str = "alt",
        regmean_rounds: int = 1,
    ) -> None:
        self.is_server = False
        self.regmean_rounds = regmean_rounds
        lr_back = lr_B if lr_B > 0 else lr
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
            lr_back,
            only_square,
            train_bias,
            clip_grad,
            regmean_rounds
        )
        self.middle_names = {}
        for name in self.gram_modules:
            self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        assert train_matrix.lower() in ["alt", "a", "b"]
        self.train_matrix = train_matrix
        if "alt" not in self.train_matrix:
            self.cur_train_matrix = self.train_matrix
        else:
            self.cur_train_matrix = "B"
        self.cur_round = 0
        self.set_train_matrix()
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        lr_back = []
        if lr_B >= 0:
            lr_back.append(lr_B)
        else:
            lr_back.append(lr)
        if lr_A >= 0:
            lr_back.append(lr_A)
        else:
            lr_back.append(lr)
        self.lr_back = lr_back
        del self.old_delta

    def split_backbone_head(self, split=False):
        Bs = []
        As = []
        head_params = []
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            Bs.append(self.cur_B[key])
            As.append(self.cur_A[key])
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
            self.head[key].requires_grad = True
            head_params.append(self.head[key])
        if split:
            return Bs, As, head_params
        else:
            return Bs + As, head_params

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
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = functional_call(self.network, optimization_dict, inputs)[
                :, self.cur_offset : self.cur_offset + self.cpt
            ]
            loss = self.loss(outputs, labels - self.cur_offset)
        if update:
            self.fabric.backward(loss)
            # torch.nn.utils.clip_grad_norm_(list(self.cur_B.values()) + list(self.cur_A.values()), 1.0)
            if self.clip_grad:
                try:
                    self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
                except:
                    pass
            self.optimizer.step()
        return loss.item()

    def begin_task(self, n_classes_per_task: int):
        BaseModel.begin_task(self, n_classes_per_task)
        # server
        if self.cur_task > 0 and self.is_server:
            self.to("cpu")
            for key in self.lora_keys:
                # filler in order to test something meaningful after each comm round (not the last one)
                self.old_delta[key] = (
                    self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                ) / self.cur_task
            self.to(self.device)
            self.init_matrices(freeze_A=False)
        else:
            if self.cur_task > 0:
                self.init_matrices(freeze_A=False)
        self.cur_round = 0

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.train()
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        for key in self.lora_keys:
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
        # self.optimization_dict = deepcopy(server_info["old_delta"])
        self.old_delta = server_info["old_delta"]
        if not self.lora_head:
            if isinstance(self.network.module.model, T5Model):
                self.network.head.load_state_dict(server_info["head"])
            else:
                self.network.model.head.load_state_dict(server_info["head"])
            # for p in self.network.model.head.parameters():
            #    p.requires_grad = True
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }
        self.set_train_matrix()
        Bs, As, head_params = self.split_backbone_head(split=True)
        back = Bs if self.cur_train_matrix == "B" else As
        back_lr = self.lr_back[0] if self.cur_train_matrix == "B" else self.lr_back[1]
        params = [{"params": back, "lr": back_lr}, {"params": head_params}]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        self.cur_round += 1
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)

    def begin_round_server(self):
        if getattr(self, "old_delta", None) is None:
            self.old_delta = {}
            for key in self.lora_keys:
                self.old_delta[key] = torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1])
        self.set_train_matrix()
        self.cur_round += 1

    def end_round_client(self, dataloader: DataLoader):
        # self.network.eval()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        Lora.end_round_client(self, dataloader)
        # setting up the parameters to correctly compute the Gram matrices for the next round
        self.set_optimization()
        for key in self.head_keys:
            self.optimization_dict[key] = self.head[key]
        for name in self.gram_modules:
            self.features[name] = self.features[name].to(self.device)
        RegMean.end_round_client(self, dataloader)  # retrieves Gram matrices from hooks
        self.to("cpu", only_trainable=False)

    def end_round_server(self, client_info: List[dict]):
        # self.network.eval()
        self.is_server = True
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
            # import gc
            dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
            # from time import time
            # start = time()
            total_mem = torch.cuda.get_device_properties(0).total_memory
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
                # eps = 5e-7
                keys = list(self.network.state_dict().keys())
                if self.cur_train_matrix == "A":
                    # merge As, Bs are all the same
                    cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
                    for key in self.lora_keys:
                        if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                            name = self.middle_names[key]
                            # print(key)
                            for i in range(len(cl_A)):
                                cl_A[i][key] = cl_A[i][key].to(self.device).to(dtype)
                                client_info[i]["grams"][name] = client_info[i]["grams"][name].to(self.device).to(dtype)
                            B = self.cur_B[key].to(self.device).to(dtype)
                            E = torch.stack(
                                [
                                    (A_[key].to(self.device).to(dtype))
                                    @ client["grams"][name].to(self.device).to(dtype)
                                    for A_, client in zip(cl_A, client_info)
                                ]
                            ).sum(0)
                            G = torch.stack(
                                [client["grams"][name].to(self.device).to(dtype) for client in client_info]
                            ).sum(0)
                            A = E @ torch.pinverse(G)  # A
                            self.cur_A[key] = A.to(torch.float32).to("cpu")
                            for i in range(len(cl_A)):
                                cl_A[i][key] = cl_A[i][key].to("cpu")
                                client_info[i]["grams"][name] = client_info[i]["grams"][name].to("cpu")
                            del E, G, A, B
                            if torch.cuda.memory_reserved() / total_mem > 0.8:
                                torch.cuda.empty_cache()
                            # gc.collect()
                        else:
                            self.cur_A[key] = nn.Parameter(
                                torch.stack(
                                    [client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]
                                ).sum(0)
                            )

                else:
                    # merge Bs, As are all the same
                    print("Merging Bs")
                    cl_B = [client["cur_B"] for client in client_info]  # list of A matrices for all clients
                    for key in self.lora_keys:
                        if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                            name = self.middle_names[key]
                            # print(key)
                            for i in range(len(cl_B)):
                                cl_B[i][key] = cl_B[i][key].to(self.device).to(dtype)
                                client_info[i]["grams"][name] = client_info[i]["grams"][name].to(self.device).to(dtype)
                            A = self.cur_A[key].to(self.device).to(dtype)
                            E2 = torch.stack(
                                [
                                    (B_[key].to(self.device).to(dtype) @ A)
                                    @ client["grams"][name].to(self.device).to(dtype)
                                    for B_, client in zip(cl_B, client_info)
                                ]
                            ).sum(0)
                            G_inv = torch.pinverse(
                                A
                                @ torch.stack(
                                    [client["grams"][name].to(self.device).to(dtype) for client in client_info]
                                ).sum(0)
                                @ A.T
                            )
                            B = E2 @ A.T @ G_inv  # B
                            self.cur_B[key] = B.to(torch.float32).to("cpu")
                            for i in range(len(cl_B)):
                                cl_B[i][key] = cl_B[i][key].to("cpu")
                                client_info[i]["grams"][name] = client_info[i]["grams"][name].to("cpu")
                            del E2, G_inv, A, B
                            # gc.collect()
                            # print(f"reserved memory: {torch.cuda.memory_reserved()}\tallocated memory: {torch.cuda.memory_allocated()}")
                            if torch.cuda.memory_reserved() / total_mem > 0.8:
                                torch.cuda.empty_cache()
                            #    print(f"reserved memory: {torch.cuda.memory_reserved()}\tallocated memory: {torch.cuda.memory_allocated()}")
                        else:
                            self.cur_B[key] = nn.Parameter(
                                torch.stack(
                                    [client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]
                                ).sum(0)
                            )
                    # bt btb eg
                    # eg ata at
            # end = time()
            torch.cuda.empty_cache()
            # print(f"Time for merging: {end - start} seconds")
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
            # end2 = time()
            torch.cuda.empty_cache()
            del cl_B, cl_A, client_info
            # print(f"Time for head: {end2 - end} seconds")
            self.network.load_state_dict(sd)
            if getattr(self, "old_delta", None) is None:
                self.old_delta = {}
                for key in self.lora_keys:
                    self.old_delta[key] = torch.zeros(
                        self.cur_B[key].shape[0], self.cur_A[key].shape[1], requires_grad=False
                    )
            self.detach()
            self.set_optimization()
            for key in self.head_keys:
                self.optimization_dict[key] = self.network.state_dict()[key]
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            self.to(self.device)
            # end3 = time()
            # print(f"Time fir the rest: {end3 - end2} seconds")

    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        client_info = {
            "cur_A": self.cur_A,
            "cur_B": self.cur_B,
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            if isinstance(self.network.module.model, T5Model):
                client_info["head"] = self.network.head.state_dict()
            else:
                client_info["head"] = self.network.model.head.state_dict()
        client_info["grams"] = self.features
        client_info["state_dict"] = self.network.state_dict()
        return client_info

    def end_task_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        if isinstance(self.network.module.model, T5Model):
            self.network.head.load_state_dict(server_info["head"])
        else:
            self.network.model.head.load_state_dict(server_info["head"])
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)
        self.set_optimization()
        gram_modules = [mod for mod in self.gram_modules if "head" not in mod]
        real_modules = deepcopy(self.gram_modules)
        self.gram_modules = gram_modules
        RegMean.end_round_client(self, dataloader)
        self.gram_modules = real_modules
        self.to("cpu", only_trainable=False)
        return {"grams": self.features}

    def end_task_server(self, client_info: List[dict] = None):
        with torch.no_grad():
            gram_modules = [mod for mod in self.gram_modules if "head" not in mod]
            grams = {}
            for module in gram_modules:
                grams[module] = torch.stack([client_info[i]["grams"][module] for i in range(len(client_info))]).sum(0)
            optimization_dict = deepcopy(self.network.state_dict())
            if getattr(self, "run_weights_gram", None) is None:
                self.run_weights_gram = {
                    key: None
                    for key in self.lora_keys
                    if "weight" in key and self.middle_names.get(key) is not None and not "head" in key
                }
            if getattr(self, "run_gram", None) is None:
                self.run_gram = {
                    key: None
                    for key in self.lora_keys
                    if "weight" in key and self.middle_names.get(key) is not None and not "head" in key
                }
            for key in self.lora_keys:
                if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                    B = self.cur_B[key].to(self.device)
                    A = self.cur_A[key].to(self.device)
                    gram = grams[self.middle_names[key]].to(self.device)
                    if self.run_weights_gram.get(key) is None:
                        self.run_weights_gram[key] = B @ A @ gram
                    else:
                        self.run_weights_gram[key] = self.run_weights_gram[key].to(self.device)
                        self.run_weights_gram[key] += B @ A @ gram
                    if self.run_gram.get(key) is None:
                        self.run_gram[key] = gram
                    else:
                        self.run_gram[key] = self.run_gram[key].to(self.device)
                        self.run_gram[key] += gram
                    if self.cur_task > 0:
                        optimization_dict[key] += self.run_weights_gram[key] @ torch.pinverse(self.run_gram[key])
            if self.cur_task > 0:
                self.optimization_dict = optimization_dict

    def set_optimization_cur_task(self, fabric=True):
        self.detach()
        self.to(self.device)
        sd = self.network.state_dict()
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        for key in self.lora_keys:
            if self.cur_task > 0 and not "individual" in self.cl_merge and not "fisher" in self.cl_merge:
                self.old_delta[key] = self.old_delta[key].to(self.device)
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        self.optimization_dict = optimization_dict

    def set_optimization(self, fabric=True):
        with torch.no_grad():
            self.optimization_dict = deepcopy(dict(self.network.state_dict()))
            for key in self.optimization_dict.keys():
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            for key in self.lora_keys:
                self.cur_B[key] = self.cur_B[key].detach()
                self.cur_A[key] = self.cur_A[key].detach()
                self.cur_B[key] = self.cur_B[key].to(self.device)
                self.cur_A[key] = self.cur_A[key].to(self.device)
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            for key in self.lora_keys:
                if self.cur_task > 0:
                    self.old_delta[key] = self.old_delta[key].to(self.device)
                    tmp = (self.old_delta[key] * self.cur_task) + (self.cur_B[key] @ self.cur_A[key])
                    self.optimization_dict[key] += tmp / (self.cur_task + 1)
                else:
                    self.optimization_dict[key] += (self.cur_B[key] @ self.cur_A[key]).detach()

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
            for key in self.head_keys:
                self.head[key] = self.head[key].to(device)
            for key in self.gram_modules:
                self.features[key] = self.features[key].to(device)
            if getattr(self, "old_delta", None) is not None:
                for key in self.lora_keys:
                    self.old_delta[key] = self.old_delta[key].to(device)
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
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
        if getattr(self, "old_delta", None) is not None:
            for key in self.lora_keys:
                self.old_delta[key] = self.old_delta[key].detach()
