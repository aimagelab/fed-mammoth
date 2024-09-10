import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models import register_model
from typing import List, Union
from models.utils import BaseModel
from networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool
from models.lora import Lora
from torch.nn.init import _calculate_correct_fan

import math
from typing import Optional, List


def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


@register_model("vera")
class Vera(Lora):
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
        d_initial: float = 0.1,
    ):
        super(Vera, self).__init__(
            fabric, network, device, optimizer, lr, wd_reg, avg_type, lora_alpha, r, lora_head, cl_merge
        )
        self.d_initial = d_initial
        self.vec_b = {}
        self.vec_d = {}
        self.pre_b = {}
        self.pre_d = {}
        for key in self.lora_keys:
            d = self.cur_B[key].shape[0]
            generator = torch.Generator().manual_seed(0)
            self.cur_B[key] = _kaiming_init(
                (self.cur_B[key].shape[0], self.cur_B[key].shape[1]), generator=generator
            ).to(self.device)
            self.cur_A[key] = _kaiming_init(
                (self.cur_A[key].shape[0], self.cur_A[key].shape[1]), generator=generator
            ).to(self.device)
            self.vec_b[key] = nn.Parameter(torch.zeros(d, 1), requires_grad=True).to(self.device)
            self.vec_d[key] = nn.Parameter(torch.ones(r, 1) * self.d_initial, requires_grad=True).to(self.device)
        # for key in self.network.state_dict().keys():
        #    # print(key, self.cur_B[key].shape, self.cur_A[key].shape, self.vec_b[key].shape, self.vec_d[key].shape)
        #    print(key, self.network.state_dict()[key].shape)

    def debug_matrices_create(self):
        for key in self.lora_keys:
            self.pre_b[key] = self.vec_b[key].detach().clone()
            self.pre_d[key] = self.vec_d[key].detach().clone()
        self.pre_head = deepcopy(self.head.state_dict())

    def debug_matrices_compare(self):
        num_equal_b = 0
        num_equal_d = 0
        for key in self.lora_keys:
            if torch.allclose(self.pre_b[key], self.vec_b[key]):
                num_equal_b += 1
                print(f"b equal with key {key}")
            if torch.allclose(self.pre_d[key], self.vec_d[key]):
                num_equal_d += 1
                print(f"d equal with key {key}")
        print(f"Number of equal b vectors: {num_equal_b} out of {len(self.lora_keys)}")
        print(f"Number of equal d vectors: {num_equal_d} out of {len(self.lora_keys)}")
        if not self.lora_head:
            num_equal_head = 0
            for key in self.pre_head.keys():
                if torch.allclose(self.pre_head[key], self.head.state_dict()[key]):
                    num_equal_head += 1
                    print(f"Head equal with key {key}")
                else:
                    print(
                        f"Head not equal with key {key}, distance: {torch.dist(self.pre_head[key], self.network.model.head.state_dict()[key])}"
                    )
            print(f"Number of equal head matrices: {num_equal_head} out of {len(self.head_keys)}")

    def get_optimization_dict(self):
        optimization_dict = deepcopy(dict(self.network.state_dict()))
        if not self.lora_head:
            for key in self.head_keys:
                # self.head[key].requires_grad = True
                optimization_dict[key] = self.head[key]
        for key in self.lora_keys:
            if self.cur_task > 0 and not "individual" in self.cl_merge:
                optimization_dict[key] += self.old_delta[key]
            # TODO: correct all instances of this by avoiding using torch.eye
            optimization_dict[key] += self.vera_mult(key)
        return optimization_dict

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict()
        # optimization_dict = super().get_optimization_dict()
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

    def vera_mult(self, key):
        return (self.vec_b[key] * self.cur_B[key]) @ (self.vec_d[key] * self.cur_A[key])

    def begin_task(self, n_classes_per_task: int):
        # BaseModel.begin_task(self, n_classes_per_task)
        self.cur_task += 1
        self.cpt = n_classes_per_task
        self.cur_offset = self.cur_task * self.cpt
        # adjust grads and move to device
        self.detach()
        self.to(self.device)
        # vera begin task
        if self.cur_task > 0:
            if self.cl_merge == "run_sum":
                for key in self.lora_keys:
                    self.old_delta[key] += self.vera_mult(key)
            elif self.cl_merge == "run_mean" or "individual" in self.cl_merge:
                for key in self.lora_keys:
                    eye_b = torch.eye(self.vec_b[key].shape[0], device=self.device, requires_grad=False)
                    eye_d = torch.eye(self.vec_d[key].shape[0], device=self.device, requires_grad=False)
                    self.old_delta[key] = (
                        self.old_delta[key] * (self.cur_task - 1) + self.vera_mult(key)
                    ) / self.cur_task
            else:
                raise ValueError("Invalid cl_merge type")
            for key in self.lora_keys:
                generator = torch.Generator().manual_seed(0)
                self.cur_B[key] = _kaiming_init(
                    (self.cur_B[key].shape[0], self.cur_B[key].shape[1]), generator=generator
                ).to(self.device)
                self.cur_A[key] = _kaiming_init(
                    (self.cur_A[key].shape[0], self.cur_A[key].shape[1]), generator=generator
                ).to(self.device)
                self.vec_b[key] = nn.Parameter(torch.zeros_like(self.vec_b[key]), requires_grad=True).to(self.device)
                self.vec_d[key] = nn.Parameter(
                    torch.ones_like(self.vec_d[key]) * self.d_initial, requires_grad=True
                ).to(self.device)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        self.vec_b = deepcopy(server_info["vec_b"])
        self.vec_d = deepcopy(server_info["vec_d"])
        self.old_delta = deepcopy(server_info["old_delta"])
        self.detach(only_lora=True)
        for key in self.lora_keys:
            self.vec_b[key].requires_grad = True
            self.vec_d[key].requires_grad = True
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.vec_b.values()) + list(self.vec_d.values()) + list(self.head.values()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(
                list(self.vec_b.values()) + list(self.vec_d.values()), lr=self.lr, weight_decay=self.wd_reg
            )
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.vec_b[key] = self.vec_b[key].detach()
            self.vec_d[key] = self.vec_d[key].detach()
        client_info = {
            "vec_b": deepcopy(self.vec_b),
            "vec_d": deepcopy(self.vec_d),
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            client_info["head"] = deepcopy(self.network.model.head.state_dict())
        return client_info

    def get_server_info(self):
        self.detach()
        server_info = super().get_server_info()
        server_info["vec_b"] = deepcopy(self.vec_b)
        server_info["vec_d"] = deepcopy(self.vec_d)
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
                self.vec_b[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_b, norm_weights)]).sum(0)
                )
                self.vec_d[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_d, norm_weights)]).sum(0)
                )

        self.set_optimization()

    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        self.detach()
        self.to(self.device)
        if "run_sum" in self.cl_merge:
            for key in self.lora_keys:
                eye_b = torch.eye(self.vec_b[key].shape[0], device=self.device, requires_grad=False)
                eye_d = torch.eye(self.vec_d[key].shape[0], device=self.device, requires_grad=False)
                if self.cur_task > 0:
                    self.optimization_dict[key] += self.old_delta[key]
                self.optimization_dict[key] += self.vera_mult(key)
        elif "run_mean" in self.cl_merge:
            for key in self.lora_keys:
                eye_b = torch.eye(self.vec_b[key].shape[0], device=self.device, requires_grad=False)
                eye_d = torch.eye(self.vec_d[key].shape[0], device=self.device, requires_grad=False)
                if self.cur_task > 0:
                    tmp = (self.old_delta[key] * self.cur_task) + (self.vera_mult(key))
                    self.optimization_dict[key] += tmp / (self.cur_task + 1)
                else:
                    self.optimization_dict[key] += self.vera_mult(key)
        elif "individual" in self.cl_merge:
            if "sum" in self.cl_merge:
                for key in self.lora_keys:
                    eye_b = torch.eye(self.vec_b[key].shape[0], device=self.device, requires_grad=False)
                    eye_d = torch.eye(self.vec_d[key].shape[0], device=self.device, requires_grad=False)
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + (self.vera_mult(key))
                        self.optimization_dict[key] += tmp
                    else:
                        self.optimization_dict[key] += self.vera_mult(key)
            elif "mean" in self.cl_merge:
                for key in self.lora_keys:
                    eye_b = torch.eye(self.vec_b[key].shape[0], device=self.device, requires_grad=False)
                    eye_d = torch.eye(self.vec_d[key].shape[0], device=self.device, requires_grad=False)
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + (self.vera_mult(key))
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += self.vera_mult(key)
        else:
            raise ValueError("Invalid cl_merge type")

    def to(self, device="cpu"):
        super().to(device)
        for key in self.lora_keys:
            self.vec_b[key] = self.vec_b[key].to(device)
            self.vec_d[key] = self.vec_d[key].to(device)
        return self

    def detach(self, only_lora: bool = False):
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
            if not only_lora:
                self.vec_b[key] = self.vec_b[key].detach()
                self.vec_d[key] = self.vec_d[key].detach()
