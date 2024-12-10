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
import math

from _models.lora import Lora


@register_model("ffa_lora")
class FfaLora(Lora):
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
        ffa: str_to_bool = True,
    ) -> None:
        super(FfaLora, self).__init__(
            fabric, network, device, optimizer, lr, wd_reg, avg_type, lora_alpha, r, lora_head, cl_merge
        )
        for key in self.lora_keys:
            self.cur_A[key] = self.cur_A[key].detach()
            # self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=False).to(self.device)
            # nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        for key in self.lora_keys:
            self.cur_A[key] = self.cur_A[key].detach()

    def get_optimization_dict(self):
        optimization_dict = deepcopy(dict(self.network.state_dict()))
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key].requires_grad = True
                optimization_dict[key] = self.head[key]
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = False
            if self.cur_task > 0 and not "individual" in self.cl_merge:
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        self.old_delta = deepcopy(server_info["old_delta"])
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            self.head = {
                key: nn.Parameter(torch.tensor(self.network.state_dict()[key]), requires_grad=True).to(self.device)
                for key in self.head_keys
            }

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.head.values()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(list(self.cur_B.values()), lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
