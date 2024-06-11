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

from models.lora import Lora

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
        enable_lora: list = [True, True, True],
        lora_head: str_to_bool = True,
        cl_merge: str = "run_sum",
        ffa: str_to_bool = True,
    ) -> None:
        super(FfaLora, self).__init__(fabric, network, device, optimizer, lr, wd_reg, avg_type, lora_alpha, r, enable_lora, lora_head, cl_merge)
        for key in self.lora_keys:
            self.cur_A[key].requires_grad = False

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        self.old_delta = deepcopy(server_info["old_delta"])
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            #for p in self.network.model.head.parameters():
            #    p.requires_grad = True
            self.head = {key: nn.Parameter(torch.tensor(self.network.state_dict()[key]), requires_grad=True).to(self.device) for key in self.head_keys}

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        #if not self.lora_head:
        #    self.optimizer = OptimizerClass(
        #        list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.network.model.head.parameters()),
        #        lr=self.lr,
        #        weight_decay=self.wd_reg,
        #    )
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.head.values()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()), lr=self.lr, weight_decay=self.wd_reg
            )
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)