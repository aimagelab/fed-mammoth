import torch
from torch import nn
from torch.nn import functional as F
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from _networks.vit import VisionTransformer
from _models.fedavg import FedAvg
from copy import deepcopy

"""
Centralized implementation of FedProto. In this case, clients also share their model, which is aggregated by the server and 
redistributed just like the prototypes. In this way, every client has the same model as the server.
"""


@register_model("fedproto")
class FedProto(FedAvg):
    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        slca: str_to_bool = False,
        num_classes: int = 100,
        ld_reg: float = 0.1,
    ) -> None:
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, avg_type, linear_probe, slca)
        self.avg_type = "proto"
        self.num_classes = num_classes
        self.proto = {}
        self.global_proto = {}
        self.ld_reg = ld_reg

    def begin_task(self, n_classes_per_task: int):
        res = super().begin_task(n_classes_per_task)
        # self.proto[self.cur_task] = torch.zeros(n_classes_per_task, 768)
        self.proto = {}
        self.global_proto[self.cur_task] = {}
        return res

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        super().begin_round_client(dataloader, server_info)
        if "prototypes" in server_info:
            self.global_proto[self.cur_task] = deepcopy(server_info["prototypes"])

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            feats, outputs = self.network(inputs, penultimate=True)
            outputs = outputs[:, self.cur_offset : self.cur_offset + self.cpt]
            loss_ce = self.loss(outputs, labels - self.cur_offset)
            classes = torch.unique(labels)
            loss_mse = 0
            if len(self.global_proto[self.cur_task]) > 0:
                for class_tensor in classes:
                    class_ = int(class_tensor.item())
                    proto_class = feats[labels == class_].mean(0)
                    loss_mse += F.mse_loss(self.global_proto[self.cur_task][class_], proto_class)
            loss = loss_ce + loss_mse * self.ld_reg
        if update:
            self.optimizer.zero_grad()
            self.fabric.backward(loss)
            self.optimizer.step()
        return loss.item()

    def end_round_client(self, dataloader: DataLoader):
        res = super().end_round_client(dataloader)
        # compute prototypes
        with torch.no_grad():
            total_feats = torch.tensor([], device=self.device)
            total_labels = torch.tensor([], device=self.device)
            for inputs, labels in dataloader:
                feats, _ = self.network(inputs, penultimate=True)
                total_feats = torch.cat((total_feats, feats))
                total_labels = torch.cat((total_labels, labels))
            classes, nums = torch.unique(total_labels, return_counts=True)
            for class_, num in zip(classes, nums):
                self.proto[int(class_.item())] = [num.item(), total_feats[total_labels == class_].mean(0)]

    def get_client_info(self, dataloader: DataLoader):
        client_info = super().get_client_info(dataloader)
        if client_info is None:
            client_info = {}
        client_info["prototypes"] = self.proto
        return client_info

    def get_server_info(self):
        server_info = super().get_server_info()
        if server_info is None:
            server_info = {}
        server_info["prototypes"] = self.global_proto[self.cur_task]
        return server_info

    def end_round_server(self, client_info: List[dict]):
        super().end_round_server(client_info)
        # compute global prototypes
        prototypes_per_class = {}
        tot_num_classes = {}
        for client in client_info:
            for class_ in client["prototypes"]:
                proto = client["prototypes"][class_]
                if class_ not in prototypes_per_class:
                    prototypes_per_class[class_] = proto[1] * proto[0]
                else:
                    prototypes_per_class[class_] += proto[1] * proto[0]
                if tot_num_classes.get(class_, 0) == 0:
                    tot_num_classes[class_] = proto[0]
                else:
                    tot_num_classes[class_] += proto[0]
        for class_ in prototypes_per_class:
            self.global_proto[self.cur_task][class_] = prototypes_per_class[class_] / tot_num_classes[class_]

    def to(self, device):
        super().to(device)
        # if self.proto is not None:
        #    for task in self.proto:
        #        self.proto[task] = self.proto[task].to(device)
        # if self.global_proto is not None:
        #    for task in self.global_proto:
        #        self.global_proto[task] = self.global_proto[task].to(device)
        return self
