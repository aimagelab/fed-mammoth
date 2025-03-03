import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit_prompt_l2p import VitL2P
import os
from utils.tools import str_to_bool
import math


@register_model("l2p")
class L2P(BaseModel):
    def __init__(
        self,
        fabric,
        network: VitL2P,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        num_epochs: int = 5,
    ) -> None:
        self.lr = lr
        params = [{"params": network.last.parameters()}, {"params": network.feat.prompt.parameters()}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        self.avg_type = avg_type
        self.clients_statistics = None
        self.mogs_per_task = {}
        for n, p in self.network.named_parameters():
            if ("prompt" in n and not "pos_embed" in n) or "last" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.num_epochs = num_epochs

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()

        return loss.item()

    def forward(self, x):  # used in evaluate, while observe is used in training
        return self.network(x)

    def linear_probe(self, dataloader: DataLoader):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.last(pre_logits)[:, self.cur_offset : self.cur_offset + self.cpt]
                labels = labels % self.cpt
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.do_linear_probe:
            self.done_linear_probe = False

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        if len(client_info) > 0:
            self.network.set_params(
                torch.stack(
                    [client["params"] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
                ).sum(0)
            )

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=1e-3, weight_decay=0)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            # restore correct optimizer
        params = [{"params": self.network.last.parameters()}, {"params": self.network.feat.prompt.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=0)
        self.optimizer = self.fabric.setup_optimizers(optimizer)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
            "client_statistics": self.clients_statistics,
        }

    def get_server_info(self):
        return {"params": self.network.get_params()}

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
