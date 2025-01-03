import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit_prompt_dual import VitDual
import os
from utils.tools import str_to_bool
import math


@register_model("dual_prompt")
class DualPrompt(BaseModel):
    def __init__(
        self,
        fabric,
        network: VitDual,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 5e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        num_epochs: int = 5,
        clip_grads: str_to_bool = False,
        use_scheduler: str_to_bool = False,
    ) -> None:
        self.clip_grads = clip_grads
        self.use_scheduler = use_scheduler
        self.lr = lr
        self.wd = wd_reg
        params = [{"params": network.model.head.parameters()}, {"params": network.model.e_prompt.parameters()}, {"params": network.model.g_prompt}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        self.avg_type = avg_type
        for n, p in self.network.original_model.named_parameters():
            p.requires_grad = False
        for n, p in self.network.model.named_parameters():
            if "prompt" in n or "head" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.num_epochs = num_epochs
        self.network : VitDual

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs, train=True, task_id = self.cur_task)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            if self.clip_grads:
                try:
                    self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
                except:
                    pass
            self.optimizer.step()

        return loss.item()

    def forward(self, x):  # used in evaluate, while observe is used in training
        return self.network(x, train=False, task_id = -1)

    def linear_probe(self, dataloader: DataLoader):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.model.head(pre_logits)[:, self.cur_offset : self.cur_offset + self.cpt]
                labels = labels % self.cpt
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        #if self.cur_task > 0:
        #    self.network.model.e_prompt.process_task_count()
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
            optimizer = self.optimizer_class(self.network.model.head.parameters(), lr=1e-3, weight_decay=0)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            # restore correct optimizer
        params = [{"params": self.network.model.head.parameters()}, {"params": self.network.model.e_prompt.parameters()}, {"params": self.network.model.g_prompt}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(optimizer)

    def end_epoch(self):
        if self.use_scheduler:
            self.scheduler.step()
        return None

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
        }

    def get_server_info(self):
        return {"params": self.network.get_params()}

    def end_round_client(self, dataloader: DataLoader):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        self.optimizer = None

    def save_checkpoint(self, output_folder: str, task: int, comm_round: int) -> None:
        training_status = self.network.training
        self.network.eval()

        checkpoint = {
            "task": task,
            "comm_round": comm_round,
            "network": self.network,
            "optimizer": self.optimizer,
        }
        name = "coda"
        name += "_linear_probe" if self.do_linear_probe else ""
        name += f"_task_{task}_round_{comm_round}_checkpoint.pt"
        self.fabric.save(os.path.join(output_folder, name), checkpoint)
        self.network.train(training_status)


    def to(self, device):
        self.network.original_model = self.network.original_model.to(device)
        self.network.model = self.network.model.to(device)
        return self