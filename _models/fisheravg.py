from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from torch.optim import SGD
from _networks.vit import VisionTransformer


@register_model("fisheravg")
class FisherAvg(BaseModel):

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        num_comm_rounds: int,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        slca: str_to_bool = False,
    ) -> None:
        if type(network) == VitHGP:
            for n, p in network.named_parameters():
                if "prompt" or "last" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        elif type(network) == VisionTransformer:
            params_backbone = []
            params_head = []
            for n, p in network.named_parameters():
                p.requires_grad = True
                if "head" not in n:
                    params_backbone.append(p)
                else:
                    params_head.append(p)
            params = [{"params": params_head}, {"params": params_backbone}]
            if slca:
                params = [{"params": params_backbone, "lr": lr / 100}, {"params": params_head}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        else:
            super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.avg_type = avg_type
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.slca = slca
        self.lr = lr
        self.wd = wd_reg
        self.num_comm_rounds = num_comm_rounds
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.optimizer_str = optimizer
        self.fish = None
        self.minibatch_size = 16

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        aug_inputs = self.augment(inputs)
        with self.fabric.autocast():
            outputs = self.network(aug_inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)
        if update:
            self.fabric.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.do_linear_probe:
            self.done_linear_probe = False

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

    def end_round_server(self, client_info: List[dict]):
        total_fisher = sum([(client["fisher"] + 1e-12) for client in client_info])
        norm_fisher = [torch.clamp((client["fisher"] + 1e-12) / total_fisher, 0, 1) for client in client_info]
        if len(client_info) > 0:
            merged_weights = 0
            for client, norm_weight in zip(client_info, norm_fisher):
                merged_weights += client["params"] * norm_weight
            self.network.set_params(merged_weights)
            
    def end_task_server(self, client_info: List[dict] = None):
        super().end_task_server(client_info)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            # restore correct optimizer
            params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
            optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
        
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer_class = OptimizerClass
        self.optimizer = OptimizerClass(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.module.get_params().data,
            "num_train_samples": len(dataloader.dataset.data),
            "fisher": self.fish,
        }

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        self.optimizer = None
        
        fish = torch.zeros_like(self.network.module.get_params())
        fake_opt = SGD(self.network.module.parameters(), lr=0)
        for j, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device).long()
            for ex, lab in zip(inputs, labels):
                output = self.network.module(ex.unsqueeze(0))[:, self.cur_offset : self.cur_offset + self.cpt]
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0) % self.cpt,
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.network.module.get_grads() ** 2
                fake_opt.zero_grad()

        fake_opt = None
        self.fish = fish.cpu()

    def get_server_info(self):
        dct = {"params": self.network.get_params().data}
        return dct

    def to(self, device):
        self.network.to(device)
        if self.fish is not None:
            self.fish.to(device)
        return self
