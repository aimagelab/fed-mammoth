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
from _networks.vit import VisionTransformer


@register_model("lwf")
class LwF(BaseModel):

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
        alpha: float = 0.5,
        softmax_temp: float = 2
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
        self.alpha = alpha
        self.softmax_temp = softmax_temp
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.optimizer_str = optimizer
    
    @staticmethod
    def modified_kl_div(old, new):
        return -torch.mean(torch.sum(old * torch.log(new), 1))

    @staticmethod
    def smooth(logits, temp, dim):
        log = logits ** (1 / temp)
        return log / torch.sum(log, dim).unsqueeze(1)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:

        aug_inputs = self.augment(inputs)
        with self.fabric.autocast():
            outputs = self.network(aug_inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)
            if self.cur_task > 0 and self.checkpoint is not None:
                with torch.no_grad():
                    old_logits = self.checkpoint(aug_inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
                loss += self.alpha * self.modified_kl_div(self.smooth(self.soft(old_logits[:, :self.cur_offset + self.cpt]).to(self.device), 2, 1),
                                                        self.smooth(self.soft(outputs[:, :self.cur_offset + self.cpt]), 2, 1))
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
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        if len(client_info) > 0:
            merged_weights = 0
            for client, norm_weight in zip(client_info, norm_weights):
                merged_weights += client["params"] * norm_weight
            self.network.set_params(merged_weights)
            
    def end_task_server(self, client_info: List[dict] = None):
        super().end_task_server(client_info)
        train_status = self.network.training
        self.checkpoint = deepcopy(self.network.eval())
        self.network.train(train_status)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])
        if server_info["checkpoint"] is not None:
            train_status = self.network.training
            self.checkpoint = deepcopy(self.network.eval())
            self.network.train(train_status)
            self.checkpoint.set_params(server_info["checkpoint"])
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
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
        }

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        self.optimizer = None
        self.checkpoint = None

    def get_server_info(self):
        dct = {"params": self.network.get_params()}
        if self.checkpoint is None:
            dct["checkpoint"] = None
        else:
            dct["checkpoint"] = self.checkpoint.get_params()
        return dct

    def to(self, device):
        self.network.to(device)
        if self.checkpoint is not None:
            self.checkpoint.to(device)
        return self
