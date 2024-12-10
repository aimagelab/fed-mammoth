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
from _models._utils import Buffer


@register_model("derpp")
class Derpp(BaseModel):

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        batch_size: int,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        slca: str_to_bool = False,
        buffer_size: int = 200,
        alpha: float=0.5,
        beta: float=0.5
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
        self.optimizer_str = optimizer
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.transform = None
        self.buffer = Buffer(buffer_size, self.device)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        
        aug_inputs = self.augment(inputs)
        with self.fabric.autocast():
            logits = self.network(aug_inputs)
            loss = self.loss(logits[:, self.cur_offset:self.cur_offset+self.cpt], labels % self.cpt)

            if len(self.buffer) > 0:
                loss_re = 0
                if self.alpha > 0:
                    buf_inputs, _, buf_logits = self.buffer.get_data(self.batch_size)
                    buf_outputs = self.network(self.augment(buf_inputs))
                    loss_re += self.alpha * F.mse_loss(buf_outputs[:, :self.cur_offset+self.cpt], buf_logits[:, :self.cur_offset+self.cpt])

                if self.beta > 0:
                    buf_inputs, buf_labels, _ = self.buffer.get_data(self.batch_size)
                    buf_outputs = self.network(self.augment(buf_inputs))
                    loss_re += self.beta * self.loss(buf_outputs[:, :self.cur_offset+self.cpt], buf_labels)

                loss += loss_re

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.buffer.add_data(examples=inputs.data,
                             labels=labels.data,
                             logits=logits.data)
        
        return loss.item()
    
    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.do_linear_probe:
            self.done_linear_probe = False

    def end_task_server(self, client_info: List[dict] = None):
        super().end_task_server(client_info)
        train_status = self.network.training
        self.checkpoint = deepcopy(self.network.eval())
        self.network.train(train_status)

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
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
        }

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        self.optimizer = None

    def get_server_info(self):
        dct = {"params": self.network.get_params()}
        return dct

    def to(self, device):
        self.network.to(device)
        return self
    