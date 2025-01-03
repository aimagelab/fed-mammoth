import torch
from torch import nn
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel


@register_model("fedsum")
class FedSum(BaseModel):
    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        sum_type: str = "full",
    ) -> None:
        self.lr = lr
        self.wd = wd_reg
        self.sum_type = sum_type

        super().__init__(fabric, network, device, optimizer, lr, wd_reg)

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

    def end_round_server(self, client_info: List[dict]):
        if self.sum_type == "treshold":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        if len(client_info) > 0:
            if self.sum_type == "trehshold":
                self.network.set_params(
                    torch.stack(
                        [
                            client["params"]
                            for client, norm_weight in zip(client_info, norm_weights)
                            if norm_weight > 1 / (len(client_info) * 10)  # treshold hyperparameter
                        ]
                    ).sum(0)
                )
            else:
                self.network.set_params(torch.stack([client["params"] for client in client_info]).sum(0))

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
        }

    def get_server_info(self):
        return {"params": self.network.get_params()}
