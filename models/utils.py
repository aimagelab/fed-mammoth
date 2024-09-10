import os
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str,
        lr: float,
        wd_reg: float,
        params: list = None,
    ):
        super().__init__()
        self.device = device
        self.network = network
        OptimizerClass = getattr(torch.optim, optimizer)
        self.optimizer_class = OptimizerClass
        if params is None:
            self.optimizer = OptimizerClass(self.network.parameters(), lr=lr, weight_decay=wd_reg)
        else:
            self.optimizer = OptimizerClass(params, lr=lr, weight_decay=wd_reg)
        self.loss = nn.CrossEntropyLoss()
        self.fabric = fabric
        self.network, self.optimizer = self.fabric.setup(self.network, self.optimizer)
        self.cur_task = -1
        self.cur_offset = 0
        self.cpt = 0

    def save_checkpoint(self, output_folder: str, task: int, comm_round: int) -> None:
        training_status = self.network.training
        self.network.eval()

        checkpoint = {
            "task": task,
            "comm_round": comm_round,
            "network": self.network,
            "optimizer": self.optimizer,
        }
        self.fabric.save(os.path.join(output_folder, "checkpoint.pt"), checkpoint)
        self.network.train(training_status)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = self.fabric.load(checkpoint_path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["task"], checkpoint["comm_round"]

    def forward(self, x):
        return self.network(x)

    def observe(self, inputs: torch.Tensor, update: bool = True) -> float:
        pass

    def begin_task(self, n_classes_per_task: int):
        self.cur_task += 1
        self.cpt = n_classes_per_task
        self.cur_offset = self.cur_task * self.cpt

    def end_task(self, dataloader: DataLoader = None, info: List[dict] = None):
        pass

    def end_task_client(self, dataloader: DataLoader = None):
        self.end_task(dataloader=dataloader)

    def end_task_server(self, client_info: List[dict] = None):
        self.end_task(info=client_info)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        pass

    def end_round_client(self, dataloader: DataLoader):
        pass

    def begin_round_server(self):
        pass

    def end_round_server(self, client_info: List[dict]):
        pass

    def end_epoch(self):
        pass

    def get_client_info(self, dataloader: DataLoader):
        pass

    def get_server_info(self):
        pass

    def to(self, device):
        # self.device = device
        self.network.to(device)
        return self
