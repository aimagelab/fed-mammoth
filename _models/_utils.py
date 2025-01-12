import os
from typing import List
import numpy as np
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
        self.augment = None
        self.test_transform = None

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
        if self.cur_task > 0:
            self.cur_offset += self.cpt
        self.cpt = n_classes_per_task

    def end_task(self, dataloader: DataLoader = None, info: List[dict] = None):
        pass

    def end_task_client(self, dataloader: DataLoader = None, server_info: dict = None):
        self.end_task(dataloader=dataloader)

    def end_task_server(self, client_info: List[dict] = None):
        self.end_task(info=client_info)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        pass

    def end_round_client(self, dataloader: DataLoader):
        pass

    def begin_round_server(self, info: List[dict] = None):
        pass

    def end_round_server(self, client_info: List[dict]):
        pass

    def end_epoch(self):
        pass

    def end_round_validation_client(self, dataloader: DataLoader):
        pass

    def end_round_validation_server(self, dataloader: DataLoader):
        pass

    def end_task_validation_client(self, dataloader: DataLoader):
        pass

    def end_task_validation_server(self, dataloader: DataLoader):
        pass

    def warmup_task_client(self, server_info, dataloader: DataLoader):
        pass

    def warmup_task_server(self, dataloaders: List[DataLoader] = None):
        pass

    def get_client_info(self, dataloader: DataLoader):
        pass

    def get_server_info(self):
        pass

    def to(self, device):
        # self.device = device
        self.network.to(device)
        return self
    
    def end_training(self):
        pass


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    def __init__(self, buffer_size, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0

    def to(self, device):
        self.device = device
        if hasattr(self, "attributes"):
            for attr_str in self.attributes:
                if hasattr(self, attr_str):
                    setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, **kwargs) -> None:
        for attr_str in self.attributes:
            setattr(self, attr_str, torch.zeros((self.buffer_size,
                    *kwargs[attr_str].shape[1:]), dtype=kwargs[attr_str].dtype,
                    device=self.device))

    def add_data(self, **kwargs):
        if hasattr(self, 'attributes'):
            assert self.attributes == list(kwargs.keys())
        else:
            self.attributes = list(kwargs.keys())
            self.init_tensors(**kwargs)

        for i in range(kwargs[self.attributes[0]].shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                for attr_str in self.attributes:
                    attribute = getattr(self, attr_str)
                    attribute[index] = kwargs[attr_str][i].to(self.device)

    def get_data(self, size: int, device=None, shuffle=True):
        target_device = self.device if device is None else device
        actual_size = min(size, len(self))

        if shuffle:
            choice = np.random.choice(len(self), size=actual_size, replace=False)
        else:
            choice = np.arange(actual_size)

        return [getattr(self, attr_str)[choice].to(target_device) for attr_str in self.attributes]

    def is_empty(self) -> bool:
        return True if self.num_seen_examples == 0 else False

    def empty(self) -> None:
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(getattr(self, attr_str), None)
                delattr(self, attr_str)
        self.attributes = None
        delattr(self, "attributes")
        self.num_seen_examples = 0
