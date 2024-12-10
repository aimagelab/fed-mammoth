import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit_prompt_coda import ViTZoo
import os
from utils.tools import str_to_bool
import math
from _models.codap import CodaPrompt


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(map(lambda group: group["initial_lr"], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        if self.last_epoch == 0:
            return base_lr
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


@register_model("powder")
class Powder(CodaPrompt):
    def __init__(
        self,
        fabric,
        network: ViTZoo,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        num_epochs: int = 5,
        clip_grads: str_to_bool = False,
        use_scheduler: str_to_bool = False,
    ) -> None:
        super().__init__(
            fabric,
            network,
            device,
            optimizer,
            lr,
            wd_reg,
            avg_type,
            linear_probe,
            num_epochs,
            clip_grads,
            use_scheduler,
        )
        self.G_task = None
        self.G_class = None
        self.mean_att = None

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels % self.cpt)

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
        return self.network(x, pen=False, train=False)

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

    def warmup_task_client(self, server_info, dataloader: DataLoader):
        # here we'll retrieve the alphas, that is, the attention vectors, and
        # average them to get the global attention vector, by passing the entire 
        # local dataset to the local model
        self.network.set_params(server_info["params"])
        self.network.eval()
        attention_vectors = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                _, att = self.network(inputs, train = True) # train = True so that only current prompts are used
                attention_vectors.append(att)
        self.network.train()
        self.mean_att = torch.stack(attention_vectors).mean(0)


    def end_task_client(self, dataloader: DataLoader = None, server_info: dict = None):
        return super().end_task_client(dataloader, server_info)

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
        params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        self.scheduler = CosineSchedule(self.optimizer, self.num_epochs)

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
        server_info = {"params": self.network.get_params(), 
                       "prompt": self.network.prompt.get_params()}

    def end_round_client(self, dataloader: DataLoader):
        pass

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
