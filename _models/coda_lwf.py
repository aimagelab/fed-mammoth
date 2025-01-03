import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit_powder import ViTZoo
import os
from utils.tools import str_to_bool
import math
from _models.codap import CodaPrompt
from copy import deepcopy
from tqdm import tqdm


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


@register_model("coda_lwf")
class Coda_LwF(CodaPrompt):
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
        linear_probe = False
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
        self.compute_similarities = False
        self.network : ViTZoo
        self.scores_per_class = None
        self.classes = None
        self.old_network = None
        self.betas = None
        #self.network.mark_forward_method(self.network.get_scores)
        self.network.mark_forward_method("get_scores")

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            with torch.no_grad():
                old_out = self.old_network(inputs, train=True)[0][:, self.cur_offset : self.cur_offset + self.cpt]
            outputs = self.network(inputs, train=True)[0][:, self.cur_offset : self.cur_offset + self.cpt]
            loss_ce = self.loss(outputs, labels - self.cur_offset)
            loss_dual_full = F.kl_div(F.log_softmax(old_out, dim=1), F.softmax(outputs, dim=1), reduction="none")
            one_hot_labels = torch.logical_not(F.one_hot(labels % self.cpt, self.cpt)).float()
            beta = (self.betas.unsqueeze(0) @ one_hot_labels.T).flatten()
            loss_dual = ((loss_dual_full * one_hot_labels).sum(1) * beta).mean()
            loss = loss_ce + loss_dual

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

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        self.compute_similarities = True

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
        self.old_network = deepcopy(self.network) #latest server model
        # restore correct optimizer
        params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        self.scheduler = CosineSchedule(self.optimizer, self.num_epochs)
        scores_per_class = torch.zeros((self.cpt, 10), device=self.device)
        classes = torch.zeros(self.cpt, device=self.device)
        eps = 1e-20
        if self.compute_similarities:
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Computing similarities")):
                    inputs = inputs.to(self.device)
                    batch_scores = self.network.get_scores(inputs, pen=True, train=True)
                    #scores = torch.cat((scores, torch.stack(batch_scores)), dim=0)
                    labels_one_hot = torch.nn.functional.one_hot(labels % self.cpt, self.cpt).float()
                    scores_per_class += labels_one_hot.T @ batch_scores
                    labs, nums = torch.unique(labels, return_counts=True)
                    classes[labs % self.cpt] += nums
            scores_per_class = (scores_per_class) / (classes.unsqueeze(1) + eps) #average prompt-selection weights (scores) per class
            self.scores_per_class = deepcopy(scores_per_class)
            not_classes = classes == 0
            scores_per_class[not_classes] = torch.Tensor([-float('Inf')]).to(self.device)
            self.betas = torch.exp(scores_per_class.sum(1)) / torch.exp(scores_per_class.sum(1)).sum()
            self.classes = classes
            self.compute_similarities = False


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
        server_info = {"params": self.network.get_params()}
        return server_info

    def end_round_client(self, dataloader: DataLoader):
        self.old_network = None
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None