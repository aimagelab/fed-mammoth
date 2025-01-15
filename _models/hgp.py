import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit_prompt_hgp import VitHGP
import os
from utils.tools import str_to_bool
import math


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


@register_model("hgp")
class HGP(BaseModel):
    def __init__(
        self,
        fabric,
        network: VitHGP,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        how_many: int = 256,
        full_cov: str_to_bool = False,
        linear_probe: str_to_bool = False,
        num_epochs: int = 5,
        rebalance_epochs: int = 5,
        rebalance_lr: float = -1,
    ) -> None:
        params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        self.avg_type = avg_type
        self.how_many = how_many
        self.clients_statistics = None
        self.mogs_per_task = {}
        for n, p in self.network.named_parameters():
            if "prompt" in n or "last" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.logit_norm = 0.1
        self.full_cov = full_cov
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.num_epochs = num_epochs
        self.scheduler = CosineSchedule(self.optimizer, self.num_epochs)
        self.lr = lr
        self.rebalance_epochs = rebalance_epochs
        if rebalance_lr == -1:
            self.rebalance_lr = lr
        else:
            self.rebalance_lr = rebalance_lr

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
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
        if self.cur_task > 0:
            self.network.prompt.process_task_count()
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
        clients_gaussians = [client["client_statistics"] for client in client_info]
        self.to(self.device)
        mogs = {}
        for clas in range(self.cur_offset, self.cur_offset + self.cpt):
            counter = 0
            for client_gaussians in clients_gaussians:
                if client_gaussians.get(clas) is not None:
                    gaus_data = []
                    gaus_mean = client_gaussians[clas][1]
                    gaus_var = client_gaussians[clas][2]
                    gaus_data.append(gaus_mean)
                    gaus_data.append(gaus_var)
                    weight = client_gaussians[clas][0]
                    if mogs.get(clas) is None:
                        mogs[clas] = [[weight], [gaus_mean], [gaus_var]]
                    else:
                        mogs[clas][0].append(weight)
                        mogs[clas][1].append(gaus_mean)
                        mogs[clas][2].append(gaus_var)
                    counter += client_gaussians[clas][0]
            if mogs.get(clas) is not None:
                mogs[clas][0] = [mogs[clas][0][i] / counter for i in range(len(mogs[clas][0]))]
        self.mogs_per_task[self.cur_task] = mogs
        optimizer = torch.optim.SGD(self.network.last.parameters(), lr=self.rebalance_lr, momentum=0.9, weight_decay=0)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)
        logits_norm = torch.tensor([], dtype=torch.float32).to(self.device)
        for epoch in range(self.rebalance_epochs):
            sampled_data = []
            sampled_label = []
            # TODO: fix the probabilities of the classes
            # since Cifar100 and Tiny-ImageNet are balanced datasets and the participation rate is 100%,
            # we can set classes_weights asequiprobable
            num_classes = (self.cur_task + 1) * self.cpt
            classes_weights = torch.ones(num_classes, dtype=torch.float32).to(self.device)
            classes_samples = torch.multinomial(classes_weights, self.how_many * num_classes, replacement=True)
            _, classes_samples = torch.unique(classes_samples, return_counts=True)
            # sample features from gaussians:
            for task in range(self.cur_task + 1):
                for clas in range(self.cpt):
                    if self.mogs_per_task[task].get([task * self.cpt + clas][0]) is None:
                        #print("No gaussian for class ", task * self.cpt + clas)
                        continue
                    weights_list = []
                    for weight in self.mogs_per_task[task][task * self.cpt + clas][0]:
                        weights_list.append(weight)
                    gaussian_samples = torch.zeros(len(weights_list), dtype=torch.int64).to(self.device)
                    weights_list = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
                    gaussian_samples_fill = torch.multinomial(
                        weights_list, classes_samples[task * self.cpt + clas], replacement=True
                    )
                    gaussian_clients, gaussian_samples_fill = torch.unique(gaussian_samples_fill, return_counts=True)
                    gaussian_samples[gaussian_clients] += gaussian_samples_fill
                    for id, (mean, variance) in enumerate(
                        zip(
                            self.mogs_per_task[task][task * self.cpt + clas][1],
                            self.mogs_per_task[task][task * self.cpt + clas][2],
                        )
                    ):
                        cls_mean = mean  # * (0.9 + decay)
                        cls_var = variance
                        if self.full_cov:
                            cov = cls_var + 1e-8 * torch.eye(cls_mean.shape[-1]).to(self.device)
                        else:
                            cov = torch.eye(cls_mean.shape[-1]).to(self.device) * (cls_var + 1e-8) * 3
                        m = MultivariateNormal(cls_mean, cov)
                        n_samples = int(torch.round(gaussian_samples[id]))
                        sampled_data_single = m.sample((n_samples,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([clas + task * self.cpt] * n_samples)
            sampled_data = torch.cat(sampled_data, 0).float().to(self.device)
            sampled_label = torch.tensor(sampled_label, dtype=torch.int64).to(self.device)
            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            crct_num = (self.cur_task + 1) * self.cpt
            for _iter in range(crct_num):
                inp = inputs[_iter * self.how_many : (_iter + 1) * self.how_many].to(self.device)
                tgt = targets[_iter * self.how_many : (_iter + 1) * self.how_many].to(self.device)
                outputs = self.network.last(inp)
                logits = outputs
                per_task_norm = []
                cur_t_size = 0
                for _ti in range(self.cur_task + 1):
                    cur_t_size += self.cpt
                temp_norm = torch.norm(logits[:, :cur_t_size], p=2, dim=-1, keepdim=True)
                per_task_norm.append(temp_norm)
                per_task_norm = torch.cat(per_task_norm, dim=-1)
                logits_norm = torch.cat((logits_norm, per_task_norm.mean(dim=0, keepdim=True)), dim=0)
                norms = per_task_norm.mean(dim=-1, keepdim=True)

                norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True)
                decoupled_logits = torch.div(logits[:, :crct_num] + 1e-12, norms + 1e-12) / self.logit_norm
                loss = F.cross_entropy(decoupled_logits, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #scheduler.step()

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=1e-3, weight_decay=0)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            # restore correct optimizer
        params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
        #params = [{"params": self.network.last.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=0)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        #self.scheduler = CosineSchedule(self.optimizer, self.num_epochs)

    def end_epoch(self):
        #self.scheduler.step()
        return None

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
            "client_statistics": self.clients_statistics,
        }

    def get_server_info(self):
        return {"params": self.network.get_params()}

    def end_round_client(self, dataloader: DataLoader):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        features = torch.tensor([], dtype=torch.float32).to(self.device)
        true_labels = torch.tensor([], dtype=torch.int64).to(self.device)
        num_epochs = 1 if not self.full_cov else 3
        with torch.no_grad():
            client_statistics = {}
            for _ in range(num_epochs):
                for id, data in enumerate(dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.network(inputs, pen=True, train=False)
                    features = torch.cat((features, outputs), 0)
                    true_labels = torch.cat((true_labels, labels), 0)
            client_labels = torch.unique(true_labels).tolist()
            for client_label in client_labels:
                number = (true_labels == client_label).sum().item()
                if number > 1:
                    gaussians = []
                    gaussians.append(number)
                    gaussians.append(torch.mean(features[true_labels == client_label], 0))
                    if self.full_cov:
                        gaussians.append(
                            torch.cov(features[true_labels == client_label].T.type(torch.float64))
                            .type(torch.float32)
                            .to(self.device)
                        )
                    else:
                        gaussians.append(torch.std(features[true_labels == client_label], 0) ** 2)
                    client_statistics[client_label] = gaussians
            self.clients_statistics = client_statistics

    def save_checkpoint(self, output_folder: str, task: int, comm_round: int) -> None:
        training_status = self.network.training
        self.network.eval()

        checkpoint = {
            "task": task,
            "comm_round": comm_round,
            "network": self.network,
            "optimizer": self.optimizer,
            "mogs": self.mogs_per_task,
        }
        name = "hgp_" + "full_cov" if self.full_cov else "diag_cov"
        name += "_linear_probe" if self.do_linear_probe else ""
        name += f"_task_{task}_round_{comm_round}_checkpoint.pt"
        self.fabric.save(os.path.join(output_folder, name), checkpoint)
        self.network.train(training_status)


    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = self.fabric.load(checkpoint_path)
        self.network.load_state_dict(checkpoint["network"].state_dict())
        self.optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
        self.mogs_per_task = checkpoint["mogs"]
        return checkpoint["task"], checkpoint["comm_round"]