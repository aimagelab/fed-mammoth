import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from models import register_model
from typing import List
from torch.utils.data import DataLoader
from models.utils import BaseModel
from networks.vit_prompt_hgp import VitHGP


@register_model("hgp_old_tasks")
class HGPOldTasks(BaseModel):
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
    ) -> None:
        params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        # self.optimizer = None
        # self.optimizer = torch.optim.AdamW(
        #    [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}],
        #    lr=lr,
        #    weight_decay=wd_reg,
        # )
        # self.network, self.optimizer = self.fabric.setup(self.network, self.optimizer)
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

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels % self.cpt)

        if update:
            self.fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()

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
            mogs[clas][0] = [mogs[clas][0][i] / counter for i in range(len(mogs[clas][0]))]
        self.mogs_per_task[self.cur_task] = mogs
        optimizer = torch.optim.SGD(self.network.last.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)
        logits_norm = torch.tensor([], dtype=torch.float32).to(self.device)
        if self.cur_task > 0:
            for epoch in range(5):
                sampled_data = []
                sampled_label = []
                # TODO: fix the prrobabilities of the classes
                # since Cifar100 and Tiny-ImageNet are balanced datasets and the participation rate is 100%,
                # we can set classes_weights asequiprobable
                num_classes = (self.cur_task + 1) * self.cpt
                classes_weights = torch.ones(num_classes, dtype=torch.float32).to(self.device)
                classes_samples = torch.multinomial(classes_weights, self.how_many * num_classes, replacement=True)
                _, classes_samples = torch.unique(classes_samples, return_counts=True)
                # sample features from gaussians:
                for task in range(self.cur_task):
                    for clas in range(self.cpt):
                        weights_list = []
                        for weight in self.mogs_per_task[task][task * self.cpt + clas][0]:
                            weights_list.append(weight)
                        gaussian_samples = torch.zeros(len(weights_list), dtype=torch.int64).to(self.device)
                        weights_list = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
                        gaussian_samples_fill = torch.multinomial(
                            weights_list, classes_samples[task * self.cpt + clas], replacement=True
                        )
                        gaussian_clients, gaussian_samples_fill = torch.unique(
                            gaussian_samples_fill, return_counts=True
                        )
                        gaussian_samples[gaussian_clients] += gaussian_samples_fill
                        for id, (mean, variance) in enumerate(
                            zip(
                                self.mogs_per_task[task][task * self.cpt + clas][1],
                                self.mogs_per_task[task][task * self.cpt + clas][2],
                            )
                        ):
                            cls_mean = mean  # * (0.9 + decay)
                            cls_var = variance
                            cov = torch.eye(cls_mean.shape[-1]).to(self.device) * cls_var * 3
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
                    outputs = self.network.last(inp)[:, : self.cur_offset]
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
                scheduler.step()

    def begin_round_client(self, _: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
            "client_statistics": self.clients_statistics,
        }

    def get_server_info(self):
        return {"params": self.network.get_params()}

    def end_round_client(self, dataloader: DataLoader):
        features = torch.tensor([], dtype=torch.float32).to(self.device)
        true_labels = torch.tensor([], dtype=torch.int64).to(self.device)
        with torch.no_grad():
            client_statistics = {}
            for data in dataloader:
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
                    gaussians.append(torch.std(features[true_labels == client_label], 0) ** 2)
                    client_statistics[client_label] = gaussians
            self.clients_statistics = client_statistics
