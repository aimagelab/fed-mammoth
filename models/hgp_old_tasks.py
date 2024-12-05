import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from models import register_model
from typing import List
from torch.utils.data import DataLoader
from models._utils import BaseModel
from networks.vit_prompt_hgp import VitHGP
from utils.tools import str_to_bool
from models.hgp import HGP


@register_model("hgp_old_tasks")
class HGPOldTasks(HGP):
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
    ) -> None:
        params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, avg_type, how_many, full_cov, linear_probe)

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
