import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit_prompt_dual import VitDual
import os
from utils.tools import str_to_bool
import math
from copy import deepcopy


@register_model("pip_dualp")
class PipDualP(BaseModel):
    def __init__(
        self,
        fabric,
        network: VitDual,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 5e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        num_epochs: int = 5,
        clip_grads: str_to_bool = False,
        use_scheduler: str_to_bool = False,
        how_many: int = 5,
    ) -> None:
        self.clip_grads = clip_grads
        self.use_scheduler = use_scheduler
        self.lr = lr
        self.wd = wd_reg
        params = [{"params": network.model.head.parameters()}, {"params": network.model.e_prompt.parameters()}, {"params": network.model.g_prompt}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        self.avg_type = avg_type
        for n, p in self.network.original_model.named_parameters():
            p.requires_grad = False
        for n, p in self.network.model.named_parameters():
            if "prompt" in n or "head" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.num_epochs = num_epochs
        self.network : VitDual
        #self.protos contains, for each class, the prototypes for that class, which are the mean and the variance of the features of the samples in that class
        self.protos = {i : [] for i in range(self.network.model.head.weight.data.shape[0])}
        #self.augmented_protos contains, for each class, the augmented prototypes for that class, which are the "how_many" augmented prototypes of that class
        self.augmented_protos = {i : [] for i in range(self.network.model.head.weight.data.shape[0])}
        self.how_many = how_many
        self.com_round = -1
        self.cur_epoch = 0
        self.cur_labels = None
        self.dataloader = None

    def augment_protos(self, classes, protos):
        for clas in classes:
            distr = torch.distributions.Uniform(0, 1)
            betas = distr.sample((self.how_many, self.protos[clas][0].shape[0])).to(self.device)
            self.augmented_protos[clas] = []
            for i in range(self.how_many):
                self.augmented_protos[clas].append(self.protos[clas][0] + betas[i] * self.protos[clas][1])
        self.augmented_protos = {clas : torch.stack(self.augmented_protos[clas]) for clas in classes}
    
    def compute_protos(self, dataloader):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        features = torch.tensor([], dtype=torch.float32).to(self.device)
        true_labels = torch.tensor([], dtype=torch.int64).to(self.device)
        num_epochs = 1
        with torch.no_grad():
            protos = {}
            for _ in range(num_epochs):
                for id, data in enumerate(dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.network(inputs, return_outputs=True, train=False, task_id=self.cur_task)["pre_logits"]
                    features = torch.cat((features, outputs), 0)
                    true_labels = torch.cat((true_labels, labels), 0)
            client_labels = torch.unique(true_labels).tolist()
            #self.cur_labels = client_labels
            client_labels_at_least = []
            for client_label in client_labels:
                number = (true_labels == client_label).sum().item()
                if number > 1:
                    client_labels_at_least.append(client_label)
                    gaussians = []
                    gaussians.append(torch.mean(features[true_labels == client_label], 0))
                    gaussians.append(torch.std(features[true_labels == client_label], 0) ** 2)
                    protos[client_label] = gaussians
            self.cur_labels = client_labels_at_least
            for label in client_labels_at_least:
                self.protos[label] = protos[label]

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            if self.com_round == 0 and self.cur_epoch == 0:
                outputs = self.network(inputs, train=True, task_id = self.cur_task)[:, self.cur_offset : self.cur_offset + self.cpt]
            else:
                features = self.network(inputs, return_outputs=True, train=True, task_id = self.cur_task)["pre_logits"]
                features = torch.cat((features, torch.cat([self.augmented_protos[clas] for clas in range(self.cur_offset, self.cur_offset + self.cpt) if clas in self.augmented_protos.keys()])), 0)
                labels = torch.cat((labels, torch.cat([torch.tensor([i] * self.how_many, dtype=torch.int64) for i in range(self.cur_offset, self.cur_offset + self.cpt) if i in self.augmented_protos.keys()]).to(self.device)), 0)
                outputs = self.network.model.head(features)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)

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
        return self.network(x, train=False, task_id = -1)

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        self.com_round = -1
        #if self.cur_task > 0:
        #    self.network.model.e_prompt.process_task_count()
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
        for i in range(self.cur_offset, self.cur_offset + self.cpt):
            mean_proto = None
            var_proto = None
            weights_sum = 0
            for idx, client in enumerate(client_info):
                if i in client["prototypes"].keys():
                    if mean_proto is None:
                        mean_proto = client["prototypes"][i][0] * norm_weights[idx]
                    else:
                        mean_proto += client["prototypes"][i][0] * norm_weights[idx]
                    weights_sum += norm_weights[idx]
            if mean_proto is not None:
                if len(self.protos[i]) == 0:
                    self.protos[i].append(mean_proto / weights_sum)
                else:
                    self.protos[i][0] = mean_proto / weights_sum
                weights_sum = 0
                for idx, client in enumerate(client_info):
                    if i in client["prototypes"].keys():
                        if var_proto is None:
                            var_proto = (client["prototypes"][i][0] ** 2 + client["prototypes"][i][1] ** 2) * norm_weights[idx]
                        else:
                            var_proto += (client["prototypes"][i][0] ** 2 + client["prototypes"][i][1] ** 2) * norm_weights[idx]
                        weights_sum += norm_weights[idx]
                if var_proto is not None:
                    if len(self.protos[i]) < 2:
                        self.protos[i].append(var_proto / weights_sum)
                    else:
                        self.protos[i][1] = (var_proto / weights_sum - self.protos[i][0] ** 2)
                    torch.where(self.protos[i][1] < 0, 0, self.protos[i][1]) # it should never be negative, but just in case

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])
        self.protos = deepcopy(server_info["prototypes"])
        augmentable_prompts = [i for i in range(self.cur_offset, self.cur_offset + self.cpt) if len(self.protos[i]) > 0]
        self.augment_protos(augmentable_prompts, self.protos)
        # restore correct optimizer
        params = [{"params": self.network.model.head.parameters()}, {"params": self.network.model.e_prompt.parameters()}, {"params": self.network.model.g_prompt}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        self.cur_epoch = 0
        self.com_round += 1
        self.dataloader = dataloader  #dataloader pointer
        b = 2 + 2

    def end_epoch(self):
        if self.use_scheduler:
            self.scheduler.step()
        if self.cur_epoch == 0 and self.com_round == 0:
            self.compute_protos(self.dataloader)
            augmentable_prompts = [i for i in range(self.cur_offset, self.cur_offset + self.cpt) if len(self.protos[i]) > 0]
            self.augment_protos(augmentable_prompts, self.protos)
        return None

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
            "prototypes": {i : self.protos[i] for i in self.cur_labels},
        }

    def get_server_info(self):
        return {"params": self.network.get_params(), "prototypes": self.protos}

    def end_round_client(self, dataloader: DataLoader):
        self.compute_protos(self.dataloader)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        self.optimizer = None

    def save_checkpoint(self, output_folder: str, task: int, comm_round: int) -> None:
        training_status = self.network.training
        self.network.eval()

        checkpoint = {
            "task": task,
            "comm_round": comm_round,
            "network": self.network,
            "optimizer": self.optimizer,
        }
        name = "pip_dualp"
        name += "_linear_probe" if self.do_linear_probe else ""
        name += f"_task_{task}_round_{comm_round}_checkpoint.pt"
        self.fabric.save(os.path.join(output_folder, name), checkpoint)
        self.network.train(training_status)


    def to(self, device):
        self.network.original_model = self.network.original_model.to(device)
        self.network.model = self.network.model.to(device)
        return self