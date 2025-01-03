import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from _models import register_model
from typing import List
from _models._utils import BaseModel
from _networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool, compute_fisher_expectation_fabric

from _models.lora import Lora, merge_AB, zero_pad
from _models.regmean import RegMean
from _models.lora import Lora
from tqdm import tqdm
import math
from transformers import T5Model


@register_model("pilora")
class PiLora(Lora):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "Adam",
        lr: float = 0.0003,
        clip_grad: str_to_bool = False,
        wd_reg: float = 0.00001,
        avg_type: str = "weighted",
        lora_alpha: float = 1,
        r: int = 16,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        temp: float = 1,
        lr_back: float = -1,
        num_tasks: int = 10,
        soft_temp: float = 0.2,
        use_l1: str_to_bool = False,
        use_pl: str_to_bool = False,
        use_ort: str_to_bool = False,
    ) -> None:
        self.num_tasks = num_tasks
        self.temp = temp
        self.soft_temp = soft_temp
        self.use_l1 = use_l1
        self.use_pl = use_pl
        self.use_ort = use_ort
        self.Q = {}
        self.K = {}
        self.V = {}
        self.lr_back = lr_back
        if self.lr_back < 0:
            self.lr_back = lr
        Lora.__init__(
            self,
            fabric,
            network,
            device,
            optimizer,
            lr,
            clip_grad,
            wd_reg,
            avg_type,
            lora_alpha,
            r,
            lora_head,
            cl_merge,
        )
        self.cur_round = 0
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        del self.old_delta
        self.old_B = []  # list of dicts, each dict contains the B matrices for a task, keys are the self.lora_keys
        self.old_Q = []  # list of dicts, each dict contains the Q matrices for a task, keys are the self.lora_keys
        self.old_V = []  # list of dicts, each dict contains the V matrices for a task, keys are the self.lora_keys
        self.old_A = []  # list of dicts, each dict contains the A matrices for a task, keys are the self.lora_keys
        self.class_protos = {}

    def forward(self, x, fabric=True):
        if fabric:
            pre, _ = functional_call(self.network, self.optimization_dict, x, kwargs={"penultimate": True})
        else:
            pre, _ = functional_call(self.network.module, self.optimization_dict, x, kwargs={"penultimate": True})
        protos = torch.cat([self.class_protos[t][i] for t in range(len(self.class_protos)) for i in range(self.cpt)])
        score = F.softmax(-torch.cdist(pre, protos, p=2), dim=1)
        return score

    def init_lora_params(self, network, r):
        for name, param in network.named_parameters():
            param.requires_grad = False  # freeze all the parameters
            if not self.lora_head and "head" in name:
                self.head_keys.append(name)
            if ("qkv" in name and "weight" in name) and ("blocks.0" in name):
                self.lora_keys.append(name)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_delta[name] = nn.Parameter(
                    torch.zeros(param.shape[0], param.shape[1]), requires_grad=False
                ).to(self.device)
                self.Q[name] = nn.Parameter(torch.zeros(param.shape[0] // 3, r, device=self.device), requires_grad=True)
                self.K[name] = nn.Parameter(
                    torch.zeros(param.shape[0] // 3, r, device=self.device), requires_grad=False
                )
                self.V[name] = nn.Parameter(torch.zeros(param.shape[0] // 3, r, device=self.device), requires_grad=True)
                self.cur_B[name] = torch.cat((self.Q[name], self.K[name], self.V[name]))  # Q  # K  # V
                self.cur_A[name] = nn.Parameter(torch.zeros(r, param.shape[1]), requires_grad=True).to(self.device)
                nn.init.kaiming_uniform_(self.cur_A[name], a=math.sqrt(5))

    def get_optimization_dict(self, fabric=True):
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key].requires_grad = True
                optimization_dict[key] = self.head[key]
        for key in self.lora_keys:
            self.Q[key].requires_grad = True
            self.K[key].requires_grad = False
            self.V[key].requires_grad = True
            # self.cur_B[key] = torch.cat((self.Q[key], self.K[key], self.V[key]))
            self.cur_A[key].requires_grad = True
            Q = self.Q[key] + (
                torch.stack([self.old_Q[i][key].detach().to(self.device) for i in range(self.cur_task)]).sum(0)
                if self.cur_task > 0
                else 0
            )
            K = self.K[key]
            V = self.V[key] + (
                torch.stack([self.old_V[i][key].detach().to(self.device) for i in range(self.cur_task)]).sum(0)
                if self.cur_task > 0
                else 0
            )
            self.cur_B[key] = torch.cat((Q, K, V))
            B = self.cur_B[
                key
            ]  # + (torch.stack([self.old_B[i][key].detach() for i in range(self.cur_task)]).sum(0) if self.cur_task > 0 else 0)
            A = self.cur_A[key] + (
                torch.stack([self.old_A[i][key].detach().to(self.device) for i in range(self.cur_task)]).sum(0)
                if self.cur_task > 0
                else 0
            )
            optimization_dict[key] += B @ A
        return optimization_dict

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        fabric = True
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict(fabric=fabric)
        eps = 1e-4
        # self.cur_B[self.lora_keys[0]].retain_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            prelogits, outputs = functional_call(
                self.network, optimization_dict, inputs, kwargs={"penultimate": True}
            )  # TODO qua rogna con t5
            outputs = outputs[:, self.cur_offset : self.cur_offset + self.cpt]
            # loss_ce = self.loss(outputs, labels - self.cur_offset)
            # loss_ce = 0
            loss = 0
            protos = torch.cat([self.class_protos[t][i] for t in range(self.cur_task + 1) for i in range(self.cpt)])
            # for idx, pre in enumerate(prelogits):
            #    loss_dce += - torch.log((torch.exp(-self.temp * (pre - protos[labels[idx]]).pow(2).sum()) + eps) / (torch.exp(-self.temp * (pre - protos).pow(2).sum(-1)).sum() + eps))
            # loss_dce /= len(labels)
            distances = (
                prelogits.pow(2).sum(1, keepdim=True)
                + protos.pow(2).sum(1, keepdim=True).T
                - 2 * (torch.matmul(prelogits, protos.T))
            )
            if isinstance(self.network.module.model, T5Model):
                distances /= 512
            else:
                distances /= 768
            distances = distances.sqrt()
            # logits = F.softmax(-distances, dim=1)self.old_Q[i][key].detach().to(self.device)
            loss_dce = F.cross_entropy(-distances / self.temp, labels)
            # loss_pl = (prelogits - protos[labels]).pow(2).sum()# / len(labels)
            loss += loss_dce
            # if self.use_pl:
            loss_pl = torch.index_select(distances, dim=1, index=(labels))
            loss_pl = torch.diagonal(loss_pl)
            loss_pl = torch.mean(loss_pl)
            loss += 0.001 * loss_pl
            # if self.use_ort and self.cur_task > 0:
            loss_ort = 0
            if self.cur_task > 0:
                for key in self.lora_keys:
                    for i in range(self.cur_task):
                        # loss_ort += torch.norm(self.cur_B[key] @ self.old_B[i][key].T, p=2)
                        # loss_ort += torch.abs(torch.mm(self.Q[key], self.old_Q[i][key].T)).sum()
                        # loss_ort += torch.abs(torch.mm(self.V[key], self.old_V[i][key].T)).sum()
                        loss_ort += torch.abs(torch.mm(self.cur_A[key], self.old_A[i][key].T)).sum()
            loss += 0.5 * loss_ort
            loss_l1 = torch.linalg.matrix_norm(self.cur_A[self.lora_keys[0]], ord=1) + torch.linalg.matrix_norm(
                self.cur_A[self.lora_keys[0]], ord=1
            )
            loss += 0.01 * loss_l1

        if update:
            if fabric:
                self.fabric.backward(loss)
            else:
                loss.backward()
            # torch.nn.utils.clip_grad_norm_(list(self.cur_B.values()) + list(self.cur_A.values()), 1.0)
            if self.clip_grad:
                try:
                    self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
                except:
                    pass
            self.optimizer.step()
        return loss.item()

    def split_backbone_head(self):
        backbone_params = []
        head_params = []
        for key in self.lora_keys:
            self.Q[key] = self.Q[key].detach()
            self.K[key] = self.K[key].detach()
            self.V[key] = self.V[key].detach()
            self.Q[key].requires_grad = True
            self.K[key].requires_grad = False
            self.V[key].requires_grad = True
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_A[key].requires_grad = True
            backbone_params.append(self.Q[key])
            backbone_params.append(self.V[key])
            backbone_params.append(self.cur_A[key])
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
            self.head[key].requires_grad = True
            head_params.append(self.head[key])
        return backbone_params, head_params

    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.Q[key] = self.Q[key].detach()
            self.K[key] = self.K[key].detach()
            self.V[key] = self.V[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        protos = torch.cat([self.class_protos[self.cur_task][i] for i in range(self.cpt)])
        client_info = {
            "cur_A": self.cur_A,
            "Q": self.Q,
            "K": self.K,
            "V": self.V,
            "proto": protos,
            "num_train_samples": len(dataloader.dataset.data),
            "classes": self.classes,
        }
        client_info["avg_feat"] = self.average_features
        return client_info

    def get_server_info(self):
        for key in self.lora_keys:
            self.Q[key] = self.Q[key].detach()
            self.K[key] = self.K[key].detach()
            self.V[key] = self.V[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        server_info = {
            "cur_A": deepcopy(self.cur_A),
            "Q": deepcopy(self.Q),
            "K": deepcopy(self.K),
            "V": deepcopy(self.V),
            "old_A": deepcopy(self.old_A),
            "old_Q": deepcopy(self.old_Q),
            "old_V": deepcopy(self.old_V),
        }
        server_info["proto"] = deepcopy(self.class_protos)
        if isinstance(self.network.module.model, T5Model):
            server_info["head"] = deepcopy(self.network.module.head.state_dict())
        else:
            server_info["head"] = deepcopy(self.network.model.head.state_dict())
        return server_info

    # def get_optimization_dict(self, fabric=True):
    #    if fabric:
    #        optimization_dict = deepcopy(dict(self.network.state_dict()))
    #    else:
    #        optimization_dict = deepcopy(dict(self.network.module.state_dict()))
    #    if not self.lora_head:
    #        for key in self.head_keys:
    #            optimization_dict[key] = self.head[key]
    #    for key in self.lora_keys:
    #
    #        optimization_dict[key] += torch.cat((self.Q[key], self.K[key], self.V[key])) @ self.cur_A[key]
    #    return optimization_dict

    def begin_task(self, n_classes_per_task: int):
        BaseModel.begin_task(self, n_classes_per_task)
        if self.cur_task > 0:
            self.old_B.append({key: self.cur_B[key].detach().clone() for key in self.lora_keys})
            self.old_A.append({key: self.cur_A[key].detach().clone() for key in self.lora_keys})
            self.old_Q.append({key: self.Q[key].detach().clone() for key in self.lora_keys})
            self.old_V.append({key: self.V[key].detach().clone() for key in self.lora_keys})
            self.init_matrices(init_K=False)
        self.cur_round = 0
        # self.class_protos[self.cur_task] = nn.ParameterList([nn.Parameter(0.1*torch.randn(self.768, 1)) for i in range(self.cpt)])
        if isinstance(self.network.module.model, T5Model):
            self.class_protos[self.cur_task] = nn.ParameterList(
                [nn.Parameter(0.1 * torch.randn(1, 512), requires_grad=True).to(self.device) for i in range(self.cpt)]
            )
        else:
            self.class_protos[self.cur_task] = nn.ParameterList(
                [nn.Parameter(0.1 * torch.randn(1, 768), requires_grad=True).to(self.device) for i in range(self.cpt)]
            )
        for i in range(self.cur_task):
            for key in self.lora_keys:
                self.old_B[i][key] = self.old_B[i][key].detach()
                self.old_A[i][key] = self.old_A[i][key].detach()
                self.old_Q[i][key] = self.old_Q[i][key].detach()
                self.old_V[i][key] = self.old_V[i][key].detach()

    def init_matrices(self, reverse=False, init_K=True):
        for key in self.lora_keys:
            self.Q[key] = nn.Parameter(torch.zeros_like(self.Q[key], device=self.device), requires_grad=True)
            if init_K:
                self.K[key] = nn.Parameter(torch.zeros_like(self.K[key], device=self.device), requires_grad=False)
            else:
                self.K[key] = self.K[key].to(self.device)
            self.V[key] = nn.Parameter(torch.zeros_like(self.V[key], device=self.device), requires_grad=True)
            self.cur_B[key] = torch.cat((self.Q[key], self.K[key], self.V[key]))  # Q  # K  # V
            self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
            if not reverse:
                nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.cur_B[key], a=math.sqrt(5))

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_A = deepcopy(server_info["cur_A"])
        self.Q = deepcopy(server_info["Q"])
        self.K = deepcopy(server_info["K"])
        self.V = deepcopy(server_info["V"])
        self.old_A = deepcopy(server_info["old_A"])
        self.old_Q = deepcopy(server_info["old_Q"])
        self.old_V = deepcopy(server_info["old_V"])
        self.class_protos = deepcopy(server_info["proto"])
        self.detach()
        for key in self.lora_keys:
            self.Q[key].requires_grad = True
            self.K[key].requires_grad = False
            self.V[key].requires_grad = True
            self.cur_B[key] = torch.cat((self.Q[key], self.K[key], self.V[key]))
            self.cur_A[key].requires_grad = True
        for i in range(self.cur_task):
            for param in self.class_protos[i]:
                param = param.detach()
        for param in self.class_protos[self.cur_task]:
            param.requires_grad = True
        if isinstance(self.network.module.model, T5Model):
            self.network.head.load_state_dict(server_info["head"])
        else:
            self.network.model.head.load_state_dict(server_info["head"])
        # for p in self.network.model.head.parameters():
        #    p.requires_grad = True
        self.head = {
            key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
            for key in self.head_keys
        }
        backbone_params, head_params = self.split_backbone_head()
        params = [
            {"params": backbone_params, "lr": self.lr_back},
            {"params": head_params},
            {"params": self.class_protos[self.cur_task]},
        ]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        self.cur_round += 1
        for i in range(self.cur_task):
            for key in self.lora_keys:
                self.old_B[i][key] = self.old_B[i][key].detach()
                self.old_A[i][key] = self.old_A[i][key].detach()
                self.old_Q[i][key] = self.old_Q[i][key].detach()
                self.old_V[i][key] = self.old_V[i][key].detach()
                self.old_B[i][key] = self.old_B[i][key].to(self.device)
                self.old_A[i][key] = self.old_A[i][key].to(self.device)
                self.old_Q[i][key] = self.old_Q[i][key].to(self.device)
                self.old_V[i][key] = self.old_V[i][key].to(self.device)

    def begin_round_server(self):
        self.cur_round += 1

    def end_round_client(self, dataloader: DataLoader):
        self.network.eval()
        self.optimizer.zero_grad()
        self.optimizer = None
        Lora.end_round_client(self, dataloader)
        # computing average class-wise features from dataset
        average_features = torch.zeros(self.cpt, 768).to(self.device)
        eps = 1e-10
        classes = []
        counts = torch.zeros(self.cpt, device=self.device)
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                prelogits, _ = self.network(inputs, penultimate=True)
                average_features[labels % self.cpt] += prelogits
                labs, nums = torch.unique(labels, return_counts=True)
                classes += labs.tolist()
                counts[labs % self.cpt] += nums
        average_features /= counts.unsqueeze(1)
        classes = list(set(classes))
        classes_indexes = [c % self.cpt for c in classes]
        self.average_features = average_features[classes_indexes].to("cpu")
        self.classes = classes
        # protos = torch.cat([self.class_protos[self.cur_task][i] for i in range(self.cpt)])
        # features_distances = (protos - average_features).pow(2).sum(-1)
        # reciprocal = 1 / (features_distances + eps)

    def end_round_server(self, client_info: List[dict]):
        with torch.no_grad():
            if self.avg_type == "weighted":
                total_samples = sum([client["num_train_samples"] for client in client_info])
                norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            else:
                weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
                norm_weights = [w / sum(weights) for w in weights]
            Qs = [client["Q"] for client in client_info]  # list of Q matrices for all clients
            Ks = [client["K"] for client in client_info]  # list of K matrices for all clients
            Vs = [client["V"] for client in client_info]  # list of V matrices for all clients
            cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
            protos = [client["proto"] for client in client_info]
            avg_feat = [client["avg_feat"] for client in client_info]
            classes = [client["classes"] for client in client_info]
            total_classes = list(set([item for sublist in classes for item in sublist]))
            total_classes.sort()
            self.to("cpu")
            # fedavg on Lora matrices
            for key in self.lora_keys:
                self.Q[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(Qs, norm_weights)]).sum(0)
                )
                self.K[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(Ks, norm_weights)]).sum(0)
                )
                self.V[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(Vs, norm_weights)]).sum(0)
                )
                self.cur_A[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(0)
                )
            torch.cuda.empty_cache()
            merging_alphas = (
                torch.tensor([-float("inf") for i in range(len(client_info) * self.cpt)])
                .reshape(len(client_info), self.cpt)
                .to(self.device)
            )
            merging_weights = []
            avg_feat_per_class = []
            eps = 1e-10
            for num_c, c in enumerate(total_classes):
                feats = torch.tensor([], device=self.device)
                for client in client_info:
                    if c in client["classes"]:
                        idx = client["classes"].index(c)
                        feats = torch.cat((feats, client["avg_feat"][idx].unsqueeze(0).to(self.device)))
                protos = torch.stack([client["proto"][num_c].to(self.device) for client in client_info])
                distances = (
                    protos.pow(2).sum(1, keepdim=True)
                    + feats.pow(2).sum(1, keepdim=True).T
                    - 2 * (torch.matmul(protos, feats.T))
                ).sqrt()
                centers_distances = distances.sum(1)
                reciprocal = 1 / (centers_distances + eps)
                normalized = (reciprocal - reciprocal.min()) / (reciprocal.max() - reciprocal.min())
                soft = F.softmax(normalized * self.soft_temp, dim=0)
                new_proto = (protos * soft.unsqueeze(1)).sum(0).unsqueeze(0)
                self.class_protos[self.cur_task][num_c] = nn.Parameter(new_proto, requires_grad=False)
            torch.cuda.empty_cache()
            del Qs, Ks, Vs, cl_A, client_info
            # print(f"Time for head: {end2 - end} seconds")
            self.detach()
            self.set_optimization()

    def end_task_client(self, dataloader: DataLoader, server_info: dict):
        pass

    def end_task_server(self, client_info: List[dict] = None):
        pass

    def set_optimization(self, fabric=True):
        with torch.no_grad():
            self.optimization_dict = deepcopy(dict(self.network.state_dict()))
            for key in self.optimization_dict.keys():
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            for key in self.lora_keys:
                self.cur_B[key] = self.cur_B[key].detach()
                self.cur_A[key] = self.cur_A[key].detach()
                self.cur_B[key] = self.cur_B[key].to(self.device)
                self.cur_A[key] = self.cur_A[key].to(self.device)
                self.Q[key] = self.Q[key].detach()
                self.K[key] = self.K[key].detach()
                self.V[key] = self.V[key].detach()
                self.Q[key] = self.Q[key].to(self.device)
                self.K[key] = self.K[key].to(self.device)
                self.V[key] = self.V[key].to(self.device)
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            for key in self.lora_keys:
                Q = self.Q[key] + (
                    torch.stack([self.old_Q[i][key].detach().to(self.device) for i in range(self.cur_task)]).sum(0)
                    if self.cur_task > 0
                    else 0
                )
                K = self.K[key]
                V = self.V[key] + (
                    torch.stack([self.old_V[i][key].detach().to(self.device) for i in range(self.cur_task)]).sum(0)
                    if self.cur_task > 0
                    else 0
                )
                self.cur_B[key] = torch.cat((Q, K, V))
                B = self.cur_B[key].to(
                    self.device
                )  # + (torch.stack([self.old_B[i][key].detach() for i in range(self.cur_task)]).sum(0) if self.cur_task > 0 else 0)
                A = self.cur_A[key] + (
                    torch.stack([self.old_A[i][key].detach().to(self.device) for i in range(self.cur_task)]).sum(0)
                    if self.cur_task > 0
                    else 0
                )
                self.optimization_dict[key] += B @ A

    def to(self, device="cpu", only_trainable=True):
        self.network = self.network.to(device)
        for key in self.lora_keys:
            self.Q[key] = self.Q[key].to(device)
            self.K[key] = self.K[key].to(device)
            self.V[key] = self.V[key].to(device)
            self.cur_A[key] = self.cur_A[key].to(device)
            self.cur_B[key] = self.cur_B[key].to(device)
        for key in self.head_keys:
            self.head[key] = self.head[key].to(device)
        for i in range(self.cur_task):
            for key in self.lora_keys:
                self.old_B[i][key] = self.old_B[i][key].to(device)
                self.old_A[i][key] = self.old_A[i][key].to(device)
        for i in range(self.cur_task):
            self.class_protos[i] = self.class_protos[i].to(device)
        return self

    def detach(self):
        for param in self.class_protos[self.cur_task]:
            param = param.detach()
        for key in self.lora_keys:
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.Q[key] = self.Q[key].detach()
            self.K[key] = self.K[key].detach()
            self.V[key] = self.V[key].detach()
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
