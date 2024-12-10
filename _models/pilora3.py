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


class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_qs,
        linear_b_qs,
        linear_a_vs,
        linear_b_vs,
    ):
        super().__init__()
        self.qkv = qkv
        for i in range(len(linear_a_qs)):
            setattr(self, f'linear_a_q_{i}', linear_a_qs[i])
            setattr(self, f'linear_b_q_{i}', linear_b_qs[i])
            setattr(self, f'linear_a_v_{i}', linear_a_vs[i])
            setattr(self, f'linear_b_v_{i}', linear_b_vs[i])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.lora_id = 0
    
    def change_lora(self, num):
        self.lora_id = num

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        for i in range(self.lora_id):
            linear_a_q = getattr(self, f'linear_a_q_{i}')
            linear_b_q = getattr(self, f'linear_b_q_{i}')
            linear_a_v = getattr(self, f'linear_a_v_{i}')
            linear_b_v = getattr(self, f'linear_b_v_{i}')
            new_q = linear_b_q(linear_a_q(x))
            new_v = linear_b_v(linear_a_v(x))
            qkv[:, :, : self.dim] += new_q
            qkv[:, :, -self.dim :] += new_v
        linear_a_q = getattr(self, f'linear_a_q_{self.lora_id}')
        linear_b_q = getattr(self, f'linear_b_q_{self.lora_id}')
        linear_a_v = getattr(self, f'linear_a_v_{self.lora_id}')
        linear_b_v = getattr(self, f'linear_b_v_{self.lora_id}')
        new_q = linear_b_q(linear_a_q(x))
        new_v = linear_b_v(linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

@register_model("pilora3")
class PiLora(BaseModel):
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
        r: int = 4,
        temp: float = 1,
        lr_back: float = -1,
        num_tasks: int = 10,
        soft_temp: float = 5,
    ) -> None:
        BaseModel.__init__(self, 
                           fabric=fabric, 
                           network=network, 
                           device=device, 
                           optimizer=optimizer, 
                           lr=lr, 
                           wd_reg=wd_reg)
        for p in self.network.parameters():
            p.requires_grad = False
        self.wd_reg = wd_reg
        self.lr = lr
        self.r = r
        self.clip_grad = clip_grad
        self.optimizer_str = optimizer
        num = num_tasks
        self.lr_back = lr_back
        if self.lr_back < 0:
            self.lr_back = lr
        blk_full = self.network.model.blocks[0] if not "T5" in str(type(self.network.module)) else self.network.model.encoder.block[0]
        w_qkv_linear = blk_full.attn.qkv
        self.dim = w_qkv_linear.in_features
        w_a_linear_qs = []
        w_b_linear_qs = []
        w_a_linear_vs = []
        w_b_linear_vs = []
        for i in range(num):
            w_a_linear_q = nn.Linear(self.dim, r, bias=False, device=self.device)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False, device=self.device)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False, device=self.device)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False, device=self.device)
            nn.init.kaiming_uniform_(w_a_linear_q.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(w_a_linear_v.weight, a=math.sqrt(5))
            nn.init.zeros_(w_b_linear_q.weight)
            nn.init.zeros_(w_b_linear_v.weight)
            w_a_linear_qs.append(w_a_linear_q)
            w_b_linear_qs.append(w_b_linear_q)
            w_a_linear_vs.append(w_a_linear_v)
            w_b_linear_vs.append(w_b_linear_v)
    
        blk_full.attn.qkv = _LoRA_qkv_timm(
            w_qkv_linear,
            w_a_linear_qs,
            w_b_linear_qs,
            w_a_linear_vs,
            w_b_linear_vs,
            )
        self.class_protos = {}
        self.cur_task = -1
        self.num_tasks = num
        self.r = r
        self.temp = temp
        self.soft_temp = soft_temp
        self.lr_back = lr_back
        self.avg_type = avg_type
        self.cur_round = 0
        self.average_features = None
        #torch.set_float32_matmul_precision("high")

    def forward(self, x, fabric=True):
        #prelogits, _ = self.network(x, penultimate=True)
        prelogits = self.network.forward(x, prelogits=True)
        protos = torch.cat([self.class_protos[t][i] for t in range(self.num_tasks) for i in range(len(self.class_protos[t]) if t < self.num_tasks - 1 else self.cpt)])
        score = F.softmax(-torch.cdist(prelogits, protos, p=2), dim=1)
        return score

    def set_requires_grad(self):
        blk : _LoRA_qkv_timm = self.network.model.blocks[0].attn.qkv
        for n, p in blk.named_parameters():
            if str(self.cur_task) in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for task in range(self.cur_task):
            for param in self.class_protos[task]:
                param.requires_grad = False
        for param in self.class_protos[self.cur_task]:
            param.requires_grad = True
        
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        fabric = True
        self.optimizer.zero_grad()
        #self.cur_B[self.lora_keys[0]].retain_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            #prelogits, outputs = functional_call(self.network, optimization_dict, inputs, kwargs={'penultimate' : True})
            #prelogits, _ = self.network(inputs, penultimate=True)
            prelogits = self.network.forward(inputs, prelogits=True)
            labels_one_hot = torch.nn.functional.one_hot(labels % self.cpt, self.cpt).float()
            self.average_features += labels_one_hot.T @ prelogits.detach()
            labs, nums = torch.unique(labels, return_counts=True)
            self.classes[labs % self.cpt] += nums
            loss = 0
            # protos = torch.cat([self.class_protos[t][i] for t in range(self.num_tasks) for i in range(self.cpt)])
            # temporal fix to make it work with cars, will need a better fix later on, but it works for now
            protos = torch.cat([self.class_protos[t][i] for t in range(self.num_tasks) for i in range(len(self.class_protos[t]) if t < self.num_tasks - 1 else self.cpt)])
            distances = prelogits.pow(2).sum(1, keepdim=True) + protos.pow(2).sum(1, keepdim=True).T - 2 *(torch.matmul(prelogits, protos.T))
            distances /= 768
            distances = distances.sqrt()
            loss_dce = F.cross_entropy(-distances / self.temp, labels)
            loss += loss_dce
            #if self.use_pl:
            loss_pl = torch.index_select(distances, dim=1, index=(labels))
            loss_pl = torch.diagonal(loss_pl)
            loss_pl = torch.mean(loss_pl)
            loss += 0.001 * loss_pl
            #if self.use_ort and self.cur_task > 0:
            blk : _LoRA_qkv_timm = self.network.model.blocks[0].attn.qkv
            loss_ort = torch.tensor(0., device=self.device)
            if self.cur_task > 0:
                for i in range(self.cur_task):
                    #loss_ort += torch.abs(torch.mm(self.cur_A[key], self.old_A[i][key].T)).sum()
                    loss_ort += torch.abs(torch.mm(getattr(blk, f'linear_a_q_{self.cur_task}').weight, getattr(blk, f'linear_a_q_{i}').weight.T)).sum() + torch.abs(torch.mm(getattr(blk, f'linear_a_v_{self.cur_task}').weight, getattr(blk, f'linear_a_v_{i}').weight.T)).sum()
            loss += 0.5 * loss_ort
            #loss_l1 = torch.linalg.matrix_norm(getattr(blk, f'linear_a_q_{self.cur_task}').weight, ord=1) + torch.linalg.matrix_norm(getattr(blk, f'linear_a_v_{self.cur_task}').weight, ord=1)
            loss_l1 = torch.norm(getattr(blk, f'linear_a_q_{self.cur_task}').weight, p=1) + torch.norm(getattr(blk, f'linear_a_v_{self.cur_task}').weight, p=1)
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
        #return loss.item()
        return {"total": loss.item(), "dce": loss_dce.item(), "pl": loss_pl.item(), "ort": loss_ort.item(), "l1": loss_l1.item()}
            
    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.cur_task == 0:
            for t in range(self.num_tasks):
                self.class_protos[t] = nn.ParameterList([nn.Parameter(0.1*torch.randn(1, 768), requires_grad=False).to(self.device) for i in range(self.cpt)])
        blk : _LoRA_qkv_timm = self.network.model.blocks[0].attn.qkv
        blk.change_lora(self.cur_task)
        #self.class_protos[self.cur_task] = nn.ParameterList([nn.Parameter(0.1*torch.randn(1, 768), requires_grad=True).to(self.device) for i in range(self.cpt)])
        self.cur_round = 0
        self.average_features = torch.zeros(self.cpt, 768).to(self.device)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.train()
        super().begin_round_client(dataloader, server_info)
        self.network.model.blocks[0].attn.qkv = deepcopy(server_info['qkv'])
        self.class_protos = deepcopy(server_info["proto"])
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.set_requires_grad()
        blk : _LoRA_qkv_timm = self.network.model.blocks[0].attn.qkv
        pars = [getattr(blk, f'linear_a_q_{self.cur_task}').weight, getattr(blk, f'linear_a_v_{self.cur_task}').weight, getattr(blk, f'linear_b_q_{self.cur_task}').weight, getattr(blk, f'linear_b_v_{self.cur_task}').weight]
        params = [{"params": pars, "lr": self.lr_back}, {"params": list(self.class_protos[self.cur_task]), "lr": self.lr}]
        #params = list(self.network.model.blocks[0].attn.qkv.parameters()) + list(self.class_protos[self.cur_task])
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        self.cur_round += 1
        self.classes = torch.tensor([0 for i in range(self.cpt)], device=self.device)
        self.average_features = torch.zeros(self.cpt, 768).to(self.device)
    
    def begin_round_server(self):
        self.cur_round += 1

    def end_round_client(self, dataloader: DataLoader):
        self.network.eval()
        self.optimizer.zero_grad()
        self.optimizer = None
        classes = []
        average_features = torch.zeros(self.cpt, 768).to(self.device)
        for i in range(self.cpt):
            if self.classes[i] > 5:
                classes.append(i)
                self.average_features[i] /= self.classes[i]
        self.average_features = self.average_features[classes].to("cpu")
        self.classes = classes

        ##computing average class-wise features from dataset
        #average_features = torch.zeros(self.cpt, 768).to(self.device)
        #eps = 1e-10
        #classes = []
        #counts = torch.zeros(self.cpt, device=self.device)
        #with torch.no_grad():
        #    for inputs, labels in tqdm(dataloader):
        #        inputs = inputs.to(self.device)
        #        labels = labels.to(self.device)
        #        prelogits = self.network(inputs, pilora=True)
        #        labels_one_hot = torch.nn.functional.one_hot(labels, self.cpt).float()
        #        average_features += labels_one_hot.T @ prelogits
        #        labs, nums = torch.unique(labels, return_counts=True)
        #        classes += labs.tolist()
        #        counts[labs % self.cpt] += nums
        #average_features /= counts.unsqueeze(1)
        #classes = list(set(classes))
        #classes.sort()
        #classes_indexes = [c % self.cpt for c in classes]
        #self.average_features = average_features[classes_indexes].to("cpu")
        #self.classes = classes

    def get_client_info(self, dataloader: DataLoader):
        client_info = {}
        client_info["qkv"] = deepcopy(self.network.model.blocks[0].attn.qkv)
        client_info["proto"] = deepcopy(self.class_protos[self.cur_task])
        client_info["avg_feat"] = deepcopy(self.average_features)
        client_info["classes"] = deepcopy(self.classes)
        client_info["num_train_samples"] = len(dataloader.dataset)
        return client_info
    
    def get_server_info(self):
        server_info = {}
        server_info["qkv"] = deepcopy(self.network.model.blocks[0].attn.qkv)
        server_info["proto"] = deepcopy(self.class_protos)
        return server_info
    
    def end_round_server(self, client_info: List[dict]):
        with torch.no_grad():
            if self.avg_type == "weighted":
                total_samples = sum([client["num_train_samples"] for client in client_info])
                norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            else:
                weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
                norm_weights = [w / sum(weights) for w in weights]
            protos = [client["proto"] for client in client_info]
            avg_feat = [client["avg_feat"] for client in client_info]
            classes = [client["classes"] for client in client_info]
            total_classes = list(set([item for sublist in classes for item in sublist]))
            total_classes.sort()
            self.to("cpu")
            blk : _LoRA_qkv_timm = self.network.model.blocks[0].attn.qkv
            linear_a_q_weights = torch.stack([getattr(client_info[i]["qkv"], f'linear_a_q_{self.cur_task}').weight * norm_weight for i, norm_weight in enumerate(norm_weights)]).sum(0)
            linear_b_q_weights = torch.stack([getattr(client_info[i]["qkv"], f'linear_b_q_{self.cur_task}').weight * norm_weight for i, norm_weight in enumerate(norm_weights)]).sum(0)
            linear_a_v_weights = torch.stack([getattr(client_info[i]["qkv"], f'linear_a_v_{self.cur_task}').weight * norm_weight for i, norm_weight in enumerate(norm_weights)]).sum(0)
            linear_b_v_weights = torch.stack([getattr(client_info[i]["qkv"], f'linear_b_v_{self.cur_task}').weight * norm_weight for i, norm_weight in enumerate(norm_weights)]).sum(0)
            a_q = getattr(blk, f'linear_a_q_{self.cur_task}')
            a_q.weight = nn.Parameter(linear_a_q_weights)
            b_q = getattr(blk, f'linear_b_q_{self.cur_task}')
            b_q.weight = nn.Parameter(linear_b_q_weights)
            a_v = getattr(blk, f'linear_a_v_{self.cur_task}')
            a_v.weight = nn.Parameter(linear_a_v_weights)
            b_v = getattr(blk, f'linear_b_v_{self.cur_task}')
            b_v.weight = nn.Parameter(linear_b_v_weights)
            self.network.model.blocks[0].attn.qkv = blk
            eps = 1e-10
            for c in total_classes:
                num_c = c % self.cpt
                feats = torch.tensor([], device=self.device)
                for client in client_info:
                    if c in client["classes"]:
                        idx = client["classes"].index(num_c)
                        feats = torch.cat((feats, client["avg_feat"][idx].unsqueeze(0).to(self.device))) #shape (num_clients_w_feature, 768)
                protos = torch.stack([client["proto"][num_c].to(self.device) for client in client_info]).squeeze(1) #shape (num_clients, 768)
                distances = (protos.pow(2).sum(-1, keepdim=True) + feats.pow(2).sum(-1, keepdim=True).T - 2 * (torch.matmul(protos, feats.T))).sqrt()
                centers_distances = distances.sum(1)
                reciprocal = 1 / (centers_distances + eps)
                normalized = (reciprocal - reciprocal.min()) / (reciprocal.max() - reciprocal.min())
                soft = F.softmax(normalized * self.soft_temp, dim=0)
                new_proto = (protos * soft.unsqueeze(1)).sum(0).unsqueeze(0)
                self.class_protos[self.cur_task][num_c] = nn.Parameter(new_proto, requires_grad=False)
            torch.cuda.empty_cache()