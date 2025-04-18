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
from utils.tools import str_to_bool
from _models.regmean import RegMean
import os
import shutil

from tqdm import tqdm

from _datasets import dataset_factory
from inspect import signature

from utils.training import evaluate


def extract_block_number_from_module_name(module_name):
    if "head" in module_name:
        return "head"
    return int(module_name.split("blocks.")[1].split(".")[0])


@register_model("regmean_v2")
class RegMean_v2(RegMean):

    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-5,
        wd_reg: float = 0.0,
        avg_type: str = "weighted",
        regmean_all: str_to_bool = True,
        alpha_regmean_head: float = 0.5,
        alpha_regmean_backbone: float = -1,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        lr_back: float = -1,
        only_square: int = 0,
        train_bias: str = "all",
        clip_grad: str_to_bool = False,
        train_only_regmean: str_to_bool = False,
        batch_size: int = 32,
        gram_fraction: float = 1.0,
        linear_probe_epochs: int = 0,
        test_lp_after_each_round: str_to_bool = False,
        lp_dataset: str = "same",
        args: dict = None,
    ) -> None:
        gram_fraction = min(1.0, max(0.0, gram_fraction))
        self.gram_fraction = gram_fraction
        self.batch_size = batch_size
        RegMean.__init__(
            self,
            fabric,
            network,
            device,
            optimizer,
            lr,
            wd_reg,
            avg_type,
            regmean_all,
            alpha_regmean_head,
            alpha_regmean_backbone,
            gram_dtype,
            reg_dtype_64,
            lr_back,
            only_square,
            train_bias,
            clip_grad,
        )
        self.blk_counter = 0
        self.blk_max = len(self.gram_modules)
        self.all_gram_modules = deepcopy(self.gram_modules)
        self.train_only_regmean = train_only_regmean
        if not self.train_only_regmean:
            for n, p in network.named_parameters():
                if n in self.middle_names:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        self.classifier = None
        self.linear_probe_epochs = linear_probe_epochs
        self.test_lp = test_lp_after_each_round
        self.lp_dataset = lp_dataset
        self.checked_validation = False


    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            if self.clip_grad:
                self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
            self.optimizer.step()

        return loss.item()
    
    def forward(self, x: torch.Tensor):
        if self.classifier is not None:
            x, _ = self.network(x, penultimate=True)
            x = self.classifier(x)
        else:
            x = self.network(x)
        return x

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        RegMean.begin_round_client(self, dataloader, server_info)

    def begin_task(self, n_classes_per_task: int):
        self.classifier = None
        return super().begin_task(n_classes_per_task)

    def compute_gram_matrices(self, dataloader: DataLoader, idx: int = 0):
        self.network.eval()
        total_samples = len(dataloader.dataset)
        needed_samples = int(total_samples * self.gram_fraction)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        hooks = {name: None for name in self.gram_modules}
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                # module.forward_handle = module.register_forward_hook(self.hook_forward)
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        with torch.no_grad():
            exit_loop = False
            for idx, (x, _) in enumerate(dataloader):
                if idx * x.size(0) > needed_samples:
                    x = x[: needed_samples - (idx - 1) * x.size(0)]
                    if x.size(0) == 0:
                        break
                    exit_loop = True
                x = x.to(self.device)
                self.network(x)
                if exit_loop:
                    break
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                if "head" in name:
                    alpha = self.alpha_regmean[1]
                else:
                    alpha = self.alpha_regmean[0]
                self.features[name] = self.features[name].to("cpu")
                shape = self.features[name].shape[-1]
                self.features[name] = (
                    self.features[name] * alpha
                    + (1 - alpha) * torch.eye(shape, dtype=self.gram_dtype) * self.features[name]
                )
                hooks[name].remove()

    def hook_handler(self, name):
        def hook_forward(module, inputs, _):
            x = inputs[0].detach().to(self.gram_dtype)
            tmp = torch.zeros(x.size(-1), x.size(-1), device=self.device, dtype=self.gram_dtype)
            if len(x.shape) == 3:
                x = x.view(-1, x.size(-1))
            torch.matmul(x.T, x, out=tmp)
            self.features[name] = self.features[name].to(self.device)
            if len(self.features[name]) == 0:
                self.features[name] = tmp
            else:
                self.features[name] += tmp

        return hook_forward

    def set_blk(self, blk: int):
        self.blk_counter = blk
        if blk == -1:
            #self.gram_modules = self.all_gram_modules
            self.gram_modules = []
        else:
            self.gram_modules = [self.all_gram_modules[blk]]

    def end_round_client(self, dataloader: DataLoader):
        return
        #self.compute_gram_matrices(dataloader)
        #self.blk_counter += 1
        #if self.blk_counter == self.blk_max:
        #    self.blk_counter = 0
        #self.set_blk(self.blk_counter)
        

    def get_client_info(self, dataloader: DataLoader):
        client_info = super().get_client_info(dataloader)
        if client_info is None:
            client_info = {}
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        client_info["grams"] = deepcopy(self.features)
        client_info["num_train_samples"] = len(dataloader.dataset.data)
        client_info["self"] = self
        client_info["dl"] = dataloader
        return client_info

    def to(self, device="cpu"):
        self.network.to(device)
        for name in self.gram_modules:
            self.features[name] = self.features[name].to(device)
        return self
    def reset_grams(self):
        for name in self.all_gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)

    def get_server_info(self):
        return {"state_dict": deepcopy(self.network.state_dict())}

    def end_round_server(self, client_info: List[dict]):
        if getattr(self, "clients", None) is None:
            self.clients = []
        self.clients : List[RegMean_v2] = [client["self"] for client in client_info]
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
        # regmean solution
        keys = list(self.network.state_dict().keys())
        regmean_keys = [key for key in keys if "weight" in key and self.middle_names.get(key) is not None]
        fedavg_keys = [key for key in keys if key not in regmean_keys]
        sd = self.network.state_dict()
        if self.train_only_regmean:
            merging_keys = regmean_keys
        else:
            merging_keys = keys
        for client in self.clients:
            client.set_blk(-1)
        #merging all not regmean-able layers with fedavg
        for key in fedavg_keys:
            sd[key] = torch.stack(
                    [client["state_dict"][key] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
                ).sum(
                    0
                ) 
            for client in self.clients:
                c_sd = client.network.state_dict()
                c_sd[key] = sd[key]
                client.network.load_state_dict(c_sd)
        self.network.load_state_dict(sd)
        #loading the state dict to the clients
        for client in self.clients:
            #client.network.load_state_dict(sd)
            client.reset_grams()
        #computing gram matrices on the client-side one layer at a time
        for blk, key in zip(range(self.blk_max), regmean_keys):
            print(f"Computing gram matrices for layer {blk}, {key}.")
            grams = []
            for i, client in enumerate(self.clients):
                client.to(self.device)
                #client.network.state_dict()[key] = sd[key]
                client.set_blk(blk)
                client.compute_gram_matrices(client_info[i]["dl"], i)
                grams.append(client.features[client.gram_modules[0]].to(dtype))
                client.reset_grams()
                client.to("cpu")
            #merging blk-th layer with regmean
            name = self.middle_names[key]
            grams = torch.stack(grams)
            grams = grams.to(self.device)
            print(grams.sum())
            sd[key] = (
                torch.stack(
                    [
                        client.network.state_dict()[key].to(self.device).to(dtype) @ grams[i]
                        for i, client in enumerate(self.clients)
                    ]
                ).sum(0)
                @ torch.pinverse(grams.sum(0))
            ).to(torch.float32)
            s_sd = self.network.state_dict()
            s_sd[key] = sd[key]
            self.network.load_state_dict(s_sd)
            for client in self.clients:
                c_sd = client.network.state_dict()
                c_sd[key] = sd[key]
                client.network.load_state_dict(c_sd)
        if self.linear_probe_epochs > 0 and self.test_lp:
            self.do_linear_probe([client["dl"] for client in client_info])
            print("Linear probe done.")
            print("Evaluating after linear probe...")
        #if not self.checked_validation:
        #    if self.linear_probe_epochs > 0 and self.lp_dataset != "same":
        #        DatasetClass = dataset_factory(self.lp_dataset)
        #        dataset_signature = list(signature(DatasetClass.__init__).parameters.keys())[1:]
        #        dataset = DatasetClass()

    def end_task_client(self, dataloader: DataLoader = None, server_info: dict = None):
        return dataloader

    def end_task_server(self, client_info: List[dict] = None):
        if self.linear_probe_epochs == 0:
            return
        return self.do_linear_probe([client["dl"] for client in client_info])
        

    def do_linear_probe(self, data_loaders: List[DataLoader]):
        self.network.train()
        features = []
        labels_ = []
        embed_dim = self.network.model.head.weight.shape[1]
        num_classes = self.network.model.head.weight.shape[0]
        self.classifier = nn.Linear(embed_dim, num_classes).to(self.device)
        nn.init.xavier_normal_(self.classifier.weight)
        torch.cuda.empty_cache()
        for dl in data_loaders:
            for i, (inputs, labels) in enumerate(dl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with self.fabric.autocast(), torch.no_grad():
                    prelogits, _ = self.network(inputs, penultimate=True)
                features.append(prelogits.to("cpu"))
                labels_.append(labels.to("cpu"))
        features = torch.cat(features)
        labels_ = torch.cat(labels_)
        batch_size = 256
        lr = 1e-3
        params = [{"params": self.classifier.parameters()}]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        optimizer = OptimizerClass(params, lr=lr, weight_decay=self.wd_reg)
        for epoch in tqdm(range(self.linear_probe_epochs)):
            for i in range(0, len(features), batch_size):
                inputs, labels = features[i : i + batch_size].to(self.device), labels_[i : i + batch_size].to(self.device)
                optimizer.zero_grad()
                outputs = self.classifier(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
                loss = self.loss(outputs, labels - self.cur_offset)
                loss.backward()
                optimizer.step()
        self.network.eval()
        return None