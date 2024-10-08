import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models import register_model
from typing import List
from models.utils import BaseModel
from networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool
from models.regmean import RegMean
import os
import shutil
from pathlib import Path

from tqdm import tqdm
from time import time, sleep
from random import random


def extract_block_number_from_module_name(module_name):
    if "head" in module_name:
        return "head"
    return int(module_name.split("blocks.")[1].split(".")[0])

def delete_files_in_directory(directory_path):
   try:
     with os.scandir(directory_path) as entries:
       for entry in entries:
         if entry.is_file():
            os.unlink(entry.path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")


@register_model("regmean_v21")
class RegMean_v2(RegMean):

    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-5,
        wd_reg: float = 0.1,
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
        save_dir: str = "regmean_v2_features",
        gram_fraction: float = 1.0,
        inverse: str_to_bool = "pinv",
    ) -> None:
        gram_fraction = min(1.0, max(0.0, gram_fraction))
        self.gram_fraction = gram_fraction
        if inverse == "inv":
            print("Using torch.inverse().")
            self.inverse = torch.inverse
        else:
            print("Using torch.pinverse().")
            self.inverse = torch.pinverse
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
        self.module_to_block = {name: extract_block_number_from_module_name(name) for name in self.all_gram_modules}
        self.blocks = list(set(self.module_to_block.values()))
        self.temp_features = torch.tensor([], dtype=self.gram_dtype)
        self.save_features = False
        self.saved_features = False
        self.last_used_block = -1
        self.batch_size = batch_size
        #self.save_dir = save_dir
        self.all_norm_modules = []
        self.save_dir = save_dir


        for name, module in self.network.named_modules():
            # if ((("qkv" in name or "mlp" in name or ("proj" in name and "attn" in name)) and self.regmean_all) or "head" in name) and not "drop" in name and not "act" in name and not "norm" in name:
            # list(module.parameters())
            if (
                len(list(module.parameters())) > 0
                and len(list(module.children())) == 0
                and (
                    (
                        ("norm1" in name or ("norm" in name and not "block" in name))
                        and (
                            only_square <= 0
                            or module.state_dict()["weight"].shape[0]
                            == module.state_dict()["weight"].shape[1]
                            == only_square
                        )
                    )
                )
            ):
                self.all_norm_modules.append(name)
                #self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        self.norm_modules = self.all_norm_modules[0:1]


    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels % self.cpt)

        if update:
            self.fabric.backward(loss)
            if self.clip_grad:
                self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
            self.optimizer.step()

        return loss.item()

    def begin_task(self, n_classes_per_task: int):
        #if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)
        return super().begin_task(n_classes_per_task)
    
    def begin_round_server(self):
        if getattr(self, "created_dir", None) is None:
            setattr(self, "created_dir", True)
        if not self.created_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                #shutil.rmtree(self.save_dir)
                sleep(random()*3)
                for i in range(10000):
                    if not os.path.exists(save_dir + "_" + str(i)):
                        save_dir = save_dir + "_" + str(i)
                        print(f"Creating save directory {save_dir}.")
                        break
                    sleep(random()*3)
                if i == 10000:
                    raise Exception("Could not create save directory.")
                self.save_dir = save_dir
        return super().begin_round_server()


    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        RegMean.begin_round_client(self, dataloader, server_info)
        self.save_features = False
        self.saved_features = False
        self.temp_features = torch.tensor([], dtype=torch.float32)
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)
        self.save_dir = server_info["save_dir"]

    def compute_gram_matrices(self, dataloader: DataLoader, idx: int = 0):
        self.network.eval()
        total_samples = len(dataloader.dataset)
        needed_samples = int(total_samples * self.gram_fraction)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        if self.save_features:
            norm_hooks = {name: None for name in self.norm_modules}
            for name, module in self.network.named_modules():
                if name in self.norm_modules:
                    norm_hooks[name] = module.register_forward_hook(self.norm_hook_handler(name))
        hooks = {name: None for name in self.gram_modules}
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                # module.forward_handle = module.register_forward_hook(self.hook_forward)
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        with torch.no_grad():
            #if not self.saved_features:
            exit_loop = False
            path = Path(self.save_dir) / ("features_" + str(idx) + ".pt")
            if not os.path.exists(path):
                for idx, (x, y) in enumerate(dataloader):
                    if idx * x.size(0) > needed_samples:
                        x = x[: needed_samples - (idx - 1) * x.size(0)]
                        y = y[: needed_samples - (idx - 1) * x.size(0)]
                        exit_loop = True
                    x, y = x.to(self.device), y.to(self.device)
                    block = self.module_to_block[self.gram_modules[0]]
                    prev_block = self.blocks[self.blocks.index(block) - 1]
                    if block != 0:
                        x = self.network.forward(x, block = prev_block)
                    self.network.forward(x, block = self.module_to_block[self.gram_modules[0]])
                    if exit_loop:
                        break
            else:
                features = torch.load(path, weights_only=True)
                dl = DataLoader(features, batch_size=self.batch_size, shuffle=False)
                for idx, x in enumerate(dl):
                    if idx * x.size(0) > needed_samples:
                        x = x[: needed_samples - (idx - 1) * x.size(0)]
                        exit_loop = True
                    x = x.to(self.device)
                    block = self.module_to_block[self.gram_modules[0]]
                    prev_block = self.blocks[self.blocks.index(block) - 1]
                    if not self.saved_features:
                        x = self.network.forward(x, block = prev_block)
                    self.network.forward(x, block = self.module_to_block[self.gram_modules[0]])
                    if exit_loop:
                        break
            if self.save_features:
                torch.save(self.temp_features, path)
                self.saved_features = True
            self.temp_features = torch.tensor([], dtype=self.gram_dtype)
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
            if self.save_features:
                if name in self.norm_modules:
                    norm_hooks[name].remove()

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
    
    def norm_hook_handler(self, name):
        name = name
        def hook_forward(module, inputs, _):
            x = inputs[0].detach().to(torch.float32).to("cpu")
            if len(self.temp_features) == 0:
                self.temp_features = x
            else:
                self.temp_features = torch.cat([self.temp_features, x], dim=0)

        return hook_forward

    def set_blk(self, blk: int):
        self.blk_counter = blk
        if blk == -1:
            #self.gram_modules = self.all_gram_modules
            self.gram_modules = []
            self.save_features = False
        else:
            self.gram_modules = [self.all_gram_modules[blk]]
            if self.last_used_block != self.module_to_block[self.gram_modules[0]]:
                self.last_used_block = self.module_to_block[self.gram_modules[0]]
                self.temp_features = torch.tensor([], dtype=self.gram_dtype)
                if self.last_used_block != 0:
                    self.save_features = True
                    if "head" in self.gram_modules[0]:
                        self.norm_modules = [self.all_norm_modules[-1]]
                    else:
                        self.norm_modules = [self.all_norm_modules[self.module_to_block[self.gram_modules[0]]]]
                    self.saved_features = False
                else:
                    self.save_features = False
            else:
                self.save_features = False

    def end_round_client(self, dataloader: DataLoader):
        return
        

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
        self.temp_features = torch.tensor([], dtype=torch.float32)

    def get_server_info(self):
        return {"state_dict": deepcopy(self.network.state_dict()), "save_dir" : self.save_dir}

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
        for blk, key in tqdm(zip(range(self.blk_max), regmean_keys), desc="Computing gram matrices", total=self.blk_max):
            # print(f"Computing gram matrices for layer {blk}, {key}.")
            grams = []
            for i, client in enumerate(self.clients):
                client.to(self.device)
                #client.network.state_dict()[key] = sd[key]
                client.set_blk(blk)
                client.compute_gram_matrices(client_info[i]["dl"], i)
                grams.append(client.features[client.gram_modules[0]].to(dtype))
                client.reset_grams()
                client.to("cpu")
                torch.cuda.empty_cache()
            #merging blk-th layer with regmean
            name = self.middle_names[key]
            grams = torch.stack(grams)
            grams = grams.to(self.device)
            # print(grams.sum(), "Computing result.")
            inv = self.inverse(grams.sum(0))
            sd[key] = (
                torch.stack(
                    [
                        client.network.state_dict()[key].to(self.device).to(dtype) @ grams[i]
                        for i, client in enumerate(self.clients)
                    ]
                ).sum(0)
                @ inv
            ).to(torch.float32)
            assert torch.dist(grams.sum(0) @ inv, torch.eye(grams.size(-1), device=self.device, dtype=dtype)) < 1e-3
            # print("Substituting result.")
            s_sd = self.network.state_dict()
            s_sd[key] = sd[key]
            self.network.load_state_dict(s_sd)
            for client in self.clients:
                c_sd = client.network.state_dict()
                c_sd[key] = sd[key]
                client.network.load_state_dict(c_sd)
        delete_files_in_directory(self.save_dir)

    def __del__(self):
        delete_files_in_directory(self.save_dir)
        if os.path.exists(self.save_dir):
            os.rmdir(self.save_dir)
        super().__del__()