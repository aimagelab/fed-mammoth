import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models import register_model
from typing import List
from models._utils import BaseModel
from networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool, compute_fisher_expectation_fabric

from models.lora import Lora, merge_AB, zero_pad
from models.regmean import RegMean
from models.lora_pre import Lora
from tqdm import tqdm
import math


@register_model("lora_reg_alt2_pre")
class LoraRegMeanAlt(Lora, RegMean):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 0.0003,
        clip_grad: str_to_bool = False,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1,
        r: int = 16,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        regmean_all: str_to_bool = True,
        alpha_regmean_head: float = 0.5,
        alpha_regmean_backbone: float = -1,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        lr_back : float = -1,
        only_square: int = 0,
        train_bias: str = "all",
        train_matrix: str = "alt",
        fisher_maxiter: int = -1,
    ) -> None:
        self.fisher_maxiter = fisher_maxiter
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
        )
        for name in self.gram_modules:
            self.middle_names[name.removeprefix("_forward_module.").removeprefix("module.") + ".weight"] = name
        assert train_matrix.lower() in ["alt", "a", "b"]
        self.train_matrix = train_matrix
        if "alt" not in self.train_matrix:
            self.cur_train_matrix = self.train_matrix
        else:
            self.cur_train_matrix = "B"
        self.cur_round = 0
        self.set_train_matrix()
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        del self.old_delta

    def split_backbone_head(self):
        backbone_params = []
        head_params = []
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            backbone_params.append(self.cur_B[key])
            backbone_params.append(self.cur_A[key])
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
            self.head[key].requires_grad = True
            head_params.append(self.head[key])
        return backbone_params, head_params

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
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict


    def __compute_fisher_hooks(self, modules, param_resolution_dict,
                    dataloader, debug_mode: bool=False, forward = 1):
        """
        all_param_lored is the list of all the parameters that are lored, so net + lora
        """
        #all_param_lored_names = list(param_resolution_dict.keys()) + list(param_resolution_dict_cls.keys())
        all_param_lored_names = modules
        def to_be_fishered(name):
            #if f"{name}.weight" in all_param_lored_names or f"{name}.bias" in all_param_lored_names:
            if f"{name}" in all_param_lored_names:
                return True
            else:
                return False

        def hook_backward(module, _, grad_output):
            grad_out = grad_output[0]
            inputs = module.inputs
            if len(grad_out.shape) > 2:
                grad_out = grad_out.reshape(-1, grad_out.shape[-1])
                inputs = inputs.reshape(-1, inputs.shape[-1])
                grad_weight = (grad_output[0].permute(0, 2, 1) @ module.inputs).pow(2).sum(0)
            else:
                grad_weight = grad_out.T.pow(2) @ inputs.pow(2)
            
            if hasattr(module, "bias") and module.__compute_bias:
                grad_bias = grad_out.T.pow(2).sum(-1)
                if not hasattr(module, "fisher_bias"):
                    setattr(module, "fisher_bias", grad_bias)
                else:
                    module.fisher_bias += grad_bias


            if not hasattr(module, "fisher_weight"):
                setattr(module, "fisher_weight", grad_weight)
            else:
                module.fisher_weight += grad_weight

        def hook_forward(module, inputs, _):
            setattr(module, "inputs", inputs[0])
        
        # insert hooks
        for name, module in self.network.named_modules():
            if to_be_fishered(name):
                if f"{name}.bias" in all_param_lored_names:
                    module.__compute_bias = True
                else:
                    module.__compute_bias = False
                module.backward_handle = module.register_full_backward_hook(hook_backward)
                module.forward_handle = module.register_forward_hook(hook_forward)

        require_grads_list = []
        for param in self.network.parameters():
            require_grads_list.append(param.requires_grad)
            param.requires_grad = False
            
        num_of_examples = 0
        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)
        for j, (examples, _) in enumerate(tqdm(dataloader, total=len(dataloader), desc='FISHER computation')):
            if j >= self.fisher_maxiter and self.fisher_maxiter > 0:
                break
            examples = examples.to(self.device)
            num_of_examples += examples.shape[0]
            if forward == 1:
                probs = torch.softmax(self.forward(examples * fake_param, fabric=False)[:, self.cur_offset : self.cur_offset + self.cpt], dim=1)
            else:
                probs = torch.softmax(self.forward_2(examples * fake_param)[:, self.cur_offset : self.cur_offset + self.cpt], dim=1)
            detached_probs = probs.detach()
            log_probs = torch.log(probs)
            fisher_sqrt = (detached_probs.sqrt() * log_probs).sum(0)
            
            for i, fish in enumerate(fisher_sqrt):
                fish.backward(
                    retain_graph=True if (i < fisher_sqrt.shape[0] - 1) else False
                )

        # remove hooks
        for name, module in self.network.named_modules():
            if to_be_fishered(name):
                module.backward_handle.remove()
                module.forward_handle.remove()
                module.inputs = None
        

        #fisher = {}
        #for (name, module), key in zip(self.network.named_modules(), self.lora_keys):
        #    for typ in ["weight", "bias"]:
        #        if f"{name}.{typ}" in param_resolution_dict:
        #            fisher[param_resolution_dict[f"{name}.{typ}"]] = getattr(module, f"fisher_{typ}")
        #            setattr(module, f"fisher_{typ}", 0)
        fisher = []
        for name, module in self.network.named_modules():
            if name in modules:
                fisher.append(getattr(module, f"fisher_weight").reshape(-1))
        fisher = torch.cat(fisher)

        #fisher_cls = {}
        #for (name, module) in self.network.named_modules():
        #    for typ in ["weight", "bias"]:
        #        if f"{name}.{typ}" in param_resolution_dict_cls:
        #            fisher_cls[param_resolution_dict_cls[f"{name}.{typ}"]] = getattr(module, f"fisher_{typ}")
        #            setattr(module, f"fisher_{typ}", 0)

        for param, req_grad in zip(self.network.parameters(), require_grads_list):
            param.requires_grad = req_grad

        return fisher, num_of_examples




    def begin_task(self, n_classes_per_task: int):
        BaseModel.begin_task(self, n_classes_per_task)
        #server
        if "fisher" in self.cl_merge and getattr(self, "old_delta_fisher", None) is not None: 
            self.to("cpu")
            for key in self.lora_keys:
                self.old_delta_fisher[key] = self.old_delta_fisher[key] + (
                    (self.cur_B[key] @ self.cur_A[key]) * self.cur_fisher[key]
                )
                self.old_fisher[key] += self.cur_fisher[key]
                # filler in order to test something meaningful after each comm round (not the last one)
                self.old_delta[key] = (
                    self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                ) / self.cur_task
            self.to(self.device)
            self.init_matrices()
        else:
            if self.cur_task > 0 :
                self.init_matrices()
        self.cur_round = 0

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        for key in self.lora_keys:
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
        #self.optimization_dict = deepcopy(server_info["old_delta"])
        self.old_delta = server_info["old_delta"]
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            # for p in self.network.model.head.parameters():
            #    p.requires_grad = True
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }
        if self.lr_back > 0:
            backbone_params, head_params = self.split_backbone_head()
            params = [{"params": backbone_params, "lr": self.lr_back}, {"params": head_params}]
        else:
            if not self.lora_head:
                params = list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.head.values())
            else:
                params = list(self.cur_B.values()) + list(self.cur_A.values())
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        self.set_train_matrix()
        self.cur_round += 1
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)

    def begin_round_server(self):
        if getattr(self, "old_delta", None) is None:
            self.old_delta = {}
            for key in self.lora_keys:
                self.old_delta[key] = torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1])
        self.set_train_matrix()
        self.cur_round += 1

    def end_round_client(self, dataloader: DataLoader):
        self.optimizer.zero_grad()
        self.optimizer = None
        Lora.end_round_client(self, dataloader)
        # setting up the parameters to correctly compute the Gram matrices for the next round
        self.set_optimization()
        for key in self.head_keys:
            self.optimization_dict[key] = self.head[key]
        for name in self.gram_modules:
            self.features[name] = self.features[name].to(self.device)
        RegMean.end_round_client(self, dataloader)  # retrieves Gram matrices from hooks
        self.to("cpu", only_trainable=False)

    def end_round_server(self, client_info: List[dict]):
        with torch.no_grad():
            if self.avg_type == "weighted":
                total_samples = sum([client["num_train_samples"] for client in client_info])
                norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            else:
                weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
                norm_weights = [w / sum(weights) for w in weights]
            cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
            cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
            self.to("cpu")
            #import gc
            dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
            #from time import time
            #start = time()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            if not self.regmean_all:
                # fedavg on Lora matrices for all layers except head
                for key in self.lora_keys:
                    self.cur_B[key] = nn.Parameter(
                        torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(
                            0
                        )
                    )
                    self.cur_A[key] = nn.Parameter(
                        torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(
                            0
                        )
                    )
            else:
                #eps = 5e-7
                keys = list(self.network.state_dict().keys())
                if self.cur_train_matrix == "A":
                    # merge As
                    cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
                    for key in self.lora_keys:
                        if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                            name = self.middle_names[key]
                        print(key)
                        for i in range(len(cl_A)):
                            cl_A[i][key] = cl_A[i][key].to(self.device).to(dtype)
                            client_info[i]["grams"][name] = client_info[i]["grams"][name].to(self.device).to(dtype)
                        B = self.cur_B[key].to(self.device).to(dtype)
                        E = torch.stack(
                            [
                                (B_[key].to(self.device).to(dtype) @ A_[key].to(self.device).to(dtype)) @ client["grams"][name].to(self.device).to(dtype)
                                for B_, A_, client in zip(cl_B, cl_A, client_info)
                            ]
                        ).sum(0)
                        G = torch.stack([client["grams"][name].to(self.device).to(dtype) for client in client_info]).sum(0)
                        A = torch.pinverse(B.T @ B) @ B.T @ E @ torch.pinverse(G)  # A
                        self.cur_A[key] = A.to(torch.float32).to("cpu")
                        for i in range(len(cl_A)):
                            cl_A[i][key] = cl_A[i][key].to("cpu")
                            client_info[i]["grams"][name] = client_info[i]["grams"][name].to("cpu")
                        del E, G, A, B
                        if torch.cuda.memory_reserved() / total_mem > 0.8:
                            torch.cuda.empty_cache()
                        #gc.collect()
                    
                else:
                    # merge Bs
                    print("Merging Bs")
                    cl_B = [client["cur_B"] for client in client_info]  # list of A matrices for all clients
                    for key in self.lora_keys:
                        if "weight" in key and self.middle_names.get(key) is not None and not "head" in key:
                            name = self.middle_names[key]
                        print(key)
                        for i in range(len(cl_B)):
                            cl_B[i][key] = cl_B[i][key].to(self.device).to(dtype)
                            client_info[i]["grams"][name] = client_info[i]["grams"][name].to(self.device).to(dtype)
                        A = self.cur_A[key].to(self.device).to(dtype)
                        E2 = torch.stack(
                            [
                                (B_[key].to(self.device).to(dtype) @ A) @ client["grams"][name].to(self.device).to(dtype)
                                for B_, client in zip(cl_B, client_info)
                            ]
                        ).sum(0)
                        G_inv = torch.pinverse(
                            A @ torch.stack([client["grams"][name].to(self.device).to(dtype) for client in client_info]).sum(0) @ A.T
                        )
                        B = E2 @ A.T @ G_inv  # B
                        self.cur_B[key] = B.to(torch.float32).to("cpu")
                        for i in range(len(cl_B)):
                            cl_B[i][key] = cl_B[i][key].to("cpu")
                            client_info[i]["grams"][name] = client_info[i]["grams"][name].to("cpu")
                        del E2, G_inv, A, B
                        #gc.collect()
                        #print(f"reserved memory: {torch.cuda.memory_reserved()}\tallocated memory: {torch.cuda.memory_allocated()}")
                        if torch.cuda.memory_reserved() / total_mem > 0.8:
                            torch.cuda.empty_cache()
                        #    print(f"reserved memory: {torch.cuda.memory_reserved()}\tallocated memory: {torch.cuda.memory_allocated()}")
                    # bt btb eg
                    # eg ata at
            #end = time()
            torch.cuda.empty_cache()
            #print(f"Time for merging: {end - start} seconds")
            keys = list(self.network.state_dict().keys())
            sd = self.network.state_dict()
            for key in keys:
                # head parameters
                if "head" in key:
                    if self.middle_names.get(key) is not None and "head" in key:  # regmean on Linear layer
                        name = self.middle_names[key]
                        sd[key] = (
                            torch.stack(
                                [
                                    client["state_dict"][key].to(dtype) @ client["grams"][name].to(dtype)
                                    for client in client_info
                                ]
                            ).sum(0)
                            @ torch.inverse(
                                torch.stack([client["grams"][name].to(dtype) for client in client_info]).sum(0)
                            )
                        ).to(torch.float32)
                    else:  # fedavg bias
                        sd[key] = torch.stack(
                            [
                                client["state_dict"][key] * norm_weight
                                for client, norm_weight in zip(client_info, norm_weights)
                            ]
                        ).sum(0)
            #end2 = time()
            torch.cuda.empty_cache()
            del cl_B, cl_A, client_info
            #print(f"Time for head: {end2 - end} seconds")
            self.network.load_state_dict(sd)
            if getattr(self, "old_delta", None) is None:
                self.old_delta = {}
                for key in self.lora_keys:
                    self.old_delta[key] = torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1], requires_grad=False)
            self.detach()
            self.set_optimization()
            for key in self.head_keys:
                self.optimization_dict[key] = self.network.state_dict()[key]
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            self.to(self.device)
            #end3 = time()
            #print(f"Time fir the rest: {end3 - end2} seconds")
    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        client_info = {
            "cur_A": self.cur_A,
            "cur_B": self.cur_B,
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            client_info["head"] = self.network.model.head.state_dict()
        client_info["grams"] = self.features
        client_info["state_dict"] = self.network.state_dict()
        return client_info

    def end_task_client(self, dataloader: DataLoader, server_info: dict):
        fisher = None
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        self.network.model.head.load_state_dict(server_info["head"])
        for name in self.gram_modules:
            self.features[name] = torch.tensor([], dtype=self.gram_dtype)
        if "fisher" in self.cl_merge:
            #classes = set()
            #num_samples = 0
            #for images, labels in dataloader:
            #    num_samples += images.shape[0]
            #    for label in labels:
            #        if label not in classes:
            #            classes.add(label.item())
            #classes = list(sorted(classes))
            #if self.lr_back > 0:
            #    backbone_params, head_params = self.split_backbone_head()
            #    params = [{"params": backbone_params, "lr": self.lr_back}, {"params": head_params}]
            #else:
            #    if not self.lora_head:
            #        params = list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.head.values())
            #    else:
            #        params = list(self.cur_B.values()) + list(self.cur_A.values())
            #OptimizerClass = getattr(torch.optim, self.optimizer_str)
            #self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
            #self.optimizer = self.fabric.setup_optimizers(self.optimizer)
            #self.optimizer.zero_grad()
            if self.regmean_all:
                precision = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision("high")
                # self.detach()
                self.detach()
                # for key in self.lora_keys:
                #    self.cur_B[key].requires_grad = True
                #    self.cur_A[key].requires_grad = True
                #merged_params = {
                #    # key: self.network.state_dict()[key] + (self.cur_B[key] @ self.cur_A[key]) for key in self.lora_keys
                #    key: torch.tensor(self.cur_B[key] @ self.cur_A[key], requires_grad=True)
                #    for key in self.lora_keys
                #}
                merged_params = {
                #    # key: self.network.state_dict()[key] + (self.cur_B[key] @ self.cur_A[key]) for key in self.lora_keys
                    key: torch.tensor(self.cur_B[key] @ self.cur_A[key], requires_grad=False)
                    for key in self.lora_keys
                }
                # adding lora parameters on the network (using .module to get rid of fabric wrapper)
                torch.cuda.empty_cache()
                #merged_params = {
                #    #key: self.network.state_dict()[key] + (self.cur_B[key] @ self.cur_A[key]) for key in self.lora_keys
                #    key: torch.tensor(self.cur_B[key] @ self.cur_A[key], requires_grad=True)
                #    for key in self.lora_keys
                #}
                self.optimization_dict = deepcopy(dict(self.network.module.state_dict()))
                for key in self.lora_keys:
                    self.optimization_dict[key] += merged_params[key]
                #fisher = compute_fisher_expectation_fabric(
                #    network=self,
                #    data_loader=dataloader,
                #    device=self.device,
                #    classes=list(set(range(self.cur_offset, self.cur_offset + self.cpt))),
                #    fabric=None,
                #    parameters=list(merged_params.values()),
                #    maxiter=self.fisher_maxiter,
                #).reshape(-1)
                merged_params = {
                    # key: self.network.state_dict()[key] + (self.cur_B[key] @ self.cur_A[key]) for key in self.lora_keys
                    key: torch.tensor(self.cur_B[key] @ self.cur_A[key], requires_grad=False)
                    for key in self.lora_keys
                }
                for key in self.lora_keys:
                    self.optimization_dict[key] += merged_params[key]
                modules_no_head = [name for name in self.gram_modules if not "head" in name]
                fisher, num_samples = self.__compute_fisher_hooks(modules_no_head, self.optimization_dict, dataloader)
                #keys = list(self.network.state_dict().keys())
                #sd = self.network.state_dict()
                #for key in keys:
                #    sd[key] = self.optimization_dict[key]
                #self.network.load_state_dict(sd)
                #for module in self.network.modules():
                #    setattr(module, "fisher_weight", 0)
                #fisher2, num_samples = self.__compute_fisher_hooks(modules_no_head, self.optimization_dict, dataloader, forward = 2)
                torch.set_float32_matmul_precision(precision)
                #for module in self.network.modules():
                #    setattr(module, "fisher_weight", 0)
                #fisher3, num_samples = self.__compute_fisher_hooks(modules_no_head, self.optimization_dict, dataloader, forward = 1)
                #for module in self.network.modules():
                #    setattr(module, "fisher_weight", 0)
                #fisher4, num_samples = self.__compute_fisher_hooks(modules_no_head, self.optimization_dict, dataloader, forward = 2)
                #fisher1 = fisher1.to("cpu")
                #fisher2 = fisher2.to("cpu")
                fisher[fisher > num_samples] = num_samples
                fisher = fisher.to("cpu")
            return {"fisher": fisher, "num_samples": num_samples}

    def end_task_server(self, client_info: List[dict] = None):
        with torch.no_grad():
            if "fisher" in self.cl_merge:
                try:
                    getattr(self, "old_delta_fisher")
                except AttributeError:
                    self.old_delta_fisher = {
                        key: torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1], requires_grad=False) for key in self.lora_keys
                    }
                try:
                    getattr(self, "old_fisher")
                except AttributeError:
                    self.old_fisher = {
                        key: torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1], requires_grad=False) for key in self.lora_keys
                    }
                fishers = torch.stack([client_info[i]["fisher"] for i in range(len(client_info))])
                num_samples = torch.tensor([client_info[i]["num_samples"] for i in range(len(client_info))]).reshape(
                    -1, 1
                )
                eps = 1e-20
                avg_fisher = fishers.sum(0) / num_samples.sum()
                avg_fisher += eps
                # avg_fisher = avg_fisher.to(self.device)
                del fishers
                self.to("cpu")
                torch.cuda.empty_cache()
                fisher_dict = {}
                counter = 0
                merged_params = {}
                for key in self.lora_keys:
                    merged_params[key] = (self.cur_B[key] @ self.cur_A[key]).detach()
                    fisher_dict[key] = avg_fisher[counter : merged_params[key].numel() + counter].reshape(
                        merged_params[key].shape
                    )
                    counter += merged_params[key].numel()
                with torch.no_grad():
                    self.optimization_dict = deepcopy(dict(self.network.state_dict()))
                    # for key in self.optimization_dict.keys():
                    #    self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                    for key in self.lora_keys:
                        self.old_delta_fisher[key].requires_grad = False
                        self.cur_B[key].requires_grad = False
                        self.cur_A[key].requires_grad = False
                        if self.cur_task > 0:
                            tmp = self.old_delta_fisher[key] + (merged_params[key].detach() * fisher_dict[key])
                            self.optimization_dict[key] += tmp / (self.old_fisher[key] + fisher_dict[key])
                        else:
                            self.optimization_dict[key] += merged_params[key].detach()
                    for key in self.network.state_dict().keys():
                        self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
                try:
                    getattr(self, "cur_fisher")
                except AttributeError:
                    self.cur_fisher = {key: fisher_dict[key] for key in self.lora_keys}
                else:
                    for key in self.lora_keys:
                        self.cur_fisher[key] = fisher_dict[key]
                del fisher_dict
                del merged_params
            # self.to("cpu")

    def set_optimization_cur_task(self, fabric=True):
        self.detach()
        self.to(self.device)
        sd = self.network.state_dict()
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        for key in self.lora_keys:
            if self.cur_task > 0 and not "individual" in self.cl_merge and not "fisher" in self.cl_merge:
                self.old_delta[key] = self.old_delta[key].to(self.device)
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        self.optimization_dict = optimization_dict

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
                self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
            for key in self.lora_keys:
                if self.cur_task > 0:
                    self.old_delta[key] = self.old_delta[key].to(self.device)
                    tmp = (self.old_delta[key] * self.cur_task) + (self.cur_B[key] @ self.cur_A[key])
                    self.optimization_dict[key] += (tmp / (self.cur_task + 1))
                else:
                    self.optimization_dict[key] += (self.cur_B[key] @ self.cur_A[key]).detach()

    def set_train_matrix(self):
        if "alt" in self.train_matrix:
            if "A" in self.cur_train_matrix or self.cur_round == 0:
                # Train B
                for key in self.lora_keys:
                    self.cur_A[key] = self.cur_A[key].detach()
                    if not self.cur_B[key].requires_grad:
                        self.cur_B[key].requires_grad = True
                self.cur_train_matrix = "B"
            else:
                # Train A
                for key in self.lora_keys:
                    if not self.cur_A[key].requires_grad:
                        self.cur_A[key].requires_grad = True
                    self.cur_B[key] = self.cur_B[key].detach()
                self.cur_train_matrix = "A"
        else:
            if "A" in self.train_matrix:
                for key in self.lora_keys:
                    if not self.cur_A[key].requires_grad:
                        self.cur_A[key].requires_grad = True
                    self.cur_B[key] = self.cur_B[key].detach()
            else:
                for key in self.lora_keys:
                    self.cur_A[key] = self.cur_A[key].detach()
                    if not self.cur_B[key].requires_grad:
                        self.cur_B[key].requires_grad = True

    def to(self, device="cpu", only_trainable=True):
        if "cpu" in device or not only_trainable:  # we move everything to the cpu
            self.network = self.network.to(device)
            for key in self.lora_keys:
                self.cur_A[key] = self.cur_A[key].to(device)
                self.cur_B[key] = self.cur_B[key].to(device)
            for key in self.head_keys:
                self.head[key] = self.head[key].to(device)
            for key in self.gram_modules:
                self.features[key] = self.features[key].to(device)
            if getattr(self, "old_delta", None) is not None:
                for key in self.lora_keys:
                    self.old_delta[key] = self.old_delta[key].to(device)
            if getattr(self, "old_delta_fisher", None) is not None:
                for key in self.lora_keys:
                    self.old_delta_fisher[key] = self.old_delta_fisher[key].to(device)
            if getattr(self, "old_fisher", None) is not None:
                for key in self.lora_keys:
                    self.old_fisher[key] = self.old_fisher[key].to(device)
            if getattr(self, "cur_fisher", None) is not None:
                for key in self.lora_keys:
                    self.cur_fisher[key] = self.cur_fisher[key].to(device)
        else:  # we move only the trainable parameters to the device
            self.network = self.network.to(device)
            for key in self.lora_keys:
                self.cur_A[key] = self.cur_A[key].to(device)
                self.cur_B[key] = self.cur_B[key].to(device)
        return self

    def detach(self):
        for key in self.lora_keys:
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
        if getattr(self, "old_delta", None) is not None:
            for key in self.lora_keys:
                self.old_delta[key] = self.old_delta[key].detach()
        if getattr(self, "old_delta_fisher", None) is not None:
            for key in self.lora_keys:
                self.old_delta_fisher[key] = self.old_delta_fisher[key].detach()
        if getattr(self, "old_fisher", None) is not None:
            for key in self.lora_keys:
                self.old_fisher[key] = self.old_fisher[key].detach()
        if getattr(self, "cur_fisher", None) is not None:
            for key in self.lora_keys:
                self.cur_fisher[key] = self.cur_fisher[key].detach()