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

from models.lora import Lora

@register_model("lora_regmean")
class LoraRegMean(Lora):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1.0,
        r: int = 16,
        enable_lora: list = [True, True, True],
        lora_head: str_to_bool = True,
        cl_merge: str = "run_sum",
        regmean_all: str_to_bool = True,
    ) -> None:
        super(LoraRegMean, self).__init__(fabric, network, device, optimizer, lr, wd_reg, avg_type, lora_alpha, r, enable_lora, lora_head, cl_merge)
        self.regmean_all = regmean_all
        self.lora_modules = []
        for name, module in self.network.named_modules():
            #if ((("qkv" in name or "mlp" in name or ("proj" in name and "attn" in name)) and self.regmean_all) or "head" in name) and not "drop" in name and not "act" in name and not "norm" in name:
            #list(module.parameters())
            if ((("qkv" in name or "mlp" in name or ("proj" in name and "attn" in name)) and self.regmean_all) or "head" in name) and len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                self.lora_modules.append(name)
        self.features = {key: torch.tensor([], dtype=torch.float32) for key in self.lora_modules}
        self.gram = {key: torch.tensor([], dtype=torch.float32) for key in self.lora_modules}



    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        hooks = {name : None for name in self.lora_modules}
        for name, module in self.network.named_modules():
            if name in self.lora_modules:
                #module.forward_handle = module.register_forward_hook(self.hook_forward)
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            #TODO handle the fact that the network is not updated
            self.network(x)
        for name, module in self.network.named_modules():
            if name in self.lora_modules:
                #hooks[name].remove()
                tmp = self.features[name].to(self.device)
                self.gram[name] = (tmp.T @ tmp).to("cpu")
                self.features[name] = torch.tensor([], dtype=torch.float32)
                hooks[name].remove()
        print("End of round")

    def hook_handler(self, name):
        def hook_forward(module, inputs, _):
            x = inputs[0].detach()
            if len(x.shape) == 3:
                x = x.view(-1, x.size(-1))
            self.features[name] = torch.cat([self.features[name], x.cpu()])
        return hook_forward
    
    def get_client_info(self, dataloader: DataLoader):
        client_info = super().get_client_info(dataloader)
        client_info["grams"] = deepcopy(self.gram)
        return client_info
    

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
        cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
        if not self.lora_head:
            heads = [client["head"] for client in client_info]  # list of head matrices for all clients
            head_sd = self.network.model.head.state_dict()
            for key in head_sd.keys():
                head_sd[key] = torch.stack(
                    [head[key] * norm_weight for head, norm_weight in zip(heads, norm_weights)]
                ).sum(0)
            self.network.model.head.load_state_dict(head_sd)

        #regmean solution
        w_solution = {name : None for name in self.lora_modules}
        for name, key in zip(self.lora_modules, self.lora_keys):
            w_solution[name] = torch.cat([client_B[key] @ client_A[key] @ client["grams"][name] for client, client_B, client_A in zip(client_info, cl_B, cl_A)]).sum(0) / torch.cat([client["grams"][name] for client in client_info]).sum(0)
            #self.features[name] = torch.stack([client["grams"][name] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)
        
        if len(client_info) > 0:
            for key in self.lora_keys:
                self.cur_B[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(0)
                )
                self.cur_A[key] = nn.Parameter(
                    torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(0)
                )
            
        lora_opt = torch.optim.SGD(list(self.cur_B.values()) + list(self.cur_A.values()), lr=1e-3, weight_decay=0)
        num_epochs = 100
        criterion = torch.nn.MSELoss()
        losses = []
        for name, key in zip(self.lora_modules, self.lora_keys):
            for epoch in range(num_epochs):
                lora_opt.zero_grad()
                res = self.cur_B[key] @ self.cur_A[key]
                loss = criterion(res, w_solution[name])
                loss.backward()
                lora_opt.step()
            losses.append(loss.detach())
        print(f"Reconstruction loss: {sum(losses) / len(losses)}")


        self.set_optimization()
