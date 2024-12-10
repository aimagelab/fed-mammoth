"""
Slow Learner with Classifier Alignment.

Note:
    SLCA USES A CUSTOM BACKBONE (see `feature_extractor_type` argument)

Arguments:
    --feature_extractor_type: the type of convnet to use. `vit-b-p16` is the default: ViT-B/16 pretrained on Imagenet 21k (**NO** finetuning on ImageNet 1k)
"""

import copy

import numpy as np
from torch.utils.data import DataLoader
from _networks.vit_ranpac import target2onehot
#from utils import binary_to_boolean_type

#from models.utils.continual_model import ContinualModel
from _models import BaseModel, register_model


import torch
import torch.nn.functional as F
#from utils.conf import get_device
from _networks.vit_ranpac import RanPAC_Model
from utils.tools import str_to_bool


@register_model("ranpac2")
class RanPAC(BaseModel):
    """RanPAC: Random Projections and Pre-trained Models for Continual Learning."""
    NAME = 'ranpac'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']
    net: RanPAC_Model
    def __init__(
        self,
        fabric,
        network: RanPAC_Model,
        device: str,
        optimizer: str = "SGD",
        momentum: float = 0.9,
        lr: float = 1e-2,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        num_epochs: int = 5,
        rp_size: int = 10000,
        num_classes: int = 100,
        use_scheduler: str_to_bool = False,
    ) -> None:
        self.lr = lr
        self.wd = wd_reg
        super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.avg_type = avg_type
        self.n_seen_classes = 0
        self.num_epochs = num_epochs
        self.rp_size = rp_size
        self.num_classes = num_classes 
        self.Q = None
        self.G = None
        self.cur_round = 0
        self.momentum = momentum
        self.optimizer_str = optimizer
        self.use_scheduler = use_scheduler

    def begin_task(self, dataset):
        # temporarily remove RP weights
        super().begin_task(dataset)
        self.cur_round = 0
        del self.network._network.fc
        self.network._network.fc = None
        self.n_seen_classes += self.cpt

        self.network._network.update_fc(self.n_seen_classes)  # creates a new head with a new number of classes (if CIL)

        if self.cur_task == 0:
            #self.optimizer = self.get_optimizer() already there
            # updating parameters to be optimized
            for n, p in self.network._network.fc.named_parameters():
                self.network._network.params_to_optimize.append(n)
            self.scheduler = self.get_scheduler()
            self.optimizer.zero_grad()
        #else:
        #    self.old_Q = copy.deepcopy(self.Q)
        #    self.old_G = copy.deepcopy(self.G)
        #    self.Q = torch.zeros_like(self.old_Q)
        #    self.G = torch.zeros_like(self.old_G)

    def compare_params(self,):
        for n, p in self.network._network.convnet.named_parameters():
            if n in self.network._network.params_to_optimize:
                if not torch.equal(p, self.old_params[self.names.index(n)]):
                    print(n, " changed", torch.dist(p, self.old_params[self.names.index(n)]))

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        super().begin_round_client(dataloader, server_info)
        pars = []
        self.old_params = []
        self.names = []
        self.Q = copy.deepcopy(server_info["Q"])
        self.G = copy.deepcopy(server_info["G"])
        self.network._network.fc.weight.data = copy.deepcopy(server_info["Wo"])
        if getattr(self.network._network.fc, "W_rand", None) is not None:
            self.network._network.fc.W_rand = copy.deepcopy(server_info["W_rand"])
        else:
            self.W_rand = copy.deepcopy(server_info["W_rand"])
        if self.cur_task == 0:
            i = 0
            for n, p in self.network._network.convnet.named_parameters():
                if n in self.network._network.params_to_optimize:
                    p.data = copy.deepcopy(server_info["backbone_params"][i])
                    i += 1
            for n, p in self.network._network.convnet.named_parameters():
                if n in self.network._network.params_to_optimize:
                    pars.append(p)
                    self.old_params.append(p.clone().detach())
                    self.names.append(n)
            for n, p in self.network._network.fc.named_parameters():
                pars.append(p)
                self.old_params.append(p.clone().detach())
                self.names.append(n)
            params = [{"params": pars}]
            if "SGD" in self.optimizer_str:
                optimizer = self.optimizer_class(params, lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
            else:
                optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
        if self.cur_task == 0 and self.cur_round == 0:
            #self.optimizer = self.get_optimizer() already there
            self.scheduler = self.get_scheduler()
            self.optimizer.zero_grad()


    def end_epoch(self):
        if self.use_scheduler:
            self.scheduler.step()

    def end_round_client(self, dataloader: DataLoader):
        self.cur_round += 1
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        if self.cur_task > 0:
            self.replace_fc(dataloader)

    def freeze_backbone(self, is_first_session=False):
        # Freeze the parameters for ViT.
        if isinstance(self.network._network.convnet, torch.nn.Module):
            for name, param in self.network._network.convnet.named_parameters():
                if is_first_session:
                    if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

    def observe(self, inputs, labels, update=True):
        if self.cur_task == 0:  # simple train on first task
            self.optimizer.zero_grad()
            with self.fabric.autocast():
                inputs = self.augment(inputs)
                #logits = self.network._network(inputs)["logits"]
                logits = self.network(inputs)["logits"]
                loss = self.loss(logits, labels)
            if update:
                self.fabric.backward(loss)
                self.optimizer.step()
            return loss.item()
        return 0, "break"

    def forward(self, x):
        return self.network._network(x)['logits']

    def get_parameters(self):
        return self.network._network.parameters()

    def get_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=0)

    def end_task_client(self, dataloader, server_info):
        if self.cur_task == 0:
            self.freeze_backbone()
            self.setup_RP()
        #dataloader.dataset.transform = self.dataset.TEST_TRANSFORM
            self.replace_fc(dataloader)
        return self.get_client_info(dataloader)
    
    def begin_round_server(self, info: F.List[dict] = None):
        if self.cur_task == 0 and self.cur_round == 0:
            M = self.rp_size
            self.freeze_backbone(is_first_session=True)
            old = copy.deepcopy(self.network._network.fc)
            self.network._network.fc.weight = torch.nn.Parameter(torch.Tensor(self.network._network.fc.out_features, M).to(self.network._network.device))  # num classes in task x M
            #self.network._network.fc.reset_parameters()
            self.network._network.fc.W_rand = torch.randn(self.network._network.fc.in_features, M).to(self.network._network.device)
            self.W_rand = copy.deepcopy(self.network._network.fc.W_rand)  # make a copy that gets passed each time the head is replaced
            del self.network._network.fc.W_rand
            self.network._network.fc = old

    def end_round_server(self, client_info):
        if self.cur_task == 0:
            if self.avg_type == "weighted":
                total_samples = sum([client["num_train_samples"] for client in client_info])
                norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            else:
                weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
                norm_weights = [w / sum(weights) for w in weights]
            if len(client_info) > 0:
                self.network._network.fc.weight.data = torch.stack([client["Wo"] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)
                i = 0
                for n, p in self.network._network.convnet.named_parameters():
                    if n in self.network._network.params_to_optimize:
                        p.data = torch.stack([client["backbone_params"][i] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)
                        i += 1
        else:
            self.end_task_server(client_info)

    def end_task_server(self, client_info):
        #if self.cur_task == 0:
        self.freeze_backbone()
        self.setup_RP()
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        if len(client_info) > 0:
            if self.cur_task == 0:
                i = 0
                for n, p in self.network._network.convnet.named_parameters():
                    if n in self.network._network.params_to_optimize:
                        p.data = torch.stack([client["backbone_params"][i] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)
                        i += 1
            self.Q = torch.stack([client["Q"] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)
            self.G = torch.stack([client["G"] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)
            #ridge = self.optimise_ridge_parameter_gpu(Features_h, Y)
            #Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T  # better nmerical stability than .inv
            #self.network._network.fc.weight.data = Wo[0:self.network._network.fc.weight.shape[0], :].to(self.network._network.device)
            self.network._network.fc.weight.data = torch.stack([client["Wo"] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]).sum(0)

    def setup_RP(self):
        self.network._network.fc.use_RP = True

        # RP with M > 0
        M = self.rp_size
        self.network._network.fc.weight = torch.nn.Parameter(torch.Tensor(self.network._network.fc.out_features, M).to(self.network._network.device))  # num classes in task x M
        self.network._network.fc.reset_parameters()
        if getattr(self, "W_rand", None) is None:
            self.network._network.fc.W_rand = torch.randn(self.network._network.fc.in_features, M).to(self.network._network.device)
            self.W_rand = copy.deepcopy(self.network._network.fc.W_rand)  # make a copy that gets passed each time the head is replaced
        else:
            self.network._network.fc.W_rand = copy.deepcopy(self.W_rand)

        self.Q = torch.zeros(M, self.num_classes)
        self.G = torch.zeros(M, M)
        self.old_Q = torch.zeros(M, self.num_classes)
        self.old_G = torch.zeros(M, M)

    def replace_fc(self, trainloader):
        self.network._network.eval()

        # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
        self.network._network.fc.use_RP = True
        self.network._network.fc.W_rand = self.W_rand

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                data, label = data[0].to(self.device), data[1].to(self.device)
                embedding = self.network._network.convnet(data)

                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.num_classes)
        # print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
        Features_h = torch.nn.functional.relu(Features_f @ self.network._network.fc.W_rand.cpu())

        # self.cur_Q = Features_h.T @ Y
        # self.cur_G = Features_h.T @ Features_h

        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        # self.Q = self.old_Q + self.cur_Q
        # self.G = self.old_G + self.cur_G
        ridge = self.optimise_ridge_parameter_gpu(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T  # better nmerical stability than .inv
        self.network._network.fc.weight.data = Wo[0:self.network._network.fc.weight.shape[0], :].to(self.network._network.device)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0**np.arange(-8, 9)
        #ridges = 10.0**np.arange(5, 7)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        #logging.info("Optimal lambda: " + str(ridge))
        print("Optimal lambda: " + str(ridge))
        return ridge
    
    def optimise_ridge_parameter_gpu(self, Features, Y):
        Y = Y.to(self.device)
        Features = Features.to(self.device)
        ridges = 10.0**np.arange(-8, 9)
        #ridges = 10.0**np.arange(5, 7)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            try:
                Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0), device=self.device), Q_val).T  # better nmerical stability than .inv
            except torch._C._LinAlgError:
                losses.append(torch.tensor(float('inf')).to("cpu"))
                continue
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]).to('cpu'))
        if torch.isinf(torch.tensor(losses)).sum() == len(losses):
            raise ValueError("All losses are inf")
        ridge = ridges[np.argmin(np.array(losses))]
        #logging.info("Optimal lambda: " + str(ridge))
        print("Optimal lambda: " + str(ridge))
        return ridge
    
    def optimise_ridge_parameter2(self, Features, Y):
        ridges = 10.0**np.arange(-8, 9)
        #ridges = 10.0**np.arange(5, 7)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        ridges = torch.tensor(ridges).to(self.device)
        Wos = torch.linalg.solve(G_val.unsqueeze(0).repeat(ridges.shape[0], 1, 1) + ridges.unsqueeze(1).unsqueeze(1) * torch.eye(G_val.size(dim=0), device=self.device).unsqueeze(0).repeat(ridges.shape[0], 1, 1), Q_val.unsqueeze(0).repeat(ridges.shape[0], 1, 1)).transpose(1, 2)
        Y_train_preds = Features[num_val_samples::, :].unsqueeze(0).repeat(ridges.shape[0], 1, 1) @ Wos.transpose(1, 2)
        losses = F.mse_loss(Y_train_preds, Y[num_val_samples::, :].unsqueeze(0).repeat(ridges.shape[0], 1, 1), reduction='none').mean(dim=(1, 2))
        #for ridge in ridges:
        #    Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T  # better nmerical stability than .inv
        #    Y_train_pred = Features[num_val_samples::, :] @ Wo.T
        #    losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        #logging.info("Optimal lambda: " + str(ridge))
        print("Optimal lambda: " + str(ridge))
        return ridge


    def get_client_info(self, dataloader: DataLoader):
        backbone_params = []
        for n, p in self.network._network.convnet.named_parameters():
            if n in self.network._network.params_to_optimize:
                backbone_params.append(p.data)
        Wo = self.network._network.fc.weight.data
        return {
            "Wo": Wo,
            "backbone_params": backbone_params,
            "Q": self.Q,
            "G": self.G,
            "num_train_samples": len(dataloader.dataset.data),
        }

    def get_server_info(self):
        W_rand = None
        if getattr(self, "W_rand", None) is not None:
            W_rand = self.W_rand
        Wo = self.network._network.fc.weight.data #if self.network._network.fc.use_RP == True else None
        backbone_params = None
        if self.cur_task == 0:
            backbone_params = []
            for n, p in self.network._network.convnet.named_parameters():
                if n in self.network._network.params_to_optimize:
                    backbone_params.append(p.data)
        return {
            "Wo": Wo,
            "Q": self.Q,
            "G": self.G,
            "W_rand": W_rand,
            "backbone_params": backbone_params,
        }