import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from models import register_model
from typing import List
from torch.utils.data import DataLoader
from models.utils import BaseModel
from networks.vit_prompt_hgp import VitHGP
from models.hgp import HGP


@register_model("hgp_regmean")
class HGPRegmean(HGP):
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
        alpha_regmean: float = 0.5,
    ) -> None:
        params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, avg_type=avg_type, how_many=how_many)
        # super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
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
        self.alpha_regmean = alpha_regmean
        self.gram = None
        self.importance = None
        self.keys = [key for key in self.network.state_dict().keys() if "last" in key]
        self.seen_classes = None

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
        client_feat_sq = [c["client_gram"] for c in client_info]
        base_class = self.cur_task * self.cpt
        clients_per_seen_class = {}
        for clas in range(base_class, base_class + self.cpt):
            clients_per_seen_class[clas] = []
            for idx, c in enumerate(client_info):
                if clas in c["seen_classes"]:
                    clients_per_seen_class[clas].append(idx)

        # clients_per_seen_class = {
        #    clas: [c["client_index"] for c in client_info if clas in c["seen_classes"]]
        #    for clas in range(base_class, base_class + self.cpt)
        # }
        ssd = self.network.state_dict()
        client_params = [c["params"] for c in client_info]  # clients' weights
        client_sd = []
        for c in client_params:
            tmp = 0
            sd = self.network.state_dict()
            for key in sd.keys():
                size = sd[key].numel()
                sd[key] = c[tmp : tmp + size].reshape(sd[key].shape)
                tmp += size
            client_sd.append(sd)
        del client_params
        # Apply regmean
        cls_weight_key = [key for key in self.keys if "last" in key and "weight" in key][0]
        for clas in range(base_class, base_class + self.cpt):
            clas_importance = []
            for client_idx in clients_per_seen_class[clas]:
                clas_importance.append(client_info[client_idx]["client_importance"][clas - base_class])
            norm_importance = torch.tensor([i / sum(clas_importance) for i in clas_importance])

            client_feats_sq = torch.stack(
                [cfsq[clas - base_class] for cfsq in client_feat_sq if type(cfsq[clas - base_class]) != int]
            )
            norm_client_feats = client_feats_sq * norm_importance.reshape(-1, 1, 1)
            norm_factor = torch.linalg.inv(norm_client_feats.sum(0))
            ensemble = torch.stack(
                [
                    feat_sq @ client_sd[c_idx][cls_weight_key][clas : clas + 1].T * norm_importance[i]
                    for i, (feat_sq, c_idx) in enumerate(zip(client_feats_sq, clients_per_seen_class[clas]))
                ]
            ).sum(0)
            ssd[cls_weight_key][clas : clas + 1] = (norm_factor @ ensemble).T.cpu()

            # fedavg_class for bias
            cls_bias_key = cls_weight_key.replace("weight", "bias")
            ssd[cls_bias_key][clas : clas + 1] = sum(
                [
                    client_sd[c_idx][cls_bias_key][clas : clas + 1] * norm_importance[i]
                    for i, c_idx in enumerate(clients_per_seen_class[clas])
                ]
            )
        self.network.load_state_dict(ssd)
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
            for task in range(self.cur_task + 1):
                for clas in range(self.cpt):
                    weights_list = []
                    for weight in self.mogs_per_task[task][task * self.cpt + clas][0]:
                        weights_list.append(weight)
                    gaussian_samples = torch.zeros(len(weights_list), dtype=torch.int64).to(self.device)
                    weights_list = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
                    gaussian_samples_fill = torch.multinomial(
                        weights_list, classes_samples[task * self.cpt + clas], replacement=True
                    )
                    gaussian_clients, gaussian_samples_fill = torch.unique(gaussian_samples_fill, return_counts=True)
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
                outputs = self.network.last(inp)
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

    def get_client_info(self, dataloader: DataLoader):
        return {
            "params": self.network.get_params(),
            "num_train_samples": len(dataloader.dataset.data),
            "client_statistics": self.clients_statistics,
            "client_gram": self.gram,
            "client_importance": self.importance,
            "seen_classes": self.seen_classes,
        }

    def get_server_info(self):
        return {"params": self.network.get_params()}

    def end_round_client(self, dataloader: DataLoader):
        self.network.eval()
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
                    print(number, torch.norm(gaussians[1], p=2), torch.norm(gaussians[2], p=2))
                    client_statistics[client_label] = gaussians
            self.clients_statistics = client_statistics
            #######################################
            # compute features
            base_class = self.cur_task * self.cpt
            training_status = self.network.training
            feature_list = []
            label_list = []
            train_dataset_data = dataloader.dataset.data.copy()
            train_dataset_targets = dataloader.dataset.targets.copy()
            for clas in range(base_class, base_class + self.cpt):
                counter = 0
                dataloader.dataset.data = train_dataset_data[train_dataset_targets == clas]
                dataloader.dataset.targets = train_dataset_targets[train_dataset_targets == clas]
                if len(dataloader.dataset.data) > 5:
                    while counter <= 1000:
                        for inputs, labels in dataloader:
                            inputs = inputs.to(self.device)
                            label_list.append(labels)
                            feature_list.append(self.network(inputs, pen=True))
                            counter += feature_list[-1].shape[0]
                            if counter >= 1000:
                                break

            dataloader.dataset.data = train_dataset_data
            dataloader.dataset.targets = train_dataset_targets
            features = torch.cat(feature_list)
            all_labels = torch.cat(label_list)
            feat_sq, importance = [], []
            base_class = self.cur_task * self.cpt
            for clas in range(base_class, base_class + self.cpt):
                if clas in all_labels.unique():
                    clas_features = features[all_labels == clas]
                    feat_sq_cur = (clas_features.T @ clas_features).cpu()
                    feat_sq_cur[torch.eye(feat_sq_cur.shape[0]) == 0] *= self.alpha_regmean
                    feat_sq.append(feat_sq_cur)
                    importance.append((all_labels == clas).sum().item())
                else:
                    importance.append(0)
                    feat_sq.append(0)

        self.network.train(training_status)
        self.gram = feat_sq
        self.importance = importance
        self.seen_classes = all_labels.unique().tolist()
