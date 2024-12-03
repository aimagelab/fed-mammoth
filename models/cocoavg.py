import torch
from torch import nn
from torch.nn import functional as F
from models import register_model
from typing import List
from torch.utils.data import DataLoader
from models.utils import BaseModel
from utils.tools import str_to_bool
from networks.vit_prompt_hgp import VitHGP
from networks.vit import VisionTransformer
import numpy as np


@register_model("cocoavg")
class CocoAvg(BaseModel):

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "class_weighted",
        linear_probe: str_to_bool = False,
        slca: str_to_bool = False,
        alpha_sample_classes: float = 0,
        beta_entropy: float = 0,
        weight_pars_dist: str_to_bool = False,
        weight_on_gradient_dist: str_to_bool = False,
        weight_on_gradient_per_layer: str_to_bool = False,
        cumulated_gradient_level: str = "all_layers",
    ) -> None:
        self.slca = slca
        self.clients_statistics = None
        self.lr = lr
        self.wd = wd_reg
        if type(network) == VitHGP:
            for n, p in network.named_parameters():
                if "prompt" or "last" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        elif type(network) == VisionTransformer:
            params_backbone = []
            params_head = []
            for n, p in network.named_parameters():
                p.requires_grad = True #TODO rimetti a posto
                if "head" not in n:
                    params_backbone.append(p)
                    #p.requires_grad = False
                else:
                    params_head.append(p)
                    #p.requires_grad = True
            params = [{"params": params_head}, {"params": params_backbone}]
            if slca:
                params = [{"params": params_backbone, "lr": lr / 100}, {"params": params_head}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        else:
            super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.avg_type = avg_type
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.alpha_sample_classes = alpha_sample_classes
        self.beta_entropy = beta_entropy
        self.weight_on_pars_distance = weight_pars_dist
        self.small_omega = 0
        self.weight_on_gradient_dist = weight_on_gradient_dist
        self.layers_names = [n for n, p in self.network.named_parameters()]
        self.params_layers = np.array([n for n, p in self.network.named_parameters() for i in range(len(p.reshape(-1)))])
        self.cumulated_gradient_level = cumulated_gradient_level
        self.weight_on_gradient_per_layer = weight_on_gradient_per_layer

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
            loss = self.loss(outputs, labels % self.cpt)

        if update:
            pre_params = self.network.get_params().detach().data.clone()
            self.fabric.backward(loss)
            grads = []
            for pp in list(self.network.parameters()):
                if pp.grad is not None:
                    grads.append(pp.grad.view(-1))
            cur_small_omega = torch.cat(grads).data
            cur_small_omega = torch.nan_to_num(cur_small_omega, 0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            cur_small_omega *= (pre_params - self.network.get_params().detach().data.clone())
            self.small_omega += cur_small_omega

        return loss.item()

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.do_linear_probe:
            self.done_linear_probe = False

    def linear_probe(self, dataloader: DataLoader):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.last(pre_logits)[:, self.cur_offset: self.cur_offset + self.cpt]
                labels = labels % self.cpt
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

    def weighted_head_per_class(self, norm_weights_per_class, clients_head_w, clients_head_b):
        updated_head_w = 0
        updated_head_b = 0
        classes_covered = set()
        for client_w, client_b, norm_weight in zip(clients_head_w, clients_head_b, norm_weights_per_class):
            updated_head_w += client_w.T * torch.tensor(list(norm_weight.values()))
            updated_head_b += client_b.T * torch.tensor(list(norm_weight.values()))
            classes_covered = classes_covered.union(set([k for k,v in norm_weight.items() if v>0]))

        #TODO need to add server weights and bias for classes not covered by none of the agents
        classes_not_covered = torch.tensor([0 if k in classes_covered else 1 for k in norm_weights_per_class[0].keys()])
        for name, param in self.network.named_parameters():
            if "head.bias" in name:
                updated_head_b += param * classes_not_covered
            if "head.weight" in name:
                updated_head_w += param.T * classes_not_covered
        return updated_head_w.T, updated_head_b

    def class_distribution_entropy(self, client_info, all_classes):
        # Evaluate entropy of the distribution of label classes for each client.
        # The idea is to penalize peacked distributions (lower entropy) wrt clients that have seen more classes

        distributions = [[single_cl_info['client_statistics'][cl][0] if cl in single_cl_info['client_statistics']
                          else 0 for cl in all_classes] for single_cl_info in client_info]
        total_samples = [np.sum(record_for_classes) for record_for_classes in distributions]
        prob_distributions = [record_for_classes/total for record_for_classes, total in zip(distributions,total_samples)]
        non_zero_prob_distr = [[x for x in probs if x>0] for probs in prob_distributions]
        clients_entropy = [-np.sum(probs * np.log2(probs)) for probs in non_zero_prob_distr]
        return clients_entropy

    def clients_class_distribution(self, client_info, all_classes):
        total_samples_per_class = {
            cla: sum([single_client_info['num_train_samples_per_class'][cla] for single_client_info in client_info])
            for cla in all_classes}
        norm_weights_per_class = [{cla: single_client_info['num_train_samples_per_class'][cla] /
                                        total_samples_per_class[cla] if total_samples_per_class[cla] > 0 else 0
                                   for cla in all_classes} for single_client_info in client_info]
        return norm_weights_per_class

    def clients_pars_distance_weights(self, client_info):
        # Evaluate the distance of every client parameters wrt to old server parameters and create a weight for each
        # parameter of each client to weight more the bigger changes

        original_parameters = self.network.get_params()
        params_distances = np.absolute(np.stack([(cl["params"] - original_parameters.cpu()).data.numpy() for cl in client_info])) + 0.1
        total_abs_change = params_distances.sum(axis=0)
        params_distances = params_distances / total_abs_change
        # for params with no change, each client will have the same impact
        params_distances = np.nan_to_num(params_distances, copy=True, nan=1 / len(client_info))
        weighted_parameters = np.stack([cl["params"].cpu().data.numpy() for cl in client_info])
        weighted_parameters = np.multiply(weighted_parameters, params_distances)
        weighted_parameters = weighted_parameters.sum(axis=0)
        return torch.from_numpy(weighted_parameters)

    def clients_little_omega_weights(self, client_info):
        # Evaluate the cumulated gradient for every parameter
        # to weight more the bigger changes

        little_omegas = np.stack([cl["small_omega"].cpu() for cl in client_info]) + 0.1
        little_omegas_sum = little_omegas.sum(axis=0)

        little_omegas_normed = little_omegas/little_omegas_sum
        little_omegas_normed = np.nan_to_num(little_omegas_normed, copy=True, nan=1 / len(client_info))
        weighted_parameters = np.stack([cl["params"].cpu().data.numpy() for cl in client_info])
        weighted_parameters = np.multiply(weighted_parameters, little_omegas_normed)
        weighted_parameters = weighted_parameters.sum(axis=0)

        return torch.from_numpy(weighted_parameters)


    def clients_layer_cumulated_gradient_weights(self, client_info):
        # Evaluate the cumulated gradient for every parameter
        # to weight more the bigger changes

        # 3 opzioni:
        # 1- su tutti i layers peso rispetto alla somma degli omegas su quel layer, caso "all_layers"
        # 2- faccio lo stesso solo sulle matrici di attention facendo avg sul resto, caso "attention_only"
        # 3- sulle matrici di attention invece di pesare tengo direttamente quelle con variazione più alta, facendo avg sul resto "attention_not_w"

        little_omegas = torch.stack([cl["small_omega"] for cl in client_info])
        original_parameters = torch.stack([cl["params"] for cl in client_info])
        weighted_parameters = torch.empty((original_parameters.shape[0], 0))
        if self.cumulated_gradient_level == "all_layers":
            for l_n in self.layers_names:
                layer_index = self.params_layers==l_n
                cumulated_gradients = little_omegas[:, layer_index].sum(axis=1)
                total_gradient = cumulated_gradients.sum()
                weights = cumulated_gradients/total_gradient
                weighted_parameters_layer = weights[:, np.newaxis] * original_parameters[:, layer_index]
                weighted_parameters_layer.sum(axis=0)
                weighted_parameters = torch.cat((weighted_parameters, weighted_parameters_layer), axis=1)

        elif self.cumulated_gradient_level == "attention_only":
            for l_n in self.layers_names:
                layer_index = self.params_layers==l_n
                if 'attn' in l_n:
                    cumulated_gradients = little_omegas[:, layer_index].sum(axis=1)
                    total_gradient = cumulated_gradients.sum()
                    weights = cumulated_gradients/total_gradient
                else:
                    weights = [1/ len(client_info) for cl in client_info]
                weighted_parameters_layer = weights[:, np.newaxis] * original_parameters[:, layer_index]
                weighted_parameters_layer.sum(axis=0)
                weighted_parameters = torch.cat((weighted_parameters, weighted_parameters_layer), axis=1)

        elif self.cumulated_gradient_level == "attention_not_w": # to be completed
            for l_n in self.layers_names:
                layer_index = self.params_layers==l_n
                if 'attn' in l_n:
                    cumulated_gradients = little_omegas[:, layer_index].sum(axis=1)
                    total_gradient = cumulated_gradients.sum()
                    weights = cumulated_gradients/total_gradient
                else:
                    weights = [1/ len(client_info) for cl in client_info]
                weighted_parameters_layer = weights[:, np.newaxis] * original_parameters[:, layer_index]
                weighted_parameters_layer.sum(axis=0)
                weighted_parameters = np.concatenate((weighted_parameters, weighted_parameters_layer), axis=1)

        else:
            raise ValueError("Cumulated gradient level is not supported.")

        return weighted_parameters

    def clients_sample_var_weights(self, client_info, all_classes):
        total_vars_per_class = {
            cla: sum([single_client_info['client_statistics'][cla][2].mean().cpu() if cla in single_client_info['client_statistics']
                      else 0 for single_client_info in client_info])
            for cla in all_classes}

        norm_vars_per_class = [{cla: single_client_info['client_statistics'][cla][2].mean().cpu() /
                                        total_vars_per_class[cla]
                                if cla in single_client_info['client_statistics'] and total_vars_per_class[cla] > 0
                                else 0
                                for cla in all_classes} for single_client_info in client_info]
        return norm_vars_per_class

    def end_round_server(self, client_info: List[dict]):
        all_classes = client_info[0]['all_classes']
        # Three possible scenarios:
        # "weighted" weights on number of samples (FedAVG), "class_weighted" weigths each client for each class
        # else if weights every client in the same way
        print(self.avg_type)
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            norm_weights_per_class = [{cla: 1 / len(client_info) for cla in all_classes} for _ in
                                      client_info]
        elif self.avg_type == "class_weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            numsamples_weights_per_class = self.clients_class_distribution(client_info, all_classes)
            samplevar_weights_per_class = self.clients_sample_var_weights(client_info, all_classes)

            norm_weights_per_class = [{cla: (self.alpha_sample_classes * nums[cla] +
                                             (1-self.alpha_sample_classes) * vars[cla])
                                       for cla in all_classes} for nums, vars in
                                       zip(numsamples_weights_per_class, samplevar_weights_per_class)]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
            norm_weights_per_class = [{cla: 1/len(client_info) for cla in client_info[0]['all_classes']} for _ in client_info]

            merged_weights = 0
            for client, norm_weight in zip(client_info, norm_weights):
                merged_weights += client["params"] * norm_weight

        # Entropy comes from the evaluation of class distribution for each client.
        # If beta_entropy = 1 -> it weights completely with respect to entropy
        print("start evaluating entropy")
        clients_entropy = self.class_distribution_entropy(client_info, all_classes)
        weights_en = clients_entropy
        if sum(weights_en) == 0:
            weights_en = [x + 0.01 for x in weights_en]
        norm_weights_en = [w / sum(weights_en) for w in weights_en]

        norm_weights = [(1 - self.beta_entropy) * w + self.beta_entropy * e_w
                        for w, e_w in zip(norm_weights, norm_weights_en)]
        print("end evaluating entropy")

        # Here I differenciate the different merged_weights techniques
        # if weight_on_pars_distance=True it uses the distance in space from the previous round for each parameter to weight more the parameters that changed more
        # if weight_on_gradient_dist=True it uses the cumulated gradient for each parameter to weight more the parameters that cumulated more gradient
        # if weight_on_gradient_per_layer=True it uses the cumulated gradient for each layer to weight more the layers that cumulated more gradient
        # else it uses the weights from previous calculation

        print("weight_on_pars_distance",self.weight_on_pars_distance)
        print("weight_on_gradient_dist",self.weight_on_gradient_dist)
        print("weight_on_gradient_per_layer",self.weight_on_gradient_per_layer)
        if len(client_info) > 0:
            if self.weight_on_pars_distance:
                merged_weights = self.clients_pars_distance_weights(client_info)
            elif self.weight_on_gradient_dist:
                merged_weights = self.clients_little_omega_weights(client_info)
            elif self.weight_on_gradient_per_layer:
                merged_weights = self.clients_layer_cumulated_gradient_weights(client_info)
            else:
                merged_weights = 0
                for client, norm_weight in zip(client_info, norm_weights):
                    merged_weights += client["params"] * norm_weight
            self.network.set_params(merged_weights)

            # After updating the parameters, I reapply the update for the head, since strategy of class_weighted parameters for the head is best
            updated_head_w, updated_head_b = self.weighted_head_per_class(norm_weights_per_class,
                                                                     [cl["params_head_w"] for cl in client_info],
                                                                     [cl["params_head_b"] for cl in client_info])
            for name, param in self.network.named_parameters():
                if "head.bias" in name:
                    param.data = updated_head_b
                if "head.weight" in name:
                    param.data = updated_head_w

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.set_params(server_info["params"])
        self.small_omega = 0
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            # restore correct optimizer
            params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
            optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)


    def end_round_client(self, dataloader: DataLoader):
        features = torch.tensor([], dtype=torch.float32).to(self.device)
        true_labels = torch.tensor([], dtype=torch.int64).to(self.device)

        # Qui vengono calcolate media e varianza degli output del modello. Pre classificatore? yes, per pen=True
        with torch.no_grad():
            client_statistics = {}
            for id, data in enumerate(dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _ = self.network(inputs, penultimate=True)
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
                    client_statistics[client_label] = gaussians
            self.clients_statistics = client_statistics


    def get_client_info(self, dataloader: DataLoader):
        client_model_features = {'num_train_samples_per_class' : {dataloader.dataset.class_to_idx[cl]:
                                np.count_nonzero(dataloader.dataset.targets == dataloader.dataset.class_to_idx[cl])
                                for cl in dataloader.dataset.classes}}
        client_model_features["params"] = self.network.get_params()
        client_model_features["params_head_w"] = [p for n, p in self.network.named_parameters() if 'head.weight' in n][0]
        client_model_features["params_head_b"] = [p for n, p in self.network.named_parameters() if 'head.bias' in n][0]
        client_model_features["num_train_samples"] = len(dataloader.dataset.data)
        client_model_features["all_classes"] = client_model_features["num_train_samples_per_class"].keys()
        client_model_features["client_statistics"] = self.clients_statistics
        client_model_features["small_omega"] = self.small_omega
        return client_model_features

    def get_server_info(self):
        return {"params": self.network.get_params()}
