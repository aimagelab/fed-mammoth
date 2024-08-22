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

from models.lora import Lora, merge_AB, zero_pad
from models.regmean import RegMean
from tqdm import tqdm


@register_model("regmean_fedavg")
class RegmeanFedavg(RegMean):
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
        slca: str_to_bool = False,
        only_square: int = 0,
        train_bias: str = "all",
        beta: float = 0.5,
    ) -> None:
        self.beta = beta
        super(RegmeanFedavg, self).__init__(
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
            slca,
            only_square,
            train_bias,
        )

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
        weights_t = torch.tensor(norm_weights, dtype=dtype)
        # regmean solution
        keys = list(self.network.state_dict().keys())
        sd = self.network.state_dict()
        for key in keys:
            if (
                "weight" in key and self.middle_names.get(key) is not None
            ):  # it means that we apply regmean to this layer
                name = self.middle_names[key]
                shape = client_info[0]["gram"][name].shape[0]
                fedavg_matrices = [torch.eye(shape).to(dtype) * weights_t[i] for i in range(len(client_info))]
                interpolated_matrices = [
                    fedavg_matrix * self.beta + client["gram"][name].to(dtype) * (1 - self.beta)
                    for fedavg_matrix, client in zip(fedavg_matrices, client_info)
                ]
                sd[key] = (
                    torch.stack(
                        [
                            client["state_dict"][key].to(dtype) @ interpolated_matrix
                            for interpolated_matrix, client in zip(interpolated_matrices, client_info)
                        ]
                    ).sum(0)
                    @ torch.pinverse(torch.stack([interpolated_matrices]).sum(0))
                ).to(torch.float32)
                # del fedavg_matrices, interpolated_matrices
            else:
                sd[key] = torch.stack(
                    [client["state_dict"][key] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
                ).sum(
                    0
                )  # fedavg for the other layers
        self.network.load_state_dict(sd)
