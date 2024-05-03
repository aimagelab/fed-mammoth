import random
from typing import Tuple
import numpy as np
import setproctitle
from argparse import ArgumentParser
from inspect import signature
import os
import getpass
import lightning as L
import torch
import json
from torch.utils.data import Dataset


from models.utils import BaseModel
from utils.training import train
from utils.args import add_args
from models import model_factory
from networks import network_factory
from datasets import dataset_factory
from datetime import datetime


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_artifacts(args: dict, fabric) -> Tuple[BaseModel, Dataset]:
    NetworkClass = network_factory(args["network"])
    DatasetClass = dataset_factory(args["dataset"])
    ModelClass = model_factory(args["model"])

    args["input_shape"] = DatasetClass.INPUT_SHAPE
    args["num_classes"] = DatasetClass.N_CLASSES_PER_TASK * DatasetClass.N_TASKS
    args["num_tasks"] = DatasetClass.N_TASKS

    network_signature = list(signature(NetworkClass.__init__).parameters.keys())[1:]
    dataset_signature = list(signature(DatasetClass.__init__).parameters.keys())[1:]
    model_signature = list(signature(ModelClass.__init__).parameters.keys())[3:]
    # TODO Questo è un po' pericoloso, dobbiamo ricordarci sempre di mettere i primi 3 argomenti fissi e dopo i nostri argomenti, che ci sta eh

    dataset = DatasetClass(**{key: args[key] for key in dataset_signature})
    network = NetworkClass(**{key: args[key] for key in network_signature})

    server_model = ModelClass(fabric, network, **{key: args[key] for key in model_signature})

    client_models = []
    for _ in range(args["num_clients"]):
        client_models.append(ModelClass(fabric, network, **{key: args[key] for key in model_signature}))

    return server_model, client_models, dataset


def main(args: dict, output_folders_root: str, nickname: str) -> None:
    set_random_seed(42)

    setproctitle.setproctitle(f"{getpass.getuser()}_{nickname}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # command = " ".join(sys.argv)
    output_folder = os.path.join(output_folders_root, f"{timestamp}_{nickname}")
    os.makedirs(output_folder)

    device = args["device"]
    if "cuda" in device:
        device, index = device.split(":")
    torch.set_float32_matmul_precision("medium")
    fabric = L.Fabric(
        accelerator=device,
        devices=1 if device == "cpu" else [int(index)],
        strategy="dp",
        precision="16-mixed",
    )
    fabric.launch()

    server_model, client_models, dataset = get_artifacts(args, fabric)

    with open(os.path.join(output_folder, "config.json"), "w") as f:
        json.dump(args, f, indent=4)

    train(fabric, server_model, client_models, dataset, args, output_folder)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="fed-mammoth",
        allow_abbrev=False,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--network", type=str, required=True)

    parser.add_argument(
        "--nickname", type=str, required=False, default="Moscow"
    )  # TODO: Change this to something more appropriate

    args = parser.parse_known_args()[0]

    args.nickname = str(args.model + "_" + args.dataset + "_" + args.network + "_" + args.nickname)

    add_args(parser, args.model, args.network, args.dataset)

    args = {**vars(parser.parse_args()), **vars(args)}

    print(
        """
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣿⣿⣿⣧⡀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⢀⣠⣴⡈⣡⣄⠉⣁⣉⢀⠀⠀⠀⠀⠀
    ⠀⠀⠀⢀⣠⣤⣼⣿⣿⣾⣿⣿⣿⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀
    ⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣹⣿⣿⠀⠀⠀⠀
    ⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢁⡀⠻⣿⣿⣿⡆⠀⡞⠀
    ⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠘⢿⣦⡈⠛⠿⠃⣼⠃⠀
    ⠀⢸⡿⣿⣿⣿⣿⣿⣿⠿⣿⣿⡿⠿⣿⣿⣿⣿⡟⠛⠁⠀⠙⠻⠷⠶⠟⠁⠀⠀
    ⠀⠘⠁⢸⣿⣿⣿⣿⠃⠰⠛⠛⠋⠀⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⣶⣶⠀⠀⠀⠀
    ⠀⠀⠀⢸⣿⣿⣿⡇⢰⣶⣿⠀⢸⡆⢸⣿⣿⣿⣟⠀⣀⣀⣠⣼⣿⡟⠀⠀⠀⠀
    ⠀⠀⠀⢸⣿⣿⣿⡇⢸⣿⣿⠀⢸⣧⠈⣿⣿⣿⡿⠀⢿⣿⠿⠛⠉⠀⠀⠀⠀⠀
    ⠀⠀⠀⢺⣿⣿⣿⠀⢸⣿⣿⠀⢸⣿⠀⢻⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠈⠉⠉⠉⠀⠈⠉⠉⠀⠈⠉⠁⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    """
    )

    main(args, args["output_folders_root"], args["nickname"])
