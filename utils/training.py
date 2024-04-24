from copy import deepcopy
import os
import torch
from torch.utils.data import DataLoader

from datasets.utils import BaseDataset
from utils.global_consts import LOG_LOSS_INTERVAL

from typing import Iterator, List, Tuple

from models.utils import BaseModel
from utils.status import progress_bar
from time import time


def get_time_str(delta_time: int):
    delta_seconds = int(delta_time % 60)
    delta_minutes = int((delta_time // 60) % 60)
    delta_hours = int((delta_time // 3600) % 24)
    delta_days = int(delta_time // (24 * 3600))

    delta_time_str = ""
    for remaining, unity in zip([delta_days, delta_hours, delta_minutes], ["d", "h", "m"]):
        if remaining > 0:
            delta_time_str += f" {remaining}{unity}"
    if delta_days == 0 and delta_hours == 0:
        delta_time_str += f" {delta_seconds}s"
    return delta_time_str[1:]


def evaluate(fabric, task, model: BaseModel, dataset: BaseDataset):
    correct, total = 0, 0
    with torch.no_grad():
        for t in range(task + 1):
            test_loaders = dataset.get_cur_dataloaders(t)[1]
            for test_loader in test_loaders:
                test_loader = fabric.setup_dataloaders(test_loader)
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    pred = torch.max(outputs, dim=1)[1]
                    correct += (pred == labels).sum().item()
                    total += labels.shape[0]

    print(f"Mean accuracy up to task {task + 1}:", round(correct / total * 100, 2), "%")


def train(
    fabric,
    server_model: BaseModel,
    client_models: List[BaseModel],
    dataset: BaseDataset,
    args: dict,
    output_folder: str,
) -> None:
    if args["checkpoint"] is not None and args["checkpoint"] != "":
        start_task, start_comm_round = server_model.load_checkpoint(args["checkpoint"])
        print(f"Loaded checkpoint at {args['checkpoint']}")
    else:
        start_task, start_comm_round = 0, 0

    server_model.train()
    for client_model in client_models:
        client_model.train()

    os.makedirs(output_folder, exist_ok=True)

    last_round_time = time()
    client_info = []
    for task in range(dataset.N_TASKS):
        if task < start_task:
            continue
        train_loaders, test_loaders = dataset.get_cur_dataloaders(task)
        server_model.begin_task(dataset.N_CLASSES_PER_TASK)
        for client_model in client_models:
            client_model.begin_task(dataset.N_CLASSES_PER_TASK)
        for comm_round in range(args["num_comm_rounds"]):
            server_model.begin_round_server()
            server_info = server_model.get_server_info()
            client_info = []
            if comm_round < start_comm_round:
                continue
            for client_idx in range(args["num_clients"]):
                train_loader = train_loaders[client_idx]
                test_loader = test_loaders[client_idx]
                train_loader = fabric.setup_dataloaders(train_loader)
                test_loader = fabric.setup_dataloaders(test_loader)
                model = client_models[client_idx]
                model.begin_round_client(train_loader, server_info)
                for epoch in range(args["num_epochs"]):
                    for i, (inputs, labels) in enumerate(train_loader):
                        train_loss = model.observe(inputs, labels)

                        if i % LOG_LOSS_INTERVAL == 0 or (
                            i == len(train_loader) - 1 and epoch == args["num_epochs"] - 1
                        ):
                            progress_bar(
                                task + 1,
                                dataset.N_TASKS,
                                comm_round + 1,
                                args["num_comm_rounds"],
                                client_idx,
                                epoch + 1,
                                args["num_epochs"],
                                train_loss,
                            )
                            # LOG LOSS HERE

                model.end_round_client(train_loader)
                client_info.append(model.get_client_info(train_loader))
                if len(train_loader):
                    print()

            print("Round time:", get_time_str(time() - last_round_time))
            last_round_time = time()
            server_model.end_round_server(client_info)
            evaluate(fabric, task, model, dataset)

            if epoch % args["checkpoint_interval"] == 0 or (comm_round + 1) == args["num_comm_rounds"]:
                server_model.save_checkpoint(output_folder, task, comm_round)

        server_model.end_task()
