import os
import torch
import wandb
from time import time
from typing import List

from datasets.utils import BaseDataset
from utils.global_consts import LOG_LOSS_INTERVAL
from models.utils import BaseModel
from utils.status import progress_bar
from utils.tools import get_time_str


def evaluate(fabric, task, model: BaseModel, dataset: BaseDataset):
    correct, total = 0, 0
    task_accuracies = []
    training_status = model.training
    model.eval()
    start_class = 0
    end_class = (task + 1) * dataset.N_CLASSES_PER_TASK
    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0
            test_loaders = dataset.get_cur_dataloaders(t)[1]
            for test_loader in test_loaders:
                test_loader = fabric.setup_dataloaders(test_loader)
                for inputs, labels in test_loader:
                    outputs = model(inputs)[:, start_class:end_class]
                    pred = torch.max(outputs, dim=1)[1]
                    task_correct += (pred == labels).sum().item()
                    task_total += labels.shape[0]
                    correct += (pred == labels).sum().item()
                    total += labels.shape[0]
            task_accuracies.append(round(task_correct / task_total * 100, 2))

    model.train(training_status)
    print(
        f"Mean accuracy up to task {task + 1}:",
        round(correct / total * 100, 2),
        "%",
        "Task accuracies:",
        task_accuracies,
    )
    return round(correct / total * 100, 2)


def train(
    fabric,
    server_model: BaseModel,
    client_models: List[BaseModel],
    dataset: BaseDataset,
    args: dict,
    output_folder: str,
) -> None:

    if args["wandb"]:
        name = f"{args["nickname"]}_{args['dataset']}_{args['model']}_rnds{args['num_comm_rounds']}_clnts{args['num_clients']}_epchs{args['num_epochs']}_bs{args['batch_size']}_lr{args['lr']}"
        wandb.init(project=args["wandb_project"], entity=args["wandb_entity"], config=args, name=name)
        # args.wandb_url = wandb.run.get_url()

    if args["checkpoint"] is not None and args["checkpoint"] != "":
        start_task, start_comm_round = server_model.load_checkpoint(args["checkpoint"])
        print(f"Loaded checkpoint at {args['checkpoint']}")
    else:
        start_task, start_comm_round = 0, 0

    server_model.train()
    for client_model in client_models:
        client_model.train()

    if not args["debug_mode"]:
        os.makedirs(output_folder, exist_ok=True)

    start_time = time()
    for task in range(dataset.N_TASKS):
        if task < start_task:
            continue
        train_loaders, test_loaders = dataset.get_cur_dataloaders(task)  # TODO: test_loaders are not used
        last_task_time = time()
        server_model.begin_task(dataset.N_CLASSES_PER_TASK)
        for client_model in client_models:
            client_model.begin_task(dataset.N_CLASSES_PER_TASK)
        for comm_round in range(args["num_comm_rounds"]):
            server_model.begin_round_server()
            server_info = server_model.get_server_info()
            if comm_round < start_comm_round:
                continue
            clients_info = []
            last_round_time = time()
            for client_idx in range(args["num_clients"]):
                train_loader = train_loaders[client_idx]
                test_loader = test_loaders[client_idx]
                train_loader = fabric.setup_dataloaders(train_loader)
                test_loader = fabric.setup_dataloaders(test_loader)
                model = client_models[client_idx]
                model.to("cuda")
                model.begin_round_client(train_loader, server_info)
                for epoch in range(args["num_epochs"]):
                    for i, (inputs, labels) in enumerate(train_loader):
                        train_loss = model.observe(inputs, labels)
                        assert not torch.isnan(
                            torch.tensor(train_loss)
                        ), f"Loss is NaN at task {task}, round{comm_round}, client {client_idx} and epoch {epoch}."
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
                        if args["wandb"]:
                            wandb.log({"train_loss": train_loss})
                    model.end_epoch()
                torch.cuda.empty_cache()
                model.end_round_client(train_loader)
                model.to("cpu")
                clients_info.append(model.get_client_info(train_loader))
                torch.cuda.empty_cache()
                if len(train_loader):
                    print()

            print("\nRound time:", get_time_str(time() - last_round_time))
            server_model.end_round_server(clients_info)
            server_model.to("cuda")
            accuracy = evaluate(fabric, task, server_model, dataset)
            if (epoch % args["checkpoint_interval"] == 0 or (comm_round + 1) == args["num_comm_rounds"]) and not args[
                "debug_mode"
            ]:
                server_model.save_checkpoint(output_folder, task, comm_round)
            torch.cuda.empty_cache()

        client_info = []
        for client_idx in range(args["num_clients"]):
            train_loader = train_loaders[client_idx]
            test_loader = test_loaders[client_idx]
            train_loader = fabric.setup_dataloaders(train_loader)
            test_loader = fabric.setup_dataloaders(test_loader)
            model = client_models[client_idx]
            model.to("cuda")
            client_info.append(model.end_task_client(train_loader))
            model.to("cpu")
            torch.cuda.empty_cache()
        server_model.end_task_server(client_info=client_info)
        server_model.to("cuda")
        accuracy = evaluate(fabric, task, server_model, dataset)
        torch.cuda.empty_cache()
        print(f"Task {task + 1} time:", get_time_str(time() - last_task_time))
        print("__________\n")

        if args["wandb"]:
            results = {
                "Mean_accuracy": accuracy,
                # TODO: add other things here
            }

            wandb.log(results)

    # TODO: it is probably needed a final evaluation here. At least for models that do something at the end_task()

    print("\nTotal training time:", get_time_str(time() - start_time))

    if args["wandb"]:
        wandb.finish()
