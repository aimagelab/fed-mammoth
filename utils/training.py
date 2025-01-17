import os
import torch
import wandb
from time import time
from typing import List

from _datasets._utils import BaseDataset
from utils.global_consts import LOG_LOSS_INTERVAL
from _models._utils import BaseModel
from utils.status import progress_bar
from utils.tools import get_time_str

import numpy as np
import matplotlib.pyplot as plt

def compute_forgetting(accuracies: List[List[float]]) -> List[float]:
    forgetting = []
    tasks = len(accuracies)
    last_accs = accuracies[-1]
    for i in range(tasks):
        forgetting.append(accuracies[i][i] - last_accs[i])
    return forgetting



def accs_and_forgetting_matrix(accuracies: List[List[float]], forgetting: List[float], output_folder: str = None) -> None:
    if output_folder is None:
        output_folder = os.getcwd()
        print("Output folder not specified, saving in current directory:", output_folder)
    tasks = len(accuracies)
    #fill list of accuracies with zeros
    for acc in accuracies:
        while len(acc) < tasks:
            acc.append(0.0)
    accs = np.array(accuracies)
    fig, ax = plt.subplots()
    ax.matshow(accs, cmap='viridis')
    for i in range(tasks):
        for j in range(tasks):
            ax.text(j, i, f'{accs[i, j]:.2f}', ha='center', va='center', color='black')
    plt.title("Task accuracies")
    plt.xlabel("Task")
    plt.ylabel("Task")
    acc_path = os.path.join(output_folder, "task_accuracies.png")
    plt.savefig(acc_path)
    plt.close()

    #now we plot forgetting as a vector, similar to what we did with accuracies
    fig, ax = plt.subplots()
    forgetting.append(round(sum(forgetting) / len(forgetting), 2))
    ax.matshow(np.array([forgetting]), cmap='viridis')
    for i in range(tasks + 1):
        ax.text(i, 0, f'{forgetting[i]:.2f}', ha='center', va='center', color='black')
    plt.title("Forgetting")
    plt.xlabel("Task")
    plt.ylabel("Forgetting")
    forg_path = os.path.join(output_folder, "forgetting.png")
    plt.savefig(forg_path)
    return [acc_path, forg_path]

def evaluate(fabric, task, model: BaseModel, dataset: BaseDataset, return_responses = False):
    correct, total = 0, 0
    task_accuracies = []
    training_status = model.training
    model.eval()
    start_class = 0
    labels_tensor = torch.tensor([], device = model.device)
    responses_tensor = torch.tensor([], device = model.device)
    if isinstance(dataset.N_CLASSES_PER_TASK, list):
        end_class = sum(dataset.N_CLASSES_PER_TASK[: task + 1])
    else:
        end_class = (task + 1) * dataset.N_CLASSES_PER_TASK
    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0
            #if isinstance(dataset, SequentialOOS):
            if dataset.IS_TEXT:
                test_loaders = dataset.get_cur_dataloaders_oos(t)[1]
            else:
                test_loaders = dataset.get_cur_dataloaders(t)[1]
            # test_loaders = dataset.get_cur_dataloaders(t)[1]
            for test_loader in test_loaders:
                test_loader = fabric.setup_dataloaders(test_loader)
                for inputs, labels in test_loader:
                    outputs = model(dataset.test_transform(inputs))[:, start_class:end_class]
                    pred = torch.max(outputs, dim=1)[1]
                    task_correct += (pred == labels).sum().item()
                    task_total += labels.shape[0]
                    correct += (pred == labels).sum().item()
                    total += labels.shape[0]
                    labels_tensor = torch.cat((labels_tensor, labels), 0)
                    responses_tensor = torch.cat((responses_tensor, pred), 0)
            task_accuracies.append(round(task_correct / task_total * 100, 2))

    model.train(training_status)
    print(
        f"Mean accuracy up to task {task + 1}:",
        round(correct / total * 100, 2),
        "%",
        "Task accuracies:",
        task_accuracies,
    )
    res = [round(correct / total * 100, 2), task_accuracies]
    return res if not return_responses else [res, labels_tensor, responses_tensor]


def evaluate_client(fabric, task, model: BaseModel, dataset: BaseDataset, idx: int):
    correct, total = 0, 0
    task_accuracies = []
    training_status = model.training
    model.eval()
    start_class = 0
    if isinstance(dataset.N_CLASSES_PER_TASK, list):
        end_class = sum(dataset.N_CLASSES_PER_TASK[: task + 1])
    else:
        end_class = (task + 1) * dataset.N_CLASSES_PER_TASK
    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0
            if dataset.IS_TEXT:
                test_loader = dataset.get_cur_dataloaders_oos(t)[1][idx]
            else:
                test_loader = dataset.get_cur_dataloaders(t)[1][idx]
            test_loader = fabric.setup_dataloaders(test_loader)
            for inputs, labels in test_loader:
                outputs = model(dataset.test_transform(inputs))[:, start_class:end_class]
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
    res = [round(correct / total * 100, 2), task_accuracies]
    return res


def evaluate_client_transfer(fabric, task, model: BaseModel, dataset: BaseDataset, idx: int):
    correct, total = 0, 0
    task_accuracies = []
    training_status = model.training
    model.eval()
    start_class = 0
    if isinstance(dataset.N_CLASSES_PER_TASK, list):
        end_class = sum(dataset.N_CLASSES_PER_TASK[: task + 1])
    else:
        end_class = (task + 1) * dataset.N_CLASSES_PER_TASK
    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0
            test_loaders = dataset.get_cur_dataloaders(t)[1]
            for i, test_loader in enumerate(test_loaders):
                if i == idx:
                    continue
                test_loader = fabric.setup_dataloaders(test_loader)
                for inputs, labels in test_loader:
                    outputs = model(dataset.test_transform(inputs))[:, start_class:end_class]
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
    res = [round(correct / total * 100, 2), task_accuracies]
    return res


def train(
    fabric,
    server_model: BaseModel,
    client_models: List[BaseModel],
    dataset: BaseDataset,
    args: dict,
    output_folder: str,
) -> None:
    
    accuracies_each_task = []

    if args["wandb"]:
        name = f"{args['nickname']}_{args['dataset']}_{args['model']}_rnds{args['num_comm_rounds']}_clnts{args['num_clients']}_epchs{args['num_epochs']}_bs{args['batch_size']}_lr{args['lr']}"
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

    total_client_indexes = torch.tensor(list(range(args["num_clients"])))
    start_time = time()
    for task in range(dataset.N_TASKS):
        if task < start_task:
            continue
        #if "oos" in args["dataset"]:
        if dataset.IS_TEXT:
            train_loaders, test_loaders = dataset.get_cur_dataloaders_oos(task)
        else:
            train_loaders, test_loaders = dataset.get_cur_dataloaders(task)  # TODO: test_loaders are not used
        last_task_time = time()
        if isinstance(dataset.N_CLASSES_PER_TASK, list):
            server_model.begin_task(dataset.N_CLASSES_PER_TASK[task])
        else:
            server_model.begin_task(dataset.N_CLASSES_PER_TASK)
        # server_model.warmup_task_server(train_loaders)
        # server_info_warmup = server_model.get_server_info()
        # client_info_warmup = []
        active_clients_sampled = torch.randperm(args["num_clients"])[
            : int(args["num_clients"] * args["participation_rate"])
        ]
        active_clients_sampled = (active_clients_sampled[torch.argsort(active_clients_sampled)]).tolist()
        for index, client_model in zip(active_clients_sampled, client_models):
            client_model.augment = dataset.train_transform
            client_model.test_transform = dataset.test_transform
            if isinstance(dataset.N_CLASSES_PER_TASK, list):
                client_model.begin_task(dataset.N_CLASSES_PER_TASK[task])
            else:
                client_model.begin_task(dataset.N_CLASSES_PER_TASK)
            train_loader = train_loaders[index]
            train_loader = fabric.setup_dataloaders(train_loader)
            # client_info_warmup.append(client_model.warmup_task_client(server_info_warmup, train_loader))
        for comm_round in range(args["num_comm_rounds"]):
            # server_model.begin_round_server(client_info_warmup)
            server_model.begin_round_server()
            server_info = server_model.get_server_info()
            if comm_round < start_comm_round:
                continue
            clients_info = []
            last_round_time = time()
            # when participation_rate is 1, all clients are active, so idx and client_idx are the same
            for idx, client_idx in enumerate(active_clients_sampled):
                train_loader = train_loaders[client_idx]
                test_loader = test_loaders[client_idx]
                train_loader = fabric.setup_dataloaders(train_loader)
                test_loader = fabric.setup_dataloaders(test_loader)
                model = client_models[idx]
                model.to(model.device)
                model.begin_round_client(train_loader, server_info)
                for epoch in range(args["num_epochs"]):
                    last_value = None
                    for i, (inputs, labels) in enumerate(train_loader):
                        train_loss = model.observe(inputs, labels)
                        t_loss = train_loss
                        if type(train_loss) == dict:
                            t_loss = train_loss[list(train_loss.keys())[0]]
                        elif type(train_loss) == list or type(train_loss) == tuple or type(train_loss) == set:
                            t_loss = train_loss[0]
                            last_value = train_loss[-1]
                        if last_value is not None and type(last_value) == str and last_value == "break":
                            break
                        assert not torch.isnan(
                            torch.tensor(t_loss)
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
                if args["test_local"]:
                    accuracy = evaluate_client(fabric, task, model, dataset, client_idx)
                    print(f"Client {client_idx} Local acc: {accuracy[0]}")
                    if args["wandb"]:
                        wandb.log({f"Client {client_idx} Local acc": accuracy[0], "comm_round": comm_round + 1 + task * args["num_comm_rounds"]})
                if args["test_local_transfer"]:
                    accuracy = evaluate_client_transfer(fabric, task, model, dataset, client_idx)
                    print(f"Client {client_idx} Local Transfer acc: {accuracy[0]}")
                    if args["wandb"]:
                        wandb.log({f"Client {client_idx} Local Transfer acc": accuracy[0], "comm_round": comm_round + 1 + task * args["num_comm_rounds"]})

                if args["validation_interval"] > 0 and (comm_round + 1) % args["validation_interval"] == 0:
                    model.end_round_validation_client(train_loader, test_loader)
                    accuracy = evaluate(fabric, task, model, dataset)
                    if args["wandb"]:
                        wandb.log({"Global client acc": accuracy, "comm_round": comm_round + 1 + task * args["num_comm_rounds"]})
                model.to("cpu")
                clients_info.append(model.get_client_info(train_loader))
                torch.cuda.empty_cache()
                if len(train_loader):
                    print()

            print("\nRound time:", get_time_str(time() - last_round_time))
            server_model.end_round_server(clients_info)
            server_model.to(server_model.device)
            accuracy = evaluate(fabric, task, server_model, dataset)
            if args["wandb"]:
                results = {
                    "Mean_accuracy": accuracy[0],
                    "task": task + 1 
                }
                for i in range(len(accuracy[1])):
                    results[f"Task_{i + 1}_accuracy"] = accuracy[1][i]
                wandb.log(results)
            if (
                (epoch % args["checkpoint_interval"] == 0 or (comm_round + 1) == args["num_comm_rounds"])
                and not args["debug_mode"]
            ) and args["save_models"]:
                server_model.save_checkpoint(output_folder, task, comm_round)
            torch.cuda.empty_cache()
            if args["validation_interval"] > 0 and (comm_round + 1) % args["validation_interval"] == 0:
                server_model.end_round_validation_server(train_loader, test_loader)
                print("Evaluation after round:")
                accuracy = evaluate(fabric, task, server_model, dataset)

        client_info = []
        server_info = server_model.get_server_info()
        for idx, client_idx in enumerate(active_clients_sampled):
            train_loader = train_loaders[client_idx]
            test_loader = test_loaders[client_idx]
            train_loader = fabric.setup_dataloaders(train_loader)
            test_loader = fabric.setup_dataloaders(test_loader)
            model = client_models[idx]
            model.to(model.device)
            client_info.append(model.end_task_client(train_loader, server_info))
            model.to("cpu")
            torch.cuda.empty_cache()
        server_model.end_task_server(client_info=client_info)
        server_model.to(model.device)
        torch.cuda.empty_cache()
        accuracy = evaluate(fabric, task, server_model, dataset)
        accuracies_each_task.append(accuracy[1])
        print(f"Task {task + 1} time:", get_time_str(time() - last_task_time))
        print("__________\n")

        if args["wandb"]:
            results = {
                "Mean_accuracy": accuracy[0],
                "End of task Accuracy": accuracy[0],
                "task": task + 1
                # TODO: add other things here
            }

            wandb.log(results)

    # TODO: it is probably needed a final evaluation here. At least for models that do something at the end_task()

    print("\nTotal training time:", get_time_str(time() - start_time))
    for client_model in client_models:
        client_model.end_training()
    server_model.end_training()
    forgetting = compute_forgetting(accuracies_each_task)
    paths = accs_and_forgetting_matrix(accuracies_each_task, forgetting, output_folder)
    if args["wandb"]:
        acc_image = wandb.Image(paths[0], caption="Accuracies Matrix")
        forg_image = wandb.Image(paths[1], caption="Forgetting Vector")

        wandb.log({"Accuracies Matrix": acc_image, "Forgetting Vector": forg_image, "task": task + 1})

    if args["wandb"]:
        wandb.finish()
