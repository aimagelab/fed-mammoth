from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseDataset:
    NAME = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRAIN_TRANSFORM = None
    TEST_TRANSFORM = None

    def __init__(
        self,
        num_clients,
        batch_size,
        partition_mode,
        distribution_alpha,
        class_quantity,
    ):
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []
        self.num_clients = num_clients
        self.batch_size = batch_size

    def _split_fcil(
        self,
        num_clients,
        partition_mode,
        distribution_alpha=None,
        class_quantity=None,
        format="numpy",
    ):
        assert partition_mode in ["distribution", "quantity"]
        if partition_mode == "distribution":
            assert distribution_alpha is not None
        elif partition_mode == "quantity":
            assert class_quantity is not None

        num_samples_per_client = []
        for split in ["train", "test"]:
            dataset = getattr(self, f"{split}_dataset")
            for task in range(0, self.N_TASKS):
                task_data = [list() for _ in range(num_clients)]
                task_targets = [list() for _ in range(num_clients)]
                base_class = task * self.N_CLASSES_PER_TASK
                cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)

                if partition_mode == "quantity":
                    trials = 0
                    while trials < 1000:
                        trials += 1
                        clients_per_class = {cls: list() for cls in cur_classes}
                        classes_set = set()
                        for client_idx in range(num_clients):
                            assert class_quantity <= len(cur_classes)
                            chosen_classes = np.random.choice(cur_classes, class_quantity, replace=False)
                            classes_set = set(list(classes_set) + list(chosen_classes))
                            for chosen_class in chosen_classes:
                                clients_per_class[chosen_class].append(client_idx)
                        if classes_set == set(cur_classes):
                            break

                    assert trials != 1000

                for clas in cur_classes:
                    class_data = dataset.data[dataset.targets == clas]
                    class_targets = dataset.targets[dataset.targets == clas]
                    num_samples = len(class_data)

                    if split == "train":
                        if partition_mode == "distribution":
                            probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))

                        elif partition_mode == "quantity":
                            probs = np.zeros((num_clients,))
                            for client_idx in clients_per_class[clas]:
                                probs[client_idx] = 1 / len(clients_per_class[clas])

                        client_distr = torch.distributions.Categorical(torch.tensor(probs))
                        assigned_client = client_distr.sample((num_samples,)).numpy()
                        num_samples_per_client.append([(assigned_client == i).sum().item() for i in range(num_clients)])
                    else:
                        train_test_ratio = sum(num_samples_per_client[clas]) / num_samples
                        num_samples_per_client[clas] = [
                            int(round(num_samples_per_client[clas][client_idx] / train_test_ratio))
                            for client_idx in range(num_clients)
                        ]
                        assigned_client = np.concatenate(
                            [np.ones((num_samples_per_client[clas][i],), dtype=int) * i for i in range(num_clients)]
                        )
                        assigned_client = assigned_client[np.random.permutation(assigned_client.shape[0])]

                        value, counts = np.unique(assigned_client, return_counts=True)
                        sample = value[torch.distributions.Categorical(torch.tensor(counts)).sample((num_clients,))]
                        assigned_client = np.concatenate([assigned_client, sample])
                        assigned_client = assigned_client[:num_samples]

                    for client_idx in range(num_clients):
                        task_data[client_idx] += [class_data[assigned_client == client_idx]]
                        task_targets[client_idx] += [class_targets[assigned_client == client_idx]]
                task_data = [np.concatenate([clas_data for clas_data in client_data]) for client_data in task_data]
                task_targets = [
                    np.concatenate([clas_data for clas_data in client_data]) for client_data in task_targets
                ]
                if format == "pytorch":
                    getattr(self, f"{split}_data").append([torch.tensor(td) for td in task_data])
                    getattr(self, f"{split}_targets").append([torch.tensor(tt) for tt in task_targets])
                else:
                    getattr(self, f"{split}_data").append(task_data)
                    getattr(self, f"{split}_targets").append(task_targets)

    def get_cur_dataloaders(self, task: int):
        self.cur_train_loaders, self.cur_test_loaders = [], []
        for split in ["train", "test"]:
            for client_idx in range(self.num_clients):
                cur_dataset = deepcopy(getattr(self, f"{split}_dataset"))  # TODO: to discuss
                cur_dataset.data = getattr(self, f"{split}_data")[task][client_idx]
                cur_dataset.targets = getattr(self, f"{split}_targets")[task][client_idx]

                # TODO: to add in the Dataloader num_workers, shuffle and potentially other params
                getattr(self, f"cur_{split}_loaders").append(DataLoader(cur_dataset, self.batch_size))

        return self.cur_train_loaders, self.cur_test_loaders
