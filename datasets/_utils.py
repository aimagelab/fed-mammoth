from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from kornia import augmentation as K


class BaseDataset:
    NAME = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRAIN_TRANSFORM = None
    TEST_TRANSFORM = None
    IS_TEXT = False

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
        self.train_transf = None
        self.test_transf = None

    def set_transforms(self, train_transform : str = None, test_transform : str = None):
        if train_transform is not None:
            self.train_transf = train_transform
        if test_transform is not None:
            self.test_transf = test_transform

    def _split_fcil_OOS(
        self,
        num_clients,
        partition_mode,
        distribution_alpha=None,
        class_quantity=None,
        format="numpy",
    ):
        assert partition_mode in ["distribution", "quantity", "extended"]
        if partition_mode == "distribution" or partition_mode == "extended":
            assert distribution_alpha is not None
        elif partition_mode == "quantity":
            assert class_quantity is not None

        num_samples_per_client = []
        for split in ["train", "test"]:
            print(f"Splitting {split} data")
            dataset = getattr(self, f"{split}_dataset")
            min_samples_split = 6 if split == "train" else 1
            for task in range(0, self.N_TASKS):
                min_samples = 0
                iterations = 0
                if split == "train" and partition_mode == "extended":
                    default_samples = 7
                    base_class = task * self.N_CLASSES_PER_TASK
                    cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)
                    cpt = self.N_CLASSES_PER_TASK
                    total_samples = np.stack(
                        [dataset.data[dataset.targets == clas].shape[0] for clas in cur_classes]
                    ).sum()
                    classes_data = [dataset.data[dataset.targets == clas] for clas in cur_classes]
                    classes_targets = [dataset.targets[dataset.targets == clas] for clas in cur_classes]
                    unrolled_assignments_per_class = np.concatenate(
                        [np.ones(len(classes_data[clas % cpt]), dtype=int) * (-1) for clas in cur_classes]
                    ).flatten()
                    clients_assignments_per_class = [
                        np.ones(len(classes_data[clas % cpt]), dtype=int) * (-1) for clas in cur_classes
                    ]
                    clients_classes_distr = np.random.dirichlet(
                        np.repeat(0.05, num_clients), size=len(cur_classes)
                    )  # num_classes x num_clients
                    classes_clients_numbers = {
                        clas % cpt: [] for clas in cur_classes
                    }  # key = class, value = [[clients], [how_many_samples_per_client]]
                    for clas in cur_classes:
                        classes_clients_numbers[clas % cpt].append([c for c in range(num_clients)])
                        classes_clients_numbers[clas % cpt].append(np.zeros((num_clients,), dtype=int))
                    for client in range(num_clients):
                        distr = torch.distributions.Categorical(torch.tensor(clients_classes_distr[:, client]))
                        classes_to_sample = distr.sample((default_samples,)).numpy()
                        for clas in classes_to_sample:
                            # classes_clients_numbers[clas][0].append(client)
                            # classes_clients_numbers[clas][1].append(1)
                            classes_clients_numbers[clas % cpt][1][client] += 1
                    # clients_selections = np.random.choice(total_samples, (num_clients, default_samples), replace=True)
                    # for client in range(num_clients):
                    #    unrolled_assignments_per_class[clients_selections[client]] = client
                    # prev = 0
                    # for clas in cur_classes:
                    #    clients_assignments_per_class[clas % cpt] = unrolled_assignments_per_class[prev:prev + len(classes_data[clas % cpt])]
                    for clas in cur_classes:
                        max_value = clients_assignments_per_class[clas % cpt].shape[0]
                        samples = np.random.choice(max_value, sum(classes_clients_numbers[clas % cpt][1]), replace=True)
                        tmp = 0
                        for client in range(num_clients):
                            how_many_samples = classes_clients_numbers[clas % cpt][1][client]
                            if how_many_samples != 0:
                                clients_assignments_per_class[clas % cpt][
                                    samples[tmp : tmp + how_many_samples]
                                ] = client
                                tmp += how_many_samples

                while min_samples < min_samples_split:
                    task_data_sentences = [list() for _ in range(num_clients)]
                    task_data_attmat = [list() for _ in range(num_clients)]
                    task_targets = [list() for _ in range(num_clients)]
                    if isinstance(self.N_CLASSES_PER_TASK, list):
                        base_class = sum(self.N_CLASSES_PER_TASK[:task])
                        cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK[task])
                    else:
                        base_class = task * self.N_CLASSES_PER_TASK
                        cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)

                    num_samples_per_client_task = []

                    iterations += 1
                    assert iterations < 10000
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

                    # we assign a number of samples to each client, so that every clients sees
                    # some samples in each task, then we use partition_mode = "distribution" to assign
                    # the rest of the samples to the clients
                    # if partition_mode == "extended":

                    for clas in cur_classes:
                        # class_data = dataset.data[dataset.targets == clas]
                        selected_tokenized_sentences = dataset.data["input_ids"][dataset.targets == clas]
                        selected_attention_masks = dataset.data["attention_mask"][dataset.targets == clas]
                        selected_data = {
                            "input_ids": selected_tokenized_sentences,
                            "attention_mask": selected_attention_masks,
                        }
                        class_data = BatchEncoding(data=selected_data)
                        class_targets = dataset.targets[dataset.targets == clas]
                        num_samples = len(selected_tokenized_sentences)

                        if split == "train":
                            if partition_mode == "distribution" or partition_mode == "extended":
                                probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))
                                while np.isnan(probs).sum() > 0:
                                    probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))

                            elif partition_mode == "quantity":
                                probs = np.zeros((num_clients,))
                                for client_idx in clients_per_class[clas]:
                                    probs[client_idx] = 1 / len(clients_per_class[clas])

                            probs2 = np.where(probs < 1e-20, 0, probs)
                            probs3 = np.clip(probs2, 0, 1)
                            client_distr = torch.distributions.Categorical(torch.tensor(probs3))
                            assigned_client = client_distr.sample((num_samples,)).numpy()
                            if partition_mode == "extended":
                                assigned_client = np.where(
                                    clients_assignments_per_class[clas % cpt] != -1,
                                    clients_assignments_per_class[clas % cpt],
                                    assigned_client,
                                )
                            num_samples_per_client_task.append(
                                [(assigned_client == i).sum().item() for i in range(num_clients)]
                            )
                        else:
                            if max(num_samples_per_client[clas]) < sum(num_samples_per_client[clas]) / num_samples:
                                train_test_ratio = 1
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
                            sample = np.array(sample).reshape(-1)
                            assigned_client = np.concatenate([assigned_client, sample])
                            assigned_client = assigned_client[:num_samples]

                        # TODO riscorporo di class_data (temo brutte cose per la gestione della memoria)
                        #   for:
                        #       task_data_sentences
                        #       task_data_attmat
                        #   task_data_sentences
                        #   task_data_attmat

                        for client_idx in range(num_clients):
                            sentences = class_data["input_ids"]
                            attmat = class_data["attention_mask"]
                            task_data_sentences[client_idx] += [sentences[assigned_client == client_idx]]
                            task_data_attmat[client_idx] += [attmat[assigned_client == client_idx]]
                            task_targets[client_idx] += [class_targets[assigned_client == client_idx]]

                    task_data_sentences = [
                        torch.cat([clas_data for clas_data in client_data]) for client_data in task_data_sentences
                    ]
                    task_data_attmat = [
                        torch.cat([clas_data for clas_data in client_data]) for client_data in task_data_attmat
                    ]

                    task_targets = [
                        np.concatenate([clas_data for clas_data in client_data]) for client_data in task_targets
                    ]

                    min_samples = min([len(client_data) for client_data in task_data_attmat])
                    # SO FAR SO GOOD

                selected_data = {"input_ids": task_data_sentences, "attention_mask": task_data_attmat}
                task_data = BatchEncoding(data=selected_data)

                for i in range(len(num_samples_per_client_task)):
                    num_samples_per_client.append(num_samples_per_client_task[i])
                if format == "pytorch":
                    getattr(self, f"{split}_data").append([torch.tensor(td) for td in task_data])
                    getattr(self, f"{split}_targets").append([torch.tensor(tt) for tt in task_targets])
                else:
                    getattr(self, f"{split}_data").append(task_data)
                    getattr(self, f"{split}_targets").append(task_targets)

        print("Data split done")

    def _split_fcil(
        self,
        num_clients,
        partition_mode,
        distribution_alpha=None,
        class_quantity=None,
        format="numpy",
    ):
        assert partition_mode in ["distribution", "quantity", "extended"]
        if partition_mode == "distribution" or partition_mode == "extended":
            assert distribution_alpha is not None
        elif partition_mode == "quantity":
            assert class_quantity is not None

        num_samples_per_client = []
        for split in ["train", "test"]:
            print(f"Splitting {split} data")
            dataset = getattr(self, f"{split}_dataset")
            min_samples_split = 6 if split == "train" else 1
            for task in range(0, self.N_TASKS):
                min_samples = 0
                iterations = 0
                if split == "train" and partition_mode == "extended":
                    default_samples = 7
                    base_class = task * self.N_CLASSES_PER_TASK
                    cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)
                    cpt = self.N_CLASSES_PER_TASK
                    total_samples = np.stack(
                        [dataset.data[dataset.targets == clas].shape[0] for clas in cur_classes]
                    ).sum()
                    classes_data = [dataset.data[dataset.targets == clas] for clas in cur_classes]
                    classes_targets = [dataset.targets[dataset.targets == clas] for clas in cur_classes]
                    unrolled_assignments_per_class = np.concatenate(
                        [np.ones(len(classes_data[clas % cpt]), dtype=int) * (-1) for clas in cur_classes]
                    ).flatten()
                    clients_assignments_per_class = [
                        np.ones(len(classes_data[clas % cpt]), dtype=int) * (-1) for clas in cur_classes
                    ]
                    clients_classes_distr = np.random.dirichlet(
                        np.repeat(0.05, num_clients), size=len(cur_classes)
                    )  # num_classes x num_clients
                    classes_clients_numbers = {
                        clas % cpt: [] for clas in cur_classes
                    }  # key = class, value = [[clients], [how_many_samples_per_client]]
                    for clas in cur_classes:
                        classes_clients_numbers[clas % cpt].append([c for c in range(num_clients)])
                        classes_clients_numbers[clas % cpt].append(np.zeros((num_clients,), dtype=int))
                    for client in range(num_clients):
                        distr = torch.distributions.Categorical(torch.tensor(clients_classes_distr[:, client]))
                        classes_to_sample = distr.sample((default_samples,)).numpy()
                        for clas in classes_to_sample:
                            # classes_clients_numbers[clas][0].append(client)
                            # classes_clients_numbers[clas][1].append(1)
                            classes_clients_numbers[clas % cpt][1][client] += 1
                    # clients_selections = np.random.choice(total_samples, (num_clients, default_samples), replace=True)
                    # for client in range(num_clients):
                    #    unrolled_assignments_per_class[clients_selections[client]] = client
                    # prev = 0
                    # for clas in cur_classes:
                    #    clients_assignments_per_class[clas % cpt] = unrolled_assignments_per_class[prev:prev + len(classes_data[clas % cpt])]
                    for clas in cur_classes:
                        max_value = clients_assignments_per_class[clas % cpt].shape[0]
                        samples = np.random.choice(max_value, sum(classes_clients_numbers[clas % cpt][1]), replace=True)
                        tmp = 0
                        for client in range(num_clients):
                            how_many_samples = classes_clients_numbers[clas % cpt][1][client]
                            if how_many_samples != 0:
                                clients_assignments_per_class[clas % cpt][
                                    samples[tmp : tmp + how_many_samples]
                                ] = client
                                tmp += how_many_samples

                while min_samples < min_samples_split:
                    task_data = [list() for _ in range(num_clients)]
                    task_targets = [list() for _ in range(num_clients)]
                    if isinstance(self.N_CLASSES_PER_TASK, list):
                        base_class = sum(self.N_CLASSES_PER_TASK[:task])
                        cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK[task])
                    else:
                        base_class = task * self.N_CLASSES_PER_TASK
                        cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)

                    num_samples_per_client_task = []

                    iterations += 1
                    assert iterations < 10000
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

                    # we assign a number of samples to each client, so that every clients sees
                    # some samples in each task, then we use partition_mode = "distribution" to assign
                    # the rest of the samples to the clients
                    # if partition_mode == "extended":

                    for clas in cur_classes:
                        class_data = dataset.data[dataset.targets == clas]
                        class_targets = dataset.targets[dataset.targets == clas]
                        num_samples = len(class_data)

                        if split == "train":
                            if partition_mode == "distribution" or partition_mode == "extended":
                                probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))
                                while np.isnan(probs).sum() > 0:
                                    probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))

                            elif partition_mode == "quantity":
                                probs = np.zeros((num_clients,))
                                for client_idx in clients_per_class[clas]:
                                    probs[client_idx] = 1 / len(clients_per_class[clas])

                            probs2 = np.where(probs < 1e-20, 0, probs)
                            probs3 = np.clip(probs2, 0, 1)
                            client_distr = torch.distributions.Categorical(torch.tensor(probs3))
                            assigned_client = client_distr.sample((num_samples,)).numpy()
                            if partition_mode == "extended":
                                assigned_client = np.where(
                                    clients_assignments_per_class[clas % cpt] != -1,
                                    clients_assignments_per_class[clas % cpt],
                                    assigned_client,
                                )
                            num_samples_per_client_task.append(
                                [(assigned_client == i).sum().item() for i in range(num_clients)]
                            )
                        else:
                            if max(num_samples_per_client[clas]) < sum(num_samples_per_client[clas]) / num_samples:
                                train_test_ratio = 1
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
                            sample = np.array(sample).reshape(-1)
                            assigned_client = np.concatenate([assigned_client, sample])
                            assigned_client = assigned_client[:num_samples]

                        for client_idx in range(num_clients):
                            task_data[client_idx] += [class_data[assigned_client == client_idx]]
                            task_targets[client_idx] += [class_targets[assigned_client == client_idx]]
                    task_data = [np.concatenate([clas_data for clas_data in client_data]) for client_data in task_data]
                    task_targets = [
                        np.concatenate([clas_data for clas_data in client_data]) for client_data in task_targets
                    ]
                    min_samples = min([len(client_data) for client_data in task_data])
                for i in range(len(num_samples_per_client_task)):
                    num_samples_per_client.append(num_samples_per_client_task[i])
                if format == "pytorch":
                    getattr(self, f"{split}_data").append([torch.tensor(td) for td in task_data])
                    getattr(self, f"{split}_targets").append([torch.tensor(tt) for tt in task_targets])
                else:
                    getattr(self, f"{split}_data").append(task_data)
                    getattr(self, f"{split}_targets").append(task_targets)
        print("Data split done")

    def get_cur_dataloaders_oos(self, task: int):
        self.cur_train_loaders, self.cur_test_loaders = [], []
        for split in ["train", "test"]:
            for client_idx in range(self.num_clients):
                cur_dataset = deepcopy(
                    getattr(self, f"{split}_dataset")
                )  # TODO: to discuss, I mean it's a deepcopy of a None object?

                tasks_data = getattr(self, f"{split}_data")[task]
                cur_input_ids = tasks_data["input_ids"][client_idx]
                cur_input_attmat = tasks_data["attention_mask"][client_idx]

                cur_data = {"input_ids": cur_input_ids, "attention_mask": cur_input_attmat}

                cur_dataset.data = BatchEncoding(cur_data)
                cur_dataset.targets = getattr(self, f"{split}_targets")[task][client_idx]

                # TODO: to add in the Dataloader num_workers, shuffle and potentially other params
                getattr(self, f"cur_{split}_loaders").append(DataLoader(cur_dataset, self.batch_size, shuffle=True))

        return self.cur_train_loaders, self.cur_test_loaders

    def get_cur_dataloaders(self, task: int):
        self.cur_train_loaders, self.cur_test_loaders = [], []
        for split in ["train", "test"]:
            for client_idx in range(self.num_clients):
                cur_dataset = deepcopy(
                    getattr(self, f"{split}_dataset")
                )  # TODO: to discuss, I mean it's a deepcopy of a None object?
                cur_dataset.data = getattr(self, f"{split}_data")[task][client_idx]
                cur_dataset.targets = getattr(self, f"{split}_targets")[task][client_idx]

                # TODO: to add in the Dataloader num_workers, shuffle and potentially other params
                getattr(self, f"cur_{split}_loaders").append(DataLoader(cur_dataset, self.batch_size, shuffle=True))

        return self.cur_train_loaders, self.cur_test_loaders
