from datasets import register_dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from datasets.utils import BaseDataset
from utils.global_consts import DATASET_PATH
import numpy as np


@register_dataset("seq-cifar100")
class SequentialCifar100(BaseDataset):
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRAIN_TRANSFORM = transforms.ToTensor()
    TEST_TRANSFORM = transforms.ToTensor()
    INPUT_SHAPE = (3, 32, 32)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        partition_mode: str,
        distribution_alpha: float = 0.05,
        class_quantity: int = 2,
    ):
        super().__init__(
            num_clients,
            batch_size,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )
        for split in ["train", "test"]:
            dataset = CIFAR100(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=getattr(self, f"{split.upper()}_TRANSFORM"),
            )
            dataset.targets = np.array(dataset.targets)
            dataset.targets = np.array(dataset.targets).astype(np.int64)
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil(num_clients, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None
