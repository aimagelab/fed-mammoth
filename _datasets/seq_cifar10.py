from _datasets import register_dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
import numpy as np
from kornia import augmentation as K

TRANSFORMS = {
    "default_train": lambda x: x,
    "default_test": lambda x: x,
}


@register_dataset("seq-cifar10")
class SequentialCifar10(BaseDataset):
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRAIN_TRANSFORM = transforms.ToTensor()
    TEST_TRANSFORM = transforms.ToTensor()
    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), interpolation=3),
            transforms.ToTensor(),
        ]
    )
    INPUT_SHAPE = (32, 32, 3)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        train_transform: str = "default_train",
        test_transform: str = "default_test",
        partition_mode: str = "distribution",
        distribution_alpha: float = 0.05,
        class_quantity: int = 1,
    ):
        super().__init__(
            num_clients,
            batch_size,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )
        self.train_transf = train_transform
        self.test_transf = test_transform

        for split in ["train", "test"]:
            dataset = CIFAR10(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=self.BASE_TRANSFORM,
            )
            dataset.targets = np.array(dataset.targets).astype(np.int64)
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil(num_clients, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None


    def train_transform(self, x):
        return TRANSFORMS[self.train_transf](x)
    
    def test_transform(self, x):
        return TRANSFORMS[self.test_transf](x)