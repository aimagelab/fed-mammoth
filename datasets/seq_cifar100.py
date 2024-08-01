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
    INPUT_SHAPE = (32, 32, 3)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        partition_mode: str = "distribution",
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
            dataset.targets = np.array(dataset.targets).astype(np.int64)
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil(num_clients, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None


@register_dataset("seq-cifar100_224")
class SequentialCifar100_224(SequentialCifar100):
    MEAN_NORM = (0.5, 0.5, 0.5)
    STD_NORM = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(MEAN_NORM, STD_NORM)
    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    TEST_TRANSFORM = transforms.Compose(
        [transforms.Resize(256, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    INPUT_SHAPE = (224, 224, 3)


@register_dataset("seq-cifar100_224_hgp")
class SequentialCifar100_224_hgp(SequentialCifar100):
    MEAN_NORM = (0.0, 0.0, 0.0)
    STD_NORM = (1.0, 1.0, 1.0)
    normalize = transforms.Normalize(MEAN_NORM, STD_NORM)
    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(224, 224), interpolation=transforms.functional.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ]
    )
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ]
    )
    INPUT_SHAPE = (224, 224, 3)


@register_dataset("seq-cifar100_224_regmean")
class SequentialCifar100_regmean(SequentialCifar100_224):
    MEAN_NORM = [0.5071, 0.4865, 0.4409]
    STD_NORM = [0.2673, 0.2564, 0.2762]
    normalize = transforms.Normalize(MEAN_NORM, STD_NORM)
    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_NORM, STD_NORM),
        ]
    )
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(224, interpolation=3), transforms.ToTensor(), normalize])


@register_dataset("joint-cifar100")
class JointCifar100(SequentialCifar100):
    N_CLASSES_PER_TASK = 100
    N_TASKS = 1
    TRAIN_TRANSFORM = transforms.ToTensor()
    TEST_TRANSFORM = transforms.ToTensor()
    INPUT_SHAPE = (32, 32, 3)


@register_dataset("joint-cifar100_224")
class JointCifar100_224(SequentialCifar100_224):
    N_CLASSES_PER_TASK = 100
    N_TASKS = 1
    INPUT_SHAPE = (224, 224, 3)
