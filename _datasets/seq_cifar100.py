from _datasets import register_dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
import numpy as np
from kornia import augmentation as K

TRANSFORMS = {
    "default_train": K.AugmentationSequential(
        K.RandomResizedCrop(size=(224, 224), resample='bicubic'),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ),
    "default_test": K.AugmentationSequential(
        K.Resize(size=(256, 256), resample='bicubic'),
        K.CenterCrop(size=(224, 224)),
        K.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    )
}

@register_dataset("seq-cifar100")
class SequentialCifar100(BaseDataset):
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    MEAN_NORM = (0.5, 0.5, 0.5)
    STD_NORM = (0.5, 0.5, 0.5)
    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), interpolation=3),
            transforms.ToTensor(),
        ]
    )
    INPUT_SHAPE = (224, 224, 3)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        train_transform: str = "default_train",
        test_transform: str = "default_test",
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
        self.train_transf= train_transform
        self.test_transf = test_transform
        
        #self.set_transforms(...)
        for split in ["train", "test"]:
            dataset = CIFAR100(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform = self.BASE_TRANSFORM,
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


@register_dataset("joint-cifar100")
class JointCifar100_224(SequentialCifar100):
    N_CLASSES_PER_TASK = 100
    N_TASKS = 1
    INPUT_SHAPE = (224, 224, 3)
