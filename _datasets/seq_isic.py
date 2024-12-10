import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import pickle
from PIL import Image
from typing import Tuple

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train": K.AugmentationSequential(
        K.Resize((256, 256), resample='bicubic'),
        K.RandomCrop((224, 224)),
        K.RandomHorizontalFlip(0.5),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ),
    "default_test": K.AugmentationSequential(
        K.Resize((256, 256), resample='bicubic'),
        K.RandomCrop((224, 224)),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    )
}


class MyISIC(Dataset):
    def __init__(self, root, train=True, transform=None, download=True) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        split = "train" if self.train else "test"
        if not os.path.exists(f"{self.root}/isic"):
            if download:
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/ERM64PkPkFtJhmiUQkVvE64BR900MbIHtJVA_CR4KKhy8A?e=OsrQr5"
                from onedrivedownloader import download

                download(
                    ln,
                    filename=os.path.join(self.root, "isic.tar.gz"),
                    unzip=True,
                    unzip_path=self.root,
                    clean=True,
                )
            else:
                raise FileNotFoundError(f"File not found: {root}/{split}_images.pkl")

        filename_labels = f"{self.root}/isic/{split}_labels.pkl"
        filename_images = f"{self.root}/isic/{split}_images.pkl"

        with open(filename_images, "rb") as f:
            self.data = pickle.load(f)

        with open(filename_labels, "rb") as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray((img * 255).astype(np.int8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-isic")
class SequentialISIC(BaseDataset):
    N_CLASSES_PER_TASK = 2
    N_TASKS = 3

    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
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
        distribution_alpha: float = 0.5,
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
            dataset = MyISIC(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=self.BASE_TRANSFORM,
            )
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil(
            num_clients,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None

    def train_transform(self, img):
        return TRANSFORMS[self.train_transf](img)
    
    def test_transform(self, img):
        return TRANSFORMS[self.test_transf](img)


@register_dataset("joint-isic")
class JointISIC(SequentialISIC):
    N_CLASSES_PER_TASK = 6
    N_TASKS = 1
