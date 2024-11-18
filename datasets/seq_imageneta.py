import os
import sys
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import google_drive_downloader as gdd


from datasets import register_dataset
from datasets.utils import BaseDataset
from utils.global_consts import DATASET_PATH


class MyImageNetA(Dataset):
    def __init__(self, root, train=True, transform=None, download=True) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if not os.path.exists(self.root + "/imagenet-a") and self.download:
            print("Downloading ImageNetA...", file=sys.stderr)
            print("This may take a while...", file=sys.stderr)

            from onedrivedownloader import download

            ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/217147_unimore_it/EUgcBS9FgYBDnBD7u-aqhCMBQUoNzZ-xwxDQosKxnSdCuw?e=YcI2EJ"
            download(
                ln,
                filename=os.path.join(self.root, "ina.zip"),
                unzip=True,
                unzip_path=self.root,
                clean=True,
            )

            print("Done.", file=sys.stderr)

        if self.train:
            dataset = datasets.ImageFolder(self.root + "/imagenet-a/train/")
        else:
            dataset = datasets.ImageFolder(self.root + "/imagenet-a/test/")

        self.data = np.array([dataset.imgs[i][0] for i in range(len(dataset.imgs))])

        self.targets = np.array([dataset.imgs[i][1] for i in range(len(dataset.imgs))]).astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-imageneta")
class SequentialImageNetA(BaseDataset):
    N_TASKS = 10
    N_CLASSES_PER_TASK = 20

    scale = (0.05, 1.0)
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    INPUT_SHAPE = (224, 224, 3)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        partition_mode: str = "distribution",
        distribution_alpha: float = 1.0,
        class_quantity: int = 4,
    ):
        super().__init__(
            num_clients,
            batch_size,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )

        for split in ["train", "test"]:
            dataset = MyImageNetA(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=getattr(self, f"{split.upper()}_TRANSFORM"),
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


@register_dataset("joint-imageneta")
class JointImagenetA(SequentialImageNetA):
    N_TASKS = 1
    N_CLASSES_PER_TASK = 200
