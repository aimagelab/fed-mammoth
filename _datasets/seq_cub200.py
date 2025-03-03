import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from PIL import Image
from typing import Tuple

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train": K.AugmentationSequential(
        K.Resize(size=(256, 256), resample="bicubic"),
        K.RandomCrop(size=(224, 224)),
        K.RandomHorizontalFlip(),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ),
    "default_test": K.AugmentationSequential(
        K.Resize(size=(256, 256), resample="bicubic"),
        K.CenterCrop(size=(224, 224)),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ),
    "angel_train": K.AugmentationSequential(
        K.Resize((300, 300), resample="bicubic"),
        K.RandomCrop((224, 224)),
        K.RandomHorizontalFlip(),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ),
    "angel_test": K.AugmentationSequential(
        K.Resize((300, 300), resample="bicubic"),
        K.RandomCrop((224, 224)),
        K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ),
}


class MyCUB200(Dataset):
    IMG_SIZE = 224

    def __init__(self, root, train=True, transform=None, download=True) -> None:
        self.root = os.path.join(root, "cub200")
        self.train = train
        self.transform = transform
        self.download = download

        if download:
            if os.path.isdir(self.root) and len(os.listdir(self.root)) > 0:
                print("Download not needed, files already on disk.")
            else:
                from onedrivedownloader import download

                ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21110&authkey=AIEfi5nlRyY1yaE" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
                print("Downloading dataset...")
                download(
                    ln,
                    filename=os.path.join(self.root, "cub_200_2011.zip"),
                    unzip=True,
                    unzip_path=self.root,
                    clean=True,
                )

        data_file = np.load(os.path.join(self.root, "train_data.npz" if train else "test_data.npz"), allow_pickle=True)

        self.data = data_file["data"]
        self.targets = data_file["targets"].astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray((img * 255).astype(np.int8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-cub200")
class SequentialCub200(BaseDataset):
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

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
            dataset = MyCUB200(
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


    def train_transform(self, x):
        return TRANSFORMS[self.train_transf](x)
    
    def test_transform(self, x):
        return TRANSFORMS[self.test_transf](x)


@register_dataset("joint-cub200")
class JointISIC(SequentialCub200):
    N_CLASSES_PER_TASK = 200
    N_TASKS = 1

