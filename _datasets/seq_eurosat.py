import os
import sys
import requests
import zipfile
import io
from torch.utils.data import Dataset
import google_drive_downloader as gdd
import pandas as pd
import json
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train": K.AugmentationSequential(
        K.RandomResizedCrop(size=(224, 224), resample='bicubic'),
        K.RandomHorizontalFlip(),
        K.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ),
    "default_test": K.AugmentationSequential(
        K.Resize(size=(224, 224), resample='bicubic'),
        K.CenterCrop(size=(224, 224)),
        K.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ),
    "lorm_iclr_train": lambda x : x,
    "lorm_iclr_test": lambda x : x,
}


class MyEuroSAT(Dataset):
    def __init__(self, root: str, train: bool = True, transform: transforms = None, download: bool = True) -> None:
        self.root = root
        self.train = train
        self.transform = transform

        if not os.path.exists(self.root + "/EuroSAT_RGB") and download:
            print("Downloading EuroSAT...", file=sys.stderr)
            # Downlaod zip from https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
            # and extract to ../data/EuroSAT_RGB
            r = requests.get("https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.root)

            # downlaod split file form https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id="1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o", dest_path=self.root + "/EuroSAT_RGB/split.json"
            )

            print("Done.", file=sys.stderr)

        self.data_split = pd.DataFrame(
            json.load(open(self.root + "/EuroSAT_RGB/split.json", "r"))["train" if self.train == True else "test"]
        )

        self.data = np.array(
            [Image.open(self.root + "/EuroSAT_RGB/" + img).convert("RGB") for img in self.data_split[0].values]
        )
        self.targets = self.data_split[1].values.astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this to have a PIL Image for the transforms
        # img = Image.fromarray(np.uint8(255 * img))
        # Altermatively, we can use the following line to convert the image to PIL format
        # transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-eurosat")
class SequentialEuroSAT(BaseDataset):
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.ToPILImage(),
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
            dataset = MyEuroSAT(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=self.BASE_TRANSFORM,
            )
            # Aggiungo .classes e .class_to_idx che uso per recuperare i nomi e indici delle classi
            dataset.classes = ["Annual Crop Land", "Forest", "Herbaceous Vegetation Land", "Highway or Road",
                               "Industrial Buildings", "Pasture Land", "Permanent Crop Land", "Residential Buildings",
                               "River", "Sea or Lake"]
            dataset.class_to_idx = {cl: i for i,cl in enumerate(dataset.classes)}
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


@register_dataset("joint-eurosat")
class JointEuroSAT224(SequentialEuroSAT):
    N_CLASSES_PER_TASK = 10
    N_TASKS = 1