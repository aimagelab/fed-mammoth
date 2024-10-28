import os
import sys
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import urllib.request
import tarfile
import google_drive_downloader as gdd
import yaml

from datasets import register_dataset
from datasets.utils import BaseDataset
from utils.global_consts import DATASET_PATH


class MyImageNetR(Dataset):
    def __init__(self, root, train=True, transform=None, download=True) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if not os.path.exists(self.root + "/imagenet-r") and self.download:
            print("Downloading ImageNetR...", file=sys.stderr)
            print("This may take a while...(Up to 5 minutes)", file=sys.stderr)

            # URL of ImageNetR tarfile
            tarfile_url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"

            # Download the tarfile
            urllib.request.urlretrieve(tarfile_url, "downloaded_file.tar")

            # move the downloaded tarfile to the root
            os.rename("downloaded_file.tar", self.root + "/imagenet-r.tar")

            # Extract the tarfile
            with tarfile.open("downloaded_file.tar", "r") as tar:
                tar.extractall(self.root)

            # Remove the downloaded tarfile
            os.remove(self.root + "/imagenet-r.tar")

            # downlaod split file form "https://drive.google.com/file/d/1iNNgknmhWQm-xAvsmMimtyGMQPNS2E_Q/view?usp=sharing"
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id="1iNNgknmhWQm-xAvsmMimtyGMQPNS2E_Q",
                dest_path=self.root + "/imagenet-r/imagenet-r_train.yaml",
                showsize=True,
            )

            # downlaod split file form "https://drive.google.com/file/d/1zXVayHfggvJfmyq-plBV6y3HhXiKVdLo/view?usp=sharing"
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id="1zXVayHfggvJfmyq-plBV6y3HhXiKVdLo",
                dest_path=self.root + "/imagenet-r/imagenet-r_test.yaml",
                showsize=True,
            )

            print("Done.", file=sys.stderr)

        if self.train:
            data_config = yaml.load(open(self.root + "/imagenet-r/imagenet-r_train.yaml"), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open(self.root + "/imagenet-r/imagenet-r_test.yaml"), Loader=yaml.Loader)

        # resize = transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC)
        # [resize(Image.open(img).convert("RGB")) for img in data_config["data"]]

        self.data = np.array(data_config["data"])

        self.targets = np.array(data_config["targets"]).astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-imagenetr")
class SequentialImageNetR(BaseDataset):
    N_TASKS = 10
    N_CLASSES_PER_TASK = 20
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
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
        distribution_alpha: float = 0.05,
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
            dataset = MyImageNetR(
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

@register_dataset("joint-imagenetr")
class JointImagenetR(SequentialImageNetR):
    N_TASKS = 1
    N_CLASSES_PER_TASK = 200