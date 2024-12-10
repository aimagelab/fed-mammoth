import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
from onedrivedownloader import download as onedrive_download

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train" : K.AugmentationSequential(
        K.RandomResizedCrop(size=(224, 224), resample="bicubic"),
        K.RandomHorizontalFlip(),
        K.Normalize(mean=(0.4802, 0.4480, 0.3975), std=(0.2770, 0.2691, 0.2821)),
    ),
    "default_test" : K.AugmentationSequential(
        K.Resize(size=(256, 256), resample="bicubic"),
        K.CenterCrop(size=(224, 224)),
        K.Normalize(mean=(0.4802, 0.4480, 0.3975), std=(0.2770, 0.2691, 0.2821)),
    ),
}


class MyTinyImageNet(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: transforms = None,
        download: bool = True,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if not os.path.exists(self.root + "/TinyImageNet") and self.download:
            ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21111&authkey=AJ9v0OmtrqOpxFA" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
            print("Downloading TinyImageNet...", file=sys.stderr)
            onedrive_download(
                ln,
                filename=os.path.join(self.root, "tiny-imagenet-processed.zip"),
                unzip=True,
                unzip_path=self.root + "/TinyImageNet",
                clean=True,
            )

        self.data = []
        for num in range(20):
            self.data.append(
                np.load(
                    os.path.join(
                        self.root + "/TinyImageNet",
                        "processed/x_%s_%02d.npy" % ("train" if self.train else "val", num + 1),
                    )
                )
            )
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(
                np.load(
                    os.path.join(
                        self.root + "/TinyImageNet",
                        "processed/y_%s_%02d.npy" % ("train" if self.train else "val", num + 1),
                    )
                )
            )
        self.targets = np.concatenate(np.array(self.targets)).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this to have a PIL Image for the transforms
        # img = Image.fromarray(np.uint8(255 * img))
        # Altermatively, we can use the following line to convert the image to PIL format
        # transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-tinyimagenet")
class SequentialTinyImageNet(BaseDataset):
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10

    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    """
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

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
            dataset = MyTinyImageNet(
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


@register_dataset("joint-tinyimagenet")
class JointTinyImageNet(SequentialTinyImageNet):
    N_CLASSES_PER_TASK = 200
    N_TASKS = 1
