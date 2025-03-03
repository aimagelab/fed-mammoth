import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

try:
    import deeplake
except ImportError:
    raise NotImplementedError("Deeplake not installed. Please install with `pip install \"deeplake<4\"` to use this dataset.")

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train": K.AugmentationSequential(
        K.RandomHorizontalFlip(),
        K.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ),
    "default_test": K.AugmentationSequential(
        K.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    )
}


def load_and_preprocess_cars196(train_str="train") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads data from deeplake and preprocesses it to be stored locally.

    Args:
        train_str (str): 'train' or 'test'.
        names_only (bool): If True, returns the class names only.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]
    """
    assert train_str in ["train", "test"], "train_str must be 'train' or 'test'"
    ds = deeplake.load(f"hub://activeloop/stanford-cars-{train_str}")
    loader = ds.pytorch()

    # Pre-process dataset
    data = []
    targets = []
    for x in tqdm(loader, desc=f"Pre-processing {train_str} dataset"):
        img = x["images"][0].permute(2, 0, 1)  # load one image at a time
        if len(img) < 3:
            img = img.repeat(3, 1, 1)  # fix rgb
        img = MyCars196.PREPROCESSING_TRANSFORM(img)  # resize
        data.append(img)
        label = x["car_models"][0].item()  # get label
        targets.append(label)

    data = torch.stack(data)  # stack all images
    targets = torch.tensor(targets)

    return data, targets


class MyCars196(Dataset):
    PREPROCESSING_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
        ]
    )

    def __init__(self, root, train=True, transform=None) -> None:
        self.root = root
        self.train = train
        self.transform = transform

        train_str = "train" if train else "test"
        if not os.path.exists(f"{root}/cars196/{train_str}_images.pt"):
            print(f"Downloading and preparing {train_str} dataset...", file=sys.stderr)
            self.load_and_preprocess_dataset(f"{root}/cars196", train_str)
        else:
            print(f"Loading pre-processed {train_str} dataset...", file=sys.stderr)
            self.data = torch.load(f"{root}/cars196/{train_str}_images.pt")
            self.targets = torch.load(f"{root}/cars196/{train_str}_labels.pt")

    def load_and_preprocess_dataset(self, root, train_str="train"):
        self.data, self.targets = load_and_preprocess_cars196(train_str)

        print(
            f"Saving pre-processed dataset in {root} ({train_str}_images.pt and {train_str}_labels.py)...",
            file=sys.stderr,
        )
        if not os.path.exists(root):
            os.makedirs(root)
        torch.save(self.data, f"{root}/{train_str}_images.pt")
        torch.save(self.targets, f"{root}/{train_str}_labels.pt")

        print("Done.", file=sys.stderr)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.transpose(1, 2, 0), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if not self.train:
            return img, target

        return img, target


@register_dataset("seq-cars")
class SequentialCars(BaseDataset):
    """
    Sequential CARS Dataset. The images are loaded from deeplake, resized to 224x224, and stored locally.
    """

    N_TASKS = 10
    N_CLASSES_PER_TASK = [20] * 9 + [16]

    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    BASE_TRANSFORM = transforms.Compose(
        [transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC), 
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
        distribution_alpha: float = 0.2,
        class_quantity: int = 2,
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
            dataset = MyCars196(
                DATASET_PATH,
                train=True if split == "train" else False,
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
