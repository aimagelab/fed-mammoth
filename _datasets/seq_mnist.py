from _datasets import register_dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train" : lambda x : x,
    "default_test" : lambda x : x,
}


@register_dataset("seq-mnist")
class SequentialMNIST(BaseDataset):
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    #TRAIN_TRANSFORM = transforms.ToTensor()
    #TEST_TRANSFORM = transforms.ToTensor()
    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), interpolation=3),
            transforms.ToTensor(),
        ]
    )
    INPUT_SHAPE = (28, 28)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
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
        for split in ["train", "test"]:
            dataset = MNIST(
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
            format="pytorch",
        )

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None


    def train_transform(self, x):
        return TRANSFORMS[self.train_transf](x)
    
    def test_transform(self, x):
        return TRANSFORMS[self.test_transf](x)
