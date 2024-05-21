from datasets import register_dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from datasets.utils import BaseDataset
from utils.global_consts import DATASET_PATH
import numpy as np
from datasets.seq_cifar100 import SequentialCifar100


@register_dataset("joint-cifar100")
class JointCifar100(SequentialCifar100):
    N_CLASSES_PER_TASK = 100
    N_TASKS = 1
    TRAIN_TRANSFORM = transforms.ToTensor()
    TEST_TRANSFORM = transforms.ToTensor()
    INPUT_SHAPE = (32, 32, 3)
