import os
from typing import Callable
from torch.utils.data import Dataset
import importlib

__all__ = ["dataset_factory"]


__DATASET_DICT__ = dict()


def register_dataset(name: str) -> Callable:
    def register_dataset_fn(cls: Dataset) -> Dataset:
        if name in __DATASET_DICT__:
            raise ValueError(f"Name {name} already registered!")
        __DATASET_DICT__[name] = cls
        return cls

    return register_dataset_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name, _ = os.path.splitext(file)
        relative_module_name = f".{module_name}"
        module = importlib.import_module(relative_module_name, package=__name__)


def dataset_factory(name: str) -> Dataset:
    assert name in __DATASET_DICT__, "Attempted to access non-registered dataset"
    return __DATASET_DICT__[name]
