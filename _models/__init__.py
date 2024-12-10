import os
import importlib
from typing import Callable
from _models._utils import BaseModel

__all__ = ["model_factory"]


__MODEL_DICT__ = dict()


def register_model(name: str) -> Callable:
    def register_model_fn(cls: BaseModel) -> BaseModel:
        if name in __MODEL_DICT__:
            raise ValueError(f"Name {name} already registered!")
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name, _ = os.path.splitext(file)
        relative_module_name = f".{module_name}"
        module = importlib.import_module(relative_module_name, package=__name__)


def model_factory(name: str) -> BaseModel:
    assert name in __MODEL_DICT__, "Attempted to access non-registered model"
    return __MODEL_DICT__[name]
