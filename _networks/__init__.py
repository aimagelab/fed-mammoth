import os
import importlib
from torch import nn
from typing import Callable

__all__ = ["network_factory"]


__NETWORK_DICT__ = dict()


def register_network(name: str) -> Callable:
    def register_network_fn(cls: nn.Module) -> nn.Module:
        if name in __NETWORK_DICT__:
            raise ValueError(f"Name {name} already registered!")
        __NETWORK_DICT__[name] = cls
        return cls

    return register_network_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name, _ = os.path.splitext(file)
        relative_module_name = f".{module_name}"
        module = importlib.import_module(relative_module_name, package=__name__)


def network_factory(name: str) -> nn.Module:
    assert name in __NETWORK_DICT__, "Attempted to access non-registered network"
    return __NETWORK_DICT__[name]
