from argparse import ArgumentParser
import inspect
from inspect import Parameter
from _networks import network_factory
from _datasets import dataset_factory
from _models import model_factory
from utils.global_consts import ADDITIONAL_ARGS

EXCLUDE_FROM_ARGS = [
    "self",
    "model",
    "dataset",
    "network",
    "input_shape",
    "num_classes",
    "fabric",
    "device",
    "num_clients",
]


def add_args(parser: ArgumentParser, model_name: str, network_name: str, dataset_name: str) -> None:
    dataset = dataset_factory(dataset_name)
    model = model_factory(model_name)
    network = network_factory(network_name)

    for module in [model, network, dataset]:
        signature = inspect.signature(module)
        for arg_name, value in list(signature.parameters.items()):
            if arg_name not in EXCLUDE_FROM_ARGS and arg_name != "args" and arg_name != "kwds":
                default = value.default
                if default is Parameter.empty:
                    parser.add_argument(f"--{arg_name}", type=value.annotation, required=True)
                else:
                    parser.add_argument(f"--{arg_name}", type=value.annotation, default=default)

    for arg_name, (kind, default) in ADDITIONAL_ARGS.items():
        if default is None:
            parser.add_argument(f"--{arg_name}", type=kind, required=True)
        else:
            parser.add_argument(f"--{arg_name}", type=kind, default=default)
