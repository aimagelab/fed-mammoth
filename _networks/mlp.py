import math
import torch
import torch.nn as nn
from _networks import register_network
from _networks._utils import BaseNetwork
import numpy as np


@register_network("mlp")
class MLP(BaseNetwork):
    HIDDEN_SIZE = 100

    def __init__(self, input_shape: int, num_classes: int) -> None:
        super().__init__()
        self.input_shape = np.prod(input_shape)
        self.output_shape = num_classes

        self.fc1 = nn.Linear(self.input_shape, self.HIDDEN_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )
        self.classifier = nn.Linear(self.HIDDEN_SIZE, self.output_shape)
        self.net = nn.Sequential(self._features, self.classifier)
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, num_flat_features(x))
        feats = self._features(x)
        out = self.classifier(feats)
        return out


def xavier(m: nn.Module) -> None:
    if m.__class__.__name__ == "Linear":
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def num_flat_features(x: torch.Tensor) -> int:
    return torch.prod(torch.tensor(x.shape[1:]))
