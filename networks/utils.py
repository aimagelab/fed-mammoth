import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 0

    def get_params(self) -> torch.Tensor:
        return torch.cat([param.reshape(-1) for param in self.parameters()])

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress : progress + torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params
