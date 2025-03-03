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
            cand_params = new_params[progress : progress + torch.tensor(pp.size()).prod()].view(pp.size()).detach().clone()
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self, discard_classifier=False) -> torch.Tensor:
        grads = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'head' in kk:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def set_grads(self, new_grads: torch.Tensor, discard_classifier=False) -> torch.Tensor:
        progress = 0
        for pp in list(self.parameters() if not discard_classifier else self._features.parameters()):
            cand_grads = new_grads[progress: progress +
                                   torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.grad = cand_grads