import torch
from torch.nn.functional import softplus


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input):
        return _input * torch.tanh(softplus(_input))
