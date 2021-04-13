import torch
from torch.nn.functional import softplus, tanh

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input):
        return _input * tanh(softplus(_input))
