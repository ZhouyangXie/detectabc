import torch
from torch.nn.functional import softplus, tanh

class Output(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args

class MergeFeature(torch.nn.Module):
    def __init__(self, dim:int=1):
        super().__init__()
        self.dim = dim

    def forward(self, _inputs):
        return torch.cat(_inputs, dim=self.dim)