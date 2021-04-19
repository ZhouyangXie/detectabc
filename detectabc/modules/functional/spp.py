import torch
from torch import nn


class SpatialPyramidPooling(nn.Module):
    _allowed_modules = [
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d]

    def __init__(self, pool_type, kernel_sizes):
        super().__init__(self)
        assert pool_type in SpatialPyramidPooling._allowed_modules
        self._pooling_layers = nn.ModuleList(
            [
                pool_type(stride=1, kernel_size=ks, padding=ks//2)
                for ks in kernel_sizes
            ]
        )

    def forward(self, x):
        pooled_x = [pooling(x) for pooling in self._pooling_layers]
        return torch.cat(pooled_x + [x], dim=1)
