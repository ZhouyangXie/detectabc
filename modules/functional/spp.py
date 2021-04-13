import torch
from torch import nn

class PyramidPooling(nn.Module):
    def __init__(self, allow=None):
        super().__init__(self)
        
        self._allow = [nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d]
        if not allow is None:
            self._allow = allow

        self._pooling_layers = []
        pass

    def forward(self, _input):
        if not len(_input.shape) == 4:
            raise ValueError

        outputs = [_input]
        for pooling in self._pooling_layers:
            outputs.append(pooling(_input))

        return torch.cat(outputs, dim=1)

    def _check_layer(self, layer):
        if not type(layer) in self._allow:
            raise TypeError
        if not layer.stride == 1:
            raise ValueError
        if not layer.kernel_size % 2 == 1:
            raise ValueError
        if not layer.padding == layer.kernel_size:
            raise ValueError

    def add_layer(self, layers):
        for layer in layers:
            self._check_layer(layer)
            self._pooling_layers.append(layer)

    @classmethod
    def from_layers(cls, layers, allowed_poolings=None):
        mod = cls(allowed_poolings)            
        mod.add_layer(layers)
        return mod
