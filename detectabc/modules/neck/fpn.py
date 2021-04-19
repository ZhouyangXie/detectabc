'''
    Adapted from:
        https://github.com/pytorch/vision/blob/master/torchvision/ops/feature_pyramid_network.py
'''
import torch.nn.functional as F
from torch import nn, Tensor

from typing import List


class FeaturePyramidNetwork(nn.Module):
    """
    Feature maps are processed and returned in a high-to-low resolution order
    Args:
        in_channels_list (list[int]): number of channels for input features
        out_channels (list[int]): number of channels for output features
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels_list: List[int],
        mid_channels: int = -1,
    ):
        super().__init__()
        assert len(in_channels_list) == len(out_channels_list)
        if mid_channels <= 0:
            mid_channels = max([*in_channels_list, *out_channels_list])

        self.inner_convs = nn.ModuleList()
        self.outter_convs = nn.ModuleList()

        for in_channels, out_channels in zip(
                in_channels_list, out_channels_list):
            self.inner_convs.append(
                nn.Conv2d(in_channels, mid_channels, 1))
            self.outter_convs.append(
                nn.Conv2d(mid_channels, out_channels, 3, padding=1))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (List[Tensor]): feature maps from high to low resolution.

        Returns:
            results (List[Tensor]): feature maps after FPN layers.
        """
        assert len(x) == len(self.inner_convs)

        last_inner = self.inner_convs[-1](x[-1])
        results = []
        results.append(self.outter_convs[-1](last_inner))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_convs[idx](x[idx])
            inner_top_down = F.interpolate(
                last_inner, size=inner_lateral.shape[-2:], mode="nearest")
            last_inner = inner_lateral + inner_top_down
            result = self.outter_convs[idx](last_inner)
            results.append(result)

        return results[::-1]
