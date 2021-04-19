'''
    Path Aggregation Network
    https://arxiv.org/abs/1803.01534
'''
import torch.nn as nn


class PathAggregationNetwork(nn.Module):
    def __init__(
        self,
        in_channels_list,
        out_channels_list,
    ):
        """
        Args:
            in_channels_list (list[int]): number of channels for each input
            out_channels_list (list[int]): number of channels for each output
            Both ranked from high to low resolution
        """
        super().__init__()
        assert len(in_channels_list) == len(out_channels_list)
        assert in_channels_list[0] == out_channels_list[0]

        self.down_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for i in range(1, len(in_channels_list)):
            self.down_convs.append(
                PathAggregationNetwork._make_downsample_conv(
                    out_channels_list[i-1], in_channels_list[i]
                ))
            self.smooth_convs.append(
                PathAggregationNetwork._make_smooth_conv(
                    in_channels_list[i], out_channels_list[i]
                ))

    @staticmethod
    def _make_downsample_conv(in_channels, out_channels):
        '''
            this layer will downsample a 2N x 2N tensor to N x N
            or a (2N + 1) x (2N + 1) tensor to (N + 1) x (N + 1)
        '''
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1)

    @staticmethod
    def _make_smooth_conv(in_channels, out_channels):
        '''
            this layer will make tensor shape unchanged
        '''
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x_list):
        """
        Args:
            x (List[Tensor]): features whose resolution is high to low

        Returns:
            results (List[Tensor]): result features, resolution high to low
        """
        assert len(x_list) - 1 == len(self.down_convs)

        results = [x_list[0]]
        for idx in range(len(x_list) - 1):
            last_result = results[-1]
            downsampled = self.down_convs[idx](last_result)
            feature_sum = x_list[idx + 1] + downsampled
            smoothed = self.smooth_convs[idx](feature_sum)
            results.append(smoothed)

        return results
