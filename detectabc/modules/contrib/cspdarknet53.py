'''
    CSPDarkNet
    adapted from:
        https://github.com/bubbliiiing/yolov4-pytorch/blob/master/nets/CSPdarknet.py
    produces 3 tensors and use Mish as activation func
    fits input size of (416, 416)
'''
import torch
import torch.nn as nn

from detectabc.modules.functional import Mish


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        self.downsample_conv = BasicConv(
            in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(
                    channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )

            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()

        # fixed params for CSPDarkNet53
        input_channel = 3
        inplanes = 32
        channels = [64, 128, 256, 512, 1024]
        layers = [1, 2, 8, 8, 4]

        self.conv1 = BasicConv(
            input_channel, inplanes, kernel_size=3, stride=1)
        self.stages = nn.ModuleList([
            Resblock_body(
                inplanes, channels[0], layers[0], first=True),
            Resblock_body(
                channels[0], channels[1], layers[1], first=False),
            Resblock_body(
                channels[1], channels[2], layers[2], first=False),
            Resblock_body(
                channels[2], channels[3], layers[3], first=False),
            Resblock_body(
                channels[3], channels[4], layers[4], first=False)
        ])

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out_low = self.stages[2](x)
        out_mid = self.stages[3](out_low)
        out_high = self.stages[4](out_mid)
        return out_low, out_mid, out_high

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
