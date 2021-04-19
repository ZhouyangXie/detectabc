'''
  Yolov4 composed of
  CSPDarkNet53, Feature Pyramid Network, Path Aggregation Network.
'''
import torch.nn as nn

from detectabc.modules.contrib import CSPDarkNet53
from detectabc.modules.neck import FeaturePyramidNetwork, \
    PathAggregationNetwork


class YoloV4(nn.Module):
    def __init__(self, num_classes, nums_anchors):
        super().__init__()

        self.backbone = CSPDarkNet53()
        self.fpn = FeaturePyramidNetwork(
            [256, 512, 1024],
            [256, 512, 1024])
        self.pan = PathAggregationNetwork(
            [256, 512, 1024],
            [256, 512, 1024])

        assert len(nums_anchors) == 3
        self.last_conv_low = YoloV4._make_last_conv(
            256, num_classes, nums_anchors[0])
        self.last_conv_mid = YoloV4._make_last_conv(
            512, num_classes, nums_anchors[1])
        self.last_conv_high = YoloV4._make_last_conv(
            1024, num_classes, nums_anchors[2])

        self._output_shapes = [
            (52, 52, nums_anchors[0], 5 + num_classes),
            (26, 26, nums_anchors[1], 5 + num_classes),
            (13, 13, nums_anchors[2], 5 + num_classes),
        ]

    @staticmethod
    def _make_last_conv(in_channels, num_classes, num_anchors):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=512,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=num_anchors*(5 + num_classes),
                kernel_size=3,
                padding=1,
                bias=False)
        )

    def forward(self, x):
        x_low, x_mid, x_high = self.backbone(x)
        fpn_low, fpn_mid, fpn_high = self.fpn([x_low, x_mid, x_high])
        pan_low, pan_mid, pan_high = self.pan([fpn_low, fpn_mid, fpn_high])

        out_low = self.last_conv_low(pan_low)
        out_mid = self.last_conv_mid(pan_mid)
        out_high = self.last_conv_high(pan_high)

        return out_low.reshape((-1, *self._output_shapes[0])),\
            out_mid.reshape((-1, *self._output_shapes[1])),\
            out_high.reshape((-1, *self._output_shapes[2]))
