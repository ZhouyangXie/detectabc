'''
  A test to construct a YOLOv4 detector using
  CSPDarkNet53, Feature Pyramid Network, Path Aggregation Network.
'''
import numpy as np

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork

from detectabc.detutils.box import BoxArray

from detectabc.modules.contrib import CSPDarkNet53
from detectabc.modules.neck import PathAggregationNetwork


class YoloV4(nn.Module):
    _neck_channels = 512

    def __init__(self, num_classes, nums_anchors):
        super().__init__()

        self.backbone = CSPDarkNet53()
        self.fpn = FeaturePyramidNetwork(
            [256, 512, 1024],
            YoloV4._neck_channels)
        self.pan = PathAggregationNetwork(
            [YoloV4._neck_channels]*3, [YoloV4._neck_channels]*3)

        assert len(nums_anchors) == 3
        self.last_conv_low = YoloV4._make_last_conv(
            YoloV4._neck_channels, num_classes, nums_anchors[0])
        self.last_conv_mid = YoloV4._make_last_conv(
            YoloV4._neck_channels, num_classes, nums_anchors[1])
        self.last_conv_high = YoloV4._make_last_conv(
            YoloV4._neck_channels, num_classes, nums_anchors[2])

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

        # torchvision FPN requires a Dict input
        features = {'low': x_low, 'mid': x_mid, 'high': x_high}
        fpn_features = self.fpn(features)
        fpn_low, fpn_mid, fpn_high = \
            fpn_features['low'], fpn_features['mid'], fpn_features['high']

        # through PAN
        pan_low, pan_mid, pan_high = self.pan([fpn_low, fpn_mid, fpn_high])

        # last conv to fit Yolo loss
        out_low = self.last_conv_low(pan_low)
        out_mid = self.last_conv_mid(pan_mid)
        out_high = self.last_conv_high(pan_high)

        return out_low.reshape((-1, *self._output_shapes[0])),\
            out_mid.reshape((-1, *self._output_shapes[1])),\
            out_high.reshape((-1, *self._output_shapes[2]))


def test_detector():
    width, hight = 416, 416
    batch_size = 2
    x = torch.rand((batch_size, 3, width, hight))

    anchors = {
        'low': BoxArray.from_array(
            img_w=416, img_h=416,
            array=np.array([[0, 16, 0, 16], [0, 8, 0, 16], [0, 16, 0, 8]])
        ),
        'mid': BoxArray.from_array(
            img_w=416, img_h=416,
            array=np.array([[0, 16, 0, 16], [0, 8, 0, 16], [0, 16, 0, 8]])
        ),
        'high': BoxArray.from_array(
            img_w=416, img_h=416,
            array=np.array([[0, 16, 0, 16], [0, 8, 0, 16], [0, 16, 0, 8]])
        )
    }
    class_names = ['a', 'b']

    model = YoloV4(len(class_names), [len(v) for _, v in anchors.items()])

    out_low, out_mid, out_high = model(x)
    assert out_low.shape == (
        batch_size, 52, 52, len(anchors['low']), 5 + len(class_names))
    assert out_mid.shape == (
        batch_size, 26, 26, len(anchors['mid']), 5 + len(class_names))
    assert out_high.shape == (
        batch_size, 13, 13, len(anchors['high']), 5 + len(class_names))

test_detector()