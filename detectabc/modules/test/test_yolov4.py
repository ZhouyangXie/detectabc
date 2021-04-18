'''
  A test to construct a YOLOv4 detector using
  CSPDarkNet53, Feature Pyramid Network, Path Aggregation Network.
'''
import numpy as np

import torch

from detectabc.detutils.box import BoxArray
from detectabc.modules.detector import YoloV4


def test_yolov4():
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
