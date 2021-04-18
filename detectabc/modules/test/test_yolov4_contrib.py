'''
  A test to construct a YOLOv4 detector using
  CSPDarkNet53, SPP, PANet by third-party contribution.
'''
import torch
from detectabc.modules.contrib import YoloV4


def test_yolov4():
    width, hight = 416, 416
    batch_size = 2
    x = torch.rand((batch_size, 3, width, hight))

    nums_anchors = [3, 3, 3]
    class_names = ['a', 'b']

    model = YoloV4(len(class_names), nums_anchors)
    out_low, out_mid, out_high = model(x)

    assert out_low.shape == (
        batch_size, nums_anchors[0] * (5 + len(class_names)), 52, 52)
    assert out_mid.shape == (
        batch_size, nums_anchors[1] * (5 + len(class_names)), 26, 26)
    assert out_high.shape == (
        batch_size, nums_anchors[2] * (5 + len(class_names)), 13, 13)
