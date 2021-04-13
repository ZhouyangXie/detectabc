'''
TODO
'''
import torch
import numpy as np

from detectutils.box import BoxArray

from .yolov2 import YoloV2


class YoloV3(YoloV2):
    """
    Compute the loss of YOLOv3
    """
    def __get_obj_error_mask(self, pred_boxarr, target_box):
        raise NotImplementedError