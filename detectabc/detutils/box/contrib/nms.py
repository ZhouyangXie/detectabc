import numpy as np

import torch
from torchvision.ops import nms

# from detectabc.detutils.box import DetectBoxArray
from detectabc.modules.torch_utils import to_numpy, to_tensor

def torchvision_nms(boxarr, iou_thre):
    boxes = to_tensor(
        np.stack([
            boxarr.xmin, 
            boxarr.ymin, 
            boxarr.xmax, 
            boxarr.ymax, 
            ])
        )
    boxes = boxes.transpose(1, 0)
    scores = to_tensor(boxarr.confs)
    kept_inds = to_numpy(nms(boxes, scores, iou_thre))

    return kept_inds
