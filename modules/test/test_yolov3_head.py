import torch.nn
import numpy as np

from detutils.box import BoxArray, LabelBoxArray, DetectBoxArray
from modules.head import YoloV3


def test_yolov3_head():
    anchors = BoxArray.from_array(
        img_w=224, img_h=224,
        array=np.array([[0, 16, 0, 16], [0, 8, 0, 16], [0, 16, 0, 8]])
    )

    class_names = ['class A', 'class B']
    yolo_head = YoloV3(
        class_names=class_names,
        anchors=anchors,
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )

    labels = LabelBoxArray(
        class_names=['class A'],
        img_w=224, img_h=224,
        xmin=[16 - 9], xmax=[16 + 9], ymin=[16 - 5], ymax=[16 + 5])

    grid_w, grid_h = 7, 7
    pred = torch.rand(grid_w, grid_h, len(anchors), 5 + len(class_names))
    pred = 2 * (pred - 0.5)  # to (-1, +1)

    # set pred[0][0][2] to be true
    # objectness
    pred[0][0][2][0] = 100
    # x-y center
    pred[0][0][2][1] = 0
    pred[0][0][2][2] = 0
    # width and height
    pred[0][0][2][3] = 0 + 0.0001
    pred[0][0][2][4] = 0 - 0.0001
    # class conf
    pred[0][0][2][5] = 100
    pred[0][0][2][6] = -100

    # let's rock
    obj_loss, nonobj_loss, coord_loss, class_loss = yolo_head(pred, labels)
    assert isinstance(obj_loss, torch.Tensor)
    assert float(obj_loss) <= 1e-10
    assert isinstance(nonobj_loss, torch.Tensor)
    # this aasertation may be broken due to the random input tensor
    assert 35 <= float(nonobj_loss) <= 45
    assert isinstance(coord_loss, torch.Tensor)
    assert 0.00389 <= float(coord_loss) <= 0.00391
    assert isinstance(class_loss, torch.Tensor)
    assert float(class_loss) <= 1e-10

    detect = yolo_head(pred)
    assert isinstance(detect, list) and len(detect) == 2
    assert isinstance(detect[0], DetectBoxArray)
    best_shot_ind = np.argmax(detect[0].confs)
    best_score, best_box = detect[0][best_shot_ind]
    best_box.rescale_to(224, 224)
    assert best_score == 1.0
    assert [np.round(c) for c in [
        best_box.xmin, best_box.xmax, best_box.ymin, best_box.ymax]
        ] == [8, 24, 12, 20]
