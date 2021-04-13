from head.yolo import Yolo
import torch.nn
from torchvision.models import resnet101

from components.backbone.resnet import ResNetBackbone
from components.head.yolov1 import YoloV1

def test_detector():
    grid_w, grid_h = 7, 7
    num_box = 2
    class_names =  ['a', 'b']

    backbone = ResNetBackbone(resnet101(pretrained=True))
    output = backbone(torch.rand(2, 3, 416, 416))
    output = output[-1] # batch_size X num_feature
    batch_size, num_feature = output.shape

    linear = torch.nn.Linear(num_feature,  (len(class_names) + num_box * Yolo.box_size) * grid_w * grid_h)
    output = linear(output)
    output = output.reshape(batch_size, grid_w, grid_h, len(class_names) + num_box * Yolo.box_size)

    assert output.shape == (batch_size, grid_w, grid_h, 12)

    yolo = YoloV1(
      class_names, num_box, grid_w, grid_h
    )
    for prediction in output:
        _ = yolo(prediction)
