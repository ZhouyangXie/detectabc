import numpy as np

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToTensor, Resize

from detectabc.detutils.box import BoxArray, LabelBoxArray
from detectabc.detutils.metrics import HitMetrics
from detectabc.modules.contrib import YoloV4
from detectabc.modules.head import YoloV3


_class_names = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]


def target_transform(label):
    label_boxarr = LabelBoxArray.from_voc(label)
    # switch X-Y axes because ToTensor does so
    label_boxarr.rotate()
    return label_boxarr


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


def run_model():
    dataset = VOCDetection(
        root="/data/sdv1/xiezhouyang/data/voc2012trainval",
        year="2012",  # or "2007"
        image_set="trainval",
        # "train", "val", "trainval" for both 2007/2012, "test" for 2007
        transform=Compose([Resize((416, 416)), ToTensor()]),
        # size required by 3rd-party CSPDarkNet53 implementation
        target_transform=target_transform
        # transform labels to LabelBoxArray
    )
    batch_size = 16
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn)

    # anchors from official implementation:
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-voc.cfg
    anchors_arr = [
        BoxArray.from_array(416, 416, np.array([
            [0, 10, 0, 13], [0, 16, 0, 30], [0, 33, 0, 23]
        ])),
        BoxArray.from_array(416, 416, np.array([
            [0, 30, 0, 61], [0, 62, 0, 45], [0, 59, 0, 119]
        ])),
        BoxArray.from_array(416, 416, np.array([
            [0, 116, 0, 90], [0, 156, 0, 198], [0, 373, 0, 326]
        ]))
    ]

    device = torch.device('cuda:5')
    model = YoloV4(
        num_classes=len(_class_names),
        nums_anchors=[len(a) for a in anchors_arr]
    )
    model = model.to(device)

    # anchor-specific YoloV3 loss for each head
    # it returns multi-part loss when labels are given
    # it returns predictions in DetectBoxArray when labels are not given
    yolov3loss_low = YoloV3(
        class_names=_class_names,
        anchors=anchors_arr[0],
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )
    yolov3loss_mid = YoloV3(
        class_names=_class_names,
        anchors=anchors_arr[1],
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )
    yolov3loss_high = YoloV3(
        class_names=_class_names,
        anchors=anchors_arr[2],
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )

    # scaling factors for multi-part loss
    # (non)objective loss
    obj_loss_scale = 1.0
    nonobj_loss_scale = 0.5
    # coordinates loss
    coord_loss_scale = 1.0
    # classification loss
    class_loss_scale = 1.0

    optimizer = SGD(model.parameters(), 1e-4)

    # train model for one step
    model.train()
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        outputs_low, outputs_mid, outputs_high = model(images)

        # use CPU for loss computing because of advanced indexing
        # inside YoloV3
        outputs_low, outputs_mid, outputs_high = \
            outputs_low.to('cpu'),\
            outputs_mid.to('cpu'),\
            outputs_high.to('cpu')
        # compute loss of each head
        loss = 0
        # iterate over heads
        for output_batch, yolov3loss in zip(
                [outputs_low, outputs_mid, outputs_high],
                [yolov3loss_low, yolov3loss_mid, yolov3loss_high]
                ):
            # iterate over images in a mini-batch
            for output, label in zip(output_batch, labels):
                obj_loss, nonobj_loss, coord_loss, class_loss = yolov3loss(
                    output, label)
                loss += (
                    obj_loss_scale * obj_loss +
                    nonobj_loss_scale * nonobj_loss +
                    coord_loss_scale * coord_loss +
                    class_loss_scale * class_loss)

        loss = loss/batch_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'iter:{i} loss:{float(loss)}')


if __name__ == "__main__":
    run_model()