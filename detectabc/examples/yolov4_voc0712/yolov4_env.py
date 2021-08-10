'''
    utils for training YoloV4 for PASCAL VOC 07+12
'''
import logging
from functools import reduce

import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToTensor, Resize

from detectabc.detutils.box import BoxArray, LabelBoxArray
from detectabc.detutils.metrics import HitMetrics
from detectabc.modules.contrib import YoloV4
from detectabc.modules.head import YoloV3


# path to VOC07/12 train/val/test set
_voc2007trainval_root = "/data/sdv1/xiezhouyang/data/voc2007trainval"
_voc2012trainval_root = "/data/sdv1/xiezhouyang/data/voc2012trainval"
_voc2007test_root = "/data/sdv1/xiezhouyang/data/voc2007test"
_class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
# above this value, a prediction is considered true-positive
_voc_detection_iou_thre = 0.5

# training parameters
# set up training params
batch_size = 64
num_epoch = 150
device = torch.device('cuda:5')
# interval for validation and checkpointing
interval = 10

# scaling factors for multi-part loss
# (non)objective loss
_obj_loss_scale = 1.0
# coordinates loss
_coord_loss_scale = 1.0
# classification loss
_class_loss_scale = 1.0

# initial learning rate
_init_lr = 1e-4
# L2 loss scale
_weight_decay = 5e-4
# minimum of learning rate decay
_min_lr = 1e-6

# set up train logging
_train_log_filename = 'train.log'


def _target_transform(label):
    label_boxarr = LabelBoxArray.from_voc(label)
    # switch X-Y axes because ToTensor does so
    label_boxarr.rotate()
    return label_boxarr


def get_dataset(phase):
    """
    Get train/val/test dataset of the task.
    Global variable _train_dataset, _val_dataset, _test_dataset
        will created for this module. Thus no need to call this
        function if using this module only.

    train: voc07trainval + voc12train
    val: voc12val
    test: voc07test

    Args:
        phase (str): 'train', 'val' or 'test'

    Returns:
        dataset (Dataset)
    """
    if phase == 'train':
        return ConcatDataset(
            [
                VOCDetection(
                    root=_voc2007trainval_root,
                    year="2007",
                    image_set="trainval",
                    transform=Compose([Resize((416, 416)), ToTensor()]),
                    target_transform=_target_transform
                ),
                VOCDetection(
                    root=_voc2012trainval_root,
                    year="2012",
                    image_set="train",
                    transform=Compose([Resize((416, 416)), ToTensor()]),
                    target_transform=_target_transform
                ),
            ]
        )
    elif phase == 'validation' or phase == 'val':
        return VOCDetection(
            root=_voc2012trainval_root,
            year="2012",
            image_set="val",
            transform=Compose([Resize((416, 416)), ToTensor()]),
            target_transform=_target_transform
        )
    elif phase == 'test':
        return VOCDetection(
            root=_voc2007test_root,
            year="2007",
            image_set="test",
            transform=Compose([Resize((416, 416)), ToTensor()]),
            target_transform=_target_transform
        )

    else:
        raise ValueError(f'phase {phase} is invalid')


_train_dataset = get_dataset('train')
_val_dataset = get_dataset('val')
_test_dataset = get_dataset('test')


def _collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


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
for anchors in anchors_arr:
    # same reason as target_transform
    anchors.rotate()


def make_model():
    return YoloV4(
        num_classes=len(_class_names),
        nums_anchors=[len(anchors) for anchors in anchors_arr]
    )


def make_yolo_loss():
    """
        Returns the loss functions for each of the 3 heads in YoloV4.
        They also act as the predictor during testing.
    """
    loss_low = YoloV3(
        class_names=_class_names,
        anchors=anchors_arr[0],
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )
    loss_mid = YoloV3(
        class_names=_class_names,
        anchors=anchors_arr[1],
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )
    loss_high = YoloV3(
        class_names=_class_names,
        anchors=anchors_arr[2],
        pred_objectness_thre=0.5,
        loss_objectness_thre=0.5
    )
    return loss_low, loss_mid, loss_high


def train(
        model,
        yolov3loss_low=None,
        yolov3loss_mid=None,
        yolov3loss_high=None,
        ):
    assert yolov3loss_low or yolov3loss_low or yolov3loss_high

    logging.basicConfig(
        filename=_train_log_filename,
        filemode='w',
        format='%(asctime)s | %(message)s',
        datefmt='%d/%m %H:%M:%S',
        level=logging.DEBUG
    )

    num_batch = len(_train_dataset)//batch_size

    train_dataloader = DataLoader(
        _train_dataset, batch_size=batch_size,
        collate_fn=_collate_fn, shuffle=True)

    # disable backbone weight gradient
    for param in model.backbone.parameters():
        param.requires_grad = False
    model = model.to(device)

    optimizer = Adam(
        model.neck.parameters(), lr=_init_lr, weight_decay=_weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, num_epoch, eta_min=_min_lr)

    for i in range(num_epoch):
        model.backbone.eval()
        model.neck.train()

        # train for one epoch
        for j, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            try:
                outputs_low, outputs_mid, outputs_high = model(images)
            except RuntimeError as err:
                logging.error(f'{err}')
                logging.info('skip this mini-batch')
                torch.cuda.empty_cache()

            # use CPU for loss computing because of advanced indexing
            # inside YoloV3
            outputs_low, outputs_mid, outputs_high = \
                outputs_low.to('cpu'),\
                outputs_mid.to('cpu'),\
                outputs_high.to('cpu')
            # compute loss of each head
            obj_loss, cent_loss, size_loss, class_loss = 0, 0, 0, 0
            # iterate over heads
            for output_batch, yolov3loss, head_name in zip(
                    [outputs_low, outputs_mid, outputs_high],
                    [yolov3loss_low, yolov3loss_mid, yolov3loss_high],
                    ['low', 'mid', 'high']
                    ):
                if yolov3loss is None:
                    continue

                b_obj_loss, b_cent_loss, b_size_loss, b_class_loss = 0, 0, 0, 0
                # iterate over images in a mini-batch
                for output, label in zip(output_batch, labels):
                    ih_obj_loss, ih_cent_loss, ih_size_loss, ih_class_loss = \
                        yolov3loss(output, label)
                    b_obj_loss += ih_obj_loss
                    b_cent_loss += ih_cent_loss
                    b_size_loss += ih_size_loss
                    b_class_loss += ih_class_loss

                obj_loss += b_obj_loss
                cent_loss += b_cent_loss
                size_loss += b_size_loss
                class_loss += b_class_loss

            loss = (
                _obj_loss_scale * obj_loss +
                _coord_loss_scale * cent_loss +
                _coord_loss_scale * size_loss +
                _class_loss_scale * class_loss)/batch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_iter_str = f'epoch:{i} iter:{j}/{num_batch} '
            logging.debug(epoch_iter_str + f'obj loss:{float(obj_loss)}')
            logging.debug(epoch_iter_str + f'box cent loss:{float(cent_loss)}')
            logging.debug(epoch_iter_str + f'box size loss:{float(size_loss)}')
            logging.debug(epoch_iter_str + f'class loss:{float(class_loss)}')
            logging.debug(epoch_iter_str + f'total loss:{float(loss)}')

        scheduler.step()

        # validation and checkpointing
        if i % interval == 0:
            mAP = test(
                'val', model,
                yolov3loss_low, yolov3loss_mid, yolov3loss_high)
            total_mAP = mAP['total']
            logging.debug(f'epoch:{i} val total AP: {total_mAP}')
            filename = f'weights/yolov4_epoch_{i}.pth'
            torch.save(model.state_dict(), filename)
            logging.debug(f'epoch:{i} save checkpoint {filename}')


def test(
        phase,
        model,
        yolov3loss_low=None,
        yolov3loss_mid=None,
        yolov3loss_high=None,
        nms_iou_thre=0.5
        ):
    assert yolov3loss_low or yolov3loss_low or yolov3loss_high

    if phase == 'val' or phase == 'validation':
        dataloader = DataLoader(
            _val_dataset, batch_size=batch_size, collate_fn=_collate_fn)
    elif phase == 'test':
        dataloader = DataLoader(
            _test_dataset, batch_size=batch_size, collate_fn=_collate_fn)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    APs = {}
    for class_name in _class_names:
        APs[class_name] = HitMetrics()

    for images, labels in dataloader:
        images = images.to(device)
        try:
            outputs_low, outputs_mid, outputs_high = model(images)
        except RuntimeError as err:
            logging.error(f'{err}')
            logging.info('breaking validation loop')
            torch.cuda.empty_cache()
            break

        outputs_low, outputs_mid, outputs_high = \
            outputs_low.to('cpu'),\
            outputs_mid.to('cpu'),\
            outputs_high.to('cpu')

        for output_low, output_mid, output_high, label in zip(
                outputs_low, outputs_mid, outputs_high, labels):

            predictions = []
            for output_batch, yolov3loss in zip(
                    [output_low, output_mid, output_high],
                    [yolov3loss_low, yolov3loss_mid, yolov3loss_high],
                    ):
                if yolov3loss is None:
                    continue
                predictions.append(yolov3loss(output_batch))

            # merge predictions across heads
            for class_name in _class_names:
                cls_predictions = reduce(
                    lambda a, b: a + b, [p[class_name] for p in predictions])

                targets = label[class_name]
                if len(targets) == 0:
                    continue

                shots = cls_predictions
                if len(shots) == 0:
                    APs[class_name].append(0.)
                    continue

                shots = shots.nms(nms_iou_thre)
                hits = []
                for target in targets:
                    ious = shots.iou(target)
                    hits.append(ious >= _voc_detection_iou_thre)

                hits = np.array(hits).transpose()
                APs[class_name].add_instance(hits, shots.confs)
                APs[class_name].append(HitMetrics(
                    np.array(hits).transpose()
                ).average_precision())

    mAP = {}
    for k, v in APs.items():
        mAP[k] = v.average_precision()

    mAP['total'] = np.mean([v for _, v in mAP.items()])
    return mAP
