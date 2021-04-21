'''
TODO
'''
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy
import numpy as np

from .yolo import Yolo


class YoloV3(Yolo):
    """
    Compute the loss of YOLOv3
    """
    def __init__(self, *args, loss_objectness_thre=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_objectness_thre = loss_objectness_thre

    def _interpret_tensor(
            self,
            objective_conf_raw,
            x_center_raw,
            y_center_raw,
            width_raw,
            height_raw,
            class_conf_raw):

        objective_conf = torch.sigmoid(objective_conf_raw)
        x_center = torch.sigmoid(x_center_raw)
        y_center = torch.sigmoid(y_center_raw)
        width_scale = torch.exp(width_raw)
        height_scale = torch.exp(height_raw)
        class_conf = torch.sigmoid(class_conf_raw)

        # get anchors ready as base
        anchor_width = torch.tensor(
            self.anchors.xmax - self.anchors.xmin,
            dtype=torch.float32).reshape(
                self._box_shape).to(self.device)

        anchor_height = torch.tensor(
            self.anchors.ymax - self.anchors.ymin,
            dtype=torch.float32).reshape(
                self._box_shape).to(self.device)

        width = anchor_width * width_scale
        height = anchor_height * height_scale

        xmin = x_center + self.x_offset - width/2
        xmax = x_center + self.x_offset + width/2
        ymin = y_center + self.y_offset - height/2
        ymax = y_center + self.y_offset + height/2

        return objective_conf, xmin, xmax, ymin, ymax, class_conf

    def _get_objective_loss(self, objective_conf, anchors_iou, preds_iou):
        num_target, _, _, _ = anchors_iou.shape

        # flatten the last 3 dims for easier indexing
        anchors_iou = anchors_iou.reshape((num_target, -1))
        objective_conf = objective_conf.reshape((-1))

        # only choose the best for each target as objectness=1
        positive_inds = np.argmax(anchors_iou, axis=1)
        positives = objective_conf[positive_inds, ]
        # choose those has low iou with any label as objectness=0
        negative_inds = anchors_iou.min(axis=0) < self.loss_objectness_thre
        negative_inds[positive_inds] = False
        negatives = objective_conf[negative_inds, ]

        # compute the two loss funcs
        obj_loss = binary_cross_entropy(
            positives, torch.ones(
                positives.shape, device=self.device, dtype=torch.float32),
            reduction='mean')

        noobj_loss = binary_cross_entropy(
            negatives, torch.zeros(
                negatives.shape, device=self.device, dtype=torch.float32),
            reduction='mean')

        return obj_loss, noobj_loss

    def _get_coord_loss(
            self, pred_coords, label_coords,
            anchors_iou, preds_iou):
        num_target, _, _, _ = anchors_iou.shape
        # only one anchor is responsible for each label
        anchors_iou = anchors_iou.reshape((num_target, -1))
        flatten_responsible_inds = np.argmax(anchors_iou, axis=1)

        coord_loss = YoloV3._box_loss(
            *[
                pcoord.reshape((-1,))[flatten_responsible_inds, ]
                for pcoord in pred_coords
            ],
            *[
                torch.tensor(tcoord, device=self.device, dtype=torch.float32)
                for tcoord in label_coords
            ]
        )

        return coord_loss

    def _get_class_loss(
            self, class_conf, label_class_inds,
            anchors_iou, preds_iou):
        label_class_inds = np.array(label_class_inds)

        # TODO: save flatten_responsible_inds
        num_target, _, _, _ = anchors_iou.shape
        # only one anchor is responsible for each label
        anchors_iou = anchors_iou.reshape((num_target, -1))
        flatten_responsible_inds = np.argmax(anchors_iou, axis=1)

        class_conf = class_conf.reshape((-1, self.num_class))
        # class_conf = class_conf[flatten_responsible_inds, :]

        class_loss = 0
        for anchor_ind in np.unique(flatten_responsible_inds):
            matched_label_inds = anchor_ind == flatten_responsible_inds
            target_labels = label_class_inds[matched_label_inds]

            t_class_score = torch.zeros(
                self.num_class, device=self.device, dtype=torch.float32)
            t_class_score[target_labels] = 1
            p_class_score = class_conf[anchor_ind]
            class_loss += YoloV3._class_loss(p_class_score, t_class_score)

        return class_loss

    @staticmethod
    def _box_loss(
        p_xmin, p_xmax, p_ymin, p_ymax,
        t_xmin, t_xmax, t_ymin, t_ymax
    ):
        t_area = (t_xmax - t_xmin) * (t_ymax - t_ymin)
        p_area = (p_xmax - p_xmin) * (p_ymax - p_ymin)
        inter_xmin = torch.max(t_xmin, p_xmin)
        inter_xmax = torch.min(t_xmax, p_xmax)
        inter_ymin = torch.max(t_ymin, p_ymin)
        inter_ymax = torch.min(t_ymax, p_ymax)
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h
        ious = inter_area / (t_area + p_area - inter_area)
        ious = torch.clamp(ious, min=0, max=1)
        return (1 - ious).sum()


    @staticmethod
    def _class_loss(p_score, t_score):
        return binary_cross_entropy(p_score, t_score, reduction='mean')
