'''
TODO
'''
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy
import numpy as np

from .yolo import Yolo
from detectabc.detutils.box import BoxArray


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

    def _get_objective_loss(self, objective_conf, labels):
        assert objective_conf.shape == \
            (self.grid_w, self.grid_h, self.num_anchor)
        # get responsible cell, cell anchors, anchor ious for each label
        labels_cell_x, labels_cell_y, labels_cell_anchors, labels_cell_ious = \
            self._get_cell_anchor(labels)

        # get the responsible objective confidence predictions
        objective_conf = objective_conf[labels_cell_x, labels_cell_y, :]
        assert objective_conf.shape == (len(labels), self.num_anchor)

        # determine the target confidence for each predictions
        target = torch.zeros(
            objective_conf.shape, device=self.device, dtype=torch.float32)
        # let the best anchor for each target to be one
        target[
            list(range(len(labels))),
            np.argmax(labels_cell_ious, axis=1)
        ] = 1.
        high_iou_mask = labels_cell_ious >= 0.5
        target[high_iou_mask, ] = 1.

        # compute the two loss funcs
        obj_loss = binary_cross_entropy(
            objective_conf, target,
            reduction='mean')

        return obj_loss

    def _get_coord_loss(self, xmin, xmax, ymin, ymax, labels):
        assert xmin.shape == (self.grid_w, self.grid_h, self.num_anchor)
        assert xmax.shape == (self.grid_w, self.grid_h, self.num_anchor)
        assert ymin.shape == (self.grid_w, self.grid_h, self.num_anchor)
        assert ymax.shape == (self.grid_w, self.grid_h, self.num_anchor)
        # get responsible cell, cell anchors, anchor ious for each label
        labels_cell_x, labels_cell_y, labels_cell_anchors, labels_cell_ious = \
            self._get_cell_anchor(labels)

        # compute the only responsible anchor in the cell
        responsible_anchor_inds = labels_cell_ious.argmax(axis=1)

        # get the responsible predictions
        p_xmin = xmin[labels_cell_x, labels_cell_y, responsible_anchor_inds]
        p_xmax = xmax[labels_cell_x, labels_cell_y, responsible_anchor_inds]
        p_ymin = ymin[labels_cell_x, labels_cell_y, responsible_anchor_inds]
        p_ymax = ymax[labels_cell_x, labels_cell_y, responsible_anchor_inds]
        assert p_xmin.shape == (len(labels),)

        cent_loss, size_loss = YoloV3._box_loss(
            p_xmin, p_xmax, p_ymin, p_ymax,
            *[
                torch.tensor(np_array, device=self.device, dtype=torch.float32)
                for np_array in
                [labels.xmin, labels.xmax, labels.ymin, labels.ymax]
            ],
            self.x_offset[labels_cell_x, labels_cell_y, 0],
            self.y_offset[labels_cell_x, labels_cell_y, 0]
        )

        return cent_loss, size_loss

    def _get_class_loss(self, class_conf, labels, label_class_inds):
        assert class_conf.shape == \
            (self.grid_w, self.grid_h, self.num_anchor, self.num_class)
        # get responsible cell, cell anchors, anchor ious for each label
        labels_cell_x, labels_cell_y, labels_cell_anchors, labels_cell_ious = \
            self._get_cell_anchor(labels)
        # compute the only responsible anchor in the cell
        responsible_anchor_inds = labels_cell_ious.argmax(axis=1)

        # filter out irresonsible class scores
        class_conf = class_conf[
            labels_cell_x, labels_cell_y, responsible_anchor_inds]
        assert class_conf.shape == (len(labels), self.num_class)

        target = torch.zeros(
            class_conf.shape, device=self.device, dtype=torch.float32)
        # TODO: label smoothing
        target[list(range(len(labels))), label_class_inds] = 1.

        class_loss = YoloV3._class_loss(class_conf, target)
        return class_loss

    @staticmethod
    def _inv_sigmoid(x):
        x = x.clamp(min=1e-4, max=1-1e-4)
        return torch.log(x/(1-x))

    @staticmethod
    def _box_loss(
        p_xmin, p_xmax, p_ymin, p_ymax,
        t_xmin, t_xmax, t_ymin, t_ymax,
        x_offset, y_offset
    ):
        loss_w = mse_loss(
            torch.log(p_xmax-p_xmin), torch.log(t_xmax-t_xmin),
            reduction='mean')
        loss_h = mse_loss(
            torch.log(p_ymax-p_ymin), torch.log(t_ymax-t_ymin),
            reduction='mean')

        pred_x = YoloV3._inv_sigmoid((p_xmax + p_xmin)/2 - x_offset)
        target_x = YoloV3._inv_sigmoid((t_xmax + t_xmin)/2 - x_offset)
        pred_y = YoloV3._inv_sigmoid((p_ymax + p_ymin)/2 - y_offset)
        target_y = YoloV3._inv_sigmoid((t_ymax + t_ymin)/2 - y_offset)

        valid_mask = ~(torch.isnan(target_x) | torch.isnan(target_y))
        if (~valid_mask).any():
            raise Warning('NaN in coord loss')

        loss_x = mse_loss(
            pred_x[valid_mask, ],
            target_x[valid_mask, ],
            reduction='mean'
        )
        loss_y = mse_loss(
            pred_y[valid_mask, ],
            target_y[valid_mask, ],
            reduction='mean'
        )

        if torch.isinf(loss_x).any() or torch.isinf(loss_y).any():
            raise ValueError

        return loss_x + loss_y, loss_w + loss_h

    def _get_cell_anchor(self, labels):
        labels_cell_x, labels_cell_y = YoloV3._get_responsible_cell_ind(labels)
        labels_cell_anchors = [
            YoloV3._make_cell_anchor_box(
                self.original_anchors, self.grid_w,
                self.grid_h, cell_x, cell_y)
            for cell_x, cell_y in zip(labels_cell_x, labels_cell_y)
        ]
        labels_cell_ious = np.stack([
            labels_cell_anchors[i].iou(labels[i]) for i in range(len(labels))
        ])
        return labels_cell_x, labels_cell_y, \
            labels_cell_anchors, labels_cell_ious

    @staticmethod
    def _class_loss(p_score, t_score):
        return binary_cross_entropy(p_score, t_score, reduction='mean')

    @staticmethod
    def _get_responsible_cell_ind(labels):
        label_center_x, label_center_y, _, _ = labels.to_xywh()
        return label_center_x.astype(int), label_center_y.astype(int)

    @staticmethod
    def _make_cell_anchor_box(anchors, grid_w, grid_h, cell_x, cell_y):
        anchors = anchors.copy()
        anchors.rescale_to(grid_w, grid_h)

        anchor_xmin = cell_x + 0.5 - (anchors.xmax - anchors.xmin)/2
        anchor_xmax = cell_x + 0.5 + (anchors.xmax - anchors.xmin)/2
        anchor_ymin = cell_y + 0.5 - (anchors.ymax - anchors.ymin)/2
        anchor_ymax = cell_y + 0.5 + (anchors.ymax - anchors.ymin)/2

        return BoxArray(
            grid_w, grid_h,
            anchor_xmin.flatten(),
            anchor_xmax.flatten(),
            anchor_ymin.flatten(),
            anchor_ymax.flatten())
