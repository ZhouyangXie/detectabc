'''
TODO
'''
from abc import ABC, abstractmethod

import torch
import numpy as np

from detutils.box import BoxArray, LabelBoxArray, DetectBoxArray

from modules.torch_utils import to_numpy


class Yolo(ABC):
    '''
    TODO
    '''
    def __init__(self,
                 class_names,
                 anchors: BoxArray,
                 pred_objectness_thre=0.5):
        """
        Parameters
        ----------
        class_names: List[str]
            a list of all class names,
            in same order as those in the prediction tensor
        anchors: BoxArray,
            a list of bounding box prior for each cell
            box location is ignored, only w/h matters
        """
        self.class_names = class_names
        self.class_name2ind = {c: i for i, c in enumerate(class_names)}
        self.num_class = len(class_names)

        # original location-agnostic windows
        self.original_anchors = anchors.copy()
        self.num_anchor = len(self.original_anchors)

        self.pred_objectness_thre = pred_objectness_thre

        # these attributes may change beween calls
        self.device = None
        self.grid_w, self.grid_h = -1, -1
        self.x_offset, self.y_offset = None, None
        # should be grid_w x grid_h x num_anchor
        self._box_shape = None
        # will be mapped to all cells
        self.anchors = None

    @staticmethod
    def _make_offset(grid_w: int, grid_h: int, num_box: int):
        x_offset, y_offset = np.mgrid[
            :float(grid_w), :float(grid_h)].astype(np.float32)

        x_offset = torch.repeat_interleave(
            torch.tensor(x_offset), num_box).reshape(
                grid_w, grid_h, num_box)
        y_offset = torch.repeat_interleave(
            torch.tensor(y_offset), num_box).reshape(
                grid_w, grid_h, num_box)

        return x_offset, y_offset

    @staticmethod
    def get_ious(pred: BoxArray, targets: BoxArray):
        """ return each box's IoU with (one of) the targets

        Parameters
        ----------
        pred : BoxArray
        label : BoxArray
        """
        ious = [pred.iou(target) for target in targets]
        return np.stack(ious)

    @staticmethod
    def _make_anchor_box(anchors: BoxArray, grid_w: int, grid_h: int):
        anchors = anchors.copy()
        anchors.rescale_to(grid_w, grid_h)

        # compute w/h of anchors and expand to _box_shape
        anchor_width = np.expand_dims(
            anchors.xmax - anchors.xmin, 0
            ).repeat(
                grid_w * grid_h, 0
            ).reshape(
                grid_w, grid_h, -1
            )
        anchor_height = np.expand_dims(
            anchors.ymax - anchors.ymin, 0
            ).repeat(
                grid_w * grid_h, 0
            ).reshape(
                grid_w, grid_h, -1
            )

        # expand cell center to _box_shape
        cell_center_x, cell_center_y = (
            np.mgrid[:grid_w, :grid_h] + 0.5)
        cell_center_x = np.expand_dims(cell_center_x, -1).repeat(
                len(anchors), -1)
        cell_center_y = np.expand_dims(cell_center_y, -1).repeat(
                len(anchors), -1)

        anchor_xmin = cell_center_x - anchor_width/2
        anchor_xmax = cell_center_x + anchor_width/2
        anchor_ymin = cell_center_y - anchor_height/2
        anchor_ymax = cell_center_y + anchor_height/2

        return BoxArray(
            grid_w, grid_h,
            anchor_xmin.flatten(),
            anchor_xmax.flatten(),
            anchor_ymin.flatten(),
            anchor_ymax.flatten())

    def _make_detection(self, xmin, xmax, ymin, ymax,
                        class_conf, objective_conf):
        '''
            Yolo's way of giving detection:
                * Choose a class for the each anchor/box
                * Filter out low objectiveness boxes.
            All inputs are numpy.array
        '''
        assert class_conf.shape == (
            self.grid_w, self.grid_h, self.num_anchor, self.num_class)
        assert objective_conf.shape == self._box_shape

        anchor_class_ind = class_conf.argmax(axis=3)
        anchor_class_conf = class_conf.max(axis=3)
        pred_box_conf = anchor_class_conf * objective_conf

        # mask against low-conf boxes
        objective_conf_mask = (objective_conf >= self.pred_objectness_thre)

        detect_boxes = []
        for i, class_name in enumerate(self.class_names):
            # mask against other classes
            anchor_class_mask = (anchor_class_ind == i)
            pred_box_mask = anchor_class_mask & objective_conf_mask

            detect_boxes.append(DetectBoxArray(
                class_name,
                pred_box_conf[pred_box_mask, ].reshape(-1),
                self.grid_w, self.grid_h,
                xmin[pred_box_mask, ].reshape(-1),
                xmax[pred_box_mask, ].reshape(-1),
                ymin[pred_box_mask, ].reshape(-1),
                ymax[pred_box_mask, ].reshape(-1)
            ))

        return detect_boxes

    def _tensor_to_detection(self, prediction):
        '''
        Unpack a W x H x (N * (5 + C)) tensor to
            class_conf: W x H x N x C
            objective_conf, xmin, xmax, ymin, ymax: W x H x N
        '''
        assert prediction.shape == (*self._box_shape, (5 + self.num_class))

        # split the prediction tensor
        objective_conf_raw = prediction[:, :, :, 0]
        x_center_raw = prediction[:, :, :, 1]
        y_center_raw = prediction[:, :, :, 2]
        width_raw = prediction[:, :, :, 3]
        height_raw = prediction[:, :, :, 4]
        class_conf_raw = prediction[:, :, :, 5:]

        objective_conf, xmin, xmax, ymin, ymax, class_conf = \
            self._interpret_tensor(
                objective_conf_raw, x_center_raw,
                y_center_raw, width_raw, height_raw, class_conf_raw)

        return objective_conf, xmin, xmax, ymin, ymax, class_conf

    def __call__(self, prediction: torch.Tensor, labels: LabelBoxArray = None):
        """return loss or detection from model output and labels

        Parameters
        ----------
        prediction : torch.Tensor
            the output of the network, one of the instances in a batch
        labels : LabelBoxArray, optional
            label bounding boxes, one of the instances in a batch,
            by default None

        Returns
        -------
        loss: dict[class name]: tuple(box_loss, objective_loss, class_loss)
            if labels are given
        detection: dict[class name]: LabelBoxArray
            if labels are not given
        """
        # make grid according to the input
        grid_w, grid_h, pred_num_anchor, pred_vector_len = prediction.shape
        assert pred_num_anchor == self.num_anchor
        assert pred_vector_len == (5 + self.num_class)

        # adjust grid and cell offset to the input
        if (grid_w, grid_h) != (self.grid_w, self.grid_h):
            self.grid_w, self.grid_h = grid_w, grid_h
            self._box_shape = (self.grid_w, self.grid_h, self.num_anchor)
            self.x_offset, self.y_offset = Yolo._make_offset(
                self.grid_w, self.grid_h, self.num_anchor)
            self.anchors = Yolo._make_anchor_box(
                self.original_anchors, grid_w, grid_h)

        # set device accordingly
        self.device = prediction.device
        if self.x_offset.device != self.device:
            self.x_offset = self.x_offset.to(self.device)
            self.y_offset = self.y_offset.to(self.device)

        # unpack the tensor and rescale to proper range
        objective_conf, xmin, xmax, ymin, ymax, class_conf = \
            self._tensor_to_detection(prediction)
        assert class_conf.shape == (*self._box_shape, self.num_class)
        assert xmin.shape == xmax.shape == ymin.shape \
            == ymax.shape == objective_conf.shape
        assert xmin.shape == self._box_shape

        # return DetectBoxArray if test only
        if labels is None:
            return self._make_detection(
                to_numpy(xmin), to_numpy(xmax),
                to_numpy(ymin), to_numpy(ymax),
                to_numpy(class_conf), to_numpy(objective_conf)
            )

        # convert coordinates to BoxArray
        pred_boxarr = BoxArray(self.grid_w, self.grid_h,
                               to_numpy(xmin).reshape(-1),
                               to_numpy(xmax).reshape(-1),
                               to_numpy(ymin).reshape(-1),
                               to_numpy(ymax).reshape(-1))
        assert len(pred_boxarr) == self.grid_w * self.grid_h * self.num_anchor

        labels.rescale_to(self.grid_w, self.grid_h)

        # IoU of anchor/predictions with the ground-truth
        preds_iou = Yolo.get_ious(pred_boxarr, labels).reshape(
            (len(labels), *self._box_shape))
        anchors_iou = Yolo.get_ious(self.anchors, labels).reshape(
            (len(labels), *self._box_shape))

        # compute objective loss
        obj_loss, nonobj_loss = self._get_objective_loss(
            objective_conf, anchors_iou, preds_iou)

        # compute coordinate loss
        coord_loss = self._get_coord_loss(
            pred_coords=(xmin, xmax, ymin, ymax),
            label_coords=(labels.xmin, labels.xmax, labels.ymin, labels.ymax),
            anchors_iou=anchors_iou,
            preds_iou=preds_iou
        )

        # compute classification loss
        label_class_inds = [
            self.class_name2ind[cn] for cn in labels.class_names]
        class_loss = self._get_class_loss(
            class_conf, label_class_inds, anchors_iou, preds_iou,
        )

        # to avoid misuse
        self.device = None

        return obj_loss, nonobj_loss, coord_loss, class_loss

    @abstractmethod
    def _interpret_tensor(
            self, objective_conf_raw, x_center_raw, y_center_raw,
            width_raw, height_raw, class_conf_raw):
        '''
            TODO
        '''
        pass

    @abstractmethod
    def _get_coord_loss(
            self, pred_coords, label_coords,
            anchors_iou, preds_iou):
        pass

    @abstractmethod
    def _get_objective_loss(self, objective_conf, anchors_iou, preds_iou):
        pass

    @abstractmethod
    def _get_class_loss(
            self, class_conf, label_class_inds,
            anchors_iou, preds_iou):
        pass
