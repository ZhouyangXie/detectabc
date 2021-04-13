'''
TODO
'''
from abc import ABC, abstractmethod

import torch
import numpy as np

from detectutils.box import _Box, BoxArray, LabelBoxArray

from components.torch_utils import to_numpy


class Yolo(ABC):
    '''
    TODO
    '''
    # size of a box prediction of a cell
    # conf, x, y, w, h
    box_size = 5

    def __init__(self,
                 class_names,
                 num_box: int,
                 coord_scale: float = 5.0,
                 obj_scale: float = 1.0,
                 noobj_scale: float = 0.5,
                 class_scale: float = 1.0,
                 objective_conf_thre: float = 0.5):
        """
        Parameters
        ----------
        class_names: List[str]
            a list of all class names,
            in same order as those in the prediction tensor
        num_box: int
            number of box of each cell (e.g. number of anchor)
        coord_scale : float, optional
            scale of coordinate loss(box loss),
            by default 5.0
        obj_scale : float, optional
            scale of objectiveness loss(existance of any object in a cell),
            by default 1.0
        noobj_scale : float, optional
            scale of non-objectiveness loss,
            by default 0.5
        class_scale : float, optional
            scale of cell classification loss,
            by default 1.0
        objective_conf_thre : float, optional
            below it, a box prediction will not be produced while inference,
            by default 0.5
        """
        self.class_names = class_names
        self.class_name2ind = {c: i for i, c in enumerate(class_names)}
        self.num_class = len(class_names)

        assert num_box > 0
        self.num_box = num_box

        # the size of each cell is 1 x 1
        # the size of the resized image under the grid is grid_w x grid_h

        self.coord_scale = np.float32(coord_scale)
        self.obj_scale = np.float32(obj_scale)
        self.noobj_scale = np.float32(noobj_scale)
        self.class_scale = np.float32(class_scale)
        # to ignore boxes with low objective conf during inference
        self.objective_conf_thre = np.float32(objective_conf_thre)

        # change with processing inputs
        self.device = None

        self.grid_w, self.grid_h = -1, -1
        self.x_offset, self.y_offset = None, None
        self._box_shape = None

    @staticmethod
    def _make_offset(grid_w: int, grid_h: int, num_box: int):
        x_offset, y_offset = np.mgrid[
            :float(grid_w),
            :float(grid_h)].astype(np.float32)

        x_offset = torch.repeat_interleave(
            torch.tensor(x_offset), num_box).reshape(
                grid_w, grid_h, num_box)
        y_offset = torch.repeat_interleave(
            torch.tensor(y_offset), num_box).reshape(
                grid_w, grid_h, num_box)

        return x_offset, y_offset

    @staticmethod
    def get_iou_mask(pred: BoxArray, label, iou_thre=0.5):
        """ return whether each box is covered by (one of) the label

        Parameters
        ----------
        pred : BoxArray
        label : _Box or BoxArray
        iou_thre : float, optional
        """
        mask = np.zeros(len(pred), dtype=np.bool)

        if isinstance(label, _Box):
            label = [label]

        for target_box in label:
            mask |= (pred.iou(target_box) > iou_thre)

        return mask

    def __call__(self, prediction: torch.Tensor, labels: LabelBoxArray = None):
        """return loss or detection from model output and labels

        Parameters
        ----------
        prediction : torch.Tensor
            the output of the network, one in a batch
        labels : LabelBoxArray, optional
            label bounding boxes, on in a batch,
            by default None

        Returns
        -------
        loss: dict[class name]: tuple(box_loss, objective_loss, class_loss)
            if labels are given
        detection: dict[class name]: LabelBoxArray
            if labels are not given
        """

        # make grid dynamically
        grid_w, grid_h, _ = prediction.shape
        if (grid_w, grid_h) != (self.grid_w, self.grid_h):
            self.grid_w, self.grid_h = grid_w, grid_h
            self._box_shape = (self.grid_w, self.grid_h, self.num_box)
            self.x_offset, self.y_offset = Yolo._make_offset(
                self.grid_w, self.grid_h, self.num_box)

        # set device dynamically
        self.device = prediction.device
        if self.x_offset.device != self.device:
            self.x_offset = self.x_offset.to(self.device)
            self.y_offset = self.y_offset.to(self.device)

        # unpack the tensor and rescale to proper range
        class_conf, objective_conf, xmin, xmax, ymin, ymax = \
            self._tensor_to_detection(prediction)
        assert xmin.shape == xmax.shape == ymin.shape == ymax.shape
        assert xmin.shape == self._box_shape

        # make custom preparation like anchors
        self._call_prepare()

        # convert coordinates to BoxArray
        pred_boxarr = BoxArray(self.grid_w, self.grid_h,
                               to_numpy(xmin).reshape(-1),
                               to_numpy(xmax).reshape(-1),
                               to_numpy(ymin).reshape(-1),
                               to_numpy(ymax).reshape(-1))
        assert len(pred_boxarr) == self.grid_w * self.grid_h * self.num_box

        if labels is None:
            return self._make_detection(
                to_numpy(xmin), to_numpy(xmax),
                to_numpy(ymin), to_numpy(ymax),
                to_numpy(class_conf), to_numpy(objective_conf)
            )

        loss = {}
        for class_name, target_box in zip(labels.class_names, labels):
            # whether each cell/box's class conf being penalized
            class_error_mask = self._get_class_error_mask(
                pred_boxarr, target_box)
            assert class_error_mask.shape == class_conf.shape[:-1]
            # classification loss for responsible box/cell(s)
            class_loss = self.class_scale * self._class_loss(
                class_conf[class_error_mask, ],
                # note this comma in [] incurs advanced indexing
                class_name)

            # whether each box's coordinates being penalized
            coord_error_mask = self._get_coord_error_mask(
                pred_boxarr, target_box)
            assert coord_error_mask.shape == self._box_shape
            # coordinates loss for responsible boxes
            box_loss = self.coord_scale * self._box_loss(
                xmin[coord_error_mask, ],
                xmax[coord_error_mask, ],
                ymin[coord_error_mask, ],
                ymax[coord_error_mask, ],
                target_box)

            # whether each cell/box's objectiveness being penalized
            obj_error_mask = self._get_obj_error_mask(pred_boxarr, target_box)
            assert obj_error_mask.shape == objective_conf.shape
            # objective loss for responsible box/cell(s)
            obj_loss, noobj_loss = self._objective_loss(
                objective_conf, obj_error_mask)
            objective_loss = self.obj_scale * obj_loss + \
                self.noobj_scale * noobj_loss

            loss[class_name] = {
                'coordinate': box_loss,
                'objective': objective_loss,
                'class': class_loss}

        # to avoid misuse
        self.device = None
        return loss

    def _call_prepare(self):
        pass

    @abstractmethod
    def _tensor_to_detection(self, prediction):
        pass

    @abstractmethod
    def _box_loss(self, xmin, xmax, ymin, ymax, target_box: _Box):
        pass

    @abstractmethod
    def _objective_loss(self, objective_conf, objective_mask):
        pass

    @abstractmethod
    def _class_loss(self, class_conf, target_class_name: str):
        pass

    @abstractmethod
    def _get_class_error_mask(self, pred_boxarr, target_box):
        pass

    @abstractmethod
    def _get_obj_error_mask(self, pred_boxarr, target_box):
        pass

    @abstractmethod
    def _get_coord_error_mask(self, pred_boxarr, target_box):
        pass

    @abstractmethod
    def _make_detection(self, xmin, xmax, ymin, ymax,
                        class_conf, objective_conf):
        pass
