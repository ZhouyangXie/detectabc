'''
TODO
'''
import torch
from torch.nn.functional import mse_loss
import numpy as np

from detectutils.box import _Box, BoxArray, LabelBoxArray, DetectBoxArray

from components.head.yolo import Yolo


class YoloV1(Yolo):
    """
    Compute the loss of detection from model output by the loss of YOLO(v1)
    """
    def __init__(self,
                 *args,
                 objective_iou_thre: float = 0.0,
                 **kwargs):
        '''
        Parameters
        ----------

        objective_iou_thre : float, optional
            with iou between target above it, a cell is considered objective,
            by default 0.0
        '''
        super().__init__(*args, **kwargs)
        self.objective_iou_thre = objective_iou_thre
        self._cell_box = None

    def _call_prepare(self):
        if self._cell_box is None or \
            (self._cell_box.grid_w, self._cell_box.grid_h) != \
                (self.grid_w, self.grid_h):
            self._cell_box = YoloV1._make_cell_box(self.grid_w, self.grid_h)

    @staticmethod
    def _make_cell_box(grid_w: int, grid_h: int):
        # make each cell in the grid a box, to determine the objectiveness
        xmin, ymin = np.mgrid[0:grid_w, 0:grid_h]
        xmin, ymin = xmin.flatten(), ymin.flatten()
        xmax, ymax = xmin + 1, ymin + 1
        return BoxArray(grid_w, grid_h, xmin, xmax, ymin, ymax)

    def _tensor_to_detection(self, prediction):
        '''
        YOLOv1 specified way of conversion
            rescale all to [0, 1], i.e. relative distance to baseline
        '''
        assert YoloV1.box_size == 5

        # split the prediction tensor
        class_conf = torch.sigmoid(
            prediction[:, :, :self.num_class])
        objective_conf = torch.sigmoid(
            prediction[:, :, self.num_class::YoloV1.box_size])
        center_x = torch.sigmoid(
            prediction[:, :, self.num_class+1::YoloV1.box_size])
        center_y = torch.sigmoid(
            prediction[:, :, self.num_class+2::YoloV1.box_size])
        width = torch.sigmoid(
            prediction[:, :, self.num_class+3::YoloV1.box_size])
        height = torch.sigmoid(
            prediction[:, :, self.num_class+4::YoloV1.box_size])

        # convert to xxyy for BoxArray utilities
        xmin = self.x_offset + center_x \
            - width * np.float32(self.grid_w/2)
        xmax = self.x_offset + center_x \
            + width * np.float32(self.grid_w/2)
        ymin = self.y_offset + center_y \
            - height * np.float32(self.grid_h/2)
        ymax = self.y_offset + center_y \
            + height * np.float32(self.grid_h/2)

        return class_conf, objective_conf, xmin, xmax, ymin, ymax

    def _box_loss(self, xmin, xmax, ymin, ymax, target_box: _Box):
        '''
        Yolov1 loss for coordinates, MSE on x y and MSE on sqrt of w h
        '''
        assert xmin.shape == xmax.shape == ymin.shape == ymax.shape
        if len(xmin) == 0:
            return np.float32(0)

        target_box.rescale_to(self.grid_w, self.grid_h)

        # convert prediction to xywh
        p_x_center, p_y_center = (xmax + xmin)/2, (ymax + ymin)/2
        p_width, p_height = xmax - xmin, ymax - ymin

        # convert target to xywh and expand to fit prediction
        t_x_center, t_y_center, t_width, t_height = target_box.to_xywh()
        t_x_center = torch.tensor(
            t_x_center, dtype=torch.float32).repeat(
                p_x_center.size()).to(self.device)

        t_y_center = torch.tensor(
            t_y_center, dtype=torch.float32).repeat(
                p_y_center.size()).to(self.device)

        t_width = torch.tensor(
            t_width, dtype=torch.float32).repeat(
                p_width.size()).to(self.device)

        t_height = torch.tensor(
            t_height, dtype=torch.float32).repeat(
                p_height.size()).to(self.device)

        assert (p_width >= 0).all() and (p_height >= 0).all() \
            and (t_width >= 0).all() and (t_height >= 0).all()

        # coord loss terms
        return mse_loss(p_x_center, t_x_center, reduction='sum') + \
            mse_loss(p_y_center, t_y_center, reduction='sum') + \
            mse_loss(p_width.sqrt(), t_width.sqrt(), reduction='sum') + \
            mse_loss(p_height.sqrt(), t_height.sqrt(), reduction='sum')

    def _objective_loss(self, objective_conf, objective_mask):
        '''
        Yolov1 loss for objectiveness of boxes
        '''
        assert objective_conf.shape == objective_mask.shape == self._box_shape

        positives = objective_conf[objective_mask, ]
        negatives = objective_conf[~objective_mask, ]

        obj_loss = mse_loss(
            positives,
            torch.ones(positives.shape, device=self.device),
            reduction='sum')
        noobj_loss = mse_loss(
            negatives,
            torch.zeros(negatives.shape, device=self.device),
            reduction='sum')

        return obj_loss, noobj_loss

    def _class_loss(self, class_conf, target_class_name: str):
        '''
        Yolov1 loss for claasification of cells
        '''
        assert class_conf.shape[-1] == self.num_class
        if len(class_conf) == 0:
            return np.float32(0)

        # set to true class label at each cell to 1.0
        label = torch.zeros(class_conf.shape, device=self.device)
        label[..., self.class_name2ind[target_class_name]] = 1

        return mse_loss(class_conf, label, reduction='sum')

    def _get_class_error_mask(self, _, target_box):
        # which cell contains this target
        return Yolo.get_iou_mask(
            self._cell_box, target_box,
            self.objective_iou_thre).reshape(
                self.grid_w, self.grid_h)
        # grid_w X grid_h

    def _get_obj_error_mask(self, pred_boxarr, target_box):
        return self._get_coord_error_mask(pred_boxarr, target_box)

    def _get_coord_error_mask(self, pred_boxarr, target_box):
        responsible_cell_mask = self._get_class_error_mask(
            pred_boxarr, target_box)

        # the box with highest iou at each cell
        responsible_box_ind = pred_boxarr.iou(
            target_box).reshape(
                self._box_shape).argmax(
                    axis=2)

        # filter out those off the target(low iou boxes)
        responsible_box_ind = responsible_box_ind[responsible_cell_mask]

        responsible_box_mask = np.zeros(self._box_shape, dtype=bool)
        responsible_box_mask[responsible_cell_mask, responsible_box_ind] = True

        return responsible_box_mask

    def _make_detection(self, xmin, xmax, ymin, ymax,
                        class_conf, objective_conf):
        '''
            Yolo's way of giving detection:
                * Choose a class for the boxes of each cell
                * Filter out low objectiveness boxes.
        '''
        assert class_conf.shape == (self.grid_w, self.grid_h, self.num_class)
        assert objective_conf.shape == self._box_shape

        cell_class_ind = class_conf.argmax(axis=2)

        box_class_conf = np.expand_dims(
            class_conf.max(axis=2),
            axis=-1).repeat(
                self.num_box, axis=-1)
        # grid_w X grid_h X num_class

        pred_box_conf = box_class_conf * objective_conf

        # mask against low-conf boxes
        objective_conf_mask = (objective_conf >= self.objective_conf_thre)

        output = []
        for i, class_name in enumerate(self.class_names):
            # mask against other classes
            cell_class_mask = np.expand_dims(
                cell_class_ind == i, axis=-1).repeat(
                    self.num_box, axis=-1)

            pred_box_mask = cell_class_mask & objective_conf_mask

            output.append(DetectBoxArray(
                class_name,
                pred_box_conf[pred_box_mask].reshape(-1),
                self.grid_w, self.grid_h,
                xmin[pred_box_mask].reshape(-1),
                xmax[pred_box_mask].reshape(-1),
                ymin[pred_box_mask].reshape(-1),
                ymax[pred_box_mask].reshape(-1)
            ))

        return output


def test_yolov1():
    class_names = ['class A', 'class B']
    num_box = 4
    grid_w = 2
    grid_h = 3

    yolo_layer = YoloV1(
        class_names=class_names,
        num_box=num_box,
        grid_w=grid_w,
        grid_h=grid_h)

    labels = LabelBoxArray(['class A'], 10, 15, [0], [5], [5], [10])

    pred = torch.rand(
        grid_w, grid_h, len(class_names) + num_box * Yolo.box_size)
    pred = 2 * (pred - 0.5)

    # set pred[0][0][0] to be true
    # classes
    pred[0][1][0] = 1.0 - 0.1
    pred[0][1][1] = 0.0 + 0.1
    # pred[0][1]'s first of the two boxes
    pred[0][1][2] = 1.0
    pred[0][1][3] = 0.5 + 0.05
    pred[0][1][4] = 0.5 - 0.05
    pred[0][1][5] = 1/2 - 0.05
    pred[0][1][6] = 1/3 + 0.05

    loss = yolo_layer(pred, labels)
    assert isinstance(loss, dict) and len(loss) == 1
    assert len(loss['class A']) == 3
    detect = yolo_layer(pred)
    assert isinstance(detect, list) and len(detect) == 2
    assert isinstance(detect[0], DetectBoxArray)
