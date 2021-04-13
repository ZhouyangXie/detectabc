'''
TODO
'''
import torch
import numpy as np

from detectutils.box import BoxArray

from .yolov1 import YoloV1


class YoloV2(YoloV1):
    """
    Compute the loss of YOLO(v2/9000)
    """
    def __init__(self,
                 anchors: BoxArray,
                 *args,
                 **kwargs):
        '''
        Parameters
        ----------
        anchors: BoxArray,
            a list of bounding box prior for each cell
            box location is ignored, only w/h matters
        '''
        kwargs['num_box'] = len(anchors)
        super().__init__(*args, **kwargs)
        # _cell_box replaced by anchors
        self.anchors = anchors.copy()
        self._anchor_box = None

    def _call_prepare(self):
        if self._anchor_box is None or \
            (self._anchor_box.grid_w, self._anchor_box.grid_h) != \
                (self.grid_w, self.grid_h):
            self.anchors.rescale_to(self.grid_w, self.grid_h)
            self._anchor_box = YoloV2._make_anchor_box(self.anchors)

    @staticmethod
    def _make_anchor_box(anchors):
        grid_w, grid_h = anchors.grid_w, anchors.grid_h

        # compute w/h of anchors and expand to _box_shape
        anchor_width = np.expand_dims(
            anchors.xmax - anchors.xmin, 0
            ).repeat(
                grid_w*grid_h, 0
            ).reshape(
                grid_w, grid_h, -1
            )
        anchor_height = np.expand_dims(
            anchors.ymax - anchors.ymin, 0
            ).repeat(
                grid_w*grid_h, 0
            ).reshape(
                grid_w, grid_h, -1
            )

        # expand cell center to _box_shape
        cell_center_x, cell_center_y = (
            np.mgrid[:grid_w, :grid_h] + 0.5)
        cell_center_x = np.expand_dims(
            cell_center_x, -1).repeat(
                len(anchors), -1)
        cell_center_y = np.expand_dims(
            cell_center_y, -1).repeat(
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

    def _tensor_to_detection(self, prediction):
        '''
        YOLOv2 specified way of conversion
        '''
        assert YoloV2.box_size == 5

        # split the prediction tensor
        class_conf = torch.sigmoid(
            prediction[:, :, :self.num_class])
        objective_conf = torch.sigmoid(
            prediction[:, :, self.num_class::YoloV2.box_size])
        x_center = torch.sigmoid(
            prediction[:, :, self.num_class+1::YoloV2.box_size])
        y_center = torch.sigmoid(
            prediction[:, :, self.num_class+2::YoloV2.box_size])
        width_scale = torch.exp(
            prediction[:, :, self.num_class+3::YoloV2.box_size])
        height_scale = torch.exp(
            prediction[:, :, self.num_class+4::YoloV2.box_size])

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

        return class_conf, objective_conf, xmin, xmax, ymin, ymax
