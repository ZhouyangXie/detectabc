'''
TODO
'''
import numpy as np

from detutils.box import BoxArray


class DetectBoxArray(BoxArray):
    '''
        storing bounding box predictions(coordinates and confidence score)
        for a class in an image
    '''

    def __init__(self, class_name, confs, img_w, img_h,
                 xmin, xmax, ymin, ymax, truncate=True) -> None:
        '''
        args:
            confs: confidence scores
        '''
        assert isinstance(class_name, str)
        self.class_name = class_name

        assert len(confs) == len(xmin)
        sorted_ind = np.argsort(confs)[::-1]
        self.confs = np.array(confs)[sorted_ind]

        super().__init__(
            img_w, img_h,
            np.array(xmin)[sorted_ind], np.array(xmax)[sorted_ind],
            np.array(ymin)[sorted_ind], np.array(ymax)[sorted_ind], truncate)

    def __getitem__(self, i):
        """
        Parameters
        ----------
        i : int or slice
            if i is an int, return float, _Box
            if i is a slice, return np.array, BoxArray
        """
        return self.confs[i], super().__getitem__(i)

    def copy(self):
        return DetectBoxArray(self.class_name,
                              self.confs, self.img_w, self.img_h, self.xmin,
                              self.xmax, self.ymin, self.ymax, False)

    @classmethod
    def from_conf_box(cls, class_name,
                      confs: np.array, box_arr: BoxArray, truncate=False):
        """
            make DetectBoxArray from BoxArray
        """
        return DetectBoxArray(class_name, confs,
                              box_arr.img_w, box_arr.img_h,
                              box_arr.xmin, box_arr.xmax,
                              box_arr.ymin, box_arr.ymax,
                              truncate)

    def filter(self, conf_thre: float):
        '''
            get boxes with confidence score above conf_thre
        '''
        return super().__getitem__(self.confs > np.float32(conf_thre))

    def nms(self, iou_thre: float):
        '''
            return the boxes after non-maximum suppression
        '''
        rest_inds = set(range(len(self)))
        output_inds = set()
        while len(rest_inds) > 0:
            max_box_id = min(rest_inds)
            _, max_box = self[max_box_id]

            # move maximum from rest to output
            rest_inds.remove(max_box_id)
            output_inds.add(max_box_id)

            # remove duplicates(high iou)
            to_remove_ind = [i for i, iou in enumerate(self.iou(max_box))
                             if iou >= iou_thre and i in rest_inds]
            rest_inds -= set(to_remove_ind)

        return DetectBoxArray.from_conf_box(
                self.class_name, *self[list(output_inds)])
