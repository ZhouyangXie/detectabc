'''
    TODO
'''
from .box import BoxArray


class LabelBoxArray(BoxArray):
    '''
        storing bounding box labels of an image(coordinates and class names)
    '''
    def __init__(self, class_names, *args, **kwargs) -> None:
        '''
        args:
            class_names: list of str
        '''
        super().__init__(*args, **kwargs)

        assert len(class_names) == len(self)
        self.class_names = class_names.copy()

    def __getitem__(self, key: str):
        """get boxes by class name as key

        Parameters
        ----------
        key : str or int or slice
            if str: class name, filter boxes by class name
            if int/slice: get boxes

        Returns
        -------
        boxes : BoxArray
            a new BoxArray instance of the class name
        """
        if isinstance(key, str):
            inds = [i for i, c in enumerate(self.class_names) if c == key]
        else:
            inds = key

        return super().__getitem__(inds)

    def copy(self):
        return LabelBoxArray(self.class_names,
                             self.img_w, self.img_h, self.xmin,
                             self.xmax, self.ymin, self.ymax, False)

    @classmethod
    def from_cls_box(cls, class_names,
                     box_arr: BoxArray, truncate=True):
        """
            make LabelBoxArray from BoxArray
        """
        return LabelBoxArray(class_names,
                             box_arr.img_w, box_arr.img_h,
                             box_arr.xmin, box_arr.xmax,
                             box_arr.ymin, box_arr.ymax, truncate)

    @classmethod
    def from_voc(cls, label):
        """load labels from PASCAL VOC

        Parameters
        ----------
        label : dict
            labels formatted as that loaded from
                torchvision.datasets.VocDetection
        """
        anno = label['annotation']

        img_w, img_h = float(anno['size']['width']), \
            float(anno['size']['height'])

        class_names, boxes = [], []
        for obj in anno['object']:
            if obj['difficult'] == '0':
                class_names.append(obj['name'])
                box = obj['bndbox']
                boxes.append(
                    (float(box['xmin']), float(box['xmax']),
                     float(box['ymin']), float(box['ymax']))
                )

        return cls.from_cls_box(class_names,
                                BoxArray.from_iter(img_w, img_h, boxes, True),
                                False)
