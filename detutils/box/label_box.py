'''
    TODO
'''
from detectutils.box import BoxArray


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


def test_label_box_array():
    '''
        unit test from VoC example
    '''
    label = {'annotation':
             {'folder': 'VOC2012',
              'filename': '2008_000008.jpg',
              'source': {
                  'database': 'The VOC2008 Database',
                  'annotation': 'PASCAL VOC2008',
                  'image': 'flickr'
              },
                 'size': {'width': '500',
                          'height': '442',
                          'depth': '3'},
                 'segmented': '0',
                 'object': [
                  {'name': 'horse',
                   'pose': 'Left',
                   'truncated': '0',
                   'occluded': '1',
                   'bndbox': {'xmin': '53',
                              'ymin': '87',
                              'xmax': '471',
                              'ymax': '420'},
                   'difficult': '0'},
                  {'name': 'person',
                   'pose': 'Unspecified',
                   'truncated': '1', 'occluded': '0',
                   'bndbox': {'xmin': '158',
                              'ymin': '44',
                              'xmax': '289',
                              'ymax': '167'},
                   'difficult': '0'
                   }
              ]
              }
             }

    obj = LabelBoxArray.from_voc(label)
    duplicate = obj.copy()

    assert obj.img_w == duplicate.img_w == 500
    assert obj.img_h == duplicate.img_h == 442

    assert obj.class_names == duplicate.class_names == ['horse', 'person']
    assert obj['person'].xmin[0] == duplicate['person'].xmin[0] == 158.0
    assert obj[0].ymax == duplicate[0].ymax == 420.
