'''
    class BoxArray for storing bounding boxes
        for an image and computing inter-box metrics
'''
import numpy as np


class _BoxMixIn:
    '''
        mix-in methods for _Box and BoxArray
    '''

    def area(self):
        '''
            compute the area of box or boxes
        '''
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def rescale_by(self, factor_w, factor_h=-1):
        '''
            rescale box and img size by a factor
        '''
        assert factor_w > 0
        if factor_h <= 0:
            factor_h = factor_w

        self.img_w, self.img_h = self.img_w * factor_w, self.img_h * factor_h
        self.xmin *= factor_w
        self.xmax *= factor_w
        self.ymin *= factor_h
        self.ymax *= factor_h

    def rescale_to(self, to_img_w: int, to_img_h: int = -1):
        '''
            rescale box and img sizes to a new size
        '''
        assert to_img_w > 0
        if to_img_h <= 0:
            to_img_h = to_img_w

        self.rescale_by(float(to_img_w/self.img_w), float(to_img_h/self.img_h))

    def rotate(self):
        '''
            rotate the img and the box
        '''
        self.img_h, self.img_w = self.img_w, self.img_h

        # swap in this way because xmin etc. are mutable in BoxArray
        temp = self.xmin
        self.xmin = self.ymin
        self.ymin = temp

        temp = self.xmax
        self.xmax = self.ymax
        self.ymax = temp

    def to_xywh(self):
        '''
            return box coordinates in xywh format
        '''
        x_center = (self.xmax + self.xmin) / 2
        y_center = (self.ymax + self.ymin) / 2
        width = (self.xmax - self.xmin)
        height = (self.ymax - self.ymin)

        return x_center, y_center, width, height


class _Box(_BoxMixIn):
    def __init__(self, *args) -> None:
        super().__init__()
        assert len(args) == 6
        self.img_w = float(args[0])
        self.img_h = float(args[1])
        self.xmin = float(args[2])
        self.xmax = float(args[3])
        self.ymin = float(args[4])
        self.ymax = float(args[5])
        # assert self.img_w >= self.xmax >= self.xmin >= 0
        # assert self.img_h >= self.ymax >= self.ymin >= 0
        assert self.xmax >= self.xmin
        assert self.ymax >= self.ymin

    def copy(self):
        '''
            return a copy of self
        '''
        return _Box(self.img_w, self.img_h, self.xmin,
                    self.xmax, self.ymin, self.ymax)


class BoxArray(_BoxMixIn):
    '''
        storing bounding boxes given an image
    '''

    def __init__(self, img_w: float, img_h: float,
                 xmin: np.array, xmax: np.array,
                 ymin: np.array, ymax: np.array,
                 truncate=False) -> None:
        '''
        args:
            img_w, img_h: the image that the boxes belong to
            xmin, xmax, ymin, ymax: np.array of int or float

        methods:

        '''
        super().__init__()

        assert img_w > 0 and img_h > 0
        self.img_w, self.img_h = float(img_w), float(img_h)

        assert len(xmin) == len(xmax) == len(ymin) == len(ymax)
        # np.array() as a deep copy constructor
        self.xmin = np.array(xmin, dtype=float)
        self.xmax = np.array(xmax, dtype=float)
        self.ymin = np.array(ymin, dtype=float)
        self.ymax = np.array(ymax, dtype=float)
        assert all(self.xmax >= self.xmin) and all(self.ymax >= self.ymin)

        if truncate:
            self._truncate()
        # else:
        #     assert all(xmin >= 0) and all(xmax <= img_w)
        #     assert all(ymin >= 0) and all(ymax <= img_h)

        self._i = None  # iter count

    def _truncate(self):
        '''
            restore coordinates to proper range and order
        '''
        self.xmin[self.xmin <= 0] = 0.
        self.xmax[self.xmax >= self.img_w] = self.img_w
        self.ymin[self.ymin <= 0] = 0.
        self.ymax[self.ymax >= self.img_h] = self.img_h

    @classmethod
    def from_array(cls, img_w: float, img_h: float, array, truncate=True):
        '''
        args:
            array: numpy.ndarray in shape of N x 4
            others as __init__
        '''
        assert len(array.shape) == 2 and array.shape[1] == 4
        return cls(img_w, img_h, *array.transpose(), truncate)

    @classmethod
    def from_iter(cls, img_w: int, img_h: int,
                  box_iter, truncate=True):
        '''
            box_iter: an iterable of tuple or list (xmin, xmax, ymin, ymax)
            others as __init__
        '''

        return cls.from_array(img_w, img_h,
                              np.array([list(box) for box in box_iter]),
                              truncate)

    def copy(self):
        '''
            return a copied instance of self
        '''
        return BoxArray(self.img_w, self.img_h, self.xmin,
                        self.xmax, self.ymin, self.ymax, False)

    def __add__(self, boxarr):
        boxarr = boxarr.copy()
        boxarr.rescale_to(self.img_w, self.img_h)
        xmin = np.concatenate((self.xmin, boxarr.xmin))
        xmax = np.concatenate((self.xmax, boxarr.xmax))
        ymin = np.concatenate((self.ymin, boxarr.ymin))
        ymax = np.concatenate((self.ymax, boxarr.ymax))

        return BoxArray(self.img_w, self.img_h, xmin, xmax, ymin, ymax)

    def __len__(self):
        return len(self.xmin)

    def __iter__(self):
        self._i = 0

        def _gen():
            for i in range(len(self)):
                yield self[i]
        return _gen()

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            return _Box(self.img_w, self.img_h, self.xmin[i],
                        self.xmax[i], self.ymin[i], self.ymax[i])
        elif isinstance(i, (list, np.ndarray, slice)):
            if isinstance(i, list):
                i = np.array(i, dtype=np.int)
            # i is slice-like, such as int array or boolean array
            return BoxArray(self.img_w, self.img_h,
                            self.xmin[i, ], self.xmax[i, ],
                            self.ymin[i, ], self.ymax[i, ], False)
        else:
            raise TypeError

    def inter(self, target: _Box):
        '''
        args:
            target: the intersection of all boxes with the target are computed,
                they are automatically rescaled to the same image size
        '''
        target = target.copy()
        target.rescale_to(self.img_w, self.img_h)

        inter_xmin = self.xmin.copy()
        inter_xmin[self.xmin < target.xmin] = target.xmin

        inter_xmax = self.xmax.copy()
        inter_xmax[self.xmax > target.xmax] = target.xmax

        inter_ymin = self.ymin.copy()
        inter_ymin[self.ymin < target.ymin] = target.ymin

        inter_ymax = self.ymax.copy()
        inter_ymax[self.ymax > target.ymax] = target.ymax

        # set those with no intersection to 0
        no_inter_ind = (inter_xmin >= inter_xmax) | (inter_ymin >= inter_ymax)
        inter_xmin[no_inter_ind] = 0
        inter_xmax[no_inter_ind] = 0
        inter_ymin[no_inter_ind] = 0
        inter_ymax[no_inter_ind] = 0

        return BoxArray(self.img_w, self.img_h,
                        inter_xmin, inter_xmax,
                        inter_ymin, inter_ymax, False).area()

    def union(self, target: _Box):
        '''
        args:
            same as inter()
        '''
        target = target.copy()
        target.rescale_to(self.img_w, self.img_h)

        union_area = self.area() + target.area() - self.inter(target)
        return union_area

    def iou(self, target: _Box):
        '''
        args:
            same as inter()
        '''
        union_area = self.union(target)
        # prevent dividing 0 for zero-size boxes
        union_area[union_area <= 0] = 1

        return self.inter(target)/union_area
