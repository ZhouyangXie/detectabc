import numpy as np
from detectabc.detutils.box import BoxArray, DetectBoxArray, LabelBoxArray


def test_boxarray():
    """
        test BoxArray
    """
    boxes = np.array([[0, 1, 0, 1], [0, 2, 0, 3], [0, 1, 0, 2]])
    box_arr = BoxArray.from_array(2, 2, boxes, True)

    assert box_arr[2].to_xywh() == (0.5, 1., 1., 2)

    assert -1e-5 < box_arr[0:2][1].xmax - box_arr[1:][0].xmax < 1e-5

    assert all(box_arr.area() == np.array([1, 4, 2]))

    iou_diffs = box_arr.iou(box_arr[2]) - np.array([0.5, 0.5, 1.])
    assert all((iou_diffs < 1e-5) | (iou_diffs > -1e-5))

    scale = 2
    rescaled = box_arr.copy()
    rescaled.rescale_to(scale*box_arr.img_w, scale*box_arr.img_h)
    assert all(rescaled.area() == box_arr.area()*scale**2)

    rotated = box_arr.copy()
    rotated.rotate()
    for box_a, box_b in zip(box_arr, rotated):
        assert box_a.xmin == box_b.ymin
        assert box_a.ymin == box_b.xmin
        assert box_a.xmax == box_b.ymax
        assert box_a.ymax == box_b.xmax

    concat = box_arr + rotated
    assert len(concat) == len(box_arr) + len(rotated)
    assert concat[-1].xmin == rotated[-1].xmin
    assert concat[-1].xmax == rotated[-1].xmax
    assert concat[-1].ymin == rotated[-1].ymin
    assert concat[-1].ymax == rotated[-1].ymax


def test_detect_box():
    '''
    TODO
    '''
    boxes = BoxArray.from_iter(10, 10, [(0, 10, 0, 10), (10, 20, 10, 20)])
    detect_boxes = DetectBoxArray.from_conf_box(
        'class C', [1., 0.], boxes)

    duplicate = detect_boxes.copy()
    assert duplicate.class_name == detect_boxes.class_name

    _, inner_boxes = detect_boxes[:]
    duplicate = DetectBoxArray.from_conf_box(
        'class C', [1., 0.], inner_boxes)
    assert duplicate.class_name == detect_boxes.class_name

    assert len(detect_boxes.filter(0.5)) == 1
    assert detect_boxes.filter(0.5)[0].xmax == 10

    detect_boxes = DetectBoxArray.from_conf_box(
        'class C', [1., 0., 3., 2., -1.],
        BoxArray.from_iter(
            10, 10,
            [(0, 1, 0, 3),
             (0, 1, 0, 2),
             (0, 5, 0, 5),
             (0, 6, 0, 6),
             (1, 1, 1, 2,)]))
    nms_boxes = detect_boxes.nms(0.5)
    assert len(nms_boxes) == 3
    assert nms_boxes[0][0] == 3.
    assert nms_boxes[1][0] == 1.
    assert nms_boxes[2][0] == -1.


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
