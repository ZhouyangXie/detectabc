import numpy as np

from detutils.metrics import HitMetrics
from detutils.box import BoxArray, DetectBoxArray, LabelBoxArray


VOC_LABEL = {'annotation':
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

DETECTION = {}
DETECTION['img_w'] = 1
DETECTION['img_h'] = 1
DETECTION['classes'] = {}

DETECTION['classes']['person'] = {}
DETECTION['classes']['person']['boxes'] = BoxArray.from_iter(
    DETECTION['img_w'],
    DETECTION['img_h'],
    [(0.01, 0.02, 0.01, 0.02),
     (0, 1, 0, 1),
     (0.316, 0.538, 0.1, 0.347)]
    )
DETECTION['classes']['person']['confs'] = [0.5, 1, 0.2]

DETECTION['classes']['dog'] = {}
DETECTION['classes']['dog']['boxes'] = BoxArray.from_iter(
    DETECTION['img_w'],
    DETECTION['img_h'],
    [(0, 1, 0, 1), ])
DETECTION['classes']['dog']['confs'] = [1]


def test_all():
    label = LabelBoxArray.from_voc(VOC_LABEL)
    for class_name in DETECTION['classes'].keys():
        DETECTION['classes'][class_name] = DetectBoxArray.from_conf_box(
            class_name,
            DETECTION['classes'][class_name]['confs'],
            DETECTION['classes'][class_name]['boxes']
        )

    APs = {}
    for class_name in set(label.class_names):
        if class_name in DETECTION['classes'].keys():
            predictions = DETECTION['classes'][class_name].filter(0.1)
            hits = []
            for target in label[class_name]:
                ious = predictions.iou(target)
                hits.append(ious >= 0.5)
            APs[class_name] = HitMetrics(
                np.array(hits).transpose()).average_precision()
        else:
            APs[class_name] = 0

    print(APs)
