import numpy as np
from detectabc.detutils.metrics import HitMetrics


def test_hit_metrics():
    '''
        test hit metrics
    '''
    hits = np.array([
        [False, False, False],
        [True,  False, False],
        [True,  False, False],  # will be ignored in AP
        [False, False, False],
        [False, False, False],
        [False, True,  False],
    ])

    metrics = HitMetrics()
    metrics.add_instance(hits, np.arange(len(hits), 0, -1))
    assert metrics.average_precision() == 0.3
    assert metrics.average_recall() == 2/3
