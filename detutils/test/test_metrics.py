import numpy as np
from detutils.metrics import HitMetrics


def test_hit_metrics():
    '''
        test hit metrics
    '''
    hits = np.array([
        [False, False, False],
        [True,  False, False],
        [True,  False, False],  # will be ignored in AP
        [False, False, False],
        [False, True,  False]])

    metrics = HitMetrics(hits)
    assert metrics.precision() == [0, 1/2, 2/3, 2/4, 3/5]
    assert metrics.recall() == [0, 1/3, 1/3, 1/3, 2/3]
    assert metrics.fscore() == [0, 2/5, 4/9, 2/5, 12/19]
    assert metrics.average_precision(method='auc') == 1/3
    assert metrics.average_precision(method='interp', sample=11) == 0.35
    assert metrics.average_recall(10) == 1/3
