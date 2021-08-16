import numpy as np
from detectabc.detutils.metrics import HitMetrics


def test_hit_metrics_A():
    '''
        test hit metrics
    '''
    hits = np.array([
        [False, False, False],
        [True,  False, False],
        [True,  False, False],
        [False, False, False],
        [False, False, False],
        [False, True,  False],
    ])

    metrics = HitMetrics()
    metrics.add_instance(hits, np.arange(len(hits), 0, -1))
    assert metrics.average_precision() == 5/18
    assert metrics.average_recall() == 2/3


def test_hit_metrics_B():
    '''
        test hit metrics
    '''
    hits = np.array([
        [False, False, False],
        [True,  False, False],
        [False, False, False],
        [False, False, False],
        [False, True,  True],
    ])

    metrics = HitMetrics()
    metrics.add_instance(hits, np.arange(len(hits), 0, -1))
    assert metrics.average_precision() == 1.3/3
    assert metrics.average_recall() == 1.0


def test_hit_metrics_C():
    '''
        test hit metrics
    '''
    hits = np.array([
        [False, False, False],
        [True,  False, False],
        [False, False, False],
        [False, True, False],
        [False, True,  True],
    ])

    metrics = HitMetrics()
    metrics.add_instance(hits, np.arange(len(hits), 0, -1))
    assert metrics.average_precision() == 0.6
    assert metrics.average_recall() == 1.0


def test_hit_metrics_D():
    '''
        test hit metrics
    '''
    hits = np.array([
        [False, False, False, False],
        [True,  False, False, False],
        [False, False, False, False],
        [False, True, False, False],
        [False, True,  False, False],
        [False, False, False, False],
        [False, False, False, False],
    ])

    metrics = HitMetrics()
    metrics.add_instance(hits, np.arange(len(hits), 0, -1))
    assert metrics.average_precision() == 0.25
    assert metrics.average_recall() == 0.5
