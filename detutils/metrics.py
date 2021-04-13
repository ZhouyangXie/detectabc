'''
TODO
'''
import numpy as np


class HitMetrics:
    '''
        metrics based on hits of predictions on multi-targets,
        such as multi-class classification or object detection
    '''

    def __init__(self, hits):
        '''
        args:
            hits: np.array dtype=bool
                hits[i][j] as a bool is whether prediction i hits target j
                hits should have been sorted by confidence score if any
                duplicate hits on same targets will be removed
        '''
        self.num_target = hits.shape[1]

        # whether the i-th prediction(duplicate-removed) hits a target
        self._hit_any = []
        # how many targets are hit at or before
        # the i-th prediction(duplicated-removed)
        self._num_hit = []
        # whether this prediction hits a target's
        # been hit before the i-th prediction
        self._duplicate_hit = []

        has_hit = np.array([False]*self.num_target)
        for row in hits:
            self._duplicate_hit.append(
                True if all(has_hit == (has_hit | row)) and any(row)
                else False)
            has_hit |= row
            self._hit_any.append(True if any(row) else False)
            self._num_hit.append(np.count_nonzero(has_hit))

        self.num_pred = len(self._hit_any)

    def precision(self, remove_duplicate=False):
        '''
            return:
                a list of float
                the i-th stands for the precision of top i predictions
        '''
        if remove_duplicate:
            hit_any = [h for i, h in enumerate(self._hit_any)
                       if not self._duplicate_hit[i]]
        else:
            hit_any = self._hit_any

        return [hit_any[:i+1].count(True)/(i+1)
                for i, n in enumerate(hit_any)]

    def recall(self, remove_duplicate=False):
        '''
            return:
                a list of float
                the i-th stands for the recall of top i predictions
        '''
        if remove_duplicate:
            num_hit = [h for i, h in enumerate(self._num_hit)
                       if not self._duplicate_hit[i]]
        else:
            num_hit = self._num_hit

        return [b/self.num_target for i, b in enumerate(num_hit)]

    def fscore(self, beta=1.0):
        '''
            return:
                a list of float
                the i-th stands for the fscore of top i predictions
        '''
        precision, recall = self.precision(), self.recall()
        scale = beta**2
        return [(1+scale)*p*r/(p*scale+r) if r > 0 else 0
                for p, r in zip(precision, recall)]

    def average_precision(self, method='auc', sample: int = 101):
        '''
            AP(Average Precision)
            args:
                method: if use auc(area under curve),
                    use precisions as sampling point;
                    if use interp(interpolation),
                    use arg sample.
                sample: number of sampling points.
        '''
        precision, recall = self.precision(True), self.recall(True)
        max_precision = [max(precision[i:]) for i, _ in enumerate(precision)]

        if method == 'auc':
            # every max-precision drop is a sample point
            curve = [(0, 1)] + list(zip(recall, max_precision)) + [(1, 0)]
            area = [(r1-r0)*min(p1, p0)
                    for (r0, p0), (r1, p1) in zip(curve[:-1], curve[1:])]
            return sum(area)

        elif method == 'interp':
            assert sample > 1
            step = 1/(sample-1)

            curve = list(zip(recall, max_precision))
            area = [0]
            # sample the min of an interval
            for r_thre in np.arange(1, 0, -step):
                sampled_p = [p for r, p in curve
                             if r_thre >= r > r_thre-step]
                area.append(
                    max(sampled_p) if len(sampled_p) > 0 else max(area))

            # -1 because of the initial place-holding 0
            return sum(area)/(len(area)-1)

        else:
            raise ValueError(f'invalid AP computation method \'{method}\'')

    def average_recall(self, top_n: int = 1) -> float:
        '''
            AR(Average Recall) of top_n predictions
        '''
        assert top_n > 0
        if top_n > self.num_pred:
            top_n = self.num_pred

        return sum(self.recall()[:top_n])/top_n


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
