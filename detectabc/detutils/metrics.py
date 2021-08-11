'''
TODO
'''
import numpy as np


class HitMetrics:
    '''
        metrics based on hits of predictions on multi-targets,
        such as multi-class classification or object detection
    '''
    def __init__(self):
        self.empty()

    def empty(self):
        self._confs = np.zeros((0,), dtype=np.float64)
        self._tp_hits = np.zeros((0,), dtype=np.int64)
        self._num_targets = 0
        self._num_instances = 0
        self._recalls = []

    def add_skipped_instance(self, num_targets):
        self._num_instances += 1
        self._num_targets += num_targets

    def add_instance(self, hits, confs):
        '''
        add the predictions for the targets in one instance
        (AP is computed across targets, AR is computed across instances)

        args:
            hits: np.array dtype=bool
            hits[i][j] as a bool is whether prediction i hits target j
            hits should have been sorted by confidence score if any
            duplicate hits on same targets will be removed
            confs: np.array dtype=float
            confs[i] is the confidence for the i-th prediction
        '''
        assert hits.ndim == 2 and hits.dtype == bool
        assert confs.ndim == 1
        assert len(hits) == len(confs)
        confs = confs.astype(np.float64)

        # how many targets in this instance
        num_targets = hits.shape[1]
        # a running variable whether the j-th target has been hit
        has_hit = np.zeros((num_targets,), dtype=bool)

        # index of non-duplicate hit in hits' rows
        valid_hits_ind = []
        # whether the corresponding hit in valid_hits_ind if tp or fp
        valid_tp_hits = []

        for i, row in enumerate(hits):
            if row.any():
                # hit at least one target
                num_new_hit_targets = np.sum(~(has_hit[row]))
                if num_new_hit_targets == 0:
                    # a duplicate hit
                    continue
                else:
                    # if a new target is hit
                    valid_tp_hits.append(num_new_hit_targets)
            else:
                # no target is hit, a false positive hit
                valid_tp_hits.append(0)

            valid_hits_ind.append(i)
            has_hit |= row

        # new hits to be added
        new_confs = confs[valid_hits_ind]
        new_tp_hits = np.array(valid_tp_hits, dtype=np.int64)

        self._num_targets += num_targets
        self._num_instances += 1
        self._recalls.append(has_hit.mean())
        self._confs = np.concatenate((self._confs, new_confs))
        self._tp_hits = np.concatenate((self._tp_hits, new_tp_hits))

    def average_recall(self):
        if self._num_instances == 0:
            return None
        else:
            return np.array(self._recalls).mean()

    def average_precision(self):
        if self._num_instances == 0:
            return None

        if len(self._tp_hits) == 0:
            return 0.0

        # sort all hits(across instances) by confidence
        hits = self._tp_hits[np.argsort(self._confs)[::-1]]
        has_hit = hits.astype(bool)

        # compute cumulative precision and recall
        precision = np.cumsum(has_hit)/(np.arange(len(has_hit)) + 1)
        recall_diff = hits/self._num_targets

        precision = precision[has_hit]
        recall_diff = recall_diff[has_hit]
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i+1])

        return (precision * recall_diff).sum()
