import torch
from torch import nn

from ..functional.placeholder import MergeFeature, Output

class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self._steps = []

    def add_step(self, steps):
        for step in steps:
            if isinstance(step, torch.nn.Module):
                self._steps.append(step)
            else:
                raise TypeError

        if not any(filter(lambda s:isinstance(s, Output), self._steps)):
            raise ValueError

    def forward(self, _inputs):
        if isinstance(_inputs, torch.Tensor):
            _inputs = [_inputs]

        n_merge = len(filter(lambda s: isinstance(s, MergeFeature), self._steps))
        if not len(_inputs) == n_merge + 1:
            raise ValueError

        outputs = []
        temp = _inputs.pop()
        for step in self._steps:
            if isinstance(step, MergeFeature):
                temp = step([_inputs.pop(), temp])
            elif isinstance(step, Output):
                outputs.append(temp)
            else:
                temp = step(temp)

        return outputs

    @classmethod
    def from_steps(cls, steps):
        """
        docstring
        """
        mod = cls()
        mod.add_step(steps)
        return mod
