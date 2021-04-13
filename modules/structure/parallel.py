from torch import nn

class Parallel(nn.Module):
    def __init__(self):
        super().__init__()
        self._opers = []

    def add_oper(self, opers):
        for oper in opers:
            self._opers.append(oper)

    def forward(self, _inputs):
        if not len(_inputs) == len(self._opers):
            raise ValueError

        return [ oper(_input) for _input, oper in zip(_inputs, self._opers) ]

    @classmethod
    def from_opers(cls, opers):
        mod = cls()
        mod.add_oper(opers)
        return mod
