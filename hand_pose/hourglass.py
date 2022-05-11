from abc import ABC
from torch import nn
import torch
from hand_pose.net_util import Residual


class Hourglass(nn.Module, ABC):
    def __init__(self, n_repeat, n_module, n_feature):

        super(Hourglass, self).__init__()
        self.n_repeat = n_repeat
        self.n_module = n_module
        self.n_feature = n_feature

        _high1_, _low1_, _low2_, _low3_ = [], [], [], []
        _low2_hourglass_ = None
        for m in range(self.n_module):
            _high1_.append(Residual(self.n_feature, self.n_feature))
            _low1_.append(Residual(self.n_feature, self.n_feature))
            _low3_.append(Residual(self.n_feature, self.n_feature))

        if n_repeat > 1:
            # build Hourglass recursively
            _low2_hourglass_ = Hourglass(n_repeat - 1, n_module, n_feature)
        else:
            for m in range(self.n_module):
                _low2_.append(Residual(self.n_feature, self.n_feature))

        self.high1 = nn.ModuleList(_high1_)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = nn.ModuleList(_low1_)
        self.low2_hourglass = _low2_hourglass_
        self.low2 = nn.ModuleList(_low2_)
        self.low3 = nn.ModuleList(_low3_)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # print('=' * 10)
        out1 = x
        for m in range(self.n_module):
            out1 = self.high1[m](out1)

        out2 = self.down(x)
        for m in range(self.n_module):
            out2 = self.low1[m](out2)

        if self.n_repeat > 1:
            out2 = self.low2_hourglass(out2)
        else:
            for m in range(self.n_module):
                out2 = self.low2[m](out2)

        for m in range(self.n_module):
            out2 = self.low3[m](out2)
        out2 = self.up(out2)
        # out1 = self.high1(x)

        out = torch.add(out1, out2)  # out1 + out2
        return out
