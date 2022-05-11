
from abc import ABC
from torch import nn
import torch
from hand_pose.hourglass import Hourglass
from hand_pose.net_util import ConvLayer, Residual


class NetHeatmap(nn.Module, ABC):
    def __init__(self, n_joint=21, n_stack=2, n_module=2, n_feature=256):
        super(NetHeatmap, self).__init__()
        self.n_joint = n_joint
        self.n_stack = n_stack
        self.n_module = n_module
        self.n_feature = n_feature

        kernel_size = 7
        padding = (kernel_size - 1) // 2
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, bias=True, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Residual(128, 128),
            Residual(128, self.n_feature)
        )

        self.hourglass = nn.ModuleList([
            Hourglass(4, 2, self.n_feature)
            for i in range(self.n_stack)
        ])

        _res_ = []
        _conv_ = []
        for i in range(self.n_stack):
            for j in range(self.n_module):
                _res_.append(Residual(self.n_feature, self.n_feature))
            _conv_.append(ConvLayer(self.n_feature, self.n_feature, 1, use_bn=True, use_relu=True))
        self.conv = nn.ModuleList(_conv_)
        self.res = nn.ModuleList(_res_)

        self.out = nn.ModuleList([
            ConvLayer(self.n_feature, self.n_joint, 1, use_bn=False, use_relu=False)
            for i in range(self.n_stack)
        ])

        self.feature_merger = nn.ModuleList([
            ConvLayer(self.n_feature, self.n_feature, 1, use_bn=False, use_relu=False)
            for i in range(self.n_stack - 1)
        ])
        self.predict_merger = nn.ModuleList([
            ConvLayer(self.n_joint, self.n_feature, 1, use_bn=False, use_relu=False)
            for i in range(self.n_stack - 1)
        ])

    def forward(self, x):
        x = self.preprocess(x)
        predicts = []
        features = []
        for i in range(self.n_stack):
            hg = self.hourglass[i](x)
            for j in range(self.n_module):
                hg = self.res[i * self.n_module + j](hg)
            feature = self.conv[i](hg)
            pred = self.out[i](feature)
            predicts.append(pred)

            if i < self.n_stack - 1:
                x = x + self.predict_merger[i](pred) + self.feature_merger[i](feature)
                features.append(x)
            else:
                features.append(feature)
        output = dict()
        output['heatmaps'] = predicts
        output['features'] = features

        return predicts, features
