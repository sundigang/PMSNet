from abc import ABC
import torch
import torch.nn as nn


class ResidualFCLayer(nn.Module, ABC):

    def __init__(self, n_channel, n_module=2, use_bn=True, use_relu=True, drop_prob=0):
        super().__init__()

        self.n_channel = n_channel
        self.n_module = n_module
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.drop_prob = drop_prob

        _fc_layers = []
        _bn_layers = []
        _dropout_layers = []
        for i in range(self.n_module):
            _fc_layers.append(nn.Linear(n_channel, n_channel))
            if use_bn:
                _bn_layers.append(nn.BatchNorm1d(n_channel))
            if drop_prob > 0:
                _dropout_layers.append(nn.Dropout(p=drop_prob))

        self.fc_layers = nn.ModuleList(_fc_layers)
        self.bn_layers = nn.ModuleList(_bn_layers)
        self.dropout_layers = nn.ModuleList(_dropout_layers)

        if use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        r = x

        for i in range(self.n_module):
            x = self.fc_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            if self.use_relu:
                x = self.relu(x)
            if self.drop_prob > 0:
                x = self.dropout_layers[i](x)

        out = x + r
        return out


class ConvLayer(nn.Module, ABC):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, use_bn=False, use_relu=True):
        super(ConvLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        # self.add_module('conv', self.conv)
        self.bn = None
        self.relu = None
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
            # self.add_module('bn', self.bn)
        if use_relu:
            self.relu = nn.ReLU()
            # self.add_module('relu', self.relu)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        # return super(ConvLayer, self).forward(x)


class ConvTransposeLayer(nn.Module, ABC):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, output_padding=1, padding=1, use_bn=True,
                 use_relu=True):
        super(ConvTransposeLayer, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = None
        self.relu = None

        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, output_padding, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        if use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class UpConvLayer(nn.Module, ABC):
    """
    Up sampling and Convolution
    """

    def __init__(self, in_channel, out_channel, scale_factor=2, kernel_size=3, stride=1, use_bn=True, use_relu=True):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = None
        self.relu = None

        self.up_sample = nn.Upsample(scale_factor=self.scale_factor)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        if use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FCLayer(nn.Module, ABC):
    def __init__(self, in_channel, out_channel, use_relu=False, use_bn=False, drop_prob=0):
        """

        :param in_channel:
        :param out_channel:
        :param use_relu:
        :param use_bn:
        :param drop_prob: dropout probability, default is 0
        """
        super(FCLayer, self).__init__()
        self.use_relu = use_relu
        # self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.drop_prob = drop_prob
        self.fc = nn.Linear(in_channel, out_channel)
        # self.add_module('fc', self.fc)
        if self.use_relu:
            self.relu = nn.ReLU(True)
            # self.add_module('relu', self.relu)
        if self.drop_prob > 0:
            self.dropout = nn.Dropout(p=self.drop_prob, inplace=False)
            # self.add_module('dropout', self.dropout)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        if self.drop_prob > 0:
            x = self.dropout(x)
        return x
        # return super(FCLayer, self).forward(x)


class Residual(nn.Module, ABC):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = ConvLayer(in_channel, out_channel // 2, 1, use_relu=False)

        self.bn2 = nn.BatchNorm2d(out_channel // 2)
        self.conv2 = ConvLayer(out_channel // 2, out_channel // 2, 3, use_relu=False)

        self.bn3 = nn.BatchNorm2d(out_channel // 2)
        self.conv3 = ConvLayer(out_channel // 2, out_channel, 1, use_relu=False)

        self.skip_conv = ConvLayer(in_channel, out_channel, 1, use_relu=False)
        self.need_skip = in_channel != out_channel

    def forward(self, x):
        r = x
        if self.need_skip:
            r = self.skip_conv(x)
        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = out + r
        return out
