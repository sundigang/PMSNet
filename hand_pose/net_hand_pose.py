from abc import ABC
import torch
from torch import nn
from hand_pose.NetHeatmap import NetHeatmap
from hand_pose.net_util import ConvLayer, Residual, FCLayer, ResidualFCLayer
import matplotlib.pyplot as plt
from hand_pose.heatmap import HeatmapHelper


class Net3DPoseManoVF1(nn.Module, ABC):

    def __init__(self, n_joint=21, n_stack=2, n_module=2, n_feature=256):
        super().__init__()
        self.n_joint = n_joint
        self.n_stack = n_stack
        self.n_module = n_module
        self.n_feature = n_feature
        self.drop_prob = 0
        self.n_pose_params = 15 * 3
        self.n_shape_params = 10

        self.net_heatmap = NetHeatmap(n_joint, n_stack, n_module, n_feature)

        self.fc_xyz = nn.Sequential(
            FCLayer(64 ** 2 * 2, 512, use_relu=True, use_bn=True, drop_prob=self.drop_prob),
            ResidualFCLayer(512, drop_prob=self.drop_prob),
            # ResidualFCLayer(512, drop_prob=self.drop_prob),
            # ResidualFCLayer(512, drop_prob=self.drop_prob),
            FCLayer(512, self.n_joint * 3)
        )

    def forward(self, x):
        output = {}
        batch_size = x.shape[0]

        hm_list, features = self.net_heatmap(x)
        output['heatmaps'] = hm_list
        output['features'] = features

        hm = hm_list[-1]
        uv = HeatmapHelper.get_uv_from_heatmap(hm)

        hm = torch.sum(hm, dim=1)
        hm = hm.reshape(hm.shape[0], -1)

        texture = torch.sum(torch.cat(features, dim=1), dim=1)
        texture = texture.reshape(texture.shape[0], -1)

        hm_texture = torch.cat([hm, texture], dim=1)
        xyz = self.fc_xyz(hm_texture)
        xy = xyz[:, :self.n_joint * 2]
        z = xyz[:, self.n_joint * 2:]

        output['uv'] = uv
        output['xy'] = xy
        output['z'] = z

        return output
