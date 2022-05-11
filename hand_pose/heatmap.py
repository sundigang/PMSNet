import time
import torch
import numpy as np
import matplotlib.pyplot as plt


class HeatmapHelper:
    @classmethod
    def get_uv_from_heatmap(cls, heatmaps):
        b, c, hm_size = heatmaps.shape[:3]
        heatmaps = heatmaps.reshape((b, c, -1))
        max_value = torch.argmax(heatmaps, dim=2, keepdim=True)
        u = max_value % hm_size
        v = max_value // hm_size
        uv = torch.cat((u, v), dim=2)

        return uv
