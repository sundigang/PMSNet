import os
from abc import ABC
import torch
import gc
import numpy as np
import time
from hand_pose.net_hand_pose import Net3DPoseManoVF1
import cv2
import glob
import PIL.Image as Image
# from common_util.helper import Helper as comHelper
import matplotlib.pyplot as plt
import os.path as osp
from torchvision import transforms as T


class Helper:
    @classmethod
    def get_uv_from_heatmap(cls, heatmaps):
        b, c, hm_size = heatmaps.shape[:3]
        heatmaps = heatmaps.reshape((b, c, -1))
        max_value = torch.argmax(heatmaps, dim=2, keepdim=True)
        u = max_value % hm_size
        v = max_value // hm_size
        uv = torch.cat((u, v), dim=2)

        return uv

    @classmethod
    def preprocess_image(cls, image_file):
        cv2_src_img = cv2.imread(image_file)
        cv2_src_img = cv2.resize(cv2_src_img, (256, 256))
        rgb_img = cv2_src_img[:, :, [2, 1, 0]]
        image = Helper.get_normal_transform()(cv2_src_img)
        image = image.unsqueeze(0)
        return image, rgb_img

    @classmethod
    def get_normal_transform(cls):
        transform = T.Compose(
            [T.ToTensor(),
             T.Normalize((0.3108, 0.3313, 0.3617), (0.1659, 0.1677, 0.1735)),
             ]
        )
        return transform

    @classmethod
    def draw_3d_joints(cls, xyz, title=None, fig_file_name=None):

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        fig = plt.figure(221)
        fig.suptitle(title, fontsize=16)
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(x, y, z, color='r', marker='s')

        colors = ['b', 'g', 'r', 'c', 'm']
        finger = 0
        line_width = 2
        for i in range(1, 21, 4):  # [1, 5, 9, 13, 17]:
            if i:  # == 5
                color = colors[finger]
                finger += 1
                ax.plot([x[0], x[i]], [y[0], y[i]], [z[0], z[i]], color, linewidth=line_width)
                for j in range(i, i + 3):
                    ax.plot([x[j], x[j + 1]], [y[j], y[j + 1]], [z[j], z[j + 1]], color, linewidth=line_width)

        data = [[x, -y], [z, -y], [-x, -z]]
        for i in range(len(data)):
            a, b = data[i]
            ax = fig.add_subplot(2, 2, i + 2)
            ax.scatter(a, b, color='r', marker='s')
            ax.axis('equal')
            finger = 0
            for i in range(1, 21, 4):  # [1, 5, 9, 13, 17]:
                if i:  # == 5
                    color = colors[finger]
                    finger += 1
                    ax.plot([a[0], a[i]], [b[0], b[i]], color, linewidth=line_width)
                    for j in range(i, i + 3):
                        ax.plot([a[j], a[j + 1]], [b[j], b[j + 1]], color, linewidth=line_width)

        plt.tight_layout()
        if fig_file_name is not None:
            plt.savefig(fig_file_name, dpi=200)
        else:
            plt.show()

    @classmethod
    def draw_2d_hand_joint(cls, image, uv, d=None, title=None, fig_file_name=None):
        # Visualize data
        fig = plt.figure(1)
        fig.suptitle(title, fontsize=16)
        ax = fig.add_subplot(1, 1, 1)

        plt.imshow(image)

        x = uv[:, 0]
        y = uv[:, 1]

        colors = ['b', 'g', 'r', 'c', 'm']
        finger = 0
        line_width = 2
        for i in range(1, 21, 4):  # [1, 5, 9, 13, 17]:
            if i:  # == 5
                color = colors[finger]
                finger += 1
                ax.plot(x[i], y[i], 'o', markersize=4., color=color)
                ax.plot([x[0], x[i]], [y[0], y[i]], color, linewidth=line_width)

                for j in range(i, i + 3):
                    ax.plot(x[j + 1], y[j + 1], 'o', markersize=4., color=color)
                    ax.plot([x[j], x[j + 1]], [y[j], y[j + 1]], color, linewidth=line_width)

        ax.plot(x[0], y[0], 'o', markersize=4., color='k')  # '#7F7F7F'
        # ax.set_xlim(0, 256)
        # ax.set_ylim(0, 256)
        # ax.set_zlim(0, 100)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        plt.tight_layout()
        if fig_file_name is not None:
            plt.savefig(fig_file_name, dpi=200)
        else:
            plt.show()

    @classmethod
    def normalize_hand_model(cls, kp_xyz, wrist_index=0, mcp_index=9, scale=None):

        wrist = kp_xyz[wrist_index]
        mcp_joint = kp_xyz[mcp_index]
        if not scale:
            scale = np.linalg.norm(wrist - mcp_joint, ord=2)
        wrist_tiled = np.tile(wrist, (len(kp_xyz), 1))
        relative_xyz = kp_xyz - wrist_tiled
        normal_xyz = relative_xyz / scale

        # kp_relative = (kp_xyz - wrist_tiled)
        # kp_normalized = kp_relative / standard_dist
        # print(normalized_xyz)
        return normal_xyz, scale

    @classmethod
    def rescale_z(cls, uv, xy, z):
        u, v = uv[:, 0], uv[:, 1]
        scale_u = u.max() - u.min()
        scale_v = v.max() - v.min()
        scale_uv = (scale_u + scale_v) / 2
        xy = cls.torch_to_numpy(xy.reshape(21, -1))
        x, y = xy[:, 0], xy[:, 1]
        scale_x = x.max() - x.min()
        scale_y = y.max() - y.min()
        scale_xy = (scale_x + scale_y) / 2
        uvz = np.zeros((uv.shape[0], 3))
        uvz[:, 0] = u
        uvz[:, 1] = v
        # z = out['z']
        # z = z.squeeze(dim=0)
        # z = dsHelper.torch_to_numpy(z)
        z = z / scale_xy * scale_uv
        return z

    @classmethod
    def get_joint_position_2d(cls, heatmaps, shape):
        row_col = []
        uv = []
        xy = []
        for i in range(len(heatmaps)):
            heatmap_i = heatmaps[i]
            row, col = np.unravel_index(np.argmax(heatmap_i), shape)
            u, v = col, row
            x, y = u, shape[0] - v

            row_col.append([row, col])
            uv.append([u, v])
            xy.append([x, y])

        return np.array(uv)

    @classmethod
    def torch_to_numpy(cls, tensor):
        return tensor.detach().cpu().numpy()


class BaseEvaluator:
    def __init__(self):
        super().__init__()

        self.use_gpu = torch.cuda.is_available()
        self.batch_size = None
        self.set_name = 'evaluation'
        # 定义loss
        self.dataset = None  # MixedHeatmapDataset(self.set_name)
        self.model = None


class PoseEvaluator(BaseEvaluator):
    def __init__(self, model_dict_file=None):
        super(PoseEvaluator, self).__init__()
        self.model_dict_file = model_dict_file

        self.load_model()

    def load_model(self):

        self.model = Net3DPoseManoVF1()

        self.model.load_state_dict(torch.load(self.model_dict_file))

        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

    def evaluate_one(self, img):

        with torch.no_grad():
            img = img.type(torch.FloatTensor)
            if self.use_gpu:
                img = img.cuda()
            out = self.model(img)

            heatmap = out['heatmaps'][-1][0].cpu().detach().numpy()
            uv = Helper.get_joint_position_2d(heatmap, heatmap.shape[1:])
            z = out['z']
            z = Helper.torch_to_numpy(z)

            return uv, z, out


def test():
    model_dict_file = f'model/Net3DPoseManoF1_on_inter_frei.pth'
    evaluator = PoseEvaluator(model_dict_file)
    pattern = r'image/input/*.*'

    image_files = glob.glob(pattern)
    image_files.sort(reverse=False)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # result_folder = f'image/output/{timestamp}'
    result_folder = f'image/output'
    n_image = len(image_files)
    # n_image = 100 * 3
    for i in range(0, n_image):
        index = i
        image_file = image_files[index]
        image, resized_rgb_img = Helper.preprocess_image(image_file)
        print(image_file)

        """ evaluate pose """
        file_name_2d = None
        file_name_3d = None

        # if not osp.exists(result_folder):
        #     os.makedirs(result_folder)
        # file_name_2d = f'{result_folder}/2D-{image_file.split("/")[-1]}'
        # file_name_3d = f'{result_folder}/3D-{image_file.split("/")[-1]}'

        uv, z, out = evaluator.evaluate_one(image)
        xy = out['xy']
        z = Helper.rescale_z(uv, xy, z)
        uvz = np.zeros((uv.shape[0], 3))
        uvz[:, :2] = uv
        uvz[:, -1] = z
        norm_uvz, _ = Helper.normalize_hand_model(uvz)

        title = 'Hand Pose Estimation'
        Helper.draw_2d_hand_joint(resized_rgb_img, uv * 4, title=title, fig_file_name=file_name_2d)
        if norm_uvz is not None:
            Helper.draw_3d_joints(norm_uvz, title=title, fig_file_name=file_name_3d)

    pass


if __name__ == '__main__':
    test()

    print('=' * 50)
