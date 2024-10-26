from typing import Tuple
from torchvision.transforms import functional as F
import numpy as np
import cv2
import torch
class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class LetterBox(object):
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        self.new_shape = new_shape  # 目标尺寸
        self.color = color  # 填充颜色
        self.auto = auto  # 是否自动计算长宽比
        self.scaleFill = scaleFill  # 是否填充图像而不保持比例
        self.scaleup = scaleup  # 是否允许放大图像

    def __call__(self, image, target):
        # 获取原始图像的宽和高
        orig_shape = image.shape[:2]  # [height, width]

        # 计算新图像的宽和高
        if self.auto:
            ratio = min(self.new_shape[0] / orig_shape[0], self.new_shape[1] / orig_shape[1])
            new_unpad = int(round(orig_shape[1] * ratio)), int(round(orig_shape[0] * ratio))
            dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # 计算填充的宽和高
        else:
            new_unpad = self.new_shape
            dw, dh = self.new_shape[1] - orig_shape[1], self.new_shape[0] - orig_shape[0]

        if self.scaleFill:  # 填充图像而不保持比例
            dw, dh = 0, 0
            new_unpad = (self.new_shape[1], self.new_shape[0])  # 固定目标尺寸

        # 计算图像填充的边界
        dw //= 2  # 计算左右填充
        dh //= 2  # 计算上下填充

        # 调整图像大小并添加填充
        if self.scaleup or ratio < 1:  # 如果允许放大或保持比例小于1
            image = cv2.resize(image, new_unpad)
        else:
            image = cv2.resize(image, orig_shape[1::-1])  # 原始大小

        # 填充图像
        top, bottom = dh, self.new_shape[0] - new_unpad[1] - dh
        left, right = dw, self.new_shape[1] - new_unpad[0] - dw
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)

        # 更新目标数据
        if target is not None:
            # 如果目标存在，更新关键点的坐标
            keypoints = target['keypoints']
            updated_keypoints = []  # 存储更新后的关键点
            
            for kpt, visibility in zip(keypoints, target['visible']):
                if visibility > 0.5:  # 只更新可见的关键点
                    # 根据缩放和填充更新坐标
                    updated_kpt = [
                        int(kpt[0] * ratio + left),
                        int(kpt[1] * ratio + top)
                    ]
                    updated_keypoints.append(updated_kpt)
                else:
                    updated_keypoints.append(kpt)  # 不可见的关键点保持不变
            
            target['keypoints'] = np.array(updated_keypoints)  # 更新 target 中的 keypoints


        return image, target

class ToTensor(object):
    """将PIL或者numpy图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
                 gaussian_sigma: int = 2,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, image, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).astype(np.int32)  # round
        for kp_id in range(num_kps):
            v = kps_weights[kp_id]
            if v < 0.5:
                # 如果该点的可见度很低，则直接忽略
                continue

            x, y = heatmap_kps[kp_id]
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
            br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
            # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格)
            if ul[0] > self.heatmap_hw[1] - 1 or \
                    ul[1] > self.heatmap_hw[0] - 1 or \
                    br[0] < 0 or \
                    br[1] < 0:
                # If not, just return the image as is
                kps_weights[kp_id] = 0
                continue

            # Usable gaussian range
            # 计算高斯核有效区域（高斯核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            # 计算heatmap中的有效区域（heatmap坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            if kps_weights[kp_id] > 0.5:
                # 将高斯核有效区域复制到heatmap对应区域
                heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                    self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        # plot_heatmap(image, heatmap, kps, kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return image, target
