from typing import Tuple
from torchvision.transforms import functional as F
import numpy as np
import cv2
import random
from draw import plot_heatmap
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



class GeometricTransform(object):
    def __init__(self, scale=(0.8, 1.2), rotation=30, translation=(0.2, 0.2), shear=10):
        self.scale = scale  # 缩放比例范围
        self.rotation = rotation  # 旋转角度范围（度数）
        self.translation = translation  # 平移范围（图像尺寸的比例）
        self.shear = shear  # 剪切角度范围（度数）

    def __call__(self, image, target):
        orig_shape = image.shape[:2]  # [height, width]
        center = (orig_shape[1] / 2, orig_shape[0] / 2)  # 图像中心

        # 随机生成缩放因子、旋转角度、平移和剪切
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        rotation_angle = random.uniform(-self.rotation, self.rotation)
        tx = random.uniform(-self.translation[0], self.translation[0]) * orig_shape[1]
        ty = random.uniform(-self.translation[1], self.translation[1]) * orig_shape[0]
        shear_angle = random.uniform(-self.shear, self.shear)

        # 计算变换矩阵
        M_scale_rotate = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
        M_translate = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        M_shear = np.array([[1, np.tan(np.radians(shear_angle)), 0],
                            [np.tan(np.radians(shear_angle)), 1, 0]], dtype=np.float32)

        # 组合所有变换
        M = M_shear @ np.vstack([M_scale_rotate, [0, 0, 1]]) @ np.vstack([M_translate, [0, 0, 1]])
        M = M[:2, :]  # 转换为 2x3 矩阵

        # 对图像应用几何变换
        transformed_image = cv2.warpAffine(image, M, (orig_shape[1], orig_shape[0]), flags=cv2.INTER_LINEAR)

        # 更新目标数据中的关键点和边界框
        if target is not None:
            # 更新关键点坐标
            keypoints = target['keypoints']
            updated_keypoints = []
            updated_visibility = []  # 用于存储更新后的可见性
            
            for (x, y), visible in zip(keypoints, target['visible']):
                if visible > 0.5:
                    # 使用变换矩阵更新坐标
                    new_x, new_y = M @ [x, y, 1]
                    if 0 <= new_x < orig_shape[1] and 0 <= new_y < orig_shape[0]:  # 检查是否在图像区域内
                        updated_keypoints.append([int(new_x), int(new_y)])
                        updated_visibility.append(1)  # 保持可见
                    else:
                        updated_keypoints.append([0, 0])  # 超出边界则设为(0,0)
                        updated_visibility.append(0)  # 标记为不可见
                else:
                    updated_keypoints.append([0, 0])  # 不可见关键点保持(0,0)
                    updated_visibility.append(0)  # 保持不可见
            
            target['keypoints'] = np.array(updated_keypoints)
            target['visible'] = np.array(updated_visibility)

        return transformed_image, target