from hrnet import Predictor, get_model
# from HR import get_model
from tools.config import load_config, Config
from PIL import Image
import matplotlib.pylab as plt
from utils import get_max_preds
import torch
import numpy as np
from draw import draw_keypoints


def visualize_channel_heatmaps(array, cols=5, cmap='viridis'):
    """
    可视化每个通道的热图并显示颜色条，同时标注值最大点的坐标。

    参数:
        array (array): 输入的多通道数组，形状为 (C, H, W)，C 为通道数。
        cols (int): 每行显示的图像数量。
        cmap (str): 热图的颜色映射（默认是 viridis）。
    """
    channels, height, width = array.shape
    rows = (channels + cols - 1) // cols  # 计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # 将 axes 变成一维数组，方便索引

    for i in range(channels):
        ax = axes[i]
        im = ax.imshow(array[i], cmap=cmap)  # 使用颜色映射显示热图

        # 获取最大值的坐标
        max_pos = np.unravel_index(np.argmax(array[i]), array[i].shape)
        ax.plot(max_pos[1], max_pos[0], 'ro')  # 在最大值处绘制红点
        ax.set_title(f'Channel {i+1} (Max: {max_pos})')
        ax.axis('off')  # 隐藏坐标轴

        # 添加颜色条（colorbar）
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)  # 设置颜色条的刻度标签大小

    # 删除多余的子图
    for i in range(channels, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def show_result(heatmap, input_shape, ori_image):
    key_points, maxval = get_max_preds(heatmap)

    # 恢复分辨率至原图
    
    scale = [int(I / i) for i, I in zip(heatmap.shape[-2:], input_shape)]

    key_points = (key_points * torch.Tensor(scale)).type(torch.uint8)

    
    image = draw_keypoints(ori_image, key_points[0].numpy(), maxval[0].numpy(), thresh=0.0)

    image.show()
    pass

    
def main(config: Config):
    
    predictor = Predictor(get_model(config), config)
    img_path = config['predict.image_path']

    
    image = Image.open(img_path)    

    key_points, maxvals = predictor(image)

    image = draw_keypoints(image, key_points, maxvals, 0.0)

    image.show()

    pass    




if __name__ == "__main__":
    main(load_config("application.yaml"))


