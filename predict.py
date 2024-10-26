from PIL import Image
from tools.config import Config
from hrnet import HRNet, get_model
from transforms import *
from tools.preprocess import letterbox
from torchvision.transforms import functional as F
from PIL import ImageDraw
from utils import get_max_preds

def visualize_keypoints_on_image(img_pil, keypoints, radius=5, color=(255, 0, 0)):
    """
    在图像上绘制关键点
    img_pil: 输入的 PIL 图像
    keypoints: 关键点坐标 (torch.tensor [num_joints, 2])，表示 (x, y)
    radius: 圆点的半径
    color: 圆点的颜色，默认为红色
    """
    draw = ImageDraw.Draw(img_pil)
    num_joints = keypoints.shape[0]

    for i in range(num_joints):
        x, y = keypoints[i]
        if x > 0 and y > 0:  # 只绘制有效的关键点
            leftUpPoint = (x - radius, y - radius)
            rightDownPoint = (x + radius, y + radius)
            draw.ellipse([leftUpPoint, rightDownPoint], fill=color)

    return img_pil

    
def main(config: Config):
        
    model = get_model(config)
    device = config['predict.device']
    input_shape = config['input_shape']
    
    image_path = config['predict.image_path']

    img_pil = Image.open(image_path)

    img_pil, _ = letterbox(img_pil, input_shape)
    
    # img_pil.show()


    tensor = F.to_tensor(img_pil)
    tensor = F.normalize(tensor, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor.to(device))
        # 获取关键点的最大响应点
        preds, maxvals = get_max_preds(outputs)
        # 可视化关键点在图像上的位置 (使用第一张图像的关键点)
        img_with_keypoints = visualize_keypoints_on_image(img_pil, preds[0])
        
        # 显示带关键点的图像
        img_with_keypoints.show()
        
        pass


if __name__ == "__main__":
    from tools.config import load_config

    main(load_config("application.yaml"))