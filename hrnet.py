from torch import nn, Tensor
import torch
from tools.config import Config
from torchvision.models.resnet import Bottleneck, BasicBlock
import os
from tools.preprocess import letterbox
from torchvision.transforms import functional as F
from PIL import Image


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=0.1),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=0.1),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=0.1)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused

class HRNet(nn.Module):

    def __init__(self, base_channel: int=32, key_points: int=17):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True)

        )


        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )


        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=0.1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential( # 兼容预训练模型
                nn.Sequential(
                    nn.Conv2d(256, base_channel * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel * 2),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        self.stage2 = nn.Sequential(
            StageModule(2, 2, base_channel)
        )

        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),


            nn.Sequential( # 适配预训练模型
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel * 4),
                    nn.ReLU(inplace=True)
                ))
            ]
        )

        self.stage3 = nn.Sequential(
            StageModule(3, 3, base_channel),
            StageModule(3, 3, base_channel),
            StageModule(3, 3, base_channel),
            StageModule(3, 3, base_channel),
        )

        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),

            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channel * 8),
                    nn.ReLU(inplace=True)
                ))
            ]
        )

        self.stage4 = nn.Sequential(
            StageModule(4, 4, base_channel),
            StageModule(4, 4, base_channel),
            StageModule(4, 1, base_channel),
        )


        self.final_layer = nn.Conv2d(base_channel, key_points, 1, 1, 0)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        x = [transition(x) for transition in self.transition1]

        x = self.stage2(x)

        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]

        x = self.stage3(x)
        
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]

        x = self.stage4(x)

        x = self.final_layer(x[-1])

        return x


from utils import get_max_preds
class Predictor:
    def __init__(self, model, config: Config):
        self.model = model
        self.input_shape = config['input_shape']
        self.device = config['device']
        self.model.eval()
        self.model = self.model.to(self.device)

        self.stride = [4, 4] # 默认的步长

    def decode_outputs(self, keypoints, scale, dx, dy):
        """
        对关键点进行反向 letterbox 操作，将其还原到原图坐标。
        
        参数:
            keypoints (list of tuple): 每个关键点的 (x, y) 坐标列表。
            scale (float): letterbox 操作的缩放比例。
            dx (int): 水平填充的像素数。
            dy (int): 垂直填充的像素数。
            
        返回:
            original_keypoints (list of tuple): 还原到原图坐标的关键点列表。
        """
        original_keypoints = []
        for x, y in keypoints:
            # 去除填充影响并按比例还原
            original_x = int((x - dx) / scale)
            original_y = int((y - dy) / scale)
            original_keypoints.append((original_x, original_y))
        
        return original_keypoints
        
    def preprocess(self, image: Image.Image) -> Tensor:
        image = image.convert('RGB')
        img_pil, scale, dx, dy = letterbox(image, self.input_shape)

        tensor = F.to_tensor(img_pil)
        tensor = F.normalize(tensor, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0)

        return tensor, scale, dx, dy
    def __call__(self, image: Image.Image):
        input_tensor, scale, dx, dy = self.preprocess(image)

        with torch.inference_mode():
            
            outputs = self.model(input_tensor.to(self.device)).cpu()

        key_points, maxvals = get_max_preds(outputs)

        maxvals = maxvals[0]
        key_points = (key_points[0] * torch.Tensor(self.stride)).type(torch.uint8)
        
        key_points = self.decode_outputs(key_points.numpy(), scale, dx, dy)
        return key_points, maxvals.numpy()
def get_model(config):

        
    device = config['predict.device']
    base_channel = config['base_channel']
    num_key_points = config['num_key_points']
    
    model_path = config['predict.model_path']
    model = HRNet(base_channel, num_key_points)
    
    dict_data = torch.load(model_path)
    if "model" in dict_data: 
        weights_dict = dict_data['model']
    else:
        weights_dict = dict_data
    

        
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

    print(f"missing_keys: {missing_keys}")
    return model.to(device)

if __name__ == "__main__":

    base_channel = 18
    model = HRNet(base_channel)
        
    model = HRNet(base_channel, key_points=17)
    
    if os.path.exists(f"model_data/hrnet_w{base_channel}_pretrained.pth"):
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(f"model_data/hrnet_w{base_channel}_pretrained.pth"), strict=False)
        # 打印未匹配的参数（帮助调试）
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    pass
