from torch import nn, Tensor
from torch.nn import MSELoss
import torch

import torch
import torch.nn as nn
from torch import Tensor

class ComputeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, outputs: Tensor, heatmaps: Tensor, kps_weights: Tensor) -> Tensor:
        
        bs = outputs.shape[0]
        
        # 计算每个关键点的 MSE 损失
        loss = self.mse(outputs, heatmaps)  # [bs, num_kps, H, W]

        # 对 H 和 W 维度求平均，并乘以每个关键点的权重
        loss = (loss.mean(dim=[2, 3]) * kps_weights).sum(dim=-1)
        
        # 防止除零的风险
        valid_kps_count = kps_weights.sum(dim=-1)
        # 添加一个小的常数以避免除零
        loss = loss / (valid_kps_count + 1e-6)

        # 最后对所有批次求和并归一化
        return loss.sum() / bs

    


        