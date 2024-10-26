
from tqdm import tqdm
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import cv2
import matplotlib.pyplot as plt
import math
def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals





def visualize_keypoints(image, ann):
    
    
    visible = ann['visible']
    
    keypoints = ann['keypoints']

    # 定义可视化参数
    for i, (kpt, vis) in enumerate(zip(keypoints, visible)):
        x, y = kpt
        if vis == 1:  # 可见的关键点
            color = (0, 255, 0)  # 绿色
            radius = 5
        elif vis == 2:  # 不确定的关键点
            color = (0, 255, 255)  # 黄色
            radius = 5
        else:  # 不可见的关键点
            continue
            color = (0, 0, 255)  # 红色
            radius = 3
        
        # 绘制关键点
        cv2.circle(image, (x, y), radius, color, -1)
        # 添加关键点的索引标签
        cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示图像
    cv2.imshow('Keypoints Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def eval_one_epoch(writer: SummaryWriter, trainer, dataLoader, epoch, EPOCH, cuda, eval_steps):
    print('Start Eval')
    trainer.eval()
    trainer.clear_loss()
    
    total_step = eval_steps * epoch
    with tqdm(total=eval_steps, desc=f'Epoch {epoch + 1} / {EPOCH}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataLoader):

            if cuda:
                # 如果 batch 是元组或列表，可以将每个元素都转移到 GPU 上
                batch = [item.cuda(non_blocking=True) for item in batch]

            with torch.no_grad():
                losses, _ = trainer(*batch)
                result_dict: dict = trainer.get_result_dict(losses)
            
            for k, v in result_dict.items():
                writer.add_scalar(f"loss/eval_{k}", v / (iteration + 1), total_step + iteration)

            
            pbar.set_postfix(**{k: v / (iteration + 1) for k, v in result_dict.items()})
            pbar.update(1)

    return trainer.total_loss / len(dataLoader)
def fit_one_epoch(writer: SummaryWriter, trainer, dataLoader, epoch, EPOCH, cuda, epoch_steps):
    print('Start Train')
    trainer.train()
    trainer.clear_loss()

    total_step = epoch_steps * epoch
    with tqdm(total=epoch_steps, desc=f'Epoch {epoch + 1} / {EPOCH}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataLoader):
            
            if iteration >= epoch_steps:
                print(iteration) 
                pass

            if cuda:
                # 如果 batch 是元组或列表，可以将每个元素都转移到 GPU 上
                batch = [item.cuda(non_blocking=True) for item in batch]


            losses = trainer.train_step(*batch)
            result_dict: dict = trainer.get_result_dict(losses)
            
            for k, v in result_dict.items():
                writer.add_scalar(f"loss/train_{k}", v / (iteration + 1), total_step + iteration)

            writer.add_scalar("lr", trainer.get_lr(), total_step + iteration)
            
            pbar.set_postfix(**{k: v / (iteration + 1) for k, v in result_dict.items()})
            pbar.update(1)
  