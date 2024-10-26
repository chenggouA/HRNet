import torch.utils
import torch.utils.data
from tools.config import Config, load_config
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from tools.train import set_seed
from tools.train import EarlyStopping
from tools.sys import create_folder_with_current_time
import os
from utils import fit_one_epoch, eval_one_epoch
from trainer import Trainer
from loss import ComputeLoss
from dataset import keypoints_dataset, collate_fn
import torch
from hrnet import HRNet 
from tools.lr_scheduler import YOLOXCosineLR
from transforms import *

torch.backends.cudnn.benchmark = True



def train(config: Config, output, writer: SummaryWriter, train_dataLoader, val_dataLoader, EPOCH, Init_lr_fit, Min_lr_fit, epoch_steps, eval_steps):
   
    save_interval = config['train.save_interval']
    resume = config['train.resume']
    device = config['train.device']
    optimizer_type = config['train.optimizer']
    momentum = config['train.momentum']
    start_epoch = 0
    base_channel = config['base_channel']
    pretrained = config['train.pretrained']

    
    cuda = False
    if device == "cuda":
        cuda = True
    

    
    
    def get_optimizer(model):
        params = [p for p in model.parameters() if p.requires_grad ]
        optimizer = {
                    'adam'  : optim.Adam(params, Init_lr_fit, betas = (momentum, 0.999), weight_decay = 0),
                    'sgd'   : optim.SGD(params, Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = 0)
        }[optimizer_type]

        return optimizer


    # 早停策略
    earlyStopping = EarlyStopping(output, save_interval, EPOCH, verbose=True)
    
    model = HRNet(base_channel, key_points=config['num_key_points'])
    
    if pretrained and os.path.exists(f"model_data/hrnet_w{base_channel}_pretrained.pth"):
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(f"model_data/hrnet_w{base_channel}_pretrained.pth"), strict=False)
        # 打印未匹配的参数（帮助调试）
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    model = model.to(device)
    trainer = Trainer(device, model, ComputeLoss())
    
    # 获得优化器
    optimizer = get_optimizer(model)

    #   获得学习率下降的公式
    lr_scheduler = YOLOXCosineLR(optimizer, Init_lr_fit, Min_lr_fit, epoch_steps * EPOCH)
    
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

        # 如果需要恢复训练，判断是否已经进入解冻阶段
        start_epoch = checkpoint['epoch'] + 1
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if device == "cuda":
            # 增加以下几行代码，将optimizer里的tensor数据全部转到GPU上
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()


        

    trainer.set_optimizer_and_lr_scheduler(optimizer, lr_scheduler)
    
    for epoch in range(start_epoch, EPOCH):
        
        fit_one_epoch(writer, trainer, train_dataLoader, epoch, EPOCH, cuda, epoch_steps)
        eval_loss = eval_one_epoch(writer, trainer, val_dataLoader, epoch, EPOCH, cuda, eval_steps)
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_scheduler': lr_scheduler.state_dict()
        }
        
        earlyStopping(eval_loss, save_files, epoch)
        if earlyStopping.early_stop:
            print(f"[{epoch}/{EPOCH}], acc={0.1} Stop!!!")
            break
    writer.close()

def main(config: Config):


    EPOCH = config['train.epoch']
    Init_lr = config['train.Init_lr']
    optimizer_type = config['train.optimizer']


    seed = config['seed']
    root_data = config['train.root_data']
    batch_size = config['train.batch_size']
    num_workers = config['train.num_workers']
    output = config['train.output']
    input_shape = config['input_shape']
    
    output = create_folder_with_current_time(output)


    set_seed(seed)

    

    # 初始化日志
    writer = SummaryWriter(log_dir=os.path.join(output, "log"))

    
    Min_lr = Init_lr * 0.01
 
    nbs             = config['train.nbs']
    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 获取数据集
    train_transforms = Compose([
        LetterBox(input_shape), KeypointToHeatMap(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = keypoints_dataset(f"{root_data}/train", train_transforms) 
    train_dataLoader = DataLoader(train_dataset, 
                                batch_size = batch_size, 
                                num_workers = num_workers, 
                                pin_memory = True,
                                collate_fn = collate_fn,
                                drop_last = True)
    
    
    val_dataset = keypoints_dataset(f"{root_data}/val", train_transforms)

    val_dataLoader = DataLoader(val_dataset, 
                                batch_size = batch_size, 
                                num_workers = num_workers, 
                                pin_memory = True,
                                collate_fn = collate_fn,
                                drop_last = True)
    
    epoch_steps = len(train_dataset) // batch_size
    eval_steps = len(val_dataset) // batch_size
    
    train(config, output, writer, train_dataLoader, val_dataLoader, EPOCH, Init_lr_fit, Min_lr_fit, epoch_steps, eval_steps)
    

if __name__ == "__main__":
    main(load_config("application.yaml"))

