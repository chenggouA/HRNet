
from tools.baseTrainer import base
from torch import Tensor

class Trainer(base):
    
    def __init__(self, device, model, loss_fn):
        super().__init__(device, model, loss_fn)
        self.total_loss = 0.0
        self.clear_loss()

    def clear_loss(self):
        self.total_loss = 0.0

    def model_forward(self, input: Tensor):
        output = super().model_forward(input)
        if isinstance(output, tuple):
            return output[0]
        
        return output

    def forward(self, imgs, *args, **kwargs):
        
        outputs = self.model(imgs)
        losses = self.loss_fn(outputs, *args, **kwargs)

        if not isinstance(losses, list):
            losses = [losses]
        return losses, outputs
    
    def get_result_dict(self, losses):
        result_dict = dict()
        self.total_loss += losses[0].item()
        result_dict['total_loss'] = self.total_loss

        return result_dict
 
    
from tools.config import Config
from hrnet import HRNet
import torch 

def get_model(config: Config):
    
    device = config['predict.device'] 
    model_path = config['predict.model_path']

    backbone = config['backbone']

    model = HRNet(backbone)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.to(device)

    model.eval()

    return model
        