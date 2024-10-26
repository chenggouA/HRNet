from torch.utils.data import Dataset
import os
import json
import cv2
import torch


def collate_fn(batch):
    
    images, target = tuple(zip(*batch))
    images = torch.stack(images)
    heatmaps = torch.stack([t['heatmap'] for t in target])
    kps_weights = torch.stack([t['kps_weights'] for t in target])
    
    return images, heatmaps, kps_weights
class keypoints_dataset(Dataset):
    
    def __init__(self, data_root, transforms = None):
        self.data_root = data_root
        self.transforms = transforms

        self.annotations = [os.path.join(self.data_root, "annotations", item) for item in os.listdir(os.path.join(self.data_root, "annotations"))]
        
        self.annotations = self.load_annotations()
        self.images = [os.path.join(self.data_root, "images", item) for item in os.listdir(os.path.join(self.data_root, "images"))]
    

    def load_annotations(self):
        
        annotations = []
        for item in self.annotations:
            with open(item,'r') as f:
                annotations.append(json.load(f))
        
        return annotations
    def load_img(self, index):
        img = cv2.imread(self.images[index])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def __getitem__(self, index):
        label = self.annotations[index]
        image = self.load_img(index)
        
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        
        return image, label

    def __len__(self):
        return len(self.annotations)
    



if __name__ == "__main__":
    from transforms import *

    transform = Compose([
        LetterBox((256, 192)), KeypointToHeatMap(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = keypoints_dataset(r"D:\code\dataset\person_keypoints\train", transform)
    
    for item in dataset:
        continue
