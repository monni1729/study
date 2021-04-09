from PIL import Image
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
import json
import torch
import os
import random
import cv2
import torchvision.models as models
import torch.nn as nn


class AlbumentationDataset(Dataset):
    def __init__(self, transform):
        self.image_list = os.listdir('/home/taylor/spo_classification/thumbnail/datasets/thumb/')
        self.transform = transform
        self.image_list.sort()

    def __getitem__(self, idx):
        original_image_dir = os.path.join('/home/taylor/spo_classification/thumbnail/datasets/original', self.image_list[idx])
        thumb_image_dir = os.path.join('/home/taylor/spo_classification/thumbnail/datasets/thumb', self.image_list[idx])
        label = np.array(1, dtype=np.long)

        original_image = Image.open(original_image_dir).convert('RGB')
        original_image = self.transform(image=np.array(original_image))['image']
        original_image = np.transpose(original_image, (2, 0, 1))

        thumb_image = Image.open(thumb_image_dir).convert('RGB')
        thumb_image = self.transform(image=np.array(thumb_image))['image']
        thumb_image = np.transpose(thumb_image, (2, 0, 1))

        label = torch.from_numpy(label)

        return torch.tensor(original_image, dtype=torch.float32), torch.tensor(thumb_image, dtype=torch.float32), label

    def __len__(self):
        return len(self.image_list)


def get_image_dataloader(batch_size, num_workers):
    transform = albumentations.Compose([albumentations.Resize(224, 224), albumentations.Normalize()])
    dataset = AlbumentationDataset(transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    return dataloader


def validate(model, dataloader):
    model.eval()
    pred1_np = np.zeros((11350, 2048), dtype=np.float32)
    pred2_np = np.zeros((11350, 2048), dtype=np.float32)
    with torch.no_grad():
        for i, (image1, image2, label) in enumerate(dataloader):
            print(i)
            image1 = image1.cuda()
            image2 = image2.cuda()

            pred1 = model(image1).detach().cpu().numpy()
            pred2 = model(image2).detach().cpu().numpy()
            
            
            pred1_np[i*10: (i+1)*10, :] = pred1
            pred2_np[i*10: (i+1)*10, :] = pred2
    
    np.save('original_vector.npy', pred1_np)
    np.save('thumb_vector.npy', pred2_np)
        
def main():

    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
    
    dataloader = get_image_dataloader(batch_size=10, num_workers=4)    
    validate(model, dataloader)
        
        
if __name__ == '__main__':
    main()

