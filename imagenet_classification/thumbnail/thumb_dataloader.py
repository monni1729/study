from PIL import Image
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
import json
import torch
import os
import random
import cv2

class AlbumentationDataset(Dataset):
    def __init__(self, transform_origianl, transform_thumb):
        self.correct_ratio = 0.5
        self.image_list = os.listdir('/home/taylor/spo_classification/thumbnail/datasets/thumb/')
        self.transform_origianl = transform_origianl
        self.transform_thumb = transform_thumb
        self.error_list = []

    def __getitem__(self, idx):
        # print(idx)
        original_image_dir = os.path.join('/home/taylor/spo_classification/thumbnail/datasets/original',
                                          self.image_list[idx])
        if random.random() > self.correct_ratio:
            thumb_image_dir = os.path.join('/home/taylor/spo_classification/thumbnail/datasets/thumb',
                                           self.image_list[int(len(self.image_list)*random.random())-1])
            label = np.array(0, dtype=np.long)
        else:
            thumb_image_dir = os.path.join('/home/taylor/spo_classification/thumbnail/datasets/thumb',
                                           self.image_list[idx])
            label = np.array(1, dtype=np.long)

        # original_image = cv2.imread(original_image_dir)
        original_image = Image.open(original_image_dir).convert('RGB')
        original_image = self.transform_origianl(image=np.array(original_image))['image']
        if original_image is None:
            print(idx)
        original_image = np.transpose(original_image, (2, 0, 1))

        # thumb_image = cv2.imread(thumb_image_dir)
        thumb_image = Image.open(thumb_image_dir).convert('RGB')
        thumb_image = self.transform_thumb(image=np.array(thumb_image))['image']
        if thumb_image is None:
            print(idx)
        # thumb_image = self.transform(image=np.array(thumb_image))['image']
        thumb_image = np.transpose(thumb_image, (2, 0, 1))

        label = torch.from_numpy(label)

        return torch.tensor(original_image, dtype=torch.float32), torch.tensor(thumb_image, dtype=torch.float32), label

    def __len__(self):
        return len(self.image_list)


def get_image_dataloader(batch_size,
                         num_workers):
    transform_original = albumentations.Compose([albumentations.Resize(600, 600),
        albumentations.Normalize()])
    transform_thumb = albumentations.Compose([albumentations.Resize(100, 100),
        albumentations.Normalize()])
    dataset = AlbumentationDataset(transform_original, transform_thumb)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    return dataloader


if __name__ == '__main__':
    '''
    image_list = os.listdir('/home/taylor/spo_classification/thumbnail/datasets/thumb/')
    print(image_list[34])
    '''
    dataloader = get_image_dataloader(1, 1)

    for i, (original_image, thumb_images, label) in enumerate(dataloader):
        # original_image = original_image.cuda()
        # thumb_images = thumb_images.cuda()
        # label = label.cuda()
        a= 1

