import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
import json
import random
import os
from taylor_transforms import (Normalizer,
                               Contrast,
                               RandomCrop,
                               Rescale,
                               Hue,
                               Saturation,
                               Brightness)


class ImageDataset(Dataset):
    def __init__(self,
                 json_path,
                 transform,
                 purpose):
        with open(json_path, 'r') as json_f:
            json_dict = json.load(json_f)
        self.annotations = json_dict['data']
        random.shuffle(self.annotations)
        self.label_map = json_dict['info']

        self.transform = transform
        self.purpose = purpose

        print('number of ' + self.purpose + ' files : ', len(self.annotations))

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        image = Image.open(annot['image_dir']).convert('RGB')
        label = self.label_map[annot['label']]

        sample = {'image': image, 'label': label}
        if self.purpose == 'train':
            sample = self.transform(sample)
        elif self.purpose == 'val':
            sample = self.transform(sample)
        else:
            raise ValueError('Purpose should be train or val.\n')

        return sample

    def __len__(self):
        return len(list(self.annotations))


def get_image_dataloader(json_path,
                         batch_size,
                         num_workers,
                         purpose,
                         transform=None):
    if not transform:
        transform = transforms.Compose([Rescale(size=(380, 380)),
                                        Normalizer()])

    dataset = ImageDataset(json_path,
                           transform=transform,
                           purpose=purpose)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)
    return dataloader


if __name__ == "__main__":
    json_path = '/home/taylor/spo_classification/imagenet_small/train_annotations.json'
    dataloader = get_image_dataloader(json_path, batch_size=1, num_workers=1, purpose='train')

    for i, data in enumerate(dataloader):
        image = data['image']
        label = data['label']

        print(image.shape)
        print(label.shape)
