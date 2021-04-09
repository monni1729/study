import os
import json
import torch
import numpy as np
import albumentations as al
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 이미지들의 정보나 변환 정보 담겨있는 클래스
class AlbumentationDataset(Dataset):
    def __init__(self, json_data, transform):
        self.annotations = json_data
        self.transform = transform

    def __getitem__(self, idx):
        annot = self.annotations[idx]

        image = Image.open(annot['image_dir']).convert('RGB')
        image = self.transform(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1))

        label = annot['category']
        label = np.array(label, dtype=np.long)
        label = torch.from_numpy(label)

        return torch.tensor(image, dtype=torch.float32), label

    def __len__(self):
        return len(self.annotations)


def get_image_dataloader(annot_dir, batch_size, num_workers):
    train_transform = al.Compose([al.HorizontalFlip(),
                            al.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
                            al.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                            al.Normalize()])
                            
    val_transform = al.Compose([al.HorizontalFlip(),
                            al.Normalize()])
                            
    with open(os.path.join(annot_dir, 'manual_train_annotations.json'), 'r') as jf:
        train_json = json.load(jf)

    with open(os.path.join(annot_dir, 'manual_val_annotations.json'), 'r') as jf:
        val_json = json.load(jf)

    max_num = 0
    for t in train_json:
        max_num = max(max_num, t['category'])
    for v in val_json:
        max_num = max(max_num, v['category'])

    train_dataset = AlbumentationDataset(train_json, train_transform)
    val_dataset = AlbumentationDataset(val_json, val_transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=False)

    return train_dataloader, val_dataloader, max_num + 1


if __name__ == '__main__':
    print()
