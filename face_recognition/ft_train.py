import os
import json
import tqdm
import time
import torch
import argparse
import numpy as np
import albumentations as al
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from utils import accuracy, custom_range
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weights/backbone_1611738287.pth')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--annot_dir', type=str, default='/home/taylor/pipeline/cropped_images')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=20)
    parser.add_argument('--val', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=512)
    args = parser.parse_args()

    return args


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


def get_image_dataloader(annot_dir, batch_size, num_workers, way, shot, val):
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

    all_json = list(val_json)

    cnt_dict = dict()
    for a in all_json:
        if a['category'] in cnt_dict.keys():
            cnt_dict[a['category']] += 1
        else:
            cnt_dict[a['category']] = 1

    candidate = list()
    for k, v in cnt_dict.items():
        if v >= shot + val:
            candidate.append(k)
        if len(candidate) == way:
            break

    if len(candidate) < way:
        raise ValueError('too much shot value or way')

    realign = dict()
    for idx, c in enumerate(candidate):
        realign[c] = idx

    shot_json = []
    val_json = []
    shot_dict = dict()
    val_dict = dict()
    for c in candidate:
        shot_dict[c] = shot
        val_dict[c] = val

    for a in all_json:
        if a['category'] in candidate and shot_dict[a['category']] > 0:
            shot_json.append(a)
            shot_dict[a['category']] -= 1
        elif a['category'] in candidate and val_dict[a['category']] > 0:
            val_json.append(a)
            val_dict[a['category']] -= 1

    for i in range(len(shot_json)):
        shot_json[i]['category'] = realign[shot_json[i]['category']]

    for i in range(len(val_json)):
        val_json[i]['category'] = realign[val_json[i]['category']]

    print('way :', way, ' shot :', shot, ' val :', val)

    train_dataset = AlbumentationDataset(shot_json, train_transform)
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

    return train_dataloader, val_dataloader


def train(backbone, head, train_dataloader, optim, criterion, epoch):
    backbone.eval()
    head.train()

    loss_list, acc_sum = [], 0
    cr_sv = custom_range(len(train_dataloader), 1)
    progress_bar = tqdm.tqdm(train_dataloader, leave=False)
    for i, (image, label) in enumerate(progress_bar):
        image = image.cuda()
        label = label.cuda()
        optim.zero_grad()

        embedding = backbone(image)
        pred = head(embedding)
        loss = criterion(pred, label)

        loss.backward()
        optim.step()

        loss_list.append(loss.item())

        acc1 = accuracy(pred, label)
        progress_bar.update()
        progress_bar.set_description(
            'Epoch [{:3}] Loss [{:02.4f}] Acc1 [{:02.4f}]]'.format(epoch, loss, acc1.item()))

        if i + 1 in cr_sv:
            print()

    if epoch % 10 == 9:
        fn = 'weights/new_head_ ' + str(int(time.time())) + '.pth'
        torch.save(head.state_dict(), fn)


def validate(backbone, head, val_dataloader, criterion, epoch):
    backbone.eval()
    head.eval()

    loss_list, acc_sum = [], 0
    acc1s = []
    with torch.no_grad():
        for i, (image, label) in enumerate(val_dataloader):
            image = image.cuda()
            label = label.cuda()

            embedding = backbone(image)
            pred = head(embedding)
            loss = criterion(pred, label)

            loss_list.append(loss.item())

            acc1 = accuracy(pred, label)
            acc1s.append(acc1.item())

        print('Epoch [{:3}] Val_Loss [{:02.4f}] Val_Acc1 [{:02.4f}]'.format(epoch,
                                                                            sum(loss_list) / len(loss_list),
                                                                            sum(acc1s) / len(loss_list)))
        return sum(loss_list) / len(loss_list)


def main():
    args = get_args()
    lr = args.lr
    num_epoch = args.num_epoch
    weight = args.weight
    way = args.way
    shot = args.shot
    val = args.val
    embedding_size = args.embedding_size
    train_dataloader, val_dataloader = get_image_dataloader(annot_dir=args.annot_dir,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            way=way, shot=shot, val=val)

    backbone = Backbone(50, 0.5, 'ir_se')
    head = torch.nn.Sequential(torch.nn.Linear(embedding_size, way))

    state_dict = torch.load(weight)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    backbone.load_state_dict(new_state_dict)

    for param in backbone.parameters():
        param.requires_grad = False

    backbone = torch.nn.DataParallel(backbone).cuda()
    head = torch.nn.DataParallel(head).cuda()

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, head.parameters()), lr)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=3,
                                                           min_lr=1e-7,
                                                           verbose=True)

    for epoch in range(num_epoch):
        train(backbone, head, train_dataloader, optim, criterion, epoch)
        val_loss = validate(backbone, head, val_dataloader, criterion, epoch)
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
