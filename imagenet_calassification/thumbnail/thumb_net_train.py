from thumb_net import ThumbNet

import torch
import tqdm
import json
import time
import os
import random
import argparse
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from thumb_dataloader import get_image_dataloader
from collections import OrderedDict



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str)

    parser.add_argument('--batch_size', type=int, default=16)  # 256 for resnet50
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    return args



def fetch_model(num_classes):
    model = models.resnet50(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(n_inputs, num_classes))

    state_dict = torch.load('trained_weights/resnet50_large_1605496700.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    return model


def train(model, train_dataloader, optim, criterion, epoch):
    model.train()
    loss_list, acc_sum = [], 0
    for i, (original_image, thumb_images, label) in enumerate(train_dataloader):
        print(i)
        original_image = original_image.cuda()
        thumb_images = thumb_images.cuda()
        label = label.cuda()

        optim.zero_grad()
        pred = model(original_image, thumb_images)

        loss = criterion(pred, label)
        loss.backward()
        optim.step()

        loss_list.append(loss.item())

        print(loss.item())


def validate(model, val_dataloader, criterion, epoch, val_len, train_len):
    model.eval()
    loss_list, acc_sum = [], 0
    acc1s = []
    acc5s = []
    max_iter = len(val_dataloader)
    with torch.no_grad():
        for i, (image, label) in enumerate(val_dataloader):
            image = image.cuda()
            label = label.cuda()

            pred = model(image)
            loss = criterion(pred, label)
            loss_list.append(loss.item())

            acc1, acc5 = accuracy(pred, label, topk=(1, 5))
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            summary.add_scalar('loss_val', loss, i + 1 + max_iter * epoch)
            summary.add_scalar('acc1_val', acc1.item(), i + 1 + max_iter * epoch)
            summary.add_scalar('acc5_val', acc5.item(), i + 1 + max_iter * epoch)
        print('Epoch [{:3}] Val_Loss [{:02.4f}] Val_Acc1 [{:02.4f}] Val_Acc5 [{:02.4f}]'.format(epoch,
                                                                                                sum(loss_list) / len(
                                                                                                    loss_list),
                                                                                                sum(acc1s) / len(
                                                                                                    loss_list),
                                                                                                sum(acc5s) / len(
                                                                                                    loss_list)))


def main():
    args = get_args()

    batch_size = args.batch_size
    num_epoch = args.num_epoch
    num_workers = args.num_workers
    learning_rate = args.lr


    train_dataloader = get_image_dataloader(batch_size=batch_size,
                                            num_workers=num_workers)

    model = ThumbNet([2, 2, 2, 2])
    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)

    for epoch in range(num_epoch):
        train(model, train_dataloader, optim, criterion, epoch)
        # validate(model, val_dataloader, criterion, epoch, val_len, train_len)


if __name__ == '__main__':
    main()
