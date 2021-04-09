import torch
import tqdm
import json
import time
import os
import random
import argparse
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms as T
import torch_optimizer as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from warmup import build_lr_scheduler
from taylor_transforms import (Normalizer,
                               Contrast,
                               RandomCrop,
                               Rescale,
                               Hue,
                               Saturation,
                               Brightness)

SAVE = False


class LSCE(torch.nn.Module):
    def __init__(self):
        super(LSCE, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str)

    parser.add_argument('--batch_size', type=int, default=(3 * 64))
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--checkpoint_dir', type=str, default='trained_weights')
    parser.add_argument('--checkpoint_interval', type=int, default=1)

    args = parser.parse_args()

    return args


def fetch_model(num_classes):
    model = models.resnet50(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(n_inputs, num_classes))

    return model


def train(model, train_dataloader, optim, criterion, epoch, scheduler):
    model.train()
    loss_list, acc_sum = [], 0

    progress_bar = tqdm.tqdm(train_dataloader, leave=False)
    for i, data in enumerate(progress_bar):
        image = data['image'].cuda()
        label = data['label'].cuda()

        optim.zero_grad()
        pred = model(image)

        loss = criterion(pred, label)
        loss.backward()
        optim.step()
        if scheduler:
            scheduler.step()

        loss_list.append(loss.item())

        pred = pred.argmax(1)
        corrects = (pred == label)
        acc = corrects.sum()
        acc_sum += acc

        progress_bar.update()
        progress_bar.set_description(
            'Epoch [{:3}]  Loss [{:02.4f}]  Acc [{:02.4f}] '.format(epoch, loss, acc_sum / len(data['label'])))

        if i % 10 == 0:
            pass
            # tensorboard

    print('Epoch [{:3}]  '.format(epoch), end='')
    print('Loss [{:02.4f}]  '.format(sum(loss_list) / len(loss_list)), end='')
    print('Acc [{:02.4f}]  '.format(acc_sum.item() / 11241324))

    return sum(loss_list) / len(loss_list), acc_sum.item() / 11241324


def validate(model, val_dataloader, criterion, epoch, val_total):
    model.eval()
    loss_list, acc_sum = [], 0
    for i, data in enumerate(val_dataloader):
        image = data['image'].cuda()
        label = data['label'].cuda()

        pred = model(image)
        loss = criterion(pred, label)
        loss_list.append(loss.item())

        pred = pred.argmax(1)
        corrects = (pred == label)
        acc = corrects.sum()
        acc_sum += acc

    print('Epoch [{:3}]  '.format(epoch), end='')
    print('Validation Loss [{:02.4f}]  '.format(sum(loss_list) / len(loss_list)), end='')
    print('Validation Acc [{:02.4f}]  '.format(acc_sum.item() / 2801723))

    return sum(loss_list) / len(loss_list), acc_sum.item() / 2801723


def main():
    args = get_args()

    batch_size = args.batch_size
    num_epoch = args.num_epoch
    num_workers = args.num_workers
    learning_rate = args.lr
    train_annotations = os.path.join(args.root_dir, 'train_annotations.json')
    val_annotations = os.path.join(args.root_dir, 'val_annotations.json')

    with open(os.path.join(args.root_dir, 'label_map.json'), 'r') as json_f:
        label_map = json.load(json_f)
    '''
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.project)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    '''
    transform = T.Compose([Rescale((256, 256)),
                           Normalizer()])

    train_dataloader = get_image_dataloader(json_path=train_annotations,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            purpose='train',
                                            transform=transform)

    val_dataloader = get_image_dataloader(json_path=val_annotations,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          purpose='val',
                                          transform=transform)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = fetch_model(len(label_map))
    model = torch.nn.DataParallel(model).cuda()
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion = LSCE().cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)


    scheduler = build_lr_scheduler(optim,
                                   scheduler='poly',
                                   iter_per_epoch=len(train_dataloader),
                                   max_epoch=100,
                                   warmup=True,
                                   warmup_epoch=10)
    # optimizer = optim.Yogi(
    #     m.parameters(),
    #     lr= 1e-2,
    #     betas=(0.9, 0.999),
    #     eps=1e-3,
    #     initial_accumulator=1e-6,
    #     weight_decay=0,
    # )

    for epoch in range(num_epoch):
        train_loss, train_acc = train(model, train_dataloader, optim, criterion, epoch, scheduler)
        val_loss, val_acc = validate(model, val_dataloader, criterion, epoch)
        '''
        print('Epoch [{:3}]  '.format(epoch), end='')
        print('Loss [{:02.4f}]  '.format(train_loss), end='')
        print('Acc [{:02.4f}]  '.format(train_acc), end='')
        print('Validation Loss [{:02.4f}]  '.format(val_loss), end='')
        print('Validation Acc [{:02.4f}]  '.format(val_acc))
        '''

        fn = 'resnet50_large' + '_' + str(int(time.time())) + '.pth'
        torch.save(model.state_dict(), fn)


class ImageDataset(Dataset):
    def __init__(self,
                 json_path,
                 transform,
                 purpose):
        with open(json_path, 'r') as json_f:
            json_dict = json.load(json_f)
        self.annotations = json_dict
        random.shuffle(self.annotations)

        with open('/mnt/sdc/imagenet_fall11/label_map.json', 'r') as json_f:
            self.label_map = json.load(json_f)

        self.transform = transform
        self.purpose = purpose

        print('number of ' + self.purpose + ' files : ', len(self.annotations))

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        image = Image.open(annot['image_dir']).convert('RGB')
        label = annot['category']

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
        transform = T.Compose([Rescale(size=(380, 380)),
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


if __name__ == '__main__':
    main()
