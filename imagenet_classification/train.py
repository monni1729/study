import torch
import tqdm
import json
import time
import os
import argparse
from Dataloader import get_image_dataloader
from EfficientNet_PyTorch.efficientnet_pytorch.model import EfficientNet
from ResNeSt_PyTorch.resnest.torch import resnest50, resnest101, resnest200

SAVE = False




def get_args():
    '''
    Supported model : EfficientNet-b0 ~ b6
                      ResNeSet50, ResNeSet101, ResNeSt200
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--validation', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--checkpoint_dir', type=str, default='trained_weights')
    parser.add_argument('--checkpoint_interval', type=int, default=1)

    args = parser.parse_args()



    return args


def fetch_model(model_name, num_classes):
    if model_name.startswith('efficientnet'):
        model = EfficientNet.from_pretrained(model_name=model_name,
                                             weights_path='/home/taylor/spo_classification/pretrained_weights/' + model_name + '.pth',
                                             num_classes=num_classes)
    elif model_name.startswith('resnest'):
        if model_name[7:] == '50':
            model = resnest50(pretrained=False)
        elif model_name[7:] == '101':
            model = resnest101(pretrained=False)
        elif model_name[7:] == '200':
            model = resnest200(pretrained=False)
        else:
            raise NameError(model_name, ' : inaccurate model name')

        weight_path = '/home/taylor/spo_classification/pretrained_weights/' + model_name + '.pth'
        model.load_state_dict(torch.load(weight_path))

        n_inputs = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Linear(n_inputs, num_classes))

    else:
        raise NameError('model should be efficientnet-bN or resnestNNN')

    return model


def train(model, train_dataloader, optim, criterion, epoch, train_total):
    model.train()
    loss_list, acc_sum = [], 0

    progress_bar = tqdm.tqdm(train_dataloader, leave=False)
    # progress_bar = train_dataloader
    for i, data in enumerate(progress_bar):
        image = data['image'].cuda()
        label = data['label'].cuda()

        optim.zero_grad()
        pred = model(image)

        loss = criterion(pred, label)
        loss.backward()
        optim.step()
        loss_list.append(loss.item())

        pred = pred.argmax(1)
        corrects = (pred == label)
        acc = corrects.sum()
        acc_sum += acc

        progress_bar.update()
        progress_bar.set_description('Epoch [{:3}]  Loss [{:02.4f}]  '.format(epoch, loss))

    print('Epoch [{:3}]  '.format(epoch), end='')
    print('Loss [{:02.4f}]  '.format(sum(loss_list) / len(loss_list)), end='')
    print('Acc [{:02.4f}]  '.format(acc_sum.item() / train_total))

    return sum(loss_list) / len(loss_list), acc_sum.item() / train_total


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
    print('Validation Acc [{:02.4f}]  '.format(acc_sum.item() / val_total))

    return sum(loss_list) / len(loss_list), acc_sum.item() / val_total


def main():
    args = get_args()

    batch_size = args.batch_size
    num_epoch = args.num_epoch
    num_workers = args.num_workers
    learning_rate = args.lr
    train_annotations = os.path.join(args.root_dir, 'train_annotations.json')
    val_annotations = os.path.join(args.root_dir, 'val_annotations.json')
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.project)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(val_annotations):
        val_annotations = False

    with open(train_annotations, 'r') as json_f:
        json_dict = json.load(json_f)
    class_info = json_dict['info']
    num_classes = len(class_info)

    train_dataloader = get_image_dataloader(json_path=train_annotations,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            purpose='train')
    train_total = sum([len(data['image']) for data in train_dataloader])

    if val_annotations:
        val_dataloader = get_image_dataloader(json_path=val_annotations,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              purpose='val')
        val_total = sum([len(data['image']) for data in val_dataloader])

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = fetch_model(args.model, num_classes)
    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)

    for epoch in range(num_epoch):
        if val_annotations:
            train_loss, train_acc = train(model, train_dataloader, optim, criterion, epoch, train_total)
            val_loss, val_acc = validate(model, val_dataloader, criterion, epoch, val_total)
            '''
            print('Epoch [{:3}]  '.format(epoch), end='')
            print('Loss [{:02.4f}]  '.format(train_loss), end='')
            print('Acc [{:02.4f}]  '.format(train_acc), end='')
            print('Validation Loss [{:02.4f}]  '.format(val_loss), end='')
            print('Validation Acc [{:02.4f}]  '.format(val_acc))
            '''
        else:
            train_loss, train_acc = train(model, train_dataloader, optim, criterion, epoch, train_total)
            '''
            print('Epoch [{:3}]]  '.format(epoch), end='')
            print('Loss [{:02.4f}]  '.format(train_loss), end='')
            print('Acc [{:02.4f}]  '.format(train_acc), end='')
            '''
        
        if SAVE:
            if train_acc > 0.95 and train_loss < 0.01:
                fn = 'trained_'+args.model+'_'+str(int(time.time()))+'.pth'
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, fn))
                exit()


if __name__ == '__main__':
    main()
