import time
import tqdm
import torch
import argparse
from utils import accuracy, custom_range
from data_loader import get_image_dataloader
from model import Backbone, Arcface


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--annot_dir', type=str, default='/home/taylor/pipeline/cropped_images')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--load_weight', type=bool)
    parser.add_argument('--emb_dim', type=int, default=512)
    args = parser.parse_args()

    return args

# num_class : label 수  
def fetch_arcface(num_classes, embedding_size=512, load_weight=False):
    backbone = Backbone(50, 0.5, 'ir_se').cuda()
    head = Arcface(embedding_size=embedding_size, classnum=num_classes).cuda()

    if load_weight:
        pass  # TODO

    return backbone, head


# 학습 
def train_fn(backbone, head, train_dataloader, optim, criterion, epoch):
    backbone.train()
    head.train()

    loss_list, acc_sum = [], 0
    cr_sv = custom_range(len(train_dataloader), 1)
    progress_bar = tqdm.tqdm(train_dataloader, leave=False) #for visualization 
    for i, (image, label) in enumerate(progress_bar):
        image = image.cuda()
        label = label.cuda()
        optim.zero_grad()

        embedding = backbone(image)
        pred = head(embedding, label)
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
        fn = 'weights/backbone_' + str(int(time.time())) + '.pth'
        torch.save(backbone.state_dict(), fn)
        fn = 'weights/head_' + str(int(time.time())) + '.pth'
        torch.save(head.state_dict(), fn)


def validate_fn(backbone, head, val_dataloader, criterion, epoch):
    backbone.eval()
    head.eval()

    loss_list, acc_sum = [], 0
    acc1s = []
    with torch.no_grad():
        for i, (image, label) in enumerate(val_dataloader):
            image = image.cuda()
            label = label.cuda()

            embedding = backbone(image)
            pred = head(embedding, label)
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
    load_weight = args.load_weight
    emb_dim = args.emb_dim

    train_dataloader, val_dataloader, num_classes = get_image_dataloader(annot_dir=args.annot_dir,
                                                                         batch_size=args.batch_size,
                                                                         num_workers=args.num_workers)
    backbone, head = fetch_arcface(num_classes, emb_dim, load_weight)
    backbone = torch.nn.DataParallel(backbone).cuda()
    head = torch.nn.DataParallel(head).cuda()

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, list(backbone.parameters()) + list(head.parameters())), lr)

    print(f'number of classes : {num_classes}')
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # 학습 성능 개선
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=3,
                                                           min_lr=1e-7,
                                                           verbose=True)

    for epoch in range(num_epoch):
        train_fn(backbone, head, train_dataloader, optim, criterion, epoch)
        val_loss = validate_fn(backbone, head, val_dataloader, criterion, epoch)
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
