import os
import shutil
import json


def fetch_imagenet_data():
    selected_labels = ['n02992211', 'n03623198',
                       'n04592741', 'n04026417',
                       'n03124170', 'n02097047',
                       'n03857828', 'n02843684',
                       'n02391049', 'n03794056']

    root_dir = '/home/taylor/imagenet_zip/'
    sub_dir = ['train', 'val']

    try:
        os.mkdir('./imagenet_small')
        os.mkdir('./imagenet_small/train')
        os.mkdir('./imagenet_small/val')
    except FileExistsError:
        pass

    for sd in sub_dir:
        for sl in selected_labels:
            folder1 = os.path.join(root_dir, sd, sl)
            folder2 = os.path.join('/home/taylor/spo_classification/imagenet_small', sd, sl)
            shutil.copytree(folder1, folder2)


if __name__ == '__main__':
    pass
