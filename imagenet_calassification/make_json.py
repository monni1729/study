import os
import json
import argparse
import random

def get_args():
    '''
    root_dir :
    The directory which is containing image data.
    The train folder must exist.
    root_dir - train - label_1
                     - label_2
                     - label_3
             - val   - label_1
                     - label_2
                     - label_3
    with_validation :
    If you want to make validation dataset, set this variable as True
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/lab/taylor/imagenet_copy')
    parser.add_argument('--with_validation', type=bool, default=True)
    args = parser.parse_args()
    return args


def make_json(root_dir, with_validation):
    with open(os.path.join(root_dir, ('label_map.json')), 'r') as jj:
        label_map = json.load(jj)
        
    if with_validation:
        sub_dir = ['train', 'val']
    else:
        sub_dir = ['train']

    selected_labels = os.listdir(os.path.join(root_dir, 'train'))

    for sd in sub_dir:
        data_list = []

        for sl in selected_labels:
            folder = os.path.join(root_dir, sd, sl)
            image_list = os.listdir(folder)

            for image in image_list:
                image_dir = os.path.join(folder, image)
                label = sl
                temp_dict = {'image_dir': image_dir, 'category': label_map[label]}
                data_list.append(temp_dict)

        json_dict = data_list
        random.shuffle(json_dict)
        with open(os.path.join(root_dir, (sd + '_annotations.json')), 'w') as jj:
            json.dump(json_dict, jj)

def make_label_map(root_dir):
    
    temp = sorted(os.listdir(os.path.join(root_dir, 'train')))
    label_map = dict()
    for idx, name in enumerate(temp):
        label_map[name] = idx

    with open(os.path.join(root_dir, ('label_map.json')), 'w') as jj:
        json.dump(label_map, jj)

if __name__ == '__main__':
    args = get_args()
    make_label_map(args.root_dir)
    make_json(args.root_dir, args.with_validation)
