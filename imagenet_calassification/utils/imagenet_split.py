import os
import json
import random


def make_json():
    validation_ratio = 0.2
    thres = 100

    root_dir = '/mnt/sdc/imagenet_fall11/fall11/'
    dir_list = os.listdir(root_dir)
    temp = []
    for n in dir_list:
        if n[-4:] == '.tar':
            pass
        else:
            temp.append(n)
    dir_list = temp
    if len(dir_list) != 21841:
        raise ValueError(len(dir_list), 'number of raw label should be 21841')

    # ------------------------------------------------------

    temp = []
    for name in dir_list:
        if len(os.listdir(os.path.join(root_dir, name))) >= thres:
            temp.append(name)

    dir_list = sorted(temp)
    print("number of labels : ", len(dir_list))

    # ------------------------------------------------------

    temp = sorted(dir_list)
    label_map = dict()
    for idx, name in enumerate(temp):
        label_map[name] = idx

    with open('label_map.json', 'w') as jj:
        json.dump(label_map, jj)

    # ------------------------------------------------------

    train_json = list()
    val_json = list()

    for dir_name in dir_list:
        sub_dir_list = os.listdir(os.path.join(root_dir, dir_name))
        random.shuffle(sub_dir_list)
        cc = int(len(sub_dir_list) * validation_ratio)
        train_list = sub_dir_list[cc:]
        val_list = sub_dir_list[:cc]

        for tl in train_list:
            train_json.append({'image_dir':os.path.join(root_dir, dir_name, tl), 'category': label_map[dir_name]})
        for vl in val_list:
            val_json.append({'image_dir':os.path.join(root_dir, dir_name, vl), 'category': label_map[dir_name]})

    with open(os.path.join('train_annotations.json'), 'w') as jj:
        json.dump(train_json, jj)

    with open(os.path.join('val_annotations.json'), 'w') as jj:
        json.dump(val_json, jj)

make_json()