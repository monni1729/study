import os
import json
import random
import shutil
import numpy as np
from PIL import Image
from skimage import transform
import skimage.io as io


def fetch_sample_images():
    with open('/mnt/sdc/imagenet_fall11/val_annotations.json', 'r') as jf:
        file_names = json.load(jf)

    cc = 0
    while True:
        idx = int((len(file_names) - 1) * random.random())
        image_name = file_names[idx]['image_dir']

        img = Image.open(image_name).convert('RGB')
        img = np.array(img)

        if img.shape[0] < 600 or img.shape[1] < 600:
            pass
        elif img.shape[0] / img.shape[1] > 1.5 or img.shape[1] / img.shape[0] > 1.5:
            pass
        else:
            cc += 1
            image_name = image_name.split('/')[-1]
            print(img.shape)
            shutil.copyfile(file_names[idx]['image_dir'],
                            os.path.join('/home/taylor/spo_classification/thumbnail/sample_images/', image_name))

        if cc == 80:
            break


def check_resolution():
    rd = '/home/taylor/spo_classification/thumbnail/sample_images/'
    images = os.listdir(rd)

    for im in images:
        img = Image.open(rd + im).convert('RGB')
        img = np.array(img)
        if img.shape[0] < 300 or img.shape[1] < 300:
            os.remove(rd + im)
        elif img.shape[0] / img.shape[1] > 1.5 or img.shape[1] / img.shape[0] > 1.5:
            os.remove(rd + im)
        else:
            pass


def rectangle(sizes):
    rd = '/home/taylor/spo_classification/thumbnail/sample_images/'
    image_names = os.listdir(rd)
    for image in image_names:
        img = Image.open(rd + image).convert('RGB')
        img = np.array(img)

        a, b = img.shape[0], img.shape[1]
        if a > b:
            img = img[(a - b) // 2:(a - b) // 2 + b, :, :]
        elif a < b:
            img = img[:, (b - a) // 2:(b - a) // 2 + a, :]
        else:
            pass
        
        for s in sizes:
            newimg = transform.resize(img, (s ,s))
            io.imsave(os.path.join('/home/taylor/spo_classification/thumbnail/', str(s)+'pix', image), newimg)

        print(img.shape)


if __name__ == '__main__':
    a = 1
    rectangle([200, 450])
    # fetch_sample_images()
    # print(len(os.listdir('/home/taylor/spo_classification/thumbnail/sample_images/')))
