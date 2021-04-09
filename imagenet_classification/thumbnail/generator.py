import random
import numpy as np
import cv2
import os
import json
import shutil
from PIL import Image

itps= [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_LINEAR_EXACT, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4] 

def fetch_sample_images():
    cnt= 0
    with open('/mnt/sdc/imagenet_fall11/val_annotations.json', 'r') as jf:
        file_names = json.load(jf)

    idx = 100000
    while True:
        image_name = file_names[idx]['image_dir']

        img = cv2.imread(image_name)

        if img.shape[0] < 600 or img.shape[1] < 600:
            pass
        elif img.shape[0] > 1000 or img.shape[1] > 1000:
            pass
        else:
            image_name = image_name.split('/')[-1]
            print(img.shape)
            cnt +=1
            shutil.copyfile(file_names[idx]['image_dir'],
                            os.path.join('/home/taylor/spo_classification/thumbnail/datasets/original/', image_name))
                            
        idx += 1 
        if cnt == 10000:
            break
            


def gen(image):
    rr = (random.random() - 0.5) * 2
    th_size = (int(100 + rr*30), int(100 - rr*30))
    th_ratio = th_size[0] / th_size[1]
    
    image_size = image.shape
    if image_size[0] > image_size[1]:
        image_gap = (image_size[0] - image_size[1]) / 2
        image_padding = image_gap * random.random()
        new_image = np.zeros((image_size[0], image_size[1] + int(2 * image_padding), 3))
        new_image[:,int(image_padding):int(image_padding)+image_size[1],:] = image
    elif image_size[0] < image_size[1]:
        image_gap = (image_size[1] - image_size[0]) / 2
        image_padding = image_gap * random.random()
        new_image = np.zeros((image_size[0] + int(2 * image_padding), image_size[1], 3))
        new_image[int(image_padding):int(image_padding)+image_size[0],:,:] = image
    else:
        new_image = image
    
    image = cv2.resize(new_image, dsize=tuple(th_size), interpolation=cv2.INTER_NEAREST)    
    return image

def main():
    root_dir = '/home/taylor/spo_classification/thumbnail/datasets/original'
    save_dir = '/home/taylor/spo_classification/thumbnail/datasets/thumb'
    img_list = os.listdir(root_dir)
    print(len(img_list))
    for img_name in img_list:
        image = cv2.imread(os.path.join(root_dir, img_name))
        new_image = gen(image)
        cv2.imwrite(os.path.join(save_dir, img_name), new_image)
    
    
if __name__ == '__main__':
    main()
    # fetch_sample_images()