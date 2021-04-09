import cv2
import random
import copy
import numpy as np


itps= [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_LINEAR_EXACT, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4] 

im_dir = '/home/taylor/spo_classification/thumbnail/600pix/n01321230_624.JPEG'
image = cv2.imread(im_dir)

th_images1 = []
for interpolation in itps:
    th_images1.append(cv2.resize(image, dsize=(96, 96), interpolation=interpolation))
    a = cv2.resize(image, dsize=(96, 96), interpolation=interpolation)

th_images2 = copy.deepcopy(th_images1)
random.shuffle(th_images2)
random_th = th_images2[0]
# print('random number :', random_th)

for idx, interpolation in enumerate(itps):
    if np.array_equal(cv2.resize(image, dsize=(96, 96), interpolation=interpolation), random_th):
        print('answer number : ', idx)

