from random import randint
import skimage
import numpy as np

for i in range(32):
    img = np.zeros((512, 512), dtype=np.int)
    for j in range(512):
        for k in range(512):
            img[j, k] = randint(0, 255)
    skimage.io.imsave("../dream/train/ran/a"+str(i)+'.png', img)
