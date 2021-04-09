import os
import json
import torch
import numpy as np
import albumentations as al
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io

from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=112, margin=10,keep_all=True, device='cuda')

detector = fetch_detector()
backbone = Backbone()
head = Arcface()
transform = al.Compose([al.Normalize()])

image = Image.open('people.jpg').convert('RGB')
# print(np.array(image)[:2, :2,:])
cropped_images = mtcnn(image)

for idx, img in enumerate(cropped_images):
    embedding = backbone(img)
    pred = head(embedding)
    cc = c.numpy()
    print(cc.shape)
    cc = cc* 255
    cc= cc.astype(np.uint8)
    cc = np.transpose(cc, (1, 2, 0))
    io.imsave(str(idx)+'_c.jpg', cc)
    '''
image = transform(image=np.array(image))['image']
image = np.transpose(image, (2, 0, 1))

detector = fetch_detector()
backbone = Backbone()
head = Arcface()

patches = detector(image)'''