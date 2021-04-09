import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from time import time
np.set_printoptions(precision=4)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


   
def main(model_name, final_layer):
    # tt = time()
    if model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise NameError('inaccurate model name')
        
    model.fc =Identity()
    model.cuda()
    model.eval()
    
    rd = '/home/taylor/spo_classification/thumbnail/'
    image_list = os.listdir(rd+'600pix')
    image_list.sort()
    
    pix_list = [200, 300, 450, 600]
    output_batch_list = [np.zeros((len(image_list), final_layer), dtype=np.float32) for _ in pix_list]
    # print(time()-tt)
    # tt = time()
    for jj, pix in enumerate(pix_list):
        for idx, image in enumerate(image_list):
            image = Image.open(rd+str(pix)+'pix/'+image).convert('RGB')
            image = np.array(image) / 255.
            image = image - 0.5
            image = image / 0.25
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)
            image = torch.unsqueeze(image, 0)
            image = image.cuda()
        
            output = model(image)
            output = output.cpu().detach().numpy()
            output_batch_list[jj][idx,:] = output

    acc = 0
    # print(time()-tt)
    for i in range(100):
        print(i)
        result = distance(output_batch_list[3], output_batch_list[1][i:i+1])
        if np.argmin(result) == i:
            acc += 1
    
    print('acc ', acc)
    
def distance(batch, thumb):
    dist_list = []
    for i in range(batch.shape[0]):
        dist_list.append(np.mean((batch[i]-thumb[0])**2))
        
    return dist_list

    

        
    
# [18, 34] : 512, [50, 101, 152] : 2048
if __name__ == '__main__':
    main('resnet34', 512)