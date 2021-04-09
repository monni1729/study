import os
import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image
import argparse
from collections import OrderedDict
import json

from EfficientNet_PyTorch.efficientnet_pytorch.model import EfficientNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--model', type=str, default='efficientnet')
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--label_mapping', type=str)

    args = parser.parse_args()

    return args


class Normalizer(object):
    def __call__(self, image):
        image = np.array(image, dtype=np.float32)
        image /= 255.

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image


class Rescale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = image.resize(self.size)

        return image


def load_model(model_name, weight_path, num_classes):
    if model_name == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        print('efficientnet model loaded')
    elif model_name == 'resnest':
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50',
                               pretrained=True)

        n_inputs = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Linear(n_inputs, num_classes))
    else:
        raise NameError('model should be efficientnet or resnest')

    state_dict = torch.load(weight_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == '__main__':
    model_name = 'resnest'
    weight_path = 'efficientdet_train_result/trained_.pth'
    num_classes = 2

    model = load_model(model_name, weight_path, num_classes)
    model = model.cuda()

    transform = transforms.Compose([Rescale(size=(380, 380)),
                                    Normalizer()])

    mapping_dict = {0: 'mountain', 1: 'river'}
    rd = '/home/taylor/spo_classification/test_data/'
    subd = ['river', 'mountain']
    for sub in subd:
        for name in os.listdir(rd + sub):
            print('file_name : ', name)
            image = Image.open(os.path.join(rd, sub, name)).convert('RGB')
            input = transform(image).cuda()
            output = model(input.unsqueeze(0))

            pred = output.detach().cpu().numpy()
            pred = softmax(pred[0])
            print('mountain : ', pred[0], '    river : ', pred[1])
            print('ground truth : ', sub, end='      ')
            print('prediction : ', mapping_dict[np.argmax(pred)], end='\n\n')

            # output = F.softmax(output.squeeze(0), dim=-1)
            # print(output.data)
