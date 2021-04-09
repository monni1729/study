import torch
from collections import OrderedDict
import numpy as np
from taylor_transforms import Normalizer, Rescale
from torchvision import transforms
from PIL import Image
import json
from EfficientNet_PyTorch.efficientnet_pytorch.model import EfficientNet
from ResNeSt_PyTorch.resnest.torch import resnest50, resnest101, resnest200
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--image_path', type=list)
    parser.add_argument('--json_path', type=str)

    args = parser.parse_args()
    return args


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inference(model, weight_path, image_path, label_map, num_classes):
    if model.startswith('efficientnet'):
        model = EfficientNet.from_pretrained(model_name=model,
                                             weights_path='/home/taylor/spo_classification/pretrained_weights/' + model + '.pth',
                                             num_classes=num_classes)

    elif model.startswith('resnest'):
        if model[7:] == '50':
            model = resnest50(pretrained=True)
        elif model[7:] == '101':
            model = resnest101(pretrained=True)
        elif model[7:] == '200':
            model = resnest200(pretrained=True)
        else:
            raise NameError(model, ' : inaccurate model name')

        n_inputs = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Linear(n_inputs, num_classes))

        state_dict = torch.load(weight_path)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    else:
        raise NameError('model should be efficientnet-bN or resnestNNN')

    model.cuda()
    model.eval()
    transform = transforms.Compose([Rescale(size=(380, 380)),
                                    Normalizer()])

    image = Image.open(image_path).convert('RGB')
    temp = {'image': image, 'label': 0}
    input = transform(temp)
    input = input['image'].cuda()
    output = model(input.unsqueeze(0))
    pred = output.detach().cpu().numpy()
    pred = softmax(pred[0])
    pred = pred.tolist()

    return [[label_map[i], pred[i]] for i in range(len(label_map))]


if __name__ == '__main__':
    '''
    model = 'efficientnet-b4'
    weight_path = '/home/taylor/spo_classification/trained_weights/pprc/trained_efficientnet-b4.pth'
    image_path = ['/home/taylor/spo_classification/test_data/pprc/passport.jpg',
                  '/home/taylor/spo_classification/test_data/pprc/registration_card.jpg']
    json_path = '/home/taylor/spo_classification/datasets/pprc/train_annotations.json'
    '''
    args = get_args()
    model = args.model
    weight_path = args.weight_path
    image_path = args.image_path
    json_path = args.json_path

    with open(json_path, 'r') as json_f:
        json_dict = json.load(json_f)
    label_map = json_dict['info']
    label_map = {v: k for k, v in label_map.items()}

    for ip in image_path:
        print(ip)
        print(inference(model, weight_path, ip, label_map, len(label_map)))
        print()
