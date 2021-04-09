import os
import json
import torch
import argparse
import numpy as np
import albumentations
import torch.nn as nn
from flask import Flask, request
from flask_cors import CORS
from collections import OrderedDict
from PIL import Image
import torchvision.models as models


app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_dir', type=str, default=' ')
# parser.add_argument('--weight_path', type=str, default='efficientdet_train_result/trained_efficientnet.pth')
# parser.add_argument('--num_classes', type=int, default=15760)

# args = parser.parse_args()

model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
model = torch.nn.DataParallel(model).cuda()


@app.route('/', methods=['POST'])
def test():
    '''

    temp = os.listdir('/home/taylor/spo_classification/thumbnail/sample_images/')
    temp.sort()
    
    for t in temp:
        dd = '/home/taylor/spo_classification/thumbnail/sample_images/' + t
        output = inference(model, dd)
        print(output)  
    '''
    image_file = request.files['image']
    data_path = 'api_image.jpg'
    image_file.save(data_path)

    output = inference(model, data_path)
    print(output)
    
    return json.dumps(output, ensure_ascii=False, indent='\t')


def l2_distance(original, thumb):
    result = []
    for i in range(original.shape[0]):
        temp = original[i] - thumb
        result.append(np.sum(temp**2))
    
    return np.array(result)


def inference(model, image_path):
    original = np.load('original_vector.npy', allow_pickle=True)
    
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    transform = albumentations.Compose([albumentations.Resize(224, 224), albumentations.Normalize()])
    image = Image.open(image_path).convert('RGB')
    image = transform(image=np.array(image))['image']
    image = np.transpose(image, (2, 0, 1))
    
    image = torch.tensor(image, dtype=torch.float32)
    image = image.cuda()
    output = model(image.unsqueeze(0))
    pred = output.detach().cpu().numpy()
    
    dts = l2_distance(original, pred[0])
    top_idx = dts.argsort()[:5]
    
    top_idx = [int(i) for i in top_idx]
    return top_idx


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=18558, debug=False)
