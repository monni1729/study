import torch
from torchvision import transforms
from PIL import ImageEnhance, Image
import numpy as np
import random



class Normalizer:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.array(image, dtype=np.float32)
        image /= 255.

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        label = np.array(label, dtype=np.long)
        label = torch.from_numpy(label)

        return {'image': image, 'label': label}


class Brightness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        '''
        factor 1 means original
        factor 0 means black image
        '''
        image, label = sample['image'], sample['label']
        enhancer = ImageEnhance.Brightness(image)
        im_output = enhancer.enhance(self.factor)

        sample['image'] = im_output
        return sample


class Contrast:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        '''
        factor 1 means original
        factor 0 means gray image
        '''
        image = sample['image']
        enhancer = ImageEnhance.Contrast(image)
        output = enhancer.enhance(self.factor)

        sample['image'] = output
        return sample


class Hue:
    def __init__(self, color, factor):
        assert color in ['red', 'green', 'blue']
        self.color = color
        self.factor = factor

    def __call__(self, sample):
        '''
        factor 1 means single color (red, green, blue)
        factor 0 means original
        '''
        image = sample['image']
        layer = Image.new('RGB', image.size, self.color)
        output = Image.blend(image, layer, self.factor)

        sample['image'] = output
        return sample


class Saturation:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        '''
        factor 1 means original
        factor 0 means gray image
        '''
        image = sample['image']
        enhancer = ImageEnhance.Color(image)
        output = enhancer.enhance(self.factor)

        sample['image'] = output
        return sample


class RandomCrop:
    def __init__(self, factor):
        assert  0 <= factor < 1
        self.factor = factor / 2.

    def __call__(self, sample):
        '''
        factor 0 means original
        factor should be lower than 1
        '''
        image = sample['image']

        image = np.array(image, dtype=np.uint8)
        h, w = image.shape[0], image.shape[1]
        h1 = int(h * (random.random() * self.factor))
        h2 = int(h * (1 - random.random() * self.factor))
        w1 = int(w * (random.random() * self.factor))
        w2 = int(w * (1 - random.random() * self.factor))

        image = image[h1:h2, w1:w2, :]
        image = Image.fromarray(np.uint8(image))
        sample['image'] = image
        return sample


class Rescale:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.resize(self.size)
        sample['image'] = image

        return sample


# visualize
if __name__ == '__main__':
    image = Image.open('/home/taylor/spo_classification/sample.jpg').convert('RGB')
    label = 0
    sample = {'image': image, 'label': label}
    transforms_com = transforms.Compose([RandomCrop(0.5), ])

    new_image = transforms_com(sample)['image']
    new_image.save('new_sample.jpg')
