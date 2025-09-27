import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from PIL.Image import Transpose
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, div_255=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.div_255 = div_255
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_A = sample['img_A']
        label_SGA = sample['label_SGA']

        img_A = np.array(img_A).astype(np.float32)
        if self.div_255:
            img_A /= 255.0
        img_A -= self.mean
        img_A /= self.std
        
        return  {'img_A': img_A,
                 'label_SGA': label_SGA,
                }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_A = sample['img_A']
        label_SGA = sample['label_SGA']
        
        img_A = np.array(img_A).astype(np.float32).transpose((2, 0, 1))
        label_SGA = np.array(label_SGA).astype(np.float32)

        img_A = torch.from_numpy(img_A).type(torch.FloatTensor)
        label_SGA = torch.from_numpy(label_SGA).type(torch.LongTensor)
        
        return  {'img_A': img_A,
                 'label_SGA': label_SGA,
                }


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_A = sample['img_A']
        label_SGA = sample['label_SGA']
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            label_SGA = label_SGA.transpose(Image.FLIP_LEFT_RIGHT)

        return  {'img_A': img_A,
                 'label_SGA': label_SGA,
                }


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img_A = sample['img_A']
        label_SGA = sample['label_SGA']
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_TOP_BOTTOM)
            label_SGA = label_SGA.transpose(Image.FLIP_TOP_BOTTOM)

        return  {'img_A': img_A,
                 'label_SGA': label_SGA,
                }


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Transpose.ROTATE_90, Transpose.ROTATE_180, Transpose.ROTATE_270]

    def __call__(self, sample):
        img_A = sample['img_A']
        label_SGA = sample['label_SGA']
        
        if random.random() < 0.5:
            rotate_degree = random.choice(self.degree)
            img_A = img_A.transpose(rotate_degree)
            label_SGA = label_SGA.transpose(rotate_degree) 

        return  {'img_A': img_A,
                 'label_SGA': label_SGA,
                }

class ColorJitterImages(object):
    def __init__(self):   
        self.colorjitter = transforms.Compose([transforms.ColorJitter(0.2, 0.4, 0.2, 0.1)])

    def __call__(self, sample):
        img_A = sample['img_A']
        label_SGA = sample['label_SGA']
        img_A = self.colorjitter(img_A)
        return  {'img_A': img_A,
                 'label_SGA': label_SGA,
                }



test_transforms_clean = transforms.Compose([ToTensor()])

def get_train_transforms_cveo(with_colorjit=False):
    if with_colorjit:
        train_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomFixRotate(),
                            ColorJitterImages(),
                            ToTensor()
                            ])
    else:
        train_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomFixRotate(),
                            ToTensor()
                            ])        
    return train_transforms

if  __name__ == '__main__':
    pass