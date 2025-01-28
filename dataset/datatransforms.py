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
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        img_A = np.array(img_A).astype(np.float32)
        img_B = np.array(img_B).astype(np.float32)
        label_BCD = np.array(label_BCD).astype(np.float32)
        if self.div_255:
            img_A /= 255.0
            img_B /= 255.0
        img_A -= self.mean
        img_A /= self.std
        img_B -= self.mean
        img_B /= self.std

        # if np.max(label_BCD) >= 2:
        #     label_BCD[label_BCD < 128] = 0
        #     label_BCD[label_BCD >= 128] = 1
        
        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, test_time_color_aug=False):
        self.test_time_color_aug = test_time_color_aug

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_A = sample['img_A']
        img_B = sample['img_B']
        if self.test_time_color_aug:
            img_A_trans = sample['img_A_trans']
            img_B_trans = sample['img_B_trans']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        img_A = np.array(img_A).astype(np.float32).transpose((2, 0, 1))
        img_B = np.array(img_B).astype(np.float32).transpose((2, 0, 1))
        if self.test_time_color_aug:
            img_A_trans = np.array(img_A_trans).astype(np.float32).transpose((2, 0, 1))
            img_B_trans = np.array(img_B_trans).astype(np.float32).transpose((2, 0, 1))
        label_BCD = np.array(label_BCD).astype(np.float32)
        label_SGA = np.array(label_SGA).astype(np.float32)
        label_SGB = np.array(label_SGB).astype(np.float32)

        img_A = torch.from_numpy(img_A).type(torch.FloatTensor)
        img_B = torch.from_numpy(img_B).type(torch.FloatTensor)
        if self.test_time_color_aug:
            img_A_trans = torch.from_numpy(img_A_trans).type(torch.FloatTensor)
            img_B_trans = torch.from_numpy(img_B_trans).type(torch.FloatTensor)
        label_BCD = torch.from_numpy(label_BCD).type(torch.LongTensor)
        label_SGA = torch.from_numpy(label_SGA).type(torch.LongTensor)
        label_SGB = torch.from_numpy(label_SGB).type(torch.LongTensor)
        
        if not self.test_time_color_aug:
            return  {'img_A': img_A,
                    'img_B': img_B,
                    'label_BCD': label_BCD,
                    'label_SGA': label_SGA,
                    'label_SGB': label_SGB
                    }
        else:
            return  {'img_A': img_A,
                    'img_B': img_B,
                    'img_A_trans': img_A_trans,
                    'img_B_trans': img_B_trans,
                    'label_BCD': label_BCD,
                    'label_SGA': label_SGA,
                    'label_SGB': label_SGB
                    }


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            label_BCD = label_BCD.transpose(Image.FLIP_LEFT_RIGHT)
            label_SGA = label_SGA.transpose(Image.FLIP_LEFT_RIGHT)
            label_SGB = label_SGB.transpose(Image.FLIP_LEFT_RIGHT)

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_TOP_BOTTOM)
            img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM)
            label_BCD = label_BCD.transpose(Image.FLIP_TOP_BOTTOM)
            label_SGA = label_SGA.transpose(Image.FLIP_TOP_BOTTOM)
            label_SGB = label_SGB.transpose(Image.FLIP_TOP_BOTTOM)

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Transpose.ROTATE_90, Transpose.ROTATE_180, Transpose.ROTATE_270]

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            rotate_degree = random.choice(self.degree)
            img_A = img_A.transpose(rotate_degree)
            img_B = img_B.transpose(rotate_degree)
            label_BCD = label_BCD.transpose(rotate_degree)
            label_SGA = label_SGA.transpose(rotate_degree)
            label_SGB = label_SGB.transpose(rotate_degree)            

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class RandomRotate(object):
    def __init__(self, degree, fillcolor=0):
        self.degree = degree
        self.fillcolor = fillcolor

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']
        
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img_A = img_A.rotate(rotate_degree, Image.BILINEAR, expand=True)
            img_B = img_B.rotate(rotate_degree, Image.BILINEAR, expand=True)
            label_BCD = label_BCD.rotate(rotate_degree, Image.NEAREST, expand=True, fillcolor=255)
            label_SGA = label_SGA.rotate(rotate_degree, Image.NEAREST, expand=True, fillcolor=self.fillcolor)
            label_SGB = label_SGB.rotate(rotate_degree, Image.NEAREST, expand=True, fillcolor=self.fillcolor)     

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']

        assert img_A.size == label_BCD.size and img_B.size == label_BCD.size == label_SGA.size == label_SGB.size

        img_A = img_A.resize(self.size, Image.BILINEAR)
        img_B = img_B.resize(self.size, Image.BILINEAR)
        label_BCD = label_BCD.resize(self.size, Image.NEAREST)
        label_SGA = label_SGA.resize(self.size, Image.NEAREST)
        label_SGB = label_SGB.resize(self.size, Image.NEAREST)    

        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }

'''
class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]
'''

class ColorJitterImages(object):
    def __init__(
        self, mutual_colortrans=False, test_time_color_aug=False
    ):  
        self.colorjitter = transforms.Compose([transforms.ColorJitter(0.2, 0.4, 0.2, 0.1)])
        self.mutual_colortrans = mutual_colortrans
        self.test_time_color_aug = test_time_color_aug

    def getavgstd(self, image, channel_last):
        avg = []
        std = []
        if not channel_last:
            for i in range(image.shape[0]):
                image_avg, image_std = np.mean(image[i,:,:]), np.std(image[i,:,:])
                avg.append(image_avg)
                std.append(image_std)
        else:
            for i in range(image.shape[2]):
                image_avg, image_std = np.mean(image[:,:,i]), np.std(image[:,:,i])
                avg.append(image_avg)
                std.append(image_std)
        return (avg, std)

    
    def multiband_color_tranfer(self, src, ref_src, channel_last=True):
        image = np.array(src, dtype=np.float32) / 255.
        original = np.array(ref_src, dtype=np.float32) / 255.

        image_avg, image_std = self.getavgstd(image, channel_last)
        original_avg, original_std = self.getavgstd(original, channel_last)

        # numpy calculation:
        if not channel_last:
            for k in range(image.shape[0]):
                channel_img = image[k,:,:]
                res = (channel_img - image_avg[k]) * (original_std[k] / image_std[k]) + original_avg[k]
                res = np.where(res<0, 0, res)
                res = np.where(res>1, 1, res)
                image[k,:,:] = res
        else:
            for k in range(image.shape[2]):
                channel_img = image[:,:,k]
                res = (channel_img - image_avg[k]) * (original_std[k] / image_std[k]) + original_avg[k]
                res = np.where(res<0, 0, res)
                res = np.where(res>1, 1, res)
                image[:,:,k] = res
        out_src = np.array(image*255., dtype=np.uint8)
        
        return Image.fromarray(out_src)


    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']

        if not self.test_time_color_aug:

            if self.mutual_colortrans:
                f = random.random() 
                if f < 0.33:
                    img_A = self.multiband_color_tranfer(img_A, img_B) 
                elif f >= 0.33 and f < 0.66:
                    img_B = self.multiband_color_tranfer(img_B, img_A)
                else:
                    pass 

            img_A = self.colorjitter(img_A)
            img_B = self.colorjitter(img_B)

            # f = random.random() 
            # if (f > 0.25) and (f <= 0.5):
            #     img_A = self.colorjitter(img_A)
            #     img_B = self.colorjitter(img_B)
            # elif (f > 0.5) and (f <= 0.75):
            #     img_A = self.colorjitter(img_A)
            # elif (f > 0.75) and (f <= 1):
            #     img_B = self.colorjitter(img_B)
            
            return  {'img_A': img_A,
                    'img_B': img_B,
                    'label_BCD': label_BCD,
                    'label_SGA': label_SGA,
                    'label_SGB': label_SGB
                    }
        else:
            img_A_trans = self.multiband_color_tranfer(img_A, img_B) 
            img_B_trans = self.multiband_color_tranfer(img_B, img_A)
            
            return  {'img_A': img_A,
                    'img_B': img_B,
                    'img_A_trans': img_A_trans,
                    'img_B_trans': img_B_trans,
                    'label_BCD': label_BCD,
                    'label_SGA': label_SGA,
                    'label_SGB': label_SGB
                    }

            

class GaussianBlur(object):
    
    def __call__(self, sample):
        img_A = sample['img_A']
        img_B = sample['img_B']
        label_BCD = sample['label_BCD']
        label_SGA = sample['label_SGA']
        label_SGB = sample['label_SGB']

        f = random.random() 
        if (f > 0.25) and (f <= 0.5):
            img_A = img_A.filter(ImageFilter.GaussianBlur(radius=1))
        elif (f > 0.5) and (f <= 0.75):
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=1))
        elif (f > 0.75) and (f <= 1):
            img_A = img_A.filter(ImageFilter.GaussianBlur(radius=1))
            img_B = img_B.filter(ImageFilter.GaussianBlur(radius=1))
        # img_A = img_A.filter(ImageFilter.GaussianBlur(radius=1))
        # img_B = img_B.filter(ImageFilter.GaussianBlur(radius=1))
        return  {'img_A': img_A,
                 'img_B': img_B,
                 'label_BCD': label_BCD,
                 'label_SGA': label_SGA,
                 'label_SGB': label_SGB
                }
        

test_transforms = transforms.Compose([ 
                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensor()])
test_transforms_clean = transforms.Compose([ 
                        ToTensor()])
# def get_test_transforms_cveo(test_time_color_aug=False):
#     if test_time_color_aug:
#         test_transforms = transforms.Compose([
#                             ColorJitterImages(test_time_color_aug=test_time_color_aug),
#                             ToTensor(test_time_color_aug=test_time_color_aug)
#                             ])
#     return test_transforms


def get_train_transforms(seg_ignore_value=None, with_colorjit=False, div_255=True):
    if with_colorjit:
        train_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomFixRotate(),
                            # RandomRotate(30, fillcolor=seg_ignore_value),
                            ColorJitterImages(),
                            # FixedResize(512), # 
                            # GaussianBlur(), # 高斯模糊
                            # Normalize(div_255=div_255, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensor()
                            ])
    else:
        train_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomFixRotate(),
                            # RandomRotate(30, fillcolor=seg_ignore_value),
                            # ColorJitterImages(),
                            # FixedResize(512), # 
                            # Normalize(div_255=div_255, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensor()
                            ])        
    return train_transforms


def get_train_transforms_cveo(with_colorjit=False, mutual_colortrans=False):
    if with_colorjit:
        train_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomFixRotate(),
                            # RandomRotate(30, fillcolor=seg_ignore_value),
                            ColorJitterImages(mutual_colortrans=mutual_colortrans),
                            # FixedResize(1024), # 512
                            # GaussianBlur(), # 高斯模糊
                            # Normalize(div_255=div_255, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensor()
                            ])
    else:
        train_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomVerticalFlip(),
                            RandomFixRotate(),
                            # RandomRotate(30, fillcolor=seg_ignore_value),
                            # ColorJitterImages(),
                            # FixedResize(1024), # 512
                            # Normalize(div_255=div_255, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensor()
                            ])        
    return train_transforms



if  __name__ == '__main__':
    
    pass