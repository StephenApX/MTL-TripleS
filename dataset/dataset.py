import os
import sys
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

# from osgeo import gdal
from PIL import Image
from torch.utils.data import Dataset

DEBUG = False
if DEBUG:
    import datatransforms as tr
    import datatransforms_seg as tr_seg
else:
    import dataset.datatransforms as tr
    import dataset.datatransforms_seg as tr_seg

import numpy as np

import random


def full_path_loader_for_txt(txt_path, train_percent=100):
    dataset = {}

    with open(txt_path, "r") as f1:
        lines = f1.read().splitlines()
    A_path = []
    B_path = []
    label_BCD_path = []
    label_SGA_path = []
    label_SGB_path = []

    random.shuffle(lines)
    if train_percent != 100:
        lines = lines[:int(len(lines)*train_percent/100.)]

    for index, line in enumerate(lines):
        path = line.split(' ')
        img_A = path[0]
        img_B = path[1]
        label_BCD = path[2]
        label_SGA = path[3]
        label_SGB = path[4]
        # assert os.path.isfile(img_A)
        # assert os.path.isfile(img_B)
        # assert os.path.isfile(label_BCD)
        # assert os.path.isfile(label_SGA)
        # assert os.path.isfile(label_SGB)
        A_path.append(img_A)
        B_path.append(img_B)
        label_BCD_path.append(label_BCD)
        label_SGA_path.append(label_SGA)
        label_SGB_path.append(label_SGB)
        dataset[index] = {'img_A': A_path[index],
                          'img_B': B_path[index],
                          'label_BCD': label_BCD_path[index],
                          'label_SGA': label_SGA_path[index],
                          'label_SGB': label_SGB_path[index]
                          }
    return dataset


def full_path_loader_from_txt_for_one_seg(txt_path, train_percent=100):
    dataset = {}

    with open(txt_path, "r") as f1:
        lines = f1.read().splitlines()
    # A_path = []
    # B_path = []
    # label_BCD_path = []
    # label_SGA_path = []
    # label_SGB_path = []

    random.shuffle(lines)
    if train_percent != 100:
        lines = lines[:int(len(lines)*train_percent/100.)]

    for index, line in enumerate(lines):
        path = line.split(' ')
        img_A = path[0]
        img_B = path[1]
        # label_BCD = path[2]
        label_SGA = path[3]
        label_SGB = path[4]
        dataset[index*2] = {'img_A': img_A,
                          'label_SGA': label_SGA,
                          }
        dataset[index*2+1] = {'img_A': img_B,
                          'label_SGA': label_SGB,
                          }
    return dataset



class CVEODataset(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.mutual_colortrans = False
        self.train_percent = args.train_percent
        self.seg_ignore_value = args.num_segclass
        self.split = split
        # self.test_time_color_aug = args.test_time_color_aug

        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path, train_percent=self.train_percent)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
        
        self.train_transforms = tr.get_train_transforms_cveo(with_colorjit=self.with_colorjit, mutual_colortrans=self.mutual_colortrans)
        # if not self.test_time_color_aug:
        self.test_transforms = tr.test_transforms_clean
        # else:
        #     self.test_transforms = tr.get_test_transforms_cveo(self.test_time_color_aug)

        # if not self.pretrained:
        #     self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
        #     self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)
        
        # data Aug.


    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_BCD_path = self.full_load[idx]['label_BCD']
        label_SGA_path = self.full_load[idx]['label_SGA']
        label_SGB_path = self.full_load[idx]['label_SGB']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_BCD = Image.open(label_BCD_path)
        # label_SGA = Image.open(label_SGA_path)
        # label_SGB = Image.open(label_SGB_path)
        # label_BCD = np.asarray(Image.open(label_BCD_path)).astype(np.uint8) - 1
        # ori 1024
        # label_SGA = Image.fromarray((np.asarray(Image.open(label_SGA_path))- 1).astype(np.uint8))
        # label_SGB = Image.fromarray((np.asarray(Image.open(label_SGB_path))- 1).astype(np.uint8))
        # new 512
        label_SGA = Image.fromarray((np.asarray(Image.open(label_SGA_path))).astype(np.uint8))
        label_SGB = Image.fromarray((np.asarray(Image.open(label_SGB_path))).astype(np.uint8))

        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                  'label_SGB': label_SGB
                  }
        
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
            sample['name'] = img_A_path
        return sample



class CVEOOnlySEGDataset(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.train_percent = args.train_percent
        self.seg_ignore_value = args.num_segclass
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_from_txt_for_one_seg(args.train_txt_path, train_percent=self.train_percent)
        elif split =='val':
            self.full_load = full_path_loader_from_txt_for_one_seg(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_from_txt_for_one_seg(args.test_txt_path)
        
        self.train_transforms = tr_seg.get_train_transforms_cveo(with_colorjit=self.with_colorjit)
        self.test_transforms = tr_seg.test_transforms_clean
        # if not self.pretrained:
        #     self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
        #     self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)
        
        # data Aug.
        

    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        # img_B_path = self.full_load[idx]['img_B']
        # label_BCD_path = self.full_load[idx]['label_BCD']
        label_SGA_path = self.full_load[idx]['label_SGA']
        # label_SGB_path = self.full_load[idx]['label_SGB']

        img_A = Image.open(img_A_path)
        # img_B = Image.open(img_B_path)
        # label_BCD = Image.open(label_BCD_path)
        # label_SGA = Image.open(label_SGA_path)
        # label_SGB = Image.open(label_SGB_path)
        # label_BCD = np.asarray(Image.open(label_BCD_path)).astype(np.uint8) - 1
        # ori 1024
        # label_SGA = Image.fromarray((np.asarray(Image.open(label_SGA_path))- 1).astype(np.uint8))
        # label_SGB = Image.fromarray((np.asarray(Image.open(label_SGB_path))- 1).astype(np.uint8))
        # new 512
        label_SGA = Image.fromarray((np.asarray(Image.open(label_SGA_path))).astype(np.uint8))
        # label_SGB = Image.fromarray((np.asarray(Image.open(label_SGB_path))).astype(np.uint8))

        sample = {'img_A': img_A,
                #   'img_B': img_B,
                #   'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                #   'label_SGB': label_SGB
                  }
                   
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
        sample['name'] = img_A_path
        return sample



class SECONDDataset512(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.seg_ignore_value = args.num_segclass
        self.split = split
        self.train_percent = args.train_percent
        if split == 'train':  
            self.full_load = full_path_loader_for_txt(args.train_txt_path, train_percent=self.train_percent)
        elif split =='val':
            self.full_load = full_path_loader_for_txt(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.test_txt_path)
        
        # self.train_transforms = tr.get_train_transforms(seg_ignore_value=9, with_colorjit=self.with_colorjit)
        self.train_transforms = tr.get_train_transforms(seg_ignore_value=self.seg_ignore_value, with_colorjit=self.with_colorjit)
        self.test_transforms = tr.test_transforms_clean
        # if not self.pretrained:
        #     self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
        #     self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)           
        
    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_BCD_path = self.full_load[idx]['label_BCD']
        label_SGA_path = self.full_load[idx]['label_SGA']
        label_SGB_path = self.full_load[idx]['label_SGB']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        # label_array = np.asarray(Image.open(label_path))
        # label_SGA = self.convert_numpy_to_Image(label_array, 0)
        # label_SGB = self.convert_numpy_to_Image(label_array, 1)
        # label_BCD = self.convert_numpy_to_Image(label_array, 2)
        # label_BCD = Image.fromarray((np.asarray(Image.open(label_BCD_path)) - 1).astype(np.uint8)) # turn 0 to 255
        label_BCD = Image.open(label_BCD_path)

        SGA_Arr = np.asarray(Image.open(label_SGA_path))
        SGB_Arr = np.asarray(Image.open(label_SGB_path))
        SGA_Arr = np.where(SGA_Arr==0,self.seg_ignore_value+1,SGA_Arr).astype(np.uint8) - 1
        SGB_Arr = np.where(SGB_Arr==0,self.seg_ignore_value+1,SGB_Arr).astype(np.uint8) - 1
        label_SGA = Image.fromarray(SGA_Arr) # turn 0 to num classes
        label_SGB = Image.fromarray(SGB_Arr) # turn 0 to num classes

            
        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                  'label_SGB': label_SGB
                  }
                   
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
        sample['name'] = img_A_path
        return sample
    
    def convert_numpy_to_Image(self, numpy_array, channel):
        return Image.fromarray(numpy_array[:, :, channel])


class SECONDOnlySEGDataset(Dataset):
    def __init__(self, args, split='train'): 
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.train_percent = args.train_percent
        self.seg_ignore_value = args.num_segclass
        self.split = split
        if split == 'train':  
            self.full_load = full_path_loader_from_txt_for_one_seg(args.train_txt_path, train_percent=self.train_percent)
        elif split =='val':
            self.full_load = full_path_loader_from_txt_for_one_seg(args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_from_txt_for_one_seg(args.test_txt_path)
        
        self.train_transforms = tr_seg.get_train_transforms_cveo(with_colorjit=self.with_colorjit)
        self.test_transforms = tr_seg.test_transforms_clean
        # if not self.pretrained:
        #     self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
        #     self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)
        
        # data Aug.
        

    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        # img_B_path = self.full_load[idx]['img_B']
        # label_BCD_path = self.full_load[idx]['label_BCD']
        label_SGA_path = self.full_load[idx]['label_SGA']
        # label_SGB_path = self.full_load[idx]['label_SGB']

        img_A = Image.open(img_A_path)
        # img_B = Image.open(img_B_path)
        # label_BCD = Image.open(label_BCD_path)
        # label_SGA = Image.open(label_SGA_path)
        # label_SGB = Image.open(label_SGB_path)
        # label_BCD = np.asarray(Image.open(label_BCD_path)).astype(np.uint8) - 1
        # ori 1024
        # label_SGA = Image.fromarray((np.asarray(Image.open(label_SGA_path))- 1).astype(np.uint8))
        # label_SGB = Image.fromarray((np.asarray(Image.open(label_SGB_path))- 1).astype(np.uint8))
        # new 512
        # label_SGA = Image.fromarray((np.asarray(Image.open(label_SGA_path))).astype(np.uint8))
        # label_SGB = Image.fromarray((np.asarray(Image.open(label_SGB_path))).astype(np.uint8))
        SGA_Arr = np.asarray(Image.open(label_SGA_path))
        SGA_Arr = np.where(SGA_Arr==0,self.seg_ignore_value+1,SGA_Arr).astype(np.uint8) - 1
        label_SGA = Image.fromarray(SGA_Arr) # turn 0 to num classes


        sample = {'img_A': img_A,
                #   'img_B': img_B,
                #   'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                #   'label_SGB': label_SGB
                  }
                   
        if self.split == 'train':       
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
        sample['name'] = img_A_path
        return sample


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from utils.parser import get_parser_with_args_from_json
    
    args = get_parser_with_args_from_json(r'F:\ContrastiveLearningCD\configs\test.json')
    
    train_dataset = SECONDDataset512(args)
      
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=None)
    print(train_dataloader.__len__())
    
    sample = next(iter(train_dataloader))
    
    colors = dict((
                    (1, (255, 255, 255, 255)), # 前三位RGB，255代表256色
                    (0, (0, 0, 0, 255)),  
                    (2, (255, 0, 0, 255)),  
                ))
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
    index_colors = [colors[key] if key in colors else
                    (255, 255, 255, 0) for key in range(0, len(colors))]
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', len(index_colors))
    
    _, axes = plt.subplots(1, 5)
    axes[0].imshow(sample['img_A'].cpu().numpy()[0, ...].transpose(1, 2, 0))
    axes[1].imshow(sample['img_B'].cpu().numpy()[0, ...].transpose(1, 2, 0))
    axes[2].imshow(sample['label_SGA'].cpu().numpy()[0, ...])
    axes[3].imshow(sample['label_SGB'].cpu().numpy()[0, ...])
    
    label_BCD = sample['label_BCD'].cpu().numpy()[0, ...]
    label_BCD[label_BCD == 255] = 2
    axes[4].imshow(label_BCD, cmap=cmap)
    plt.show()
    print(1)