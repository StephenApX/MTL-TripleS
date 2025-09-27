import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math

class BCDEvaluator(object):
    '''only see change class'''
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
        
    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self): # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre==0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre[-1]

    def Recall(self): # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec==0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec[-1]
    
    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP[1] + self.FN[1] + self.FP[1]
        IoU = np.where(IoU==0, 0, self.TP[1] / IoU)
        IoU = np.round(np.nanmean(IoU), 5)
        return IoU
    
    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2* self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1[-1] # return change class
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False
        
        
class SEGEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
        
    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self): # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre==0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre

    def Recall(self): # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec==0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec

    def Mean_Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        MIoU = self.TP + self.FN + self.FP
        MIoU = np.where(MIoU==0, 0, self.TP / MIoU)
        MIoU = np.round(np.nanmean(MIoU), 5)
        return MIoU
    
    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP + self.FN + self.FP
        IoU = np.where(IoU==0, 0, self.TP / IoU)
        return IoU
    
    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2* self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False


class SCDEvaluator(object): #  Sek
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2) # row is True col is pred
        self.pre_cal = False
    
    @staticmethod    
    def cal_kappa(hist):
        if hist.sum() == 0:
            po = 0
            pe = 1
            kappa = 0
        else:
            po = np.diag(hist).sum() / hist.sum()
            pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
            print('po, pe:', po, pe)
            
            if pe == 1:
                kappa = 0
            else:
                kappa = (po - pe) / (1 - pe)
        return kappa
    

    def mIoU_and_SeK(self):
        hist = self.confusion_matrix
        print('hist:', hist)

        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        print('c2hist:', c2hist)

        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        print('hist_n0:', hist_n0)


        kappa_n0 = self.cal_kappa(hist_n0)
        print('kappa_n0:', kappa_n0)

        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

        return iu[0], iu[1], IoU_mean, Sek

    
    def _generate_matrix(self, gt_image, pre_image): # 0:nochange, 1-c:change-lc-class. 
        # So input cls must be: 1(nochange) + C(change-lc) = C+1
        mask = (gt_image >= 0) & (gt_image <= self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pre_cal = False


if __name__ == '__main__':
    pass
    
