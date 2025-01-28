class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_half(self, val):
        self.sum += val
    
    def update_otherhalf(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

class AverageList(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val_list):
        self.val = val_list
        if isinstance(self.sum, int):
            self.sum = self.val
        else:
            self.sum = [val_in+val_sum for val_in, val_sum in zip(self.val, self.sum)]
        self.count += 1
        # self.avg = [val_sum/self.count for val_sum in self.sum]

    def get_avg(self):

        return [val_sum/self.count for val_sum in self.sum]
