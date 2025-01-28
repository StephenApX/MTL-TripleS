import numpy as np
import torch
import random
import os
from functools import reduce
import argparse as ag

import sys
sys.path.append("..") 
from model import *
# from torchstat import stat
from thop import profile
import time


def seed_torch(seed=6):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
 
 
def count_model_parameters(module, _default_logger=None):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    print('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))
    return cnt  


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()
 
    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0
 
    # benchmark with 2000 image and take the average
    for i in range(max_iter):
 
        torch.cuda.synchronize()
        start_time = time.perf_counter()
 
        with torch.no_grad():
            model(*data)
 
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
 
        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)
 
        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps
 
    
def get_model(args):
    if args.model == 'CVEOScd4':
        return CVEOScd4(args)

   
if __name__ == '__main__':
    pass