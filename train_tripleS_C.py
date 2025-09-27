import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from loss.losses import BCDLoss
from utils.callbacks import AverageMeter
from utils.evaluator import BCDEvaluator, SEGEvaluator
from utils.helper import get_lr, seed_torch, get_model
from utils.saver import Saver
from utils.logger import Logger as Log

from dataset.dataset import CVEODataset, CVEOOnlySEGDataset, HRSCDDataset512, HRSCDOnlySEGDataset


def split_sample(sample):
    img_A = sample['img_A'].cuda(non_blocking=True)
    img_B = sample['img_B'].cuda(non_blocking=True)
    label_BCD = sample['label_BCD'].cuda(non_blocking=True)
    label_SGA = sample['label_SGA'].cuda(non_blocking=True)
    label_SGB = sample['label_SGB'].cuda(non_blocking=True)   
    return img_A, img_B, label_BCD, label_SGA.long(), label_SGB.long()
    
def split_single_sample(sample):
    img_A = sample['img_A'].cuda(non_blocking=True)
    label_SGA = sample['label_SGA'].cuda(non_blocking=True)
    return img_A, label_SGA.long()


def get_dataset(args):
    if args.dataset == 'SCSCD7' or args.dataset == 'CCSCD5':
        train_dataset = CVEODataset(args, split='train')
        val_dataset = CVEODataset(args, split='val')   
    elif args.dataset == 'HRSCD':
        train_dataset = HRSCDDataset512(args, split='train')
        val_dataset = HRSCDDataset512(args, split='val')    
    return train_dataset, val_dataset

def get_singleSEG_dataset(args):
    if args.dataset == 'SCSCD7' or args.dataset == 'CCSCD5':
        train_dataset = CVEOOnlySEGDataset(args, split='train')
        val_dataset = CVEOOnlySEGDataset(args, split='val') 
    elif args.dataset == 'HRSCD':
        train_dataset = HRSCDOnlySEGDataset(args, split='train')
        val_dataset = HRSCDOnlySEGDataset(args, split='val')     
    return train_dataset, val_dataset


def main(args):  
    seed_torch()
    
    model = get_model(args).cuda()
    train_double_dataset, val_double_dataset = get_dataset(args)
    train_single_dataset, _ = get_singleSEG_dataset(args) 
    
    drop_last = True
    
    # double input.
    train_double_loader = DataLoader(train_double_dataset, batch_size=args.batch_size, shuffle=True, 
                            persistent_workers=True, pin_memory=True, num_workers=args.num_workers, drop_last=drop_last)
    val_double_loader = DataLoader(val_double_dataset, batch_size=args.val_batch_size, shuffle=False, 
                            persistent_workers=True, pin_memory=True, num_workers=args.val_num_workers, drop_last=drop_last)
    # single input.
    train_single_loader = DataLoader(train_single_dataset, batch_size=args.batch_size, shuffle=True, 
                            persistent_workers=True, pin_memory=True, num_workers=args.num_workers, drop_last=drop_last)

    
    loss_func_bcd = BCDLoss()
    loss_func_seg = torch.nn.CrossEntropyLoss(ignore_index=args.num_segclass)  # 
    

    seg_params = []
    for named_parameter in [model.encoder.named_parameters(), model.seg_decoder.named_parameters(), model.head_seg.named_parameters()]:
        for pname, p in named_parameter:
            seg_params += [p]
    seg_params_id = list(map(id, seg_params))
    other_params = list(filter(lambda p: id(p) not in seg_params_id, model.parameters())) # bcd


    # seg task optim.
    optimizer_SEG = torch.optim.Adam(seg_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler_SEG = StepLR(optimizer_SEG, step_size=10, gamma=0.9)
    # bcd task optim.
    optimizer_BCD = torch.optim.Adam(other_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler_BCD = StepLR(optimizer_BCD, step_size=10, gamma=0.9)
    # global optim.
    optimizer_whole = torch.optim.Adam(model.parameters(), lr=args.learning_rate_whole, weight_decay=args.weight_decay)


    saver = Saver(args)    
    evaluator_bcd = BCDEvaluator()
    evaluator_seg_A = SEGEvaluator(args.num_segclass)
    evaluator_seg_B = SEGEvaluator(args.num_segclass)
    evaluator_seg_total = SEGEvaluator(args.num_segclass)
    metric_best = -1
    metric_best_dict = {}
    start_epoch = 1
    Log.init(logfile_level="info", log_file=saver.experiment_dir + '/log.log')

    if isinstance(args.resume, str):
        checkpoint = torch.load(args.resume)
        checkpoint['epoch'] = 1   
        model.load_state_dict(checkpoint['state_dict']) 
        Log.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        del checkpoint      
    epoch_best = start_epoch   
    

    sta3_sub_loss = np.zeros((3))
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc=args.congfig_name):     
        
        ''' traning '''
        losses_seg = AverageMeter()
        losses_bcd = AverageMeter()
        losses_total = AverageMeter()
        
        model.train()

        sort_idx = np.argsort(-sta3_sub_loss)
        for sub_case in list(sort_idx):
            if sub_case == 0:
                # situation 0: iterate seg samples and update seg-related params.
                for batch_idx, sample in enumerate(train_single_loader): # single seg.
                    imgs, label_seg = split_single_sample(sample)
                    hook = 'seg'
                    outputs = model(imgs, hook=hook)

                    loss_seg = loss_func_seg(outputs['seg_A'], label_seg)
                    optimizer_SEG.zero_grad()
                    loss_seg.backward() # Computes the gradient 
                    optimizer_SEG.step() # update params by the gradient
                    
                    losses_seg.update(loss_seg.item())
                    sta3_sub_loss[0] = loss_seg.item()
                        
                    if batch_idx % args.print_step == 0 or batch_idx == len(train_single_loader)-1:
                        print('[Epoch:%3d/%3d | Batch:%4d/%4d | SEG] loss_seg: %.4f  lr: %5f' % 
                            (epoch, args.epochs, batch_idx+1, train_single_loader.__len__(), losses_seg.avg, get_lr(optimizer_SEG))
                        )
                        if batch_idx == len(train_single_loader)-1:
                            Log.info('[Epoch:%3d/%3d | Batch:%4d/%4d | SEG] loss_seg: %.4f  lr: %5f' % 
                            (epoch, args.epochs, batch_idx+1, train_single_loader.__len__(), losses_seg.avg, get_lr(optimizer_SEG))
                            )
            
            if sub_case == 1:
                # situation 1: iterate bcd samples and update bcd-related params.
                for batch_idx, sample in enumerate(train_double_loader): # double 
                    img_A, img_B, label_BCD, label_SGA, label_SGB = split_sample(sample)
                
                    outputs = model(img_A=img_A, img_B=img_B)
                    
                    # whether only segmentation
                    loss_cd = loss_func_bcd(outputs['BCD'], label_BCD)
                    optimizer_BCD.zero_grad()
                    loss_cd.backward() # Computes the gradient 
                    optimizer_BCD.step() # update params by the gradient

                    losses_bcd.update(loss_cd.item())
                    sta3_sub_loss[1] = loss_cd.item()
                    
                    if batch_idx % args.print_step == 0 or batch_idx == len(train_double_loader)-1:
                        print('[Epoch:%3d/%3d | Batch:%4d/%4d | BCD] loss_bcd: %.4f | lr: %5f' % 
                            (epoch, args.epochs, batch_idx+1, train_double_loader.__len__(), losses_bcd.avg, get_lr(optimizer_BCD))
                        )
                        if batch_idx == len(train_single_loader)-1:
                            Log.info('[Epoch:%3d/%3d | Batch:%4d/%4d | BCD] loss_bcd: %.4f | lr: %5f' % 
                                (epoch, args.epochs, batch_idx+1, train_double_loader.__len__(), losses_bcd.avg, get_lr(optimizer_BCD))
                            )

            if sub_case == 2:
                # situation 2: iterate scd samples and update whole params.
                for batch_idx, sample in enumerate(train_double_loader): # double 
                    img_A, img_B, label_BCD, label_SGA, label_SGB = split_sample(sample)
                    outputs = model(img_A=img_A, img_B=img_B)
                    
                    loss_seg = loss_func_seg(outputs['seg_A'], label_SGA) + loss_func_seg(outputs['seg_B'], label_SGB)
                    loss_cd = loss_func_bcd(outputs['BCD'], label_BCD)
                    loss = (loss_seg * 0.5 + loss_cd) * 0.5

                    optimizer_whole.zero_grad()
                    loss.backward() # Computes the gradient 
                    optimizer_whole.step() # update params by the gradient

                    losses_total.update(loss.item())
                    sta3_sub_loss[2] = loss.item()
                    
                    if batch_idx % args.print_step == 0 or batch_idx == len(train_double_loader)-1:
                        print('[Epoch:%3d/%3d | Batch:%4d/%4d | STA3] losses_total: %.4f | lr: %5f' % 
                            (epoch, args.epochs, batch_idx+1, train_double_loader.__len__(), losses_total.avg, get_lr(optimizer_whole))
                        )
                        if batch_idx == len(train_single_loader)-1:
                            Log.info('[Epoch:%3d/%3d | Batch:%4d/%4d | STA3] losses_total: %.4f | lr: %5f' % 
                                (epoch, args.epochs, batch_idx+1, train_double_loader.__len__(), losses_total.avg, get_lr(optimizer_whole))
                            )


        lr_scheduler_SEG.step() 
        lr_scheduler_BCD.step()

        Log.info('[Training Epoch:%3d/%3d] loss_bcd: %.4f loss_seg: %.4f loss_sta3: %.4f  | lr: %f/%f/%f' % (epoch, args.epochs, losses_bcd.avg, losses_seg.avg, losses_total.avg, get_lr(optimizer_SEG), get_lr(optimizer_BCD), get_lr(optimizer_whole))) 
        

        '''
        validation
        '''
        evaluator_bcd.reset()
        evaluator_seg_A.reset()
        evaluator_seg_B.reset()
        evaluator_seg_total.reset()
        
        valosses_seg_A = AverageMeter()
        valosses_seg_B = AverageMeter()
        valosses_bcd = AverageMeter()
        valosses_total = AverageMeter()
        
        model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_double_loader):
                
                img_A, img_B, label_BCD, label_SGA, label_SGB = split_sample(sample)
                outputs = model(img_A=img_A, img_B=img_B)
                
                # loss.
                loss_cd = loss_func_bcd(outputs['BCD'], label_BCD)
                loss_seg_A = loss_func_seg(outputs['seg_A'], label_SGA)
                loss_seg_B = loss_func_seg(outputs['seg_B'], label_SGB)
                loss = loss_cd + loss_seg_A + loss_seg_B 
                valosses_bcd.update(loss_cd.item())    
                valosses_seg_A.update(loss_seg_A.item())
                valosses_seg_B.update(loss_seg_B.item())
                valosses_total.update(loss.item())
                

                pred_bcd = outputs['BCD'].sigmoid().squeeze().cpu().detach().numpy().round().astype('int')
                evaluator_bcd.add_batch(label_BCD.cpu().numpy().astype('int').squeeze(), pred_bcd)
                
                pred_seg_A = torch.argmax(outputs['seg_A'], 1).cpu().detach().numpy().astype('int')
                pred_seg_B = torch.argmax(outputs['seg_B'], 1).cpu().detach().numpy().astype('int')
                pred_seg = np.concatenate([pred_seg_A, pred_seg_B], axis=0)
                label_seg = torch.cat([label_SGA, label_SGB], dim=0)
                evaluator_seg_A.add_batch(label_SGA.cpu().numpy().astype('int'), pred_seg_A)
                evaluator_seg_B.add_batch(label_SGB.cpu().numpy().astype('int'), pred_seg_B)
                evaluator_seg_total.add_batch(label_seg.cpu().numpy().astype('int'), pred_seg)
                
                if batch_idx % args.print_step == 0 or batch_idx == len(val_double_loader)-1:
                    print('[Epoch:%3d/%3d | Batch:%4d/%4d] loss_total: %.4f loss_bcd: %.4f loss_segA: %.4f loss_segB: %.4f' %
                        (epoch, args.epochs, batch_idx+1, val_double_loader.__len__(), valosses_total.avg, 
                         valosses_bcd.avg, valosses_seg_A.avg, valosses_seg_B.avg)
                    )
                    
            Log.info('[Validation Epoch:%3d/%3d] loss_total: %.4f loss_bcd: %.4f loss_segA: %.4f loss_segB: %.4f' % 
                    (epoch, args.epochs, valosses_total.avg, valosses_bcd.avg, valosses_seg_A.avg, valosses_seg_B.avg))
            
            OA_bcd = evaluator_bcd.Overall_Accuracy()
            IoU_bcd = evaluator_bcd.Intersection_over_Union()
            
            mIoU_seg_A = evaluator_seg_A.Mean_Intersection_over_Union()
            mIoU_seg_B = evaluator_seg_B.Mean_Intersection_over_Union()
            OA_seg_total = evaluator_seg_total.Overall_Accuracy()
            mIoU_seg_total = evaluator_seg_total.Mean_Intersection_over_Union()
            
            Log.info('[Validation Epoch:%3d/%3d] OA_BCD: %.4f IoU_BCD: %.4f mIoU_SEG_total: %.4f mIoU_seg_A: %.4f mIoU_seg_B: %.4f OA_SEG_total: %.4f ' % 
            (epoch, args.epochs, OA_bcd, IoU_bcd, mIoU_seg_total, mIoU_seg_A, mIoU_seg_B, OA_seg_total))
        

        metric_current = IoU_bcd + mIoU_seg_total
        if (metric_current > metric_best) or (epoch == 1):
            metric_best_dict = {}
            metric_best_dict["IoU_BCD"] = IoU_bcd
            metric_best_dict["mIoU_SEG_A"] = mIoU_seg_A
            metric_best_dict["mIoU_SEG_B"] = mIoU_seg_B
            metric_best_dict["mIoU_SEG_total"] = mIoU_seg_total
            epoch_best = epoch
            metric_best = metric_current

            # save ckpt when achieve higheset perf.
            saver.save_checkpoint({
                'state_dict': model.state_dict(),
            }, epoch, metric_current)         
            
        print('=> Current metric %.4f Best metric %.4f' % (metric_current, metric_best))
        Log.info('=> Best epoch %3d Best metric: IoU_BCD: %.4f; mIoU_SEG_total: %.4f, mIoU_SEG_A: %.4f, mIoU_SEG_B: %.4f' % (epoch_best, metric_best_dict["IoU_BCD"], metric_best_dict["mIoU_SEG_total"], metric_best_dict["mIoU_SEG_A"], metric_best_dict["mIoU_SEG_B"]))
          
                       
if __name__ == '__main__':   

    parser = argparse.ArgumentParser(description='Training MTL semantic change detection model.')
    
    parser.add_argument('-c', '--congfig_file', type=str, help='congfigs_name')
    params = parser.parse_args()  

    with open(str(params.congfig_file), 'r') as fin:
        configs = json.load(fin)
        parser.set_defaults(**configs)
        parser.add_argument('--congfig_name', default=str(os.path.basename(params.congfig_file).split('.')[0]), type=str)
        params = parser.parse_args()  

    main(params)