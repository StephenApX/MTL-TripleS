import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import cv2

from utils.evaluator import BCDEvaluator, SEGEvaluator, SCDEvaluator
from utils.helper import seed_torch, get_model

from dataset.dataset import CVEODataset, HRSCDDataset512


def convert_color(src_val, color_map):
    return tuple(color_map[src_val])
vfunc = np.vectorize(convert_color, otypes=["uint8","uint8","uint8"])
def int2rgb(src, color_map):
    (width, height) = src.shape
    src = src.reshape(width * height)
    output = np.array(vfunc(src, color_map), "uint8")
    return output.reshape(3, width, height).transpose([1,2,0])


def split_sample(sample):
    img_A = sample['img_A'].cuda(non_blocking=True)
    img_B = sample['img_B'].cuda(non_blocking=True)
    label_BCD = sample['label_BCD'].cuda(non_blocking=True)
    label_SGA = sample['label_SGA'].cuda(non_blocking=True)
    label_SGB = sample['label_SGB'].cuda(non_blocking=True)   
    return img_A, img_B, label_BCD, label_SGA.long(), label_SGB.long()

def get_dataset(args):
    if args.dataset == 'SCSCD7' or args.dataset == 'CCSCD5':
        val_dataset = CVEODataset(args, split='val')   
    elif args.dataset == 'HRSCD':
        val_dataset = HRSCDDataset512(args, split='val')    
    return val_dataset

def get_color_map(args):
    if args.dataset == 'SCSCD7':
        color_map = {
            0:(255,255,255), # nochange
            1:(255,128,0), # bareland
            2:(0,0,255), # blue, water
            3:(255,0,0), # red
            4:(255,255,0), # yellow
            5:(0,255,0), # green
            6:(0,128,0), # dark green
            7:(128,128,128) # grey
        } # RGB
    elif args.dataset == 'CCSCD5':
        color_map = { # of 
            0:(255,255,255), # nochange
            1:(255,128,0), # orange: others: bareland
            2:(0,0,255), # blue: water
            3:(255,0,0), # red: bld
            4:(0,255,0), # light green: plant
            5:(255,255,0), # yellow: road
        } # RGB
    elif args.dataset == 'HRSCD':
        color_map = { # of 
            0:(255,255,255), # nochange
            1:(255,0,0),# red: Artificial surface. bld
            2:(255,128,0), # orange: Agricultural areas
            3:(0,255,0),# light green: forest
            4:(255,0,255),# purple: wetland
            5:(0,0,255) # blue: water
        } # RGB
    return color_map


def main(args):
    seed_torch()

    model = get_model(args).cuda()
    val_dataset = get_dataset(args)
    color_map = get_color_map(args)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, persistent_workers=True, pin_memory=True, num_workers=args.val_num_workers, drop_last=False)

    evaluator_bcd = BCDEvaluator()
    evaluator_seg_A = SEGEvaluator(args.num_segclass)
    evaluator_seg_B = SEGEvaluator(args.num_segclass)
    evaluator_seg_total = SEGEvaluator(args.num_segclass)
    evaluator_scd = SCDEvaluator(args.num_segclass+1)
    from_to_mat = np.zeros((args.num_segclass+1,)*2)
    
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

    out_dir = str(args.congfig_file).replace('.json', '').replace('configs', 'infer')
    os.makedirs(out_dir, exist_ok=True)
    prev_seg_dir = os.path.join(out_dir, 'prev_seg')
    curr_seg_dir = os.path.join(out_dir, 'curr_seg')
    bcd_dir = os.path.join(out_dir, 'bcd')
    masked_prev_seg_dir = os.path.join(out_dir, 'mask_prev_seg')
    masked_curr_seg_dir = os.path.join(out_dir, 'mask_curr_seg')
    os.makedirs(prev_seg_dir, exist_ok=True)
    os.makedirs(curr_seg_dir, exist_ok=True)
    os.makedirs(bcd_dir, exist_ok=True) 
    os.makedirs(masked_prev_seg_dir, exist_ok=True)
    os.makedirs(masked_curr_seg_dir, exist_ok=True)

       
    
    evaluator_bcd.reset()
    evaluator_seg_A.reset()
    evaluator_seg_B.reset()
    evaluator_seg_total.reset()
    evaluator_scd.reset()

    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            print(sample['name'])
            img_name = os.path.basename(sample['name'][0])
            
            img_A, img_B, label_BCD, label_SGA, label_SGB = split_sample(sample)
            outputs = model(img_A=img_A, img_B=img_B)
            label_seg = torch.cat([label_SGA, label_SGB], dim=0)
            np_label_SGA = label_SGA.cpu().numpy().astype('int')
            np_label_SGB = label_SGB.cpu().numpy().astype('int')
            np_label_BCD = label_BCD.cpu().numpy().astype('int').squeeze()
            mask_label_seg_A = np.multiply(np.squeeze(np_label_SGA)+1, np_label_BCD)
            mask_label_seg_B = np.multiply(np.squeeze(np_label_SGB)+1, np_label_BCD)

            pred_bcd = outputs['BCD'].sigmoid().squeeze().cpu().detach().numpy().round().astype('int')
            pred_seg_A = torch.argmax(outputs['seg_A'], 1).cpu().detach().numpy().astype('int')
            pred_seg_B = torch.argmax(outputs['seg_B'], 1).cpu().detach().numpy().astype('int')
            pred_seg = np.concatenate([pred_seg_A, pred_seg_B], axis=0)
            mask_pred_seg_A = np.multiply(np.squeeze(pred_seg_A)+1, pred_bcd)
            mask_pred_seg_B = np.multiply(np.squeeze(pred_seg_B)+1, pred_bcd)


            # eval.
            evaluator_bcd.add_batch(label_BCD.cpu().numpy().astype('int').squeeze(), pred_bcd)
            evaluator_seg_A.add_batch(label_SGA.cpu().numpy().astype('int'), pred_seg_A)
            evaluator_seg_B.add_batch(label_SGB.cpu().numpy().astype('int'), pred_seg_B)
            evaluator_seg_total.add_batch(label_seg.cpu().numpy().astype('int'), pred_seg)
            evaluator_scd.add_batch(mask_label_seg_A, mask_pred_seg_A)
            evaluator_scd.add_batch(mask_label_seg_B, mask_pred_seg_B)

            # from-to mat.
            from_to_pred = (args.num_segclass+1) * mask_pred_seg_A + mask_pred_seg_B
            from_to_count = np.bincount(from_to_pred.flatten(), minlength=(args.num_segclass+1)**2)
            from_to_mat += from_to_count.reshape(args.num_segclass+1, args.num_segclass+1)


            # int2rgb
            cv2.imwrite(os.path.join(bcd_dir, img_name), pred_bcd * 255)
            clsA_map = int2rgb(np.squeeze(pred_seg_A)+1, color_map)[:,:,::-1]
            cv2.imwrite(os.path.join(prev_seg_dir, img_name), clsA_map)
            clsB_map = int2rgb(np.squeeze(pred_seg_B)+1, color_map)[:,:,::-1]
            cv2.imwrite(os.path.join(curr_seg_dir, img_name), clsB_map)
            mask_clsA_map = int2rgb(np.squeeze(mask_pred_seg_A), color_map)[:,:,::-1]
            mask_clsB_map = int2rgb(np.squeeze(mask_pred_seg_B), color_map)[:,:,::-1]
            cv2.imwrite(os.path.join(masked_prev_seg_dir, img_name), mask_clsA_map)
            cv2.imwrite(os.path.join(masked_curr_seg_dir, img_name), mask_clsB_map)

            
        # eval BCD.
        OA_bcd = evaluator_bcd.Overall_Accuracy()
        IoU_bcd = evaluator_bcd.Intersection_over_Union()
        F1_bcd = evaluator_bcd.F1_score()


        # eval SEG.
        OA_seg_A = evaluator_seg_A.Overall_Accuracy()
        mIoU_seg_A = evaluator_seg_A.Mean_Intersection_over_Union()
        F1_seg_A = evaluator_seg_A.F1_score().mean()
        OA_seg_B = evaluator_seg_B.Overall_Accuracy()
        mIoU_seg_B = evaluator_seg_B.Mean_Intersection_over_Union()
        F1_seg_B = evaluator_seg_B.F1_score().mean()
        
        OA_seg_total = evaluator_seg_total.Overall_Accuracy()
        mIoU_seg_total = evaluator_seg_total.Mean_Intersection_over_Union()
        F1_seg_total = evaluator_seg_total.F1_score().mean()
        

        # eval SCD.
        iou_nochange, iou_change, IoU_mean, Sek = evaluator_scd.mIoU_and_SeK()
        # save from-to
        scd_cm = from_to_mat[1:,1:].astype(np.float32)
        sum_scd_cm = np.sum(scd_cm)
        perc_cm = scd_cm / float(sum_scd_cm)
        config_name = str(os.path.basename(args.congfig_file)).replace('.json', '')
        np.save(os.path.join(out_dir, f'{config_name}_cm.npy'), from_to_mat)
        np.save(os.path.join(out_dir, f'{config_name}_perc_cm.npy'), perc_cm)
            

        assessment = {}
        assessment['BCD'] = {}
        assessment['BCD']['OA_bcd'] = np.round(OA_bcd,5)
        assessment['BCD']['IoU_bcd'] = np.round(IoU_bcd,5)
        assessment['BCD']['F1_bcd'] = np.round(F1_bcd,5)

        assessment['SEG'] = {}
        assessment['SEG']['prev'] = {}
        assessment['SEG']['prev']['OA_seg_A'] = np.round(OA_seg_A,5)
        assessment['SEG']['prev']['mIoU_seg_A'] = np.round(mIoU_seg_A,5)
        assessment['SEG']['prev']['mF1_seg_A'] = np.round(F1_seg_A,5)
        assessment['SEG']['curr'] = {}
        assessment['SEG']['curr']['OA_seg_B'] = np.round(OA_seg_B,5)
        assessment['SEG']['curr']['mIoU_seg_B'] = np.round(mIoU_seg_B,5)
        assessment['SEG']['curr']['mF1_seg_B'] = np.round(F1_seg_B,5)
        assessment['SEG']['whole'] = {}
        assessment['SEG']['whole']['OA_seg_total'] = np.round(OA_seg_total,5)
        assessment['SEG']['whole']['mIoU_seg_total'] = np.round(mIoU_seg_total,5)
        assessment['SEG']['whole']['mF1_seg_total'] = np.round(F1_seg_total,5)

        assessment['SCD'] = {}
        assessment['SCD']['iou_nochange'] = np.round(iou_nochange,5)
        assessment['SCD']['iou_change'] = np.round(iou_change,5)
        assessment['SCD']['IoU_mean'] = np.round(IoU_mean,5)
        assessment['SCD']['Sek'] = np.round(Sek,5)


        with open(os.path.join(out_dir, 'eval.log'), 'w')as file:
            for task, eval_dict in assessment.items():
                file.write('{0}\n'.format(task))
                for metric, value in eval_dict.items():
                    file.write('{0}\t{1}\n'.format(metric, value))
                file.write('\n')
            file.close()




if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Training change detection network')
    parser.add_argument('-c', '--congfig_file', type=str, help='congfigs_name')
    parser.add_argument('-p', '--ckpt_path', type=str, help='ckpt_path')

    params = parser.parse_args()  

    with open(str(params.congfig_file), 'r') as fin:
        configs = json.load(fin)
        parser.set_defaults(**configs)
        parser.add_argument('--congfig_name', default=str(os.path.basename(params.congfig_file).split('.')[0]), type=str)
        params = parser.parse_args()  

    main(params)
