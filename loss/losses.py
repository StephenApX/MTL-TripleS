import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn
import torch.nn.functional as F


class BCEWithIgnoreLoss(nn.Module):
    def __init__(self, ignore_index=255, OHEM=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.OHEM = OHEM

    def forward(self, logits, target):
        if len(logits.shape) != len(target.shape) and logits.shape[1] == 1:
            logits = logits.squeeze(1)
            
        target = target.float()
        valid_mask = (target != self.ignore_index)
        loss = self.bce(logits, target)
        
        # OHEM
        if self.OHEM:
            loss_, _ = loss.contiguous().view(-1).sort()
            min_value = loss_[int(0.5 * loss.numel())]
            
            loss = loss[valid_mask]
            loss = loss[loss >= min_value]
        else:
            loss = loss[valid_mask]
        
        return loss.mean()
 

class BCDLoss(nn.Module):
    def __init__(self,
                 losses=[BCEWithIgnoreLoss(), DiceLoss(mode='binary', ignore_index=255)],
                 loss_weight=[1, 1]):
        super(BCDLoss, self).__init__()
        self.loss_weights = loss_weight
        self.losses = losses

    def forward(self, logits, target):
        losses = {}
        for i in range(len(self.losses)):
            loss = self.losses[i](logits, target)
            losses[i] = loss * self.loss_weights[i]
        losses["loss"] = sum(losses.values())
        return losses["loss"]
    
