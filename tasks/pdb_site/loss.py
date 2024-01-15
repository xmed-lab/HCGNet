import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class BinaryWCELoss(nn.Module):
    def __init__(self, postive_weight=5.0, reduction='mean'):
        super().__init__()
        self.postive_weight = postive_weight
        self.reduction = reduction

    def forward(self, predict, target): # only support gpu
        '''
        inputs:
        - predict: N, 1
        - target: N, ?
        '''
        loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')

        target_np = target.data.cpu().numpy()
        weights = np.ones([len(target_np), 1])
        weights[target_np == 1] = self.postive_weight
        weights = torch.from_numpy(weights).float().cuda()

        loss *= weights
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class ProteinSegLoss(nn.Module):
    def __init__(self, postive_weight=5.0, gamma=0.5):
        super().__init__()
        assert gamma >= 0 and gamma <= 1, 'invalid gamma {}'.format(gamma)
        
        self.gamma = gamma
        self.bce_func = BinaryWCELoss(postive_weight=postive_weight, reduction='mean')
        self.dice_func = BinaryDiceLoss(reduction='mean')

    def forward(self, predict, target):
        bce_loss = self.bce_func(predict, target)

        predict = torch.sigmoid(predict.unsqueeze(0))
        target = target.unsqueeze(0).float()
        dice_loss = self.dice_func(predict, target)
        
        loss = self.gamma * bce_loss + (1 - self.gamma) * dice_loss
        return loss
