# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        # https://forums.fast.ai/t/focalloss-with-multi-class/35588
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        loss = (self.alpha * (1 - pt)**self.gamma * ce_loss).mean() # mean over the batch

        return loss.sum()