"""Focal loss implementation."""

import torch
import torch.nn as nn


# Support: ['FocalLoss']


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7, weight = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        # weight lets callers class-weight the inner CE the same way they would a
        # plain nn.CrossEntropyLoss.
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
