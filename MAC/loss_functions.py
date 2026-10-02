"""Multi-attribute branch losses for the Massive Attribute Classifier (MAC)."""

import torch
import torch.nn as nn

class BranchCrossEntropyLoss(nn.Module):
    '''
        Categorical cross-entropy loss for each attribute branch followed by
        equal weighting between each of these attribute-related losses.
    '''

    def __init__(self, class_weights):
        super(BranchCrossEntropyLoss, self).__init__()

        losses = []
        for weights in class_weights:
            losses.append(torch.nn.CrossEntropyLoss(weight=weights))
        self.balanced_losses = nn.ModuleList(losses)

    def forward(self,  logits, target):
        assert logits.shape == target.shape
        nb_attributes = logits.shape[1]

        branch_loss = []
        for i in range(nb_attributes):
            branch_loss.append(self.balanced_losses[i](logits[:, i, :], target[:, i, :].to(torch.float32)))
        return torch.mean(torch.stack(branch_loss)), branch_loss


class BranchCrossEntropyLossBioTask(nn.Module):
    '''
        Categorical cross-entropy loss for each attribute branch followed by
        equal weighting between each of these attribute-related losses.
    '''

    def __init__(self, class_weights):
        super(BranchCrossEntropyLossBioTask, self).__init__()

        losses = []
        for weights in class_weights:
            losses.append(torch.nn.CrossEntropyLoss(weight=weights))
        self.balanced_losses = nn.ModuleList(losses)

    def forward(self,  logits, target):
        branch_loss = []
        for i in range(len(logits)):
            branch_loss.append(self.balanced_losses[i](logits[i], target[i]))
        return torch.mean(torch.stack(branch_loss)), branch_loss
