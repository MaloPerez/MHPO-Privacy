"""BioTask_*Layers classifier heads and the Massive Attribute Classifier (MAC) branch,
used as the privacy adversary / auxiliary attribute model across the baselines."""

import math

import torch
import torch.nn as nn
from torch.nn import Parameter


class MACBranch(nn.Module):
    """
        Branch of the Massive Attribute Classifier (MAC). One branch per attribute,
        outputs prediction for each attribute.

        Described: https://ieeexplore.ieee.org/document/9469725
    """

    def __init__(self, num_classes=2):
        super(MACBranch, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MACFull(nn.Module):
    """
        Full Massive Attribute Classifier (MAC).
        Contains initial layers and every subsequent branch.

        Described: https://ieeexplore.ieee.org/document/9469725
    """

    def __init__(self, num_attributes=40, embedding_size=512):
        super(MACFull, self).__init__()
        self.num_attributes = num_attributes
        self.layer1 = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout()
        )

        self.branches = nn.ModuleList([MACBranch() for i in range(num_attributes)])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        attributes_logits = []
        for branch in self.branches:
            x_branch = branch.forward(x).unsqueeze(1)
            attributes_logits.append(x_branch)

        return torch.cat(attributes_logits, dim=1)

    def init_weights_model(self, type_init):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if type_init == "normal":
                    nn.init.normal_(layer.weight, 0, 1 / math.sqrt(layer.weight.size(1)))
                elif type_init == 'xavier':
                    nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain("relu"))


class BioTasks(nn.Module):
    """
        Full Massive Attribute Classifier (MAC).
        Contains initial layers and every subsequent branch.

        Described: https://ieeexplore.ieee.org/document/9469725
        NUM_CLASS_DICT = {'AGE': 9,
                      'RACE': 7,
                      'GENDER': 2}
    """

    def __init__(self, num_classes=[2, 7, 9], embedding_size=512):
        super(BioTasks, self).__init__()
        self.num_classes = num_classes
        self.num_attributes = len(self.num_classes)
        self.layer1 = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout()
        )

        self.branches = nn.ModuleList([MACBranch(self.num_classes[i]) for i in range(self.num_attributes)])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        attributes_logits = []
        for branch in self.branches:
            x_branch = branch.forward(x)
            attributes_logits.append(x_branch)

        return attributes_logits

    def init_weights_model(self, type_init):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if type_init == "normal":
                    nn.init.normal_(layer.weight, 0, 1 / math.sqrt(layer.weight.size(1)))
                elif type_init == 'xavier':
                    nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain("relu"))


class BioTask_2Layers(nn.Module):

    def __init__(self, in_features, out_features):
        super(BioTask_2Layers, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, out_features),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class BioTask_1Layer(nn.Module):

    def __init__(self, in_features, out_features):
        super(BioTask_1Layer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class BioTask_5Layers(nn.Module):

    def __init__(self, in_features, out_features):
        super(BioTask_5Layers, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 1064),
            nn.ReLU(),
            nn.BatchNorm1d(1064),
            nn.Dropout()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1064, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(128, out_features),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class BioTask_7Layers(nn.Module):

    def __init__(self, in_features, out_features):
        super(BioTask_7Layers, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 1064),
            nn.ReLU(),
            nn.BatchNorm1d(1064),
            nn.Dropout()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(1064, 1064),
            nn.ReLU(),
            nn.BatchNorm1d(1064),
            nn.Dropout()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(1064, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout()
        )

        self.layer6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(128, out_features),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
