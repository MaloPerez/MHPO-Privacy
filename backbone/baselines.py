# based on vgg16

import torch
import torch.nn as nn
from torch.distributions import Laplace


class Encoder(nn.Module):

    def __init__(self, layers = 1):
        super(Encoder, self).__init__()

        self.layers = layers

        self.features1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features3 = nn.Sequential(

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features4 = nn.Sequential(

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features5 = nn.Sequential(

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

    def forward(self, x):
        p1 = self.features1(x)
        x, p1_idx = self.pool1(p1)
        p2 = self.features2(x)
        x, p2_idx = self.pool2(p2)

        if self.layers == 4:
            return x

        p3 = self.features3(x)
        x, p3_idx = self.pool3(p3)

        if self.layers == 3:
            return x

        p4 = self.features4(x)
        x, p4_idx = self.pool4(p4)

        if self.layers == 2:
            return x

        p5 = self.features5(x)
        x, p5_idx = self.pool5(p5)
        x = x.view(x.size(0), -1)

        return x  # cp1_idx, p2_idx  , p3_idx, p4_idx, p5_idx


class Public_Classifier(nn.Module):

    def __init__(self, input=4608, output=1, layers=1):
        super(Public_Classifier, self).__init__()

        self.layers = layers

        if layers < 1 or layers > 4:
            raise "Layers has to be between 1 and 4"

        if layers == 4:
            self.features3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )

            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                      return_indices=True)

        if layers >= 3:
            self.features4 = nn.Sequential(

                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )

            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                      return_indices=True)

        if layers >= 2:
            self.features5 = nn.Sequential(

                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )

            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                      return_indices=True)

        self.classifier = nn.Sequential(
            nn.Linear(input, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, output, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        if self.layers == 4:
            p3 = self.features3(x)
            x, p3_idx = self.pool3(p3)
        if self.layers >= 3:
            p4 = self.features4(x)
            x, p4_idx = self.pool4(p4)
        if self.layers >= 2:
            p5 = self.features5(x)
            x, p5_idx = self.pool5(p5)
            x = x.view(x.size(0), -1)

        out = self.classifier(x)

        return out


class PCAReduction(nn.Module):
    def __init__(self, input_dim, output_dim, pretrained_matrix=None, freeze=True):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

        if pretrained_matrix is not None:
            # Initialize with pretrained PCA matrix
            self.projection.weight.data = pretrained_matrix.T  # shape: (output_dim, input_dim)
            if freeze:
                self.projection.weight.requires_grad = False

    def forward(self, x):
        return self.projection(x)


class PCADecompression(nn.Module):
    def __init__(self, pca_matrix):
        super().__init__()
        self.pca_matrix = pca_matrix.t()

    def forward(self, x):
        x = torch.matmul(x, self.pca_matrix)
        return x


class NoiseInjection(nn.Module):
    def __init__(self, scale=2.0, device='cuda:0'):
        super().__init__()
        self.dist = Laplace(loc=0.0, scale=scale)
        self.device = device

    def forward(self, x):
        noise = self.dist.sample(x.shape).to(self.device)
        return x + noise

class PrivacyEmbedding(nn.Module):
    def __init__(self, pca_module, noise_module):
        super().__init__()
        self.pca = pca_module
        self.noise = noise_module

    def forward(self, x):
        x = self.pca(x)
        x = self.noise(x)
        return x