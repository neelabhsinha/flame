import torch
from models.residual_block import ResidualBlock
import torch.nn as nn
import numpy as np

# The defined CNN Backbone for extracting features from image
class CNNBackbone(nn.Module):
    def __init__(self, in_channels):
        super(CNNBackbone, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.l1 = ResidualBlock(in_channels=64, out_channels=64)
        self.l2 = ResidualBlock(in_channels=64, out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.l3 = ResidualBlock(in_channels=64, out_channels=128)
        self.l4 = ResidualBlock(in_channels=128, out_channels=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.l5 = ResidualBlock(in_channels=128, out_channels=256)
        self.l6 = ResidualBlock(in_channels=256, out_channels=256)


    def forward(self, img):
        img = self.initial(img)
        img = self.maxpool1(img)
        img = self.l1(img)
        img = self.l2(img)
        img = self.maxpool2(img)
        img = self.l3(img)
        img = self.l4(img)
        img = self.maxpool3(img)
        img = self.l5(img)
        img = self.l6(img)
        # img = self.maxpool4(img)
        return img
