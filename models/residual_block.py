import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.residual_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
        )
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None

        self.relu = nn.ReLU()

    def forward(self, x):
        input_matrix = x
        x = self.residual_connection(x)
        if self.skip is not None:
            input_matrix = self.skip(input_matrix)
        out = x + input_matrix
        out = self.relu(out)
        return out
