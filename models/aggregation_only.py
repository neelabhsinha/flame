import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_backbone import CNNBackbone
from models.residual_block import ResidualBlock

# FLAME - Aggregation only (F-AO model)
class ConcatenatedFusionNet(nn.Module):
    def __init__(self):
        super(ConcatenatedFusionNet, self).__init__()
        self.rgbbackbone = CNNBackbone(in_channels=3)
        self.heatmapbackbone = CNNBackbone(in_channels=28)
        self.cnn = ResidualBlock(in_channels=256 * 2, out_channels=256)
        self.predictorfc1 = nn.Linear(in_features=14 * 14 * 256 + 3, out_features=512, bias=True)
        self.drop1 = nn.Dropout(p=0.2)
        self.predictorfc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, img, fl, hp):
        img = self.rgbbackbone(img)
        heatmap = self.heatmapbackbone(fl)
        combined_feature = torch.cat((img, heatmap), 1)
        combined_feature = self.cnn(combined_feature)
        flattened_img = combined_feature.reshape(-1, combined_feature.shape[1] * combined_feature.shape[2] * combined_feature.shape[3])
        fmap_final = torch.cat((flattened_img, hp), 1)
        fmap_final = fmap_final.float()
        fmap_final = self.predictorfc1(fmap_final)
        fmap_final = F.relu(fmap_final)
        fmap_final = self.drop1(fmap_final)
        fmap_final = self.predictorfc2(fmap_final)
        fmap_final = F.relu(fmap_final)
        fmap_final = self.drop2(fmap_final)
        output = self.out(fmap_final)
        return output