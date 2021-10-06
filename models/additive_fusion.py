import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_backbone import CNNBackbone

# FLAME-Additive Fusion (F-AF) Model
class AdditiveFusionNet(nn.Module):
    def __init__(self):
        super(AdditiveFusionNet, self).__init__()
        self.rgbbackbone = CNNBackbone(in_channels=3)
        self.heatmapbackbone = CNNBackbone(in_channels=28)
        self.predictorfc1 = nn.Linear(in_features=14 * 14 * 256 + 3, out_features=512, bias=True)  # 3 head pose angles
        self.drop1 = nn.Dropout(p=0.2)
        self.predictorfc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, img, fl, hp):
        img = self.rgbbackbone(img)
        heatmap = self.heatmapbackbone(fl)
        img_modified = img + heatmap
        flattened_img = img_modified.reshape(-1, img_modified.shape[1] * img_modified.shape[2] * img_modified.shape[3])
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
