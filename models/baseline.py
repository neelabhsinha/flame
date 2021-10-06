import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_backbone import CNNBackbone


class BaselineNetwork(nn.Module):
    def __init__(self, in_channels):
        super(BaselineNetwork, self).__init__()
        self.backbone = CNNBackbone(in_channels=in_channels)
        self.predictorfc1 = nn.Linear(in_features=14*14 * 256 + 3, out_features=512, bias=True)  # 3 head pose angles
        self.drop1 = nn.Dropout(p=0.2)
        self.predictorfc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, img, fl, hp):
        img = self.backbone(img)
        flattened_img = img.reshape(-1, img.shape[1] * img.shape[2] * img.shape[3])
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
