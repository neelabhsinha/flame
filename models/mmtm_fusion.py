import torch
import torch.nn as nn
import torch.nn.functional as F

from models.residual_block import ResidualBlock


class MMTMFusionNet(nn.Module):
    def __init__(self, input_size):
        super(MMTMFusionNet, self).__init__()
        out_size = int((input_size - 2) / 2 ** 3)
        self.initial_rgb = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1_rgb = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.initial_heatmap = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1_heatmap = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.m1_rgb = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.m1_heatmap = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.m2_rgb = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.m2_heatmap = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        # MMTM
        self.z_1 = nn.Linear(in_features=2 * 128, out_features=int(128 / 2), bias=True)
        self.e1_rgb = nn.Linear(in_features=int(128 / 2), out_features=128, bias=True)
        self.e1_heatmap = nn.Linear(in_features=int(128 / 2), out_features=128, bias=True)
        self.m3_rgb = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256)
        )
        self.m3_heatmap = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256)
        )
        # MMTM
        self.z_2 = nn.Linear(in_features=2 * 256, out_features=int(256 / 2), bias=True)
        self.e2_rgb = nn.Linear(in_features=int(256 / 2), out_features=256, bias=True)
        self.e2_heatmap = nn.Linear(in_features=int(256 / 2), out_features=256, bias=True)
        # Post Processing
        self.postprocess = ResidualBlock(in_channels=256 * 2, out_channels=256)
        # Regression
        self.predictorfc1 = nn.Linear(in_features=out_size * out_size * 256 + 3, out_features=512, bias=True)
        self.drop1 = nn.Dropout(p=0.2)
        self.predictorfc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, rgb, heatmap, hp):
        rgb = self.initial_rgb(rgb)
        heatmap = self.initial_heatmap(heatmap)
        rgb = self.maxpool1_rgb(rgb)
        heatmap = self.maxpool1_heatmap(heatmap)
        rgb = self.m1_rgb(rgb)
        heatmap = self.m1_heatmap(heatmap)
        rgb = self.m2_rgb(rgb)
        heatmap = self.m2_heatmap(heatmap)
        # Multimodal Transfer 1
        s_a1 = torch.mean(torch.mean(rgb, 3), 2)
        s_b1 = torch.mean(torch.mean(heatmap, 3), 2)
        z1 = self.z_1(torch.cat((s_a1, s_b1), 1))
        e1rgb = self.e1_rgb(z1)
        e1heatmap = self.e1_heatmap(z1)
        e1rgb = F.sigmoid(e1rgb)
        e1heatmap = F.sigmoid(e1heatmap)
        for batch in range(0, rgb.shape[0]):
            for channel in range(0, rgb.shape[1]):
                rgb[batch, channel] = 2 * rgb[batch, channel] * e1rgb[batch, channel]
                heatmap[batch, channel] = 2 * heatmap[batch, channel] * e1heatmap[batch, channel]
        # Next Layer
        rgb = self.m3_rgb(rgb)
        heatmap = self.m3_heatmap(heatmap)
        # Multimodal Transfer 2
        s_a2 = torch.mean(torch.mean(rgb, 3), 2)
        s_b2 = torch.mean(torch.mean(heatmap, 3), 2)
        z2 = self.z_2(torch.cat((s_a2, s_b2), 1))
        e2rgb = self.e2_rgb(z2)
        e2heatmap = self.e2_heatmap(z2)
        e2rgb = F.sigmoid(e2rgb)
        e2heatmap = F.sigmoid(e2heatmap)
        for batch in range(0, rgb.shape[0]):
            for channel in range(0, rgb.shape[1]):
                rgb[batch, channel] = 2 * rgb[batch, channel] * e2rgb[batch, channel]
                heatmap[batch, channel] = 2 * heatmap[batch, channel] * e2heatmap[batch, channel]
        # Next Layer
        combined_feature = torch.cat((rgb, heatmap), 1)
        img = self.postprocess(combined_feature)
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
