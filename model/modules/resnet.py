import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import *


class ModifiedResNet18(nn.Module):
    def __init__(self, in_channel=3, pretrained=False):
        super(ModifiedResNet18, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x2s = self.resnet.relu(x)
        x = self.resnet.maxpool(x2s)

        x4s = self.resnet.layer1(x)
        x8s = self.resnet.layer2(x4s)
        x16s = self.resnet.layer3(x8s)
        x = self.resnet.layer4(x16s)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.resnet.fc(x)

        return x2s, x4s, x8s, x16s, x


class Decoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, input):
        return self.fc(input)


class SM_rn18_v2(nn.Module):
    def __init__(self, class_num=41):
        super(SM_rn18_v2, self).__init__()

        # Encoder: Modified ResNet18 with pretrained weights
        self.encoder = ModifiedResNet18(in_channel=3, pretrained=True)

        # Decoder blocks as direct layers in the model
        self.conv16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # Final convolutional output layer
        self.output_conv = nn.Conv2d(64, class_num, kernel_size=1, stride=1)

    def forward(self, x):
        # Extract features from encoder
        x2s, x4s, x8s, x16s, xfc = self.encoder(x)

        # Apply Decoder blocks
        fm1 = self.conv16(xfc)
        fm1 = F.interpolate(fm1, size=[x16s.size(2), x16s.size(
            3)], mode="bilinear", align_corners=False)

        fm1 = self.conv8(torch.cat([fm1, x16s], dim=1))
        fm1 = F.interpolate(fm1, size=[x8s.size(2), x8s.size(
            3)], mode="bilinear", align_corners=False)

        fm1 = self.conv4(torch.cat([fm1, x8s], dim=1))
        fm1 = F.interpolate(fm1, size=[x4s.size(2), x4s.size(
            3)], mode="bilinear", align_corners=False)

        fm1 = self.conv2(torch.cat([fm1, x4s], dim=1))
        fm1 = F.interpolate(fm1, size=[x2s.size(2), x2s.size(
            3)], mode="bilinear", align_corners=False)

        fm1 = self.conv1(torch.cat([fm1, x2s], dim=1))
        fm1 = F.interpolate(fm1, size=[x.size(2), x.size(
            3)], mode="bilinear", align_corners=False)

        # Output prediction
        output = self.output_conv(fm1)

        return output

    def postprocess(self, output):
        probs = torch.softmax(output, dim=1)
        class_masks = (probs > 0.1).float()
        segMask_pre = torch.argmax(class_masks, dim=1)
        segMask_pre = torch.unsqueeze(segMask_pre, 1)
