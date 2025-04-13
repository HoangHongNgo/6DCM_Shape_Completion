import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from .modules import *


class SM_SCM_rn18(nn.Module):
    def __init__(self, class_num=41, sm_path="/media/dsp520/Grasp_2T/6DCM_Grasp/6DCM/checkpoints/SM/SM_checkpoint_0040.pt"):
        super(SM_SCM_rn18, self).__init__()

        # Load SM net
        self.sm_net = SM_rn18_v2()
        checkpoint = torch.load(sm_path)
        self.sm_net.load_state_dict(checkpoint["state_dict"])

        self.intrinsic = np.load(
            "/media/dsp520/Grasp_2T/graspnet/scenes/scene_0000/realsense/camK.npy")
        self.camera = CameraInfo(640, 360, self.intrinsic[0][0] / 2, self.intrinsic[1][1] / 2, self.intrinsic[0][2],
                                 self.intrinsic[1][2], 1000)

        # Load ResNet18 backbone
        self.enc = ModifiedResNet18(in_channel=2, pretrained=False)

        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv = nn.Conv2d(64, 1, 1, 1)

    def forward(self, input):
        rgb_input = input[:, :3, :, :]

        depth_input = input[:, 3, :, :].unsqueeze(1)

        sm_res = self.sm_net(rgb_input)
        segMask_pre = self.sm_net.postprocess(sm_res)

        fm1, fm2, fm3, fm4, fm5 = self.enc(
            torch.cat([segMask_pre, depth_input, ], 1))

        x = F.interpolate(fm5, [fm4.size(2), fm4.size(3)],
                          mode="bilinear", align_corners=False)

        x = self.dec1(torch.cat([x, fm4], 1))
        x = F.interpolate(x, [fm3.size(2), fm3.size(3)],
                          mode="bilinear", align_corners=False)

        x = self.dec2(torch.cat([x, fm3], 1))
        x = F.interpolate(x, [fm2.size(2), fm2.size(3)],
                          mode="bilinear", align_corners=False)

        x = self.dec3(torch.cat([x, fm2], 1))
        x = F.interpolate(x, [fm1.size(2), fm1.size(3)],
                          mode="bilinear", align_corners=False)

        x = self.dec4(torch.cat([x, fm1], 1))
        x = F.interpolate(x, [input.size(2), input.size(3)],
                          mode="bilinear", align_corners=False)

        x = self.conv(x)
        # x = F.interpolate(x, [input.size(2)*2, input.size(3)*2],
        #                   mode="bilinear", align_corners=False)

        # depth_input = F.interpolate(depth_input, [input.size(
        #     2)*2, input.size(3)*2], mode="bilinear", align_corners=False)

        # cloud = create_point_cloud_from_depth_image_tensor(
        #     depth=depth_input, fx=self.camera.fx, fy=self.camera.fy, cx=self.camera.cx, cy=self.camera.cy, scale=self.camera.scale)
        # # visualize_pc(cloud)
        # rear_pt_pre = ssd2pointcloud_tensor(cloud=cloud.permute(0, 3, 1, 2), mask=F.interpolate(
        #     segMask_pre.float(), [input.size(2)*2, input.size(3)*2], mode="bilinear", align_corners=False), diff=x)
        # rear_pt_pre = F.interpolate(rear_pt_pre, [input.size(
        #     2), input.size(3)], mode="bilinear", align_corners=False)
        # x = F.interpolate(x, [input.size(2), input.size(3)],
        #                   mode="bilinear", align_corners=False)

        return {
            'semantic_mask': sm_res,
            'diff': x
        }

    def postprocess(self, input, output):
        depth_input = input[:, 3, :, :].unsqueeze(1)
        diff = output['diff']
        segMask_pre = output['semantic_mask']

        x = F.interpolate(diff, [input.size(2)*2, input.size(3)*2],
                          mode="bilinear", align_corners=False)

        depth_input = F.interpolate(depth_input, [input.size(
            2)*2, input.size(3)*2], mode="bilinear", align_corners=False)

        cloud = create_point_cloud_from_depth_image_tensor(
            depth=depth_input, fx=self.camera.fx, fy=self.camera.fy, cx=self.camera.cx, cy=self.camera.cy, scale=self.camera.scale)
        # visualize_pc(cloud)
        rear_pt_pre = ssd2pointcloud_tensor(cloud=cloud.permute(0, 3, 1, 2), mask=F.interpolate(
            segMask_pre.float(), [input.size(2)*2, input.size(3)*2], mode="bilinear", align_corners=False), diff=x, format='cloud_rear')
        rear_pt_pre = F.interpolate(rear_pt_pre, [input.size(
            2), input.size(3)], mode="bilinear", align_corners=False)

        return rear_pt_pre
