import torch
import torch.nn as nn
from utils import *


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=2):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(
            1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(
            self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(
            1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyzc):
        B, N, _ = xyzc.shape

        # encoder
        feature = self.first_conv(xyzc[:, :, :3].transpose(
            2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[
            0]                          # (B,  256, 1)
        # (B,  512, N)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)
        # (B, 1024, N)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[
            0]                           # (B, 1024)

        # decoder
        # (B, num_coarse, 3), coarse point cloud
        diff = self.mlp(feature_global).reshape(-1, self.num_coarse, 1)
        coarse = generate_pc_from_diff(xyzc[:, :, :3], diff)
        visualize_pc(coarse)

        # (B, num_coarse, S, 3)
        point_feat = coarse.unsqueeze(
            2).expand(-1, -1, self.grid_size ** 2, -1)
        # (B, 3, num_fine)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(
            B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        # (B, 2, num_fine)
        seed = seed.reshape(B, -1, self.num_dense)

        feature_global = feature_global.unsqueeze(
            2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        # (B, 1024+2+3, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)

        # (B, 3, num_fine), fine point cloud
        fine = self.final_conv(feat) + point_feat

        return diff.contiguous(), fine.transpose(1, 2).contiguous()
