import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Optional


class ShortcutProjection(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):

        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class BottleneckResidualBlock(nn.Module):

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        self.act3 = nn.ReLU()

    def forward(self, x: torch.Tensor):

        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act3(x + shortcut)


class ResNetBase(nn.Module):

    def __init__(self, n_blocks: List[int], n_channels: List[int],
                 bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, first_kernel_size: int = 7, output_dim=100):

        super().__init__()

        first_channel = 64

        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv = nn.Conv2d(img_channels, first_channel,
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(first_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        blocks = []
        prev_channels = first_channel
        for i, channels in enumerate(n_channels):
            stride = 1 if len(blocks) == 0 else 2

            if bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels,
                                                      stride=stride))

            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                    nn.Linear(n_channels[-1], output_dim))

    def forward(self, x: torch.Tensor):

        x = self.bn(self.conv(x))
        x = self.blocks(x)
        x = self.linear(x)
        return x


class ResidualBlock_V2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, num_group_norm_groups):

        super().__init__()

        self.gn1 = nn.GroupNorm(num_group_norm_groups, in_channels)
        self.act1 = GeLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.gn2 = nn.GroupNorm(num_group_norm_groups, out_channels)
        self.act2 = GeLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):

        shortcut = self.shortcut(x)
        x = self.conv1(self.act1(self.g1(x)))
        x = self.conv2(self.act2(self.g2(x)))
        return x + shortcut


class GeLU(nn.Module):

    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, input):
        return F.gelu(input)
