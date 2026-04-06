"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Reusable Max Pool Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 1
        layers1 = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers1.append(nn.BatchNorm2d(64))
        layers1.append(nn.ReLU(inplace=True))        
        self.block1 = nn.Sequential(*layers1)

        # Block 2
        layers2 = [nn.Conv2d(64, 128, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers2.append(nn.BatchNorm2d(128))
        layers2.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*layers2)

        # Block 3
        layers3 = [nn.Conv2d(128, 256, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers3.append(nn.BatchNorm2d(256))
        layers3.append(nn.ReLU(inplace=True))
        layers3.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if use_batchnorm:
            layers3.append(nn.BatchNorm2d(256))
        layers3.append(nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(*layers3)

        # Block 4
        layers4 = [nn.Conv2d(256, 512, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers4.append(nn.BatchNorm2d(512))
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if use_batchnorm:
            layers4.append(nn.BatchNorm2d(512))
        layers4.append(nn.ReLU(inplace=True))
        self.block4 = nn.Sequential(*layers4)

        # Block 5
        layers5 = [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers5.append(nn.BatchNorm2d(512))
        layers5.append(nn.ReLU(inplace=True))
        layers5.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if use_batchnorm:
            layers5.append(nn.BatchNorm2d(512))
        layers5.append(nn.ReLU(inplace=True))
        self.block5 = nn.Sequential(*layers5)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        feature_maps = {}

        x1 = self.block1(x)
        feature_maps["block1"] = x1
        p1 = self.pool1(x1)

        x2 = self.block2(p1)
        feature_maps["block2"] = x2
        p2 = self.pool2(x2)

        x3 = self.block3(p2)
        feature_maps["block3"] = x3
        p3 = self.pool3(x3)

        x4 = self.block4(p3)
        feature_maps["block4"] = x4
        p4 = self.pool4(x4)

        x5 = self.block5(p4)
        feature_maps["block5"] = x5
        bottleneck = self.pool5(x5)

        if return_features:
            return bottleneck, feature_maps
        else:
            return bottleneck
        

VGG11 = VGG11Encoder