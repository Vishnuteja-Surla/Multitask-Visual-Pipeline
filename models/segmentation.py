"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        # VGG11 Encoder
        self.encoder = VGG11Encoder(in_channels)

        # Decoder Blocks
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(            
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(            
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.outConv = nn.Conv2d(32, num_classes, kernel_size=1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, feature_maps = self.encoder(x, return_features=True)

        u1 = self.up1(bottleneck)
        cat1 = torch.cat([u1, feature_maps['block5']], dim=1)
        x1 = self.dec1(cat1)

        u2 = self.up2(x1)
        cat2 = torch.cat([u2, feature_maps['block4']], dim=1)
        x2 = self.dec2(cat2)

        u3 = self.up3(x2)
        cat3 = torch.cat([u3, feature_maps['block3']], dim=1)
        x3 = self.dec3(cat3)

        u4 = self.up4(x3)
        cat4 = torch.cat([u4, feature_maps['block2']], dim=1)
        x4 = self.dec4(cat4)

        u5 = self.up5(x4)
        cat5 = torch.cat([u5, feature_maps['block1']], dim=1)
        x5 = self.dec5(cat5)

        out = self.outConv(x5)

        return out