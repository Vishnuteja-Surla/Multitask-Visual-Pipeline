"""Localization modules
"""

import torch
import torch.nn as nn
from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, use_batchnorm: bool = True):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            use_batchnorm: True if the layers should include batch normalization else False
        """
        super().__init__()

        # VGG11 Encoder
        self.encoder = VGG11Encoder(in_channels, use_batchnorm)

        # FC Layers
        self.layer1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        bottleneck = self.encoder(x)
        x1 = self.layer1(bottleneck)
        x2 = self.layer2(x1)
        compressed_coords = self.layer3(x2)
        coords = compressed_coords * 224.0

        return coords