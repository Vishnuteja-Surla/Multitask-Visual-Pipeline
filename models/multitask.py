"""Unified multi-task model
"""
import os
import gdown
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth", dropout_p: int = 0.5, encoder_backbone: str = "classifier", use_batchnorm: bool = True):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
            dropout_p: Dropout Probability to be followed in FC layers.
            encoder_backbone: The name of architecture from which encoder is to be extracted from. One of ["unet", "classifier", "localizer"].
            use_batchnorm: True if the layers should include batch normalization else False
        """
        super().__init__()

        if not os.path.exists(classifier_path):
            gdown.download(id="1em8sltDyPe1uLhu0LzdjrU41IrGic4R8", output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="1a6CQNBaL0-wAx5DSoPk1WYXJjmmjlLzY", output=localizer_path, quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1pJ1k_lXAhNsZIkvuuVpL_X39XACaVvtB", output=unet_path, quiet=False)

        # 1. Shared Backbone
        self.encoder = VGG11Encoder(in_channels, use_batchnorm)

        # 2. Rebuilding the Heads

        # --- Classification Head ---
        self.class_layer1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.class_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.class_layer3 = nn.Linear(in_features=4096, out_features=num_breeds, bias=True)

        # --- Localization Head ---
        self.loc_layer1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.loc_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.loc_layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4, bias=True),
            nn.Sigmoid()
        )

        # --- Segmentation Head ---
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        dec_1 = [nn.Conv2d(1024, 512, kernel_size=3, padding=1)]
        if use_batchnorm:
            dec_1.append(nn.BatchNorm2d(512))
        dec_1.extend([nn.ReLU(inplace=True), CustomDropout(dropout_p), nn.Conv2d(512, 512, kernel_size=3, padding=1)])
        if use_batchnorm:
            dec_1.append(nn.BatchNorm2d(512))
        dec_1.extend([nn.ReLU(inplace=True), CustomDropout(dropout_p)])
        self.dec1 = nn.Sequential(*dec_1)

        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        dec_2 = [nn.Conv2d(1024, 256, kernel_size=3, padding=1)]
        if use_batchnorm:
            dec_2.append(nn.BatchNorm2d(256))
        dec_2.extend([nn.ReLU(inplace=True), CustomDropout(dropout_p), nn.Conv2d(256, 256, kernel_size=3, padding=1)])
        if use_batchnorm:
            dec_2.append(nn.BatchNorm2d(256))
        dec_2.extend([nn.ReLU(inplace=True), CustomDropout(dropout_p)])
        self.dec2 = nn.Sequential(*dec_2)

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        dec_3 = [nn.Conv2d(512, 128, kernel_size=3, padding=1)]
        if use_batchnorm:
            dec_3.append(nn.BatchNorm2d(128))
        dec_3.extend([nn.ReLU(inplace=True), CustomDropout(dropout_p), nn.Conv2d(128, 128, kernel_size=3, padding=1)])
        if use_batchnorm:
            dec_3.append(nn.BatchNorm2d(128))
        dec_3.extend([nn.ReLU(inplace=True), CustomDropout(dropout_p)])
        self.dec3 = nn.Sequential(*dec_3)

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        dec_4 = [nn.Conv2d(256, 64, kernel_size=3, padding=1)]
        if use_batchnorm:
            dec_4.append(nn.BatchNorm2d(64))
        dec_4.append(nn.ReLU(inplace=True))
        self.dec4 = nn.Sequential(*dec_4)

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        dec_5 = [nn.Conv2d(128, 32, kernel_size=3, padding=1)]
        if use_batchnorm:
            dec_5.append(nn.BatchNorm2d(32))
        dec_5.append(nn.ReLU(inplace=True))
        self.dec5 = nn.Sequential(*dec_5)

        self.outConv = nn.Conv2d(32, seg_classes, kernel_size=1)

        # 3. Loading the raw state dicts from the files

        if os.path.exists(unet_path) and os.path.exists(classifier_path) and os.path.exists(localizer_path):
            unet_checkpoint = torch.load(unet_path, map_location="cpu")
            class_checkpoint = torch.load(classifier_path, map_location="cpu")
            loc_checkpoint = torch.load(localizer_path, map_location="cpu")

            unet_state = unet_checkpoint.get("state_dict", unet_checkpoint)
            class_state = class_checkpoint.get("state_dict", class_checkpoint)
            loc_state = loc_checkpoint.get("state_dict", loc_checkpoint)

            # 4. Filtering encoder from states and renaming the layers in paths
            class_head = {k.replace("layer", "class_layer"): v for k, v in class_state.items() if not k.startswith("encoder.")}
            loc_head = {k.replace("layer", "loc_layer"): v for k, v in loc_state.items() if not k.startswith("encoder.")}
            unet_head = {k: v for k, v in unet_state.items() if not k.startswith("encoder.")}

            class_encoder = {k.replace("encoder.", ""): v for k, v in class_state.items() if k.startswith("encoder.")}
            loc_encoder = {k.replace("encoder.", ""): v for k, v in loc_state.items() if k.startswith("encoder.")}
            unet_encoder = {k.replace("encoder.", ""): v for k, v in unet_state.items() if k.startswith("encoder.")}

            # 5. Loading only the encoder
            if encoder_backbone == "unet":
                self.encoder.load_state_dict(unet_encoder, strict=False)
            elif encoder_backbone == "classifier":
                self.encoder.load_state_dict(class_encoder, strict=False)
            elif encoder_backbone == "localizer":
                self.encoder.load_state_dict(loc_encoder, strict=False)
            else:
                raise ValueError("Invalid Backbone to extract Encoder from")
            
            # 6. Loading the heads
            self.load_state_dict(class_head, strict=False)
            self.load_state_dict(loc_head, strict=False)
            self.load_state_dict(unet_head, strict=False)
        else:
            print("[WARNING] One or more checkpoints missing. Initializing Multi-task model with random weights for architecture verification.")



    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # 1. Shared Backbone
        bottleneck, feature_maps = self.encoder(x, return_features=True)

        # 2. Classification
        x1 = self.class_layer1(bottleneck)
        x2 = self.class_layer2(x1)
        logits = self.class_layer3(x2)

        # 3. Localization
        x1 = self.loc_layer1(bottleneck)
        x2 = self.loc_layer2(x1)
        compressed_coords = self.loc_layer3(x2)
        coords = compressed_coords * 224.0

        # 4. Segmentation
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

        return {
            "classification": logits,
            "localization": coords,
            "segmentation": out
        }