import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class RestorationNetV2(nn.Module):
    def __init__(self, num_features=64, num_blocks=20):
        super().__init__()
        # 1. Feature Extractor Head
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)

        # 2. Deep Residual Backbone Body
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 3. 2x Learned Sub-Pixel Feature Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # 4. Residual Reconstruction Tail
        self.tail = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Bicubic baseline (2x scale)
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        # CNN Feature extraction
        features = self.head(x)
        body = self.body(features)
        body = self.body_conv(body)
        features = features + body

        # Upsample features & predict correction map
        features = self.upsample(features)
        correction = self.tail(features)

        # Final Restored Output = Bicubic Base + Learned Residual Correction
        output = base + correction
        return output
