%%writefile models/attention_unet.py
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.net(w)
        return x * w

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.attn(self.conv(x))

class AttentionUNet(nn.Module):
    """
    Simple Attention U-Net (stable):
    - Normal U-Net skip connections
    - SE (channel attention) in conv blocks
    """
    def __init__(self, in_channels=3, out_channels=1, base=64, use_se=True):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base, use_se)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2, use_se)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2, base*4, use_se)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base*4, base*8, use_se)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base*8, base*16, use_se)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8, use_se)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4, use_se)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2, use_se)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base, use_se)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x); x = self.pool1(s1)
        s2 = self.enc2(x); x = self.pool2(s2)
        s3 = self.enc3(x); x = self.pool3(s3)
        s4 = self.enc4(x); x = self.pool4(s4)

        x = self.bottleneck(x)

        x = self.up4(x); x = torch.cat([x, s4], dim=1); x = self.dec4(x)
        x = self.up3(x); x = torch.cat([x, s3], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, s2], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, s1], dim=1); x = self.dec1(x)

        return self.out(x)
