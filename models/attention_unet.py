import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.relu = nn.ReLU()
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, g, x):
        g1 = self.W_g(g)
        l1 = self.W_l(x)
        output = self.relu(g1 + l1)
        return torch.sigmoid(self.psi(output)) * x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.ag3 = AttentionGate(512, 256, 256)
        self.ag2 = AttentionGate(256, 128, 128)
        self.ag1 = AttentionGate(128, 64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        bottleneck_out = self.bottleneck(enc3_out)
        dec3_out = self.dec3(bottleneck_out, self.ag3(enc3_out, bottleneck_out))
        dec2_out = self.dec2(dec3_out, self.ag2(enc2_out, dec3_out))
        dec1_out = self.dec1(dec2_out, self.ag1(enc1_out, dec2_out))
        return self.final_conv(dec1_out)

# Example Usage
if __name__ == '__main__':
    model = AttentionUNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Example input
    preds = model(x)
    print(preds.shape)  # Should print torch.Size([1, 1, 256, 256])
