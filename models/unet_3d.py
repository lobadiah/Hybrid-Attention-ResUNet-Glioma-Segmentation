import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv_block(x)
        return self.pool(x)

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock3D, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv_block(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.encoder1 = EncoderBlock3D(n_channels, 64)
        self.encoder2 = EncoderBlock3D(64, 128)
        self.encoder3 = EncoderBlock3D(128, 256)
        self.encoder4 = EncoderBlock3D(256, 512)
        self.bottleneck = ConvBlock3D(512, 1024)
        self.decoder4 = DecoderBlock3D(1024, 512)
        self.decoder3 = DecoderBlock3D(512, 256)
        self.decoder2 = DecoderBlock3D(256, 128)
        self.decoder1 = DecoderBlock3D(128, 64)
        self.final_conv = nn.Conv3d(64, n_classes, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        bottleneck = self.bottleneck(skip4)
        x = self.decoder4(bottleneck, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.final_conv(x)

# Example usage:
if __name__ == '__main__':
    model = UNet3D(n_channels=1, n_classes=2)
    input_tensor = torch.randn(1, 1, 64, 64, 64)  # Batch size 1, 1 channel, 64x64x64 volume
    output = model(input_tensor)
    print(f'Output shape: {output.shape}')  # Should be (1, 2, 64, 64, 64) for 2 classes
