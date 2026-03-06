import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    """
    Calculates the Dice Loss between the predictions and the target.

    Args:
        pred (torch.Tensor): Predicted output from the model (logits).
        target (torch.Tensor): Ground truth labels.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Computed Dice Loss.
    """
    pred = F.sigmoid(pred)
    pred_f = pred.view(-1)
    target_f = target.view(-1)
    intersection = (pred_f * target_f).sum()
    return 1 - (2. * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth)


def dice_coefficient(pred, target, smooth=1.0):
    """
    Calculates the Dice Coefficient between the predictions and the target.

    Args:
        pred (torch.Tensor): Predicted output from the model (logits).
        target (torch.Tensor): Ground truth labels.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Computed Dice Coefficient.
    """
    pred = F.sigmoid(pred)
    pred_f = pred.view(-1)
    target_f = target.view(-1)
    intersection = (pred_f * target_f).sum()
    return (2. * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth)


class UNetWithDiceLoss(nn.Module):
    """
    Implementation of U-Net architecture with Dice Loss.

    Attributes:
        encoder (nn.Module): Encoder path.
        decoder (nn.Module): Decoder path.
        bottleneck (nn.Module): Bottleneck layer.

    Methods:
        forward(x): Forward pass through the network.
    """

    def __init__(self):
        super(UNetWithDiceLoss, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits.
        """
        enc = self.encoder(x)
        bottleneck_output = self.bottleneck(enc)
        return self.decoder(bottleneck_output)


# Example usage:
if __name__ == '__main__':
    model = UNetWithDiceLoss()
    input_tensor = torch.randn((1, 1, 256, 256))  # Example input
    output = model(input_tensor)
    print(output.shape)  # Should output: torch.Size([1, 1, 256, 256])
    
