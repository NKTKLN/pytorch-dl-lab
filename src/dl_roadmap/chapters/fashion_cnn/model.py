"""Small convolutional network for Fashion-MNIST classification."""

from torch import Tensor, nn


class CnnBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool block, halving spatial dims."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the block layers.

        Args:
            in_channels: Number of channels in the input feature map.
            out_channels: Number of channels produced by the convolution.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the conv/norm/relu/maxpool sequence.

        Args:
            x: Input feature map of shape (batch, in_channels, H, W).

        Returns:
            Feature map of shape (batch, out_channels, H/2, W/2).
        """
        block_out = self.conv(x)
        block_out = self.norm(block_out)
        block_out = self.relu(block_out)
        block_out = self.maxpool(block_out)
        return block_out


class CnnFashionMNIST(nn.Module):
    """Convolutional classifier for 28x28 grayscale Fashion-MNIST images.

    Passes the input through two `CnnBlock`s (halving spatial dims each
    time) before flattening and classifying with a small MLP head.
    """

    def __init__(self) -> None:
        """Build the network layers."""
        super().__init__()

        self.features = nn.Sequential(
            CnnBlock(1, 16),
            CnnBlock(16, 8),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Logits of shape (batch, 10).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
