"""Simple multilayer perceptron for MNIST digit classification."""

import torch
from torch import nn


class MLP_MNIST(nn.Module):
    """Fully-connected classifier for 28x28 grayscale MNIST digits.

    Flattens the input image and passes it through two hidden ReLU
    layers before producing raw logits over the 10 digit classes.
    """

    def __init__(self) -> None:
        """Build the network layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784).

        Returns:
            Logits of shape (batch, 10).
        """
        return self.net(x)
