"""Synthetic linear regression dataset generation utilities."""

import torch

from dl_roadmap.data.train_test_split import train_test_split


def make_synthetic_regression_dataset(
    n_samples: int = 250,
    n_features: int = 1,
    noise: float = 25.0,
    test_size: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Generate a synthetic linear regression dataset split into train/val.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of input features.
        noise: Standard deviation of the Gaussian noise added to targets.
        test_size: Fraction of samples to allocate to the validation split.

    Returns:
        A tuple of (X_train, X_val, y_train, y_val, real_coef, real_bias):
        the train/val tensors and the ground-truth coefficients and bias
        used to generate the data.
    """
    real_coef = torch.rand(n_features) * 100
    real_bias = (torch.randn(1) * 100).item()

    X = torch.randn(n_samples, n_features)
    y = X @ real_coef + real_bias + torch.randn(n_samples) * noise

    X = X.float()
    y = y.float().unsqueeze(1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    return X_train, X_val, y_train, y_val, real_coef, real_bias
