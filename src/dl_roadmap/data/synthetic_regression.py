"""Synthetic linear regression dataset generation utilities."""

import random

import torch
from sklearn.datasets import make_regression

from dl_roadmap.data.train_test_split import train_test_split


def make_synthetic_regression_dataset(
    n_samples: int = 250,
    n_features: int = 1,
    noise: float = 25.0,
    test_size: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Generate a synthetic linear regression dataset split into train/val.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of input features.
        noise: Standard deviation of the Gaussian noise added to targets.
        test_size: Fraction of samples to allocate to the validation split.

    Returns:
        A tuple of (X_train, X_val, y_train, y_val, real_coef, real_bias):
        the train/val tensors and the ground-truth coefficient and bias
        used to generate the data.
    """
    real_bias = random.random() * 100  # noqa: S311
    x_numpy, y_numpy, real_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        bias=real_bias,
        coef=True,
    )

    X = torch.from_numpy(x_numpy).float()
    y = torch.from_numpy(y_numpy).float().unsqueeze(1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    return X_train, X_val, y_train, y_val, float(real_coef), real_bias
