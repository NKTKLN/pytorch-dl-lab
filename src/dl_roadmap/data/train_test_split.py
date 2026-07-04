"""Train/test splitting utility for tensor datasets."""

import torch


def train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: float = 0.2,
    shuffle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split features and targets into train and test tensors.

    Args:
        X: Feature tensor of shape (n_samples, ...).
        y: Target tensor of shape (n_samples, ...); must share `X`'s
            first dimension.
        test_size: Fraction of samples to allocate to the test split, in (0, 1).
        shuffle: If True, shuffle sample indices before splitting;
            otherwise take the last `test_size` fraction as-is.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If `X` and `y` don't have the same number of samples.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, got "
            f"{X.shape[0]} and {y.shape[0]}"
        )

    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)

    indeces = torch.arange(n_samples)
    if shuffle:
        indeces = torch.randperm(n_samples)

    train_indeces = indeces[:-n_test]
    test_indeces = indeces[-n_test:]

    return X[train_indeces], X[test_indeces], y[train_indeces], y[test_indeces]
