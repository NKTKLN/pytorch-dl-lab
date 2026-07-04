"""Training history plotting utilities."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ema(values: Sequence[float], alpha: float = 0.2) -> list[float]:
    """Compute an exponential moving average over `values`.

    Args:
        values: Sequence of raw values to smooth.
        alpha: Smoothing factor in (0, 1]; higher values track the raw
            signal more closely, lower values smooth more aggressively.

    Returns:
        The EMA-smoothed values, one per input value.
    """
    smoothed = []
    last = values[0]

    for value in values:
        last = alpha * value + (1 - alpha) * last
        smoothed.append(last)

    return smoothed


def plot_training_history(
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    ema_alpha: float = 0.2,
) -> None:
    """Plot train/validation loss curves.

    The train loss is smoothed with an exponential moving average to
    reduce per-epoch noise; the validation loss is plotted as-is. Each
    curve gets its own epoch axis, so `train_loss` and `val_loss` don't
    need to have the same length.

    Args:
        train_loss: Per-epoch training loss values.
        val_loss: Optional per-epoch validation loss values.
        ema_alpha: Smoothing factor passed to the EMA applied to the
            train loss curve.
    """
    train_epochs = np.arange(1, len(train_loss) + 1)
    smooth_train_loss = _ema(train_loss, alpha=ema_alpha)
    sns.lineplot(x=train_epochs, y=smooth_train_loss, label="train_loss EMA")

    if val_loss is not None:
        val_epochs = np.arange(1, len(val_loss) + 1)
        sns.lineplot(x=val_epochs, y=val_loss, label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
