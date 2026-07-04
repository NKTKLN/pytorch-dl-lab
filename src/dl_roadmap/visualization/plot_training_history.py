"""Training history plotting utilities."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    filename: str | None = None,
    show_fig: bool = True,
) -> None:
    """Plot train/validation loss curves.

    Args:
        train_loss: Per-epoch training loss values.
        val_loss: Optional per-epoch validation loss values.
        filename: If given, save the figure to this path.
        show_fig: If True, display the figure with `plt.show()`.
    """
    train_epochs = np.arange(1, len(train_loss) + 1)
    sns.lineplot(x=train_epochs, y=train_loss, label="train_loss")

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

    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show_fig:
        plt.show()
