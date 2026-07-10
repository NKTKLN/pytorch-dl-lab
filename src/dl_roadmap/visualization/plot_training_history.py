"""Training history plotting utilities."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes


def plot_training_history(
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    best_epoch: int | None = None,
    filename: str | None = None,
    show_fig: bool = True,
) -> None:
    """Plot train/validation loss curves.

    Args:
        train_loss: Per-epoch training loss values.
        val_loss: Optional per-epoch validation loss values.
        best_epoch: If given, draw a vertical marker at this epoch.
        filename: If given, save the figure to this path.
        show_fig: If True, display the figure with `plt.show()`.
    """
    _, ax = plt.subplots()

    train_epochs = np.arange(1, len(train_loss) + 1)
    sns.lineplot(x=train_epochs, y=train_loss, label="train_loss", ax=ax)

    if val_loss is not None:
        val_epochs = np.arange(1, len(val_loss) + 1)
        sns.lineplot(x=val_epochs, y=val_loss, label="val_loss", ax=ax)

    if best_epoch is not None:
        ax.axvline(
            x=best_epoch,
            linestyle="--",
            linewidth=1.5,
            label=f"best_epoch ({best_epoch})",
            c="r",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show_fig:
        plt.show()


def plot_training_history_on_ax(
    ax: Axes,
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    best_epoch: int | None = None,
    title: str = "Training and Validation Loss",
) -> None:
    """Plot train/validation loss curves onto an existing Axes.

    Args:
        ax: Axes to draw the curves on.
        train_loss: Per-epoch training loss values.
        val_loss: Optional per-epoch validation loss values.
        best_epoch: If given, draw a vertical marker at this epoch.
        title: Title to display above the plot.
    """
    train_epochs = np.arange(1, len(train_loss) + 1)
    sns.lineplot(x=train_epochs, y=train_loss, label="train_loss", ax=ax)

    if val_loss is not None:
        val_epochs = np.arange(1, len(val_loss) + 1)
        sns.lineplot(x=val_epochs, y=val_loss, label="val_loss", ax=ax)

    if best_epoch is not None:
        ax.axvline(
            x=best_epoch,
            linestyle="--",
            linewidth=1.5,
            label=f"best_epoch ({best_epoch})",
            c="r",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
