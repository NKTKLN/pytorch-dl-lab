"""Plotting utilities for visualizing classification confusion matrices."""

import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from torcheval.metrics.functional import multiclass_confusion_matrix

_purple_cmap = sns.light_palette("#7e44d3", as_cmap=True)


def plot_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    filename: str | None = None,
    show_fig: bool = True,
) -> None:
    """Display a confusion matrix using seaborn heatmap.

    Args:
        preds: Predicted class labels.
        target: True class labels.
        filename: If given, save the figure to this path.
        show_fig: If True, display the figure with `plt.show()`.
    """
    num_classes = int(target.max().item() + 1)
    conf_matrix = multiclass_confusion_matrix(preds, target, num_classes)

    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap=_purple_cmap, cbar=False, square=True
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show_fig:
        plt.show()
