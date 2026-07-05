import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    filename: str | None = None,
    show_fig: bool = True,
) -> None:
    """Display a confusion matrix using seaborn heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        filename: If given, save the figure to this path.
        show_fig: If True, display the figure with `plt.show()`.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show_fig:
        plt.show()
