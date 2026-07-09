"""Classification metrics helpers for dl_roadmap."""

from torch import Tensor
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)


def evaluate_multiclass_classification(
    preds: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1-score for multiclass predictions.

    Args:
        preds: Predicted class labels or logits, shape (n_samples,) or
            (n_samples, n_classes).
        target: True class labels, shape (n_samples,).

    Returns:
        Mapping of metric name to its scalar value.
    """
    return {
        "Accuracy": multiclass_accuracy(preds, target).item(),
        "Precision": multiclass_precision(preds, target).item(),
        "Recall": multiclass_recall(preds, target).item(),
        "F1-score": multiclass_f1_score(preds, target).item(),
    }


def evaluate_binary_classification(
    preds: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1-score for binary predictions.

    Args:
        preds: Predicted labels or scores, shape (n_samples,).
        target: True binary labels, shape (n_samples,).

    Returns:
        Mapping of metric name to its scalar value.
    """
    return {
        "Accuracy": binary_accuracy(preds, target).item(),
        "Precision": binary_precision(preds, target).item(),
        "Recall": binary_recall(preds, target).item(),
        "F1-score": binary_f1_score(preds, target).item(),
    }
