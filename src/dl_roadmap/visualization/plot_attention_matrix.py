"""Plotting utilities for visualizing seq2seq attention weights."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor

_yellow_cmap = sns.light_palette("#F2C94C", as_cmap=True)


def plot_attention_matrix(
    input_tokens: list[str],
    output_tokens: list[str],
    attentions: Tensor,
    filename: str | Path | None = None,
    show_fig: bool = True,
) -> None:
    """Plots an attention heatmap between input and output subword tokens.

    Args:
        input_tokens: Source-sentence subword pieces, labeling columns.
        output_tokens: Decoded subword pieces, labeling rows.
        attentions: Attention weights, shaped
            ``len(output_tokens) x len(input_tokens)``.
        filename: If given, save the figure to this path.
        show_fig: If True, display the figure with `plt.show()`.
    """
    if not output_tokens:
        print("No output tokens to plot attention for (empty translation).")
        return

    plt.figure(figsize=(len(input_tokens) * 0.7 + 2, len(output_tokens) * 0.7 + 2))
    sns.heatmap(
        attentions.cpu().numpy(),
        annot=True,
        fmt=".2f",
        cmap=_yellow_cmap,
        cbar=False,
        square=True,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Attention Matrix")

    if filename is not None:
        plt.savefig(filename, dpi=300)

    if show_fig:
        plt.show()
