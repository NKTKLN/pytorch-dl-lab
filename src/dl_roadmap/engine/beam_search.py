"""Beam search decoding for autoregressive sequence models."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

StepFn = Callable[[torch.Tensor], torch.Tensor]
"""Scores the next token for a partially decoded sequence.

Takes the tokens decoded so far, shaped ``1 x cur_len``, and returns the
logits for the position that follows them, shaped ``vocab_size`` or
``1 x vocab_size``. Everything model-specific (the encoded source, cached
encoder memory, sampling temperature) is expected to be captured by the
closure, which keeps `beam_search` itself architecture-agnostic.
"""


@dataclass
class BeamNode:
    """A single hypothesis in the beam, with its running log-probability.

    Attributes:
        sequence: Token ids decoded so far, shaped ``1 x cur_len``,
            including the leading ``<BOS>``.
        score: Sum of the log-probabilities of every token in `sequence`,
            without any length normalization.
    """

    sequence: torch.Tensor
    score: float = 0.0


def _next_token_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """Normalizes a step function's output into next-token log-probabilities.

    Args:
        logits: Next-token logits, shaped ``vocab_size`` or
            ``1 x vocab_size``.

    Returns:
        Log-probabilities over the vocabulary, shaped ``vocab_size``.

    Raises:
        ValueError: If `logits` has neither of the accepted shapes.
    """
    if logits.ndim == 2 and logits.shape[0] == 1:  # noqa: PLR2004
        logits = logits[0]

    if logits.ndim != 1:
        raise ValueError(
            "step must return next-token logits shaped `vocab_size` or "
            f"`1 x vocab_size`, got {tuple(logits.shape)}"
        )

    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def beam_search(  # noqa: PLR0913
    step: StepFn,
    bos_id: int,
    eos_id: int,
    max_length: int,
    beam_width: int = 3,
    length_penalty_alpha: float = 0.6,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Decodes the highest-scoring sequence for a single example.

    Args:
        step: Callable scoring the next token for a partially decoded
            sequence; see `StepFn`.
        bos_id: Token id used to seed decoding.
        eos_id: Token id that terminates a hypothesis.
        max_length: Maximum decoded length, counting the leading ``<BOS>``,
            before a hypothesis is forced into the completed set.
        beam_width: Number of hypotheses kept alive at each step.
        length_penalty_alpha: Strength of the length penalty applied when
            ranking completed hypotheses; 0 disables it.
        device: Device the decoder tensors are created on; defaults to CPU.
            Should match the device `step` expects its input on.

    Returns:
        The token ids of the highest-scoring hypothesis, including the
        leading ``<BOS>``, shaped ``1 x seq_len``.

    Raises:
        ValueError: If `beam_width` is below 1, or `max_length` leaves no
            room for a token after ``<BOS>``.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    if max_length < 2:  # noqa: PLR2004
        raise ValueError(
            f"max_length must be >= 2 to decode a token after <BOS>, got {max_length}"
        )

    device = device or torch.device("cpu")

    def length_penalty(length: int) -> float:
        return float(((5 + length) ** length_penalty_alpha) / (6**length_penalty_alpha))

    def normalized_score(node: BeamNode) -> float:
        return node.score / length_penalty(node.sequence.shape[1])

    beam = [
        BeamNode(sequence=torch.tensor([[bos_id]], dtype=torch.long, device=device))
    ]
    completed: list[BeamNode] = []

    while beam:
        candidates: list[BeamNode] = []

        for node in beam:
            log_probs = _next_token_log_probs(step(node.sequence))
            top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)

            for log_prob, token in zip(top_log_probs, top_tokens):
                next_token = token.view(1, 1)
                sequence = torch.cat([node.sequence, next_token], dim=1)
                candidate = BeamNode(
                    sequence=sequence,
                    score=node.score + float(log_prob),
                )

                if int(token) == eos_id or sequence.shape[1] >= max_length:
                    completed.append(candidate)
                else:
                    candidates.append(candidate)

        candidates.sort(key=lambda node: node.score, reverse=True)
        beam = candidates[:beam_width]

    return max(completed, key=normalized_score).sequence
