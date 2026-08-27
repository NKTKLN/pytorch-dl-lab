"""Beam search decoding for autoregressive sequence models."""

import math
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


def _next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalizes a step function's output into a flat logit vector.

    Args:
        logits: Next-token logits, shaped ``vocab_size`` or
            ``1 x vocab_size``.

    Returns:
        Logits over the vocabulary, shaped ``vocab_size``, safe to modify
        in place.

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

    return logits.float().clone()


def _apply_repetition_penalty(
    logits: torch.Tensor, sequence: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Discounts the logits of tokens the hypothesis already contains.

    Args:
        logits: Next-token logits, shaped ``vocab_size``, modified in place.
        sequence: Tokens decoded so far, shaped ``1 x cur_len``.
        penalty: Divisor for positive logits and multiplier for negative
            ones; 1 disables the penalty.

    Returns:
        The penalized `logits`.
    """
    if penalty == 1.0:
        return logits

    tokens = torch.unique(sequence)
    scores = logits[tokens]
    logits[tokens] = torch.where(scores < 0, scores * penalty, scores / penalty)

    return logits


def _banned_ngram_tokens(sequence: torch.Tensor, ngram_size: int) -> list[int]:
    """Finds the tokens that would repeat an n-gram already in `sequence`.

    Args:
        sequence: Tokens decoded so far, shaped ``1 x cur_len``.
        ngram_size: Length of the n-grams that may not repeat; 0 disables
            the check.

    Returns:
        Token ids that must not be generated next.
    """
    tokens = sequence[0].tolist()

    if ngram_size <= 0 or len(tokens) < ngram_size:
        return []

    prefix = tuple(tokens[len(tokens) - ngram_size + 1 :])
    banned = {
        tokens[start + ngram_size - 1]
        for start in range(len(tokens) - ngram_size + 1)
        if tuple(tokens[start : start + ngram_size - 1]) == prefix
    }

    return sorted(banned)


def _validate_args(
    max_length: int,
    beam_width: int,
    min_length: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> None:
    """Checks the decoding arguments of `beam_search`.

    Args:
        max_length: Maximum decoded length, counting the leading ``<BOS>``.
        beam_width: Number of hypotheses kept alive at each step.
        min_length: Decoded length below which ``<EOS>`` is suppressed.
        repetition_penalty: Discount applied to already generated tokens.
        no_repeat_ngram_size: Length of the n-grams that may not repeat.

    Raises:
        ValueError: If any argument is outside its accepted range.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    if max_length < 2:  # noqa: PLR2004
        raise ValueError(
            f"max_length must be >= 2 to decode a token after <BOS>, got {max_length}"
        )

    if min_length > max_length:
        raise ValueError(
            f"min_length must be <= max_length, got {min_length} > {max_length}"
        )

    if repetition_penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {repetition_penalty}")

    if no_repeat_ngram_size < 0:
        raise ValueError(
            f"no_repeat_ngram_size must be >= 0, got {no_repeat_ngram_size}"
        )


@torch.no_grad()
def beam_search(  # noqa: PLR0913
    step: StepFn,
    bos_id: int,
    eos_id: int,
    max_length: int,
    beam_width: int = 3,
    length_penalty_alpha: float = 0.6,
    min_length: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
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
        min_length: Decoded length, counting the leading ``<BOS>``, below
            which ``<EOS>`` is suppressed; 0 disables it.
        repetition_penalty: Discount applied to the logits of tokens
            already present in a hypothesis; 1 disables it.
        no_repeat_ngram_size: Length of the n-grams that may not repeat
            within a hypothesis; 0 disables the check.
        device: Device the decoder tensors are created on; defaults to CPU.
            Should match the device `step` expects its input on.

    Returns:
        The token ids of the highest-scoring hypothesis, including the
        leading ``<BOS>``, shaped ``1 x seq_len``.

    Raises:
        ValueError: If `beam_width` is below 1, if `max_length` leaves no
            room for a token after ``<BOS>``, if `min_length` exceeds
            `max_length`, if `repetition_penalty` is not positive, or if
            `no_repeat_ngram_size` is negative.
    """
    _validate_args(
        max_length=max_length,
        beam_width=beam_width,
        min_length=min_length,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    device = device or torch.device("cpu")

    def length_penalty(length: int) -> float:
        return float(((5 + length) ** length_penalty_alpha) / (6**length_penalty_alpha))

    def normalized_score(node: BeamNode) -> float:
        return node.score / length_penalty(node.sequence.shape[1])

    def next_token_log_probs(node: BeamNode) -> torch.Tensor:
        logits = _next_token_logits(step(node.sequence))
        logits = _apply_repetition_penalty(logits, node.sequence, repetition_penalty)

        banned = _banned_ngram_tokens(node.sequence, no_repeat_ngram_size)
        if banned:
            logits[banned] = -math.inf

        if node.sequence.shape[1] < min_length:
            logits[eos_id] = -math.inf

        return F.log_softmax(logits, dim=-1)

    beam = [
        BeamNode(sequence=torch.tensor([[bos_id]], dtype=torch.long, device=device))
    ]
    completed: list[BeamNode] = []

    while beam:
        candidates: list[BeamNode] = []

        for node in beam:
            log_probs = next_token_log_probs(node)
            width = min(beam_width, log_probs.shape[-1])
            top_log_probs, top_tokens = log_probs.topk(width, dim=-1)

            for log_prob, token in zip(top_log_probs, top_tokens):
                if not math.isfinite(float(log_prob)):
                    continue

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
