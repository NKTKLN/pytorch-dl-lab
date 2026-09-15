"""Length-aware batch sampling utilities."""

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """Group similar-length examples while retaining epoch-level shuffling."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size_multiplier: int = 100,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize the sampler.

        Args:
            lengths: Sequence length for every dataset example.
            batch_size: Number of examples in each batch.
            bucket_size_multiplier: Number of batches placed in a sorting pool.
            shuffle: Whether to shuffle pools, batches, and examples within batches.
            drop_last: Whether to discard the final incomplete batch.
            seed: Base seed used for deterministic epoch-level shuffling.
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of length-bucketed index batches."""
        rng = random.Random(self.seed + self.epoch)  # noqa: S311
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    if self.shuffle:
                        rng.shuffle(batch)
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        """Return the number of batches produced in one epoch."""
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive the shuffle seed."""
        self.epoch = epoch
