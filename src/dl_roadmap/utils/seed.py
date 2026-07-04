import random

import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducible runs.

    Seeds Python's `random`, PyTorch (CPU and CUDA), and NumPy (if
    installed).

    Args:
        seed: Seed value to apply to all RNGs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except Exception:  # noqa: S110
        pass
