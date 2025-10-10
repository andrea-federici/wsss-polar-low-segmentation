"""Utilities for deterministic seeding across the project."""
from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from lightning import seed_everything as pl_seed_everything


def configure_seed(seed: Optional[int], *, deterministic: bool = True) -> None:
    """Configure global RNG state for full reproducibility.

    Args:
        seed: Seed to apply across python, numpy, torch and Lightning. If ``None``
            no action is performed.
        deterministic: When ``True`` additional flags are set to force PyTorch to
            use deterministic algorithms when available.
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by PyTorch for deterministic cuBLAS executions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    pl_seed_everything(seed, workers=True)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Create a deterministic worker initialization function for DataLoaders."""

    def _init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn
