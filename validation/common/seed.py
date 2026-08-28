"""Deterministic seeding so every reported number is reproducible."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism flags (CPU is already deterministic for the ops we use).
    torch.use_deterministic_algorithms(False)  # avoid spurious errors on Conv1d.
