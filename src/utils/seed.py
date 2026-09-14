"""Deterministic seeding across python/numpy/torch (and CUDA where available)."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed every RNG we touch so a (config, split, seed) triple is reproducible.

    deterministic_cudnn=True trades a little speed for run-to-run reproducibility on
    the same GPU; set False if you only care about statistical (not bit-exact)
    reproducibility and want the extra throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
