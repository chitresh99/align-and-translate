import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

###############################
# Config
###############################

@dataclass
class Config:
    # Model sizes use the paper's symbols (Appendix A.2.3).
    # Paper values in comments; defaults are scaled down to run on the CPU.
    n: int = 128
    m: int = 80
    l: int = 64
    n_prime: int = 128

    # training
    batch_size: int = 80
    batches_per_block : int = 20
    rho: float = 0.95
    eps: float = 1e-6
    clip_threshold : float = 1.0
    epochs: int = 30

    # initialisation
    init_std: float = 0.01
    align_init_std: float = 0.001

    # data
    train_data: int = 6000
    valid_pairs: int = 300
    min_phrases: int = 2
    max_phrases: int = 5

    # decoding
    beam_width: int = 5
    max_decode_len: int = 24
    overfit_epochs: int = 1000
    seed: int = 0
