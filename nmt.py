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
    # model sizes use the paper's symbols (appendix A.2.3)
    # paper values in comments; defaults are scaled down to run on the CPU

    n: int = 128  # hidden units, paper 1000
    m: int = 80  # word embedding, paper 620
    l: int = 64  # maxout units in the deep output, paper 500
    n_prime: int = 128  # alignment model hidden size, paper 1000

    # training (Appendix B.2)
    batch_size: int = 80
    batches_per_block: int = 20  # sort 20 * 80 = 1600 pairs at a time
    rho: float = 0.95  # adadelta decay
    eps: float = 1e-6  # adadelta epsilon
    clip_threshold: float = 1.0  # gradient L2 norm cap
    epochs: int = 30

    # initialisation (appendix B.1)
    init_std: float = 0.01  # variance 0.01^2
    align_init_std: float = 0.001  # variance 0.001^2, for W_a and U_a

    # data
    train_data: int = 6000
    valid_pairs: int = 300
    min_phrases: int = 2
    max_phrases: int = 5

    # decoding
    beam_width: int = 5  # the paper uses beam search but never states the width
    max_decode_len: int = 24

    overfit_epochs: int = 1000
    seed: int = 0


###############################
# Data
###############################

SRC_ADJ = ["red", "big", "small", "old", "new", "green", "cold", "bright"]
SRC_NOUN = ["cat", "house", "tree", "river", "bird", "stone", "road", "field"]

TGT_ADJ = {
    "red": "rouge",
    "big": "grand",
    "small": "petit",
    "old": "vieux",
    "new": "neuf",
    "green": "vert",
    "cold": "froid",
    "bright": "clair",
}

TGT_NOUN = {
    "cat": ["le", "chat"],
    "house": ["la", "maison"],
    "tree": ["arbre"],
    "river": ["la", "riviere"],
    "bird": ["oiseau"],
    "stone": ["la", "pierre"],
    "road": ["route"],
    "field": ["le", "champ"],
}

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Vocab:
    def __init__(self, tokens):
        self.itos = list(SPECIALS) + sorted(set(tokens) - set(SPECIALS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def make_pair(rng, min_phrases, max_phrases):
    n = rng.randint(min_phrases, max_phrases)
    src, tgt = [], []
    for _ in range(n):
        adj = rng.choice(SRC_ADJ)
        noun = rng.choice(SRC_NOUN)
        src += [adj, noun]
        tgt += TGT_NOUN[noun] + [TGT_ADJ[adj]]  # noun first, adjective after
    return src, tgt


def build_dataset(cfg):
    rng = random.Random(cfg.seed)
    train = [
        make_pair(rng, cfg.min_phrases, cfg.max_phrases) for _ in range(cfg.train_pairs)
    ]
    rng_valid = random.Random(cfg.seed + 10_000)
    valid = [
        make_pair(rng_valid, cfg.min_phrases, cfg.max_phrases)
        for _ in range(cfg.valid_pairs)
    ]
    src_vocab = Vocab(SRC_ADJ + SRC_NOUN)
    tgt_vocab = Vocab(
        [t for v in TGT_NOUN.values() for t in v] + list(TGT_ADJ.values())
    )
    return train, valid, src_vocab, tgt_vocab


def tensorize(pair, src_vocab, tgt_vocab):
    src_tokens, tgt_tokens = pair
    src_ids = src_vocab.encode(src_tokens) + [EOS]
    tgt_ids = [SOS] + tgt_vocab.encode(tgt_tokens) + [EOS]
    return src_ids, tgt_ids


def collate(chunk):
    """
    turns a list of variable-length sentence pairs into two rectangular tensors, which is what a batch has to be.
    Returns src (B, max_src_len) and tgt (B, max_tgt_len), padded at the end with PAD.
    """
    src_len = max(len(s) for s, _ in chunk)  # longest source in the batch
    tgt_len = max(len(t) for _, t in chunk)  # longest target, measured separately
    src = torch.full((len(chunk), src_len), PAD, dtype=torch.long)  # all PAD to start
    tgt = torch.full((len(chunk), tgt_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(chunk):
        src[i, : len(s)] = torch.tensor(s)  # copy left-aligned, rest stays PAD
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt


def iterate_batches(data, cfg):
    """It takes 1600 sentence pairs at a time,
    sorts them by length, cuts them into 20 batches of 80,
    and pads each batch, so every batch contains sentences
    of about the same length and needs very little padding."""
    block = cfg.batch_size * cfg.batches_per_block
    for start in range(0, len(data), block):
        chunk = sorted(data[start : start + block], key=lambda x: len(x[0]))
        for b in range(0, len(chunk), cfg.batch_size):
            yield collate(chunk[b : b + cfg.batch_size])

###############################
# Model
###############################

class GatedUnit(nn.Module):
    def __init__(self,input_dim,n,context_dim = 0):
        super().__init__()
        self.W = nn.Linear(input_dim, 3 * n) 
        self.U_zr = nn.Linear(n, 2 * n, bias=False)  # holds the weights for both gates
        self.U = nn.Linear(n, n, bias=False) # U holds the weight for the candidate state
        self.C = nn.Linear(context_dim, 3 * n, bias=False) if context_dim else None

    def forward(self,x,h_prev,c=None):
        """
        it takes the input x and the previous state h_prev and returns the new state
        z: is the update gate
        r: is the reset gate
        """
        proj = self.W(x)                          # W_z x, W_r x, W x (plus biases), stacked
        if self.C is not None:
            proj = proj + self.C(c)               # decoder only: add C_z c, C_r c, C c
        p_z, p_r, p_h = proj.chunk(3, dim=-1)     # split input parts for z, r, candidate
        u_z, u_r = self.U_zr(h_prev).chunk(2, dim=-1)   # U_z h, U_r h

        z = torch.sigmoid(p_z + u_z)              # update gate
        r = torch.sigmoid(p_r + u_r)              # reset gate
        h_tilde = torch.tanh(p_h + self.U(r * h_prev))  # candidate, reset before U
        return (1 - z) * h_prev + z * h_tilde     # blend old and new

class Encoder(nn.Module):
    def __init__(self, K_x, cfg):
        super().__init__()
        self.n = cfg.n
        self.embedding = nn.Embedding(K_x, cfg.m, padding_idx=PAD)  # shared by both directions
        self.forward_rnn = GatedUnit(cfg.m, cfg.n)
        self.backward_rnn = GatedUnit(cfg.m, cfg.n)

    def forward(self, src, src_mask):
        B, T_x = src.shape
        emb = self.embedding(src)

        h = emb.new_zeros(B, self.n)
        fwd = []
        for j in range(T_x):
            h_new = self.forward_rnn(emb[:, j], h)
            h = torch.where(src_mask[:, j : j + 1], h_new, h)  # keep old state on padding
            fwd.append(h)
        h = emb.new_zeros(B, self.n)
        bwd = [None] * T_x
        for j in reversed(range(T_x)):
            h_new = self.backward_rnn(emb[:, j], h)
            h = torch.where(src_mask[:, j : j + 1], h_new, h) 
            bwd[j] = h
        fwd = torch.stack(fwd, dim=1)
        bwd = torch.stack(bwd, dim=1) 
        annotations = torch.cat([fwd, bwd], dim=-1)  
        return annotations, bwd[:, 0]

