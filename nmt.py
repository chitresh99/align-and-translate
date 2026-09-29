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

class AlignmentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_a = nn.Linear(cfg.n, cfg.n_prime, bias=False)
        self.U_a = nn.Linear(2 * cfg.n, cfg.n_prime)
        self.v_a = nn.Linear(cfg.n_prime, 1, bias=False)

    def precompute(self, annotations):
        return self.U_a(annotations)  

    def forward(self, s_prev, annotations, U_h, src_mask):
        e = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + U_h)).squeeze(-1)
        e = e.masked_fill(~src_mask, float("-inf"))  
        alpha = torch.softmax(e, dim=-1)
        c = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)
        return c, alpha

class Decoder(nn.Module):
    def __init__(self, K_y, cfg):
        super().__init__()
        self.l = cfg.l
        self.embedding = nn.Embedding(K_y, cfg.m, padding_idx=PAD)
        self.W_s = nn.Linear(cfg.n, cfg.n)
        self.alignment = AlignmentModel(cfg)
        self.rnn = GatedUnit(cfg.m, cfg.n, context_dim=2 * cfg.n)

        self.U_o = nn.Linear(cfg.n, 2 * cfg.l)
        self.V_o = nn.Linear(cfg.m, 2 * cfg.l, bias=False)
        self.C_o = nn.Linear(2 * cfg.n, 2 * cfg.l, bias=False)
        self.W_o = nn.Linear(cfg.l, K_y)

    def initial_state(self, bwd_first):
        return torch.tanh(self.W_s(bwd_first))

    def readout(self, s_prev, e_y, c):
        t_tilde = self.U_o(s_prev) + self.V_o(e_y) + self.C_o(c)  # (B, 2l)
        t = t_tilde.view(-1, self.l, 2).max(dim=-1).values  # (B, l)
        logits = self.W_o(t)
        return t_tilde, t, logits

    def step(self, y_prev, s_prev, annotations, U_h, src_mask):
        c, alpha = self.alignment(s_prev, annotations, U_h, src_mask)
        e_y = self.embedding(y_prev)  # (B, m)
        _, _, logits = self.readout(s_prev, e_y, c)
        s = self.rnn(e_y, s_prev, c)
        return logits, s, alpha

class RNNsearch(nn.Module):
    def __init__(self, K_x, K_y, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(K_x, cfg)
        self.decoder = Decoder(K_y, cfg)
        self.init_weights()

    def init_weights(self):
        cfg = self.cfg
        for p in self.parameters():
            if p.dim() == 1:
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, 0.0, cfg.init_std)

        with torch.no_grad():
            for cell in [
                self.encoder.forward_rnn,
                self.encoder.backward_rnn,
                self.decoder.rnn,
            ]:
                nn.init.orthogonal_(cell.U.weight)
                blocks = [nn.init.orthogonal_(torch.empty(cfg.n, cfg.n)) for _ in range(2)]
                cell.U_zr.weight.copy_(torch.cat(blocks, dim=0))

            align = self.decoder.alignment
            nn.init.normal_(align.W_a.weight, 0.0, cfg.align_init_std)
            nn.init.normal_(align.U_a.weight, 0.0, cfg.align_init_std)
            nn.init.zeros_(align.v_a.weight)

            self.encoder.embedding.weight[PAD].zero_()
            self.decoder.embedding.weight[PAD].zero_()

    def encode(self, src, src_mask):
        annotations, bwd_first = self.encoder(src, src_mask)
        U_h = self.decoder.alignment.precompute(annotations)
        s0 = self.decoder.initial_state(bwd_first)
        return annotations, U_h, s0

    def forward(self, src, src_mask, tgt_in):
        annotations, U_h, s = self.encode(src, src_mask)
        logits, alphas = [], []
        for i in range(tgt_in.size(1)):
            step_logits, s, alpha = self.decoder.step(
                tgt_in[:, i], s, annotations, U_h, src_mask
            )
            logits.append(step_logits)
            alphas.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)

###############################
# Training
###############################

def compute_loss(model, src, tgt):
    src_mask = src != PAD
    tgt_in = tgt[:, :-1]  # <sos> y1 ... yT
    tgt_out = tgt[:, 1:]  # y1 ... yT <eos>

    logits, _ = model(src, src_mask, tgt_in)
    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=PAD,
        reduction="sum",
    )
    num_tokens = (tgt_out != PAD).sum().item()
    return loss_sum / src.size(0), loss_sum.item(), num_tokens

def train(model, pairs, cfg, src_vocab, tgt_vocab, epochs, log_every=5):
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=1.0, rho=cfg.rho, eps=cfg.eps
    )

    data = [tensorize(p, src_vocab, tgt_vocab) for p in pairs]
    random.Random(cfg.seed).shuffle(data)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in iterate_batches(data, cfg):
            optimizer.zero_grad()
            loss, loss_sum, n_tokens = compute_loss(model, src, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_threshold)
            optimizer.step()
            total_loss += loss_sum
            total_tokens += n_tokens

        if epoch % log_every == 0 or epoch == 1:
            per_token = total_loss / total_tokens
            print(
                f"  epoch {epoch:4d}  loss/token {per_token:6.4f}  "
                f"ppl {math.exp(per_token):8.3f}"
            )
    return model

def train(model, pairs, cfg, src_vocab, tgt_vocab, epochs, log_every=5):
    # Zeiler's Adadelta has no learning rate; PyTorch's version multiplies the
    # update by lr, so lr=1.0 recovers the original.
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=1.0, rho=cfg.rho, eps=cfg.eps
    )

    data = [tensorize(p, src_vocab, tgt_vocab) for p in pairs]
    random.Random(cfg.seed).shuffle(data)  # shuffled once before training

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in iterate_batches(data, cfg):
            optimizer.zero_grad()
            loss, loss_sum, n_tokens = compute_loss(model, src, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_threshold)
            optimizer.step()
            total_loss += loss_sum
            total_tokens += n_tokens

        if epoch % log_every == 0 or epoch == 1:
            per_token = total_loss / total_tokens
            print(
                f"  epoch {epoch:4d}  loss/token {per_token:6.4f}  "
                f"ppl {math.exp(per_token):8.3f}"
            )
    return model

#####################################
# Decoding
#####################################

@torch.no_grad()
def greedy_decode(model, src, cfg):
    model.eval()
    src_mask = src != PAD
    annotations, U_h, s = model.encode(src, src_mask)  # computed once, reused every step
    y = torch.tensor([SOS])
    tokens, alphas = [], []

    for _ in range(cfg.max_decode_len):
        logits, s, alpha = model.decoder.step(y, s, annotations, U_h, src_mask)
        alphas.append(alpha[0])
        y = logits.argmax(dim=-1)
        if y.item() == EOS:
            break
        tokens.append(y.item())

    return tokens, alphas


@torch.no_grad()
def beam_search(model, src, cfg):
    model.eval()
    src_mask = src != PAD
    annotations, U_h, s0 = model.encode(src, src_mask)

    beams = [([], 0.0, s0, [])] 
    finished = []

    for _ in range(cfg.max_decode_len):
        candidates = []
        for tokens, score, s, alphas in beams:
            last = torch.tensor([tokens[-1] if tokens else SOS])
            logits, s_new, alpha = model.decoder.step(
                last, s, annotations, U_h, src_mask
            )
            log_probs = torch.log_softmax(logits[0], dim=-1)
            topk = torch.topk(log_probs, cfg.beam_width)
            for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                candidates.append(
                    (tokens + [idx], score + lp, s_new, alphas + [alpha[0]])
                )

        candidates.sort(key=lambda x: x[1], reverse=True)

        beams = []
        for tokens, score, s, alphas in candidates:
            if tokens[-1] == EOS:
                finished.append((tokens[:-1], score, alphas))
            else:
                beams.append((tokens, score, s, alphas))
            if len(beams) == cfg.beam_width:
                break

        if not beams:
            break
        if finished and max(f[1] for f in finished) >= beams[0][1]:
            break

    finished += [(t, sc, a) for t, sc, _, a in beams]
    best = max(finished, key=lambda x: x[1])
    return best[0], best[2]

