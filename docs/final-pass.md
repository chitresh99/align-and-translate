# Code Explanation: Model

Line-by-line walkthrough of the Model section of the RNNsearch implementation: `GatedUnit`, `Encoder`, `AlignmentModel`, `Decoder` and `RNNsearch`.

Symbols follow the paper: `n` hidden size, `m` embedding size, `l` maxout size, `n'` alignment hidden size, `B` batch size, `T_x` source length, `T_y` target length, `K_x` and `K_y` source and target vocabulary sizes.

## GatedUnit

The paper's gated hidden unit, the GRU. One call is one time step.

### Setup (`__init__`)

- `self.W = nn.Linear(input_dim, 3 * n)` holds the three input matrices $W_z$, $W_r$ and $W$ stacked into one layer, so a single matrix multiply gives the input part of the update gate, the reset gate and the candidate. It also holds the biases for all three.
- `self.U_zr = nn.Linear(n, 2 * n, bias=False)` holds the recurrent matrices $U_z$ and $U_r$ stacked. Both multiply the previous state, so they can share one multiply.
- `self.U = nn.Linear(n, n, bias=False)` is the candidate's recurrent matrix $U$. It is kept separate because it multiplies $r \circ h$, not $h$, and $r$ is only known after `U_zr` has run.
- `self.C = nn.Linear(context_dim, 3 * n, bias=False) if context_dim else None` holds the context matrices $C_z$, $C_r$ and $C$ stacked. Only the decoder passes a `context_dim`; the encoder gets `None`, which drops the context terms.

### Step (`forward`)

- `proj = self.W(x)` computes the input contributions for all three parts at once, shape `(B, 3n)`.
- `if self.C is not None: proj = proj + self.C(c)` adds the context contribution. This only runs in the decoder, where the attention context enters every gate.
- `p_z, p_r, p_h = proj.chunk(3, dim=-1)` splits the stacked result into the parts for the update gate, reset gate and candidate, each `(B, n)`.
- `u_z, u_r = self.U_zr(h_prev).chunk(2, dim=-1)` computes $U_z h$ and $U_r h$ in one multiply and splits them.
- `z = torch.sigmoid(p_z + u_z)` is the update gate: per dimension, how much the state should change.
- `r = torch.sigmoid(p_r + u_r)` is the reset gate: per dimension, how much of the old state the candidate may use.
- `h_tilde = torch.tanh(p_h + self.U(r * h_prev))` is the candidate state. The reset gate is applied before the matrix, $U[r \circ h]$, which is the paper's form and the reason this cell is written by hand instead of using `nn.GRU`.
- `return (1 - z) * h_prev + z * h_tilde` blends old and new per dimension. Where $z$ is near 0 the old value is copied forward; near 1 it is replaced.

## Encoder

The bidirectional RNN that turns the source sentence into one annotation per word.

### Setup (`__init__`)

- `self.n = cfg.n` stores the hidden size so `forward` can create zero states of that size.
- `self.embedding = nn.Embedding(K_x, cfg.m, padding_idx=PAD)` is a lookup table that turns each source word id into a vector of size `m`. `padding_idx=PAD` keeps the PAD row at zero and never updates it. There is one table, shared by both directions.
- `self.forward_rnn = GatedUnit(cfg.m, cfg.n)` is the GRU that reads left to right.
- `self.backward_rnn = GatedUnit(cfg.m, cfg.n)` is a second GRU, with its own separate weights, that reads right to left.

### Embedding (`forward`)

- `B, T_x = src.shape` gets the batch size and the padded source length.
- `emb = self.embedding(src)` turns word ids into vectors, shape `(B, T_x, m)`.

### Forward pass, left to right

- `h = emb.new_zeros(B, self.n)` sets the starting state to all zeros, shape `(B, n)`. `new_zeros` puts it on the same device and dtype as `emb`.
- `fwd = []` collects the state at every position.
- `for j in range(T_x):` visits positions 0, 1, 2, ... in order.
- `h_new = self.forward_rnn(emb[:, j], h)` runs one GRU step on word `j` for the whole batch.
- `h = torch.where(src_mask[:, j : j + 1], h_new, h)` means: for each sentence, if position `j` is a real word, take the new state; if it is padding, keep the old one. The `j : j + 1` slice keeps shape `(B, 1)` so it broadcasts across all `n` dimensions.
- `fwd.append(h)` saves this position's forward state.

### Backward pass, right to left

- `h = emb.new_zeros(B, self.n)` resets to zeros for the second direction.
- `bwd = [None] * T_x` pre-sizes the list, because it will be filled from the end, not appended.
- `for j in reversed(range(T_x)):` visits positions from last to first.
- `h_new = self.backward_rnn(emb[:, j], h)` runs one step of the backward GRU.
- `h = torch.where(...)` is the same keep-on-padding rule. Here it matters: reading from the right, a short sentence meets its padding first, and the state stays at zero through it, so the real reading begins at the sentence's actual last word.
- `bwd[j] = h` stores the state at position `j`, so `bwd` ends up in normal left-to-right order.

### Combining

- `fwd = torch.stack(fwd, dim=1)` turns the list into one tensor, `(B, T_x, n)`.
- `bwd = torch.stack(bwd, dim=1)` does the same, `(B, T_x, n)`.
- `annotations = torch.cat([fwd, bwd], dim=-1)` joins both directions at each position, giving $h_j = [\overrightarrow{h}_j ; \overleftarrow{h}_j]$ with shape `(B, T_x, 2n)`. Each word's vector now knows what came before it and what comes after it.
- `return annotations, bwd[:, 0]` returns the annotations for attention, plus the backward state at position 0 ($\overleftarrow{h}_1$ in the paper), which has read the whole sentence and is used to initialise the decoder: $s_0 = \tanh(W_s \overleftarrow{h}_1)$.

## AlignmentModel

The attention mechanism, equations 5 to 7. Scores every source annotation against the decoder state and returns the context vector.

### Setup (`__init__`)

- `self.W_a = nn.Linear(cfg.n, cfg.n_prime, bias=False)` projects the decoder state (size `n`) into the alignment space (size `n'`).
- `self.U_a = nn.Linear(2 * cfg.n, cfg.n_prime)` projects each annotation (size `2n`) into the same space. It carries the alignment model's only bias.
- `self.v_a = nn.Linear(cfg.n_prime, 1, bias=False)` collapses each `n'`-dim vector to a single score.

### Precompute (`precompute`)

- `return self.U_a(annotations)` computes $U_a h_j$ for every source position, shape `(B, T_x, n')`. It does not depend on the target step, so it is done once per sentence and reused at every decoder step.

### Scoring and context (`forward`)

- `self.W_a(s_prev).unsqueeze(1)` projects the decoder state to `(B, n')` and adds a middle axis, giving `(B, 1, n')`, so it broadcasts across all `T_x` source positions.
- `+ U_h` adds the cached source projections, giving $W_a s_{i-1} + U_a h_j$ for every $j$ at once, `(B, T_x, n')`.
- `torch.tanh(...)` applies the MLP's nonlinearity.
- `self.v_a(...).squeeze(-1)` reduces each position to one score $e_{ij}$, going from `(B, T_x, 1)` to `(B, T_x)`.
- `e = e.masked_fill(~src_mask, float("-inf"))` sets padded positions to minus infinity, so after the softmax they get exactly zero weight.
- `alpha = torch.softmax(e, dim=-1)` turns scores into weights over source positions, `(B, T_x)`. Each row sums to 1.
- `c = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)` takes the weighted sum $c_i = \sum_j \alpha_{ij} h_j$ as a batched matrix multiply: `(B, 1, T_x)` times `(B, T_x, 2n)` gives `(B, 1, 2n)`, squeezed to `(B, 2n)`.
- `return c, alpha` returns the context vector for this step and the weights, which are kept for the alignment heatmap.

## Decoder

Generates the translation one word at a time. Per step: build the context from $s_{i-1}$, predict $y_i$ from $(s_{i-1}, y_{i-1}, c_i)$, then update to $s_i$. This order follows the Appendix A.2.2 formula, which uses $s_{i-1}$ for the prediction even though equation 4 writes $s_i$.

### Setup (`__init__`)

- `self.l = cfg.l` stores the maxout size, needed to reshape in `readout`.
- `self.embedding = nn.Embedding(K_y, cfg.m, padding_idx=PAD)` is the target-side lookup table, separate from the encoder's.
- `self.W_s = nn.Linear(cfg.n, cfg.n)` is the matrix that builds the initial decoder state from $\overleftarrow{h}_1$.
- `self.alignment = AlignmentModel(cfg)` is the attention module.
- `self.rnn = GatedUnit(cfg.m, cfg.n, context_dim=2 * cfg.n)` is the decoder GRU. Its input is the previous word's embedding, and it also receives the `2n`-dim context in every gate.
- `self.U_o = nn.Linear(cfg.n, 2 * cfg.l)` maps the decoder state into the deep output layer, `2l` wide.
- `self.V_o = nn.Linear(cfg.m, 2 * cfg.l, bias=False)` maps the previous word's embedding into the same layer.
- `self.C_o = nn.Linear(2 * cfg.n, 2 * cfg.l, bias=False)` maps the context into the same layer.
- `self.W_o = nn.Linear(cfg.l, K_y)` maps the maxout output to one logit per target word.

### Initial state (`initial_state`)

- `return torch.tanh(self.W_s(bwd_first))` computes $s_0 = \tanh(W_s \overleftarrow{h}_1)$, shape `(B, n)`.

### Output layer (`readout`)

- `t_tilde = self.U_o(s_prev) + self.V_o(e_y) + self.C_o(c)` combines the previous state, previous word and context into $\tilde{t}_i$, shape `(B, 2l)`.
- `t = t_tilde.view(-1, self.l, 2).max(dim=-1).values` is the maxout. `view` groups adjacent pairs, elements 0 and 1, 2 and 3, and so on, into shape `(B, l, 2)`, and `max` keeps the larger of each pair, giving `(B, l)`.
- `logits = self.W_o(t)` gives one score per vocabulary word, `(B, K_y)`. The softmax is applied inside the loss, not here.
- `return t_tilde, t, logits` returns all three so the shape checks can inspect each stage.

### One decoding step (`step`)

- `c, alpha = self.alignment(s_prev, annotations, U_h, src_mask)` scores the source against $s_{i-1}$ and builds the context $c_i$.
- `e_y = self.embedding(y_prev)` embeds the previous target word, `(B, m)`.
- `_, _, logits = self.readout(s_prev, e_y, c)` predicts the current word from $(s_{i-1}, y_{i-1}, c_i)$.
- `s = self.rnn(e_y, s_prev, c)` updates the state to $s_i$, after the prediction.
- `return logits, s, alpha` returns the word scores, the new state for the next step, and the attention weights.

## RNNsearch

The full model: ties the encoder and decoder together and applies the paper's initialisation.

### Setup (`__init__`)

- `self.cfg = cfg` stores the config for use in `init_weights`.
- `self.encoder = Encoder(K_x, cfg)` builds the bidirectional encoder.
- `self.decoder = Decoder(K_y, cfg)` builds the decoder, which contains the alignment model.
- `self.init_weights()` overwrites PyTorch's default initialisation with the paper's.

### Initialisation (`init_weights`, Appendix B.1)

- `for p in self.parameters():` visits every parameter tensor in the model.
- `if p.dim() == 1: nn.init.zeros_(p)` sets every bias vector to zero. Biases are the only 1-D parameters.
- `else: nn.init.normal_(p, 0.0, cfg.init_std)` draws every matrix from a Gaussian with standard deviation 0.01. This is the default; the lines below overwrite the special cases.
- `with torch.no_grad():` makes the direct weight edits below invisible to autograd.
- `for cell in [...]` visits the three GRUs: forward encoder, backward encoder, decoder.
- `nn.init.orthogonal_(cell.U.weight)` makes the candidate's recurrent matrix orthogonal.
- `blocks = [nn.init.orthogonal_(torch.empty(cfg.n, cfg.n)) for _ in range(2)]` builds two separate `n x n` orthogonal matrices, one for $U_z$ and one for $U_r$.
- `cell.U_zr.weight.copy_(torch.cat(blocks, dim=0))` stacks them into the `(2n, n)` tensor. Initialising the stacked tensor directly would make the pair orthogonal as a whole, not each block on its own.
- `nn.init.normal_(align.W_a.weight, 0.0, cfg.align_init_std)` and the same for `U_a` draw the alignment projections with standard deviation 0.001.
- `nn.init.zeros_(align.v_a.weight)` sets $v_a$ to zero, so every score starts at 0 and attention starts exactly uniform.
- `self.encoder.embedding.weight[PAD].zero_()` and the decoder equivalent restore the PAD rows to zero, since the Gaussian loop above overwrote them.

### Encoding (`encode`)

- `annotations, bwd_first = self.encoder(src, src_mask)` runs the bidirectional encoder.
- `U_h = self.decoder.alignment.precompute(annotations)` caches $U_a h_j$ once for the whole sentence.
- `s0 = self.decoder.initial_state(bwd_first)` builds the decoder's starting state.
- `return annotations, U_h, s0` returns everything the decoder needs. Both training and decoding call this, so the source is encoded once and reused at every step.

### Training forward pass (`forward`)

- `annotations, U_h, s = self.encode(src, src_mask)` encodes the source and gets $s_0$.
- `logits, alphas = [], []` collect results per target step.
- `for i in range(tgt_in.size(1)):` loops over target positions. It has to be a loop because $c_i$ depends on $s_{i-1}$, which only exists after the previous step.
- `self.decoder.step(tgt_in[:, i], s, annotations, U_h, src_mask)` runs one step, feeding in the reference word, not the model's own prediction. This is teacher forcing.
- `logits.append(step_logits)` and `alphas.append(alpha)` save this step's outputs; `s` is carried to the next step.
- `return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)` returns logits `(B, T_y, K_y)` for the loss and the full alignment matrix `(B, T_y, T_x)`, where row `i` holds the attention weights used for target word `i`.

## Training

### Loss (`compute_loss`)

Computes the loss for one batch.

- `src_mask = src != PAD` marks the real source tokens, used by the encoder and by attention.
- `tgt_in = tgt[:, :-1]` drops the last token, giving the decoder inputs `<sos> y1 ... yT`.
- `tgt_out = tgt[:, 1:]` drops the first token, giving the targets `y1 ... yT <eos>`. The shift by one means that at every step the correct answer is the next word.
- `logits, _ = model(src, src_mask, tgt_in)` runs the full model with teacher forcing: every step is fed the reference word, not the model's own prediction. Logits have shape `(B, T_y, K_y)`; the attention weights are not needed here and are discarded.
- `logits.reshape(-1, logits.size(-1))` flattens all positions of all sentences into one list, `(B * T_y, K_y)`.
- `tgt_out.reshape(-1)` flattens the targets the same way, `(B * T_y,)`, so each row of logits lines up with its correct word.
- `nn.functional.cross_entropy(...)` applies a softmax over the vocabulary and takes the negative log probability of the correct word at every position.
- `ignore_index=PAD` skips padded target positions, so padding adds nothing to the loss or the gradient.
- `reduction="sum"` adds the losses of every real token in the batch into one number.
- `num_tokens = (tgt_out != PAD).sum().item()` counts the real target tokens, used for per-token logging.
- `return loss_sum / src.size(0), loss_sum.item(), num_tokens` returns three things:
  - the loss per sentence, which is what `backward()` uses. This matches the paper's objective, the log-likelihood of each sentence averaged over the batch.
  - the raw sum as a plain Python number, for logging without keeping the graph alive.
  - the token count, so `train` can compute loss per token across the epoch.

### Training loop (`train`)

Runs the optimiser over the data for a number of epochs.

- `optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=cfg.rho, eps=cfg.eps)` builds the paper's optimiser (Appendix B.2). Zeiler's Adadelta has no learning rate; PyTorch's version multiplies the update by `lr`, so `lr=1.0` recovers the original.
- `data = [tensorize(p, src_vocab, tgt_vocab) for p in pairs]` converts every pair of word lists into id lists, adding `<eos>` to the source and `<sos>`, `<eos>` to the target.
- `random.Random(cfg.seed).shuffle(data)` shuffles the data once before training, as in Appendix B.2. A seeded generator makes the order reproducible.
- `for epoch in range(1, epochs + 1):` loops over full passes of the data. The data is not reshuffled between epochs.
- `model.train()` puts the model in training mode. This model has no dropout or batch norm, so it changes nothing here, but it is the standard habit and undoes the `model.eval()` set by the decoding functions.
- `total_loss, total_tokens = 0.0, 0` reset the epoch's running totals for logging.
- `for src, tgt in iterate_batches(data, cfg):` gets length-sorted, padded batches one at a time.
- `optimizer.zero_grad()` clears the gradients left over from the previous batch. PyTorch accumulates gradients by default, so without this they would add up.
- `loss, loss_sum, n_tokens = compute_loss(model, src, tgt)` runs the model and computes the loss.
- `loss.backward()` backpropagates through the whole model: decoder, attention, alignment MLP and both encoder directions.
- `torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_threshold)` treats all gradients as one vector and, if its L2 norm is above 1, scales it down to norm 1. The direction is unchanged. This guards against exploding gradients from backprop through time.
- `optimizer.step()` applies the Adadelta update to every parameter.
- `total_loss += loss_sum` and `total_tokens += n_tokens` accumulate the epoch's totals.
- `if epoch % log_every == 0 or epoch == 1:` logs on the first epoch and then every `log_every` epochs.
- `per_token = total_loss / total_tokens` is the average loss per real target token across the epoch. Per token, not per sentence, so the number is comparable across batches and datasets with different sentence lengths.
- `math.exp(per_token)` is the perplexity: roughly how many words the model is choosing between on average at each step. 1 would mean it is always certain and correct.
- `return model` returns the trained model.