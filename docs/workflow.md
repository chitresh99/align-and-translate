# Workflow

Step by step trace of one full pass through the Bahdanau attention model, encoder input to emitted word.

Running example: a 7-word English source sentence being translated into French. Dimensions follow the paper (encoder hidden size 1000 per direction, embeddings 620, vocabulary 30000).

## Encoder: runs once per sentence

The encoder does not re-run during decoding. These four steps happen a single time and their output is then frozen.

### 1. Embed

Each source word ID is looked up in an embedding table and becomes a vector of 620 numbers.

```
7 word IDs -> 7 embeddings, each (620,)
```

### 2. Forward RNN

A GRU reads the embeddings left to right, producing one state per position.

```
forward_h_1 .. forward_h_7, each (1000,)
```

`forward_h_3` has seen words 1 through 3.

### 3. Backward RNN

A separate GRU with its own parameters reads the same embeddings right to left.

```
backward_h_1 .. backward_h_7, each (1000,)
```

`backward_h_3` has seen words 3 through 7.

### 4. Concatenate into annotations

```
h_j = [forward_h_j ; backward_h_j]     shape (2000,)
```

Seven annotations of 2000 numbers each. Each is centred on its own source word but carries context from both directions. These are the fixed reference set the decoder will attend over for the entire output sentence.

## Decoder: loops once per output word

Walking through the production of target word 4. At entry the decoder state is `s_3` and the last emitted word is `y_3`.

### 5. Score every source position

Run the alignment MLP once per annotation, 7 times in total:

```
e_j = v_a^T tanh(W_a s_3 + U_a h_j)
```

| term | shapes | result |
| --- | --- | --- |
| `W_a s_3` | (1000, 1000) x (1000,) | (1000,) |
| `U_a h_j` | (1000, 2000) x (2000,) | (1000,) |
| sum, then tanh | | (1000,) |
| `v_a^T (...)` | (1, 1000) x (1000,) | scalar |

Output: 7 raw scores, for example `[0.2, 1.1, 4.8, 0.9, 0.1, -0.3, 0.4]`.

`W_a s_3` is computed once and reused across all 7 positions. In a batched implementation `U_a h_j` is computed once for the whole source sentence and cached across all decoder timesteps, since the annotations never change.

### 6. Softmax over source positions

```
alpha_j = exp(e_j) / sum_k exp(e_k)
```

Output: 7 weights summing to 1, for example `[0.01, 0.03, 0.89, 0.03, 0.01, 0.01, 0.02]`.

This is the soft alignment for this output step.

### 7. Blend into the context vector

```
c_4 = sum_j alpha_j h_j
```

Output: one vector of 2000 numbers, dominated by `h_3` but containing a trace of every other annotation. No annotation is excluded and nothing is selected.

### 8. Update the decoder state

```
s_4 = GRU(s_3, y_3, c_4)
```

Three inputs: where the decoder was, the word it last emitted, and the context blend just computed. Output: one vector of 1000 numbers.

### 9. Project to vocabulary scores

Combine `s_4`, `y_3` and `c_4` through the output layer, then project with a matrix of shape `(30000, hidden)`.

Output: 30000 raw logits, one per candidate French word.

### 10. Softmax over vocabulary

Turns those logits into a probability distribution over the full target vocabulary.

```
accord       0.71
convention   0.12
traite       0.06
pacte        0.03
...          29996 more, each small
```

### 11. Select

Greedy decoding takes the argmax. Beam search keeps the top k partial translations alive and scores them by the total log probability of the sentence so far, which searches for the highest probability sentence rather than stacking highest probability words. Sampling draws from the distribution and is not used for translation, where there is a correct answer.

The selected word is `y_4`.

### 12. Loop

`y_4` and `s_4` become the inputs to step 5 of the next iteration. Stop when the model emits the end of sentence token.

## Two distributions, two different jobs

Easy to fuse these, worth keeping apart.

| | step 6 | step 10 |
| --- | --- | --- |
| over what | 7 source positions | 30000 vocabulary entries |
| size | source length | vocabulary size |
| question answered | where should I look | what should I write |
| used for | weighted averaging | selecting the output word |
| recomputed | every decoder step | every decoder step |

Attention never picks anything. The only hard selection in the whole pipeline is step 11, and at training time even that does not happen.

## What changes at training time

Steps 1 through 10 are identical. After that:

- Step 11 does not happen. No word is selected.
- Loss: take the probability the model assigned to the correct word from the reference translation, take the negative log. Sum or average over positions.
- Step 12 feeds the reference word as the next input rather than the model's own prediction. This is teacher forcing.
- After the full sentence, backpropagate.

The gradient path runs:

```
loss
  -> output layer
  -> s_i
  -> c_i
  -> alpha_ij
  -> alignment MLP (v_a, W_a, U_a)
  -> annotations h_j
  -> forward and backward encoder GRUs
  -> embeddings
```

Every parameter is updated by one gradient signal. The alignment MLP receives no alignment supervision at any point. It is trained purely because bad alignments produce bad translations. That is what "jointly learning to align and translate" refers to.

## Implementation notes

**Cache the annotation projections.** `U_a h_j` depends only on the encoder output, so compute it once per sentence rather than once per decoder step. This is the single largest easy saving, turning a per-step matrix multiply of shape (1000, 2000) into a lookup.

**Cost is quadratic in sentence length.** The alignment MLP runs source length times target length times per sentence. Fine at sentence scale. The Transformer inherits this same quadratic cost.

**Masking.** With batched variable-length sources, padded positions must be masked to negative infinity before the softmax in step 6, otherwise padding receives attention weight and leaks into the context vector.

**Shape checklist for debugging.**

| tensor | shape |
| --- | --- |
| source embeddings | (batch, src_len, 620) |
| annotations `h` | (batch, src_len, 2000) |
| decoder state `s` | (batch, 1000) |
| scores `e` | (batch, src_len) |
| weights `alpha` | (batch, src_len) |
| context `c` | (batch, 2000) |
| logits | (batch, 30000) |

The attention weight tensor over a full sentence is (batch, tgt_len, src_len), which is the tensor to save if the alignment heatmaps are going to be plotted.