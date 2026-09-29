# Implementing Bahdanau, Cho and Bengio (2014): Neural Machine Translation by Jointly Learning to Align and Translate

Paper: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

## Quick start

```bash
uv sync
python3 -u nmt.py | tee run.txt
```

## Goal

Implement the paper from scratch in PyTorch: the bidirectional GRU encoder, the additive attention (alignment) model, the GRU decoder with a maxout deep output, the paper's initialisation and training procedure, and beam search decoding. The goal was to implement the method faithfully and understand every component, not to match the paper's BLEU score, which would require its data and compute.

## The synthetic task

Instead of WMT'14 English to French, the model is trained on a synthetic translation task.

- Each source sentence is 2 to 5 adjective-noun phrases, for example `red house`.
- The target translates each word and swaps the order to noun then adjective, as in French, for example `la maison rouge`.
- Some nouns carry an article (`le`, `la`) and some do not, so source and target lengths differ.
- There is exactly one correct translation per input.
- 6000 training pairs and 300 held-out pairs, generated from separately seeded random generators.
- Vocabulary: 16 source words and 18 target words, plus 4 special tokens (`<pad>`, `<sos>`, `<eos>`, `<unk>`), giving `K_x = 20` and `K_y = 22`.

This keeps the parts of the problem the paper cares about (word translation, local reordering, variable length) while making it small enough to train on a CPU. The local reordering is what makes the alignment heatmaps interesting: a correct model has to look at the noun before the adjective.

## What follows the paper

| Component | Paper | This implementation |
|---|---|---|
| Encoder | Bidirectional RNN, annotations $h_j = [\overrightarrow{h}_j ; \overleftarrow{h}_j]$ | Same |
| Encoder embeddings | One matrix shared by both directions | Same |
| Recurrent cell | Gated hidden unit (GRU), reset gate applied before the matrix, $U[r \circ h]$ | Same, written by hand since `nn.GRU` applies the reset gate after the matrix |
| Decoder context | $c_i$ enters the update gate, reset gate and candidate | Same |
| Alignment model | $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$ | Same |
| $U_a h_j$ | Precomputed once per sentence | Same |
| Initial decoder state | $s_0 = \tanh(W_s \overleftarrow{h}_1)$ | Same |
| Output layer | Deep output with one maxout layer | Same |
| Order of operations | Predict $y_i$ from $(s_{i-1}, y_{i-1}, c_i)$, then update to $s_i$ | Same (follows the A.2.2 formula) |
| Initialisation | Orthogonal recurrent matrices, $W_a$ and $U_a$ with std 0.001, $v_a$ and biases zero, all else std 0.01 | Same |
| Optimiser | Adadelta, $\rho = 0.95$, $\epsilon = 10^{-6}$ | Same |
| Gradient clipping | L2 norm rescaled to at most 1 | Same |
| Batch size | 80 | Same |
| Batching | Shuffle once, then sort each block of 1600 pairs by length and split into 20 minibatches | Same |
| Source order | Not reversed | Same |
| Decoding | Beam search | Same, plus greedy for comparison |

## Deviations

| | Paper | This implementation | Reason |
|---|---|---|---|
| Hidden size $n$ | 1000 | 128 | Compute |
| Embedding size $m$ | 620 | 80 | Compute |
| Maxout size $l$ | 500 | 64 | Compute |
| Alignment hidden size $n'$ | 1000 | 128 | Compute |
| Parameters | Not stated | 469,302 | Compute |
| Data | 348M-word selected subset of WMT'14 | 6000 synthetic pairs | Compute and data |
| Vocabulary | 30,000 per language | 16 source / 18 target words | Synthetic task |
| Training length | About 5 days per model, 2.2 to 6.4 epochs | 55 epochs | Much smaller dataset, and Adadelta starts slowly |
| Beam width | Not stated | 5 | Paper does not give a value |
| Metric | BLEU | Exact match | One correct answer per input, so partial credit hides whether the mapping was learned |

## Pipeline overview

1. **Data generation.** Seeded generators produce training and held-out pairs. The held-out set uses its own seed, so it does not change when the training set size changes.
2. **Vocabularies.** One per language, mapping words to integer ids, with reserved ids for the special tokens.
3. **Tensorisation.** `<eos>` is appended to the source, which is not reversed. The target is wrapped in `<sos>` and `<eos>`.
4. **Batching.** The data is shuffled once. Blocks of 1600 pairs are sorted by source length, split into 20 minibatches of 80, and padded at the end.
5. **Encoder.** Forward and backward GRUs read the embedded source. On padded positions each direction keeps its previous state, so the backward GRU effectively starts at each sentence's real last word. The two directions are concatenated into annotations of size $2n$.
6. **Attention.** At each decoder step, every annotation is scored against $s_{i-1}$, padded positions are masked to $-\infty$, the scores are softmaxed over source positions, and the context $c_i$ is the weighted sum of annotations.
7. **Decoder.** Predict $y_i$ through the maxout deep output, then update the GRU state with the previous word and the context.
8. **Training.** Teacher forcing: the decoder input is the target shifted right, the labels are the target shifted left. Loss is summed over real tokens and divided by the number of sentences.
9. **Decoding.** Greedy decoding and beam search, both starting from `<sos>` and stopping at `<eos>` or a length cap. Each beam hypothesis carries its own attention rows, so the printed heatmap belongs to the sentence that was actually output.
10. **Evaluation.** Exact match on held-out pairs, sample translations, and a text alignment table in the style of the paper's Figure 3.

## Sanity checks

These run before training and are printed at the top of every run.

- **Shape check.** Every tensor boundary is asserted in the paper's symbols, including that each attention row sums to 1 and that padded positions receive exactly zero weight.
- **Uniform attention at initialisation.** With $v_a = 0$ every score is 0, so attention must be exactly $1/T_x$. The check asserts this and prints the attention entropy next to $\log T_x$, the maximum possible entropy.
- **Overfit check.** A fresh model trained on 20 pairs must memorise them. This proves gradients reach every component, the alignment model included.

```
Uniform attention at init
  attention entropy 2.3979  log(T_x) 2.3979  (uniform)

Overfit check
  epoch    1  loss/token 3.0910  ppl   22.000
  epoch  250  loss/token 1.2594  ppl    3.523
  epoch  500  loss/token 0.1124  ppl    1.119
  epoch  750  loss/token 0.0003  ppl    1.000
  epoch 1000  loss/token 0.0001  ppl    1.000
  memorised 100% of 20 training pairs
```

The starting perplexity of exactly 22.000 equals `K_y`: before training the model assigns equal probability to every target word, which is what the initialisation should give.

## Results

All three runs use the same code and configuration, changing only the number of epochs.

### Run 1: 30 epochs

```
Training
  epoch    1  loss/token 2.3191  ppl   10.167
  epoch    5  loss/token 1.6056  ppl    4.981
  epoch   10  loss/token 1.5510  ppl    4.716
  epoch   15  loss/token 1.3978  ppl    4.046
  epoch   20  loss/token 1.1476  ppl    3.151
  epoch   25  loss/token 0.9556  ppl    2.600
  epoch   30  loss/token 0.8296  ppl    2.292

Held-out exact match
  greedy          0.0%
  beam (B=5)      0.0%
```

```
source          red house red cat new stone small house
reference       la maison rouge le chat rouge la pierre neuf la maison petit
predicted       la pierre rouge le chat rouge la pierre rouge la maison froid
```

```
Alignment (rows: target, columns: source)
           red house   red   cat   new stone small house <eos>
la        0.10  0.12  0.11  0.11  0.11  0.12  0.11  0.11  0.11
pierre    0.10  0.11  0.11  0.11  0.11  0.11  0.11  0.11  0.11
rouge     0.11  0.11  0.11  0.11  0.11  0.12  0.11  0.11  0.11
le        0.11  0.11  0.11  0.11  0.11  0.12  0.11  0.11  0.11
chat      0.08  0.11  0.11  0.11  0.11  0.13  0.12  0.12  0.11
rouge     0.06  0.12  0.10  0.11  0.12  0.14  0.12  0.12  0.11
la        0.03  0.09  0.05  0.06  0.11  0.23  0.16  0.16  0.10
pierre    0.03  0.04  0.03  0.03  0.04  0.40  0.18  0.21  0.05
rouge     0.02  0.04  0.02  0.02  0.04  0.33  0.24  0.25  0.04
la        0.04  0.05  0.02  0.03  0.05  0.28  0.22  0.29  0.03
maison    0.08  0.09  0.06  0.07  0.08  0.18  0.14  0.23  0.07
froid     0.05  0.07  0.04  0.05  0.06  0.22  0.21  0.25  0.05
<eos>     0.04  0.07  0.04  0.05  0.07  0.22  0.21  0.24  0.06
```

### Run 2: 100 epochs

```
Training
  epoch   30  loss/token 0.8296  ppl    2.292
  epoch   35  loss/token 0.5954  ppl    1.814
  epoch   40  loss/token 0.1080  ppl    1.114
  epoch   45  loss/token 0.0013  ppl    1.001
  epoch   50  loss/token 0.0001  ppl    1.000
  epoch   55  loss/token 0.0000  ppl    1.000
  ...
  epoch  100  loss/token 0.0000  ppl    1.000

Held-out exact match
  greedy          100.0%
  beam (B=5)      100.0%
```

Epochs 1 to 30 are identical to run 1. The alignment table differs from run 3's only by a few hundredths per cell.

### Run 3: 55 epochs (final)

```
Training
  epoch    1  loss/token 2.3191  ppl   10.167
  epoch   10  loss/token 1.5510  ppl    4.716
  epoch   20  loss/token 1.1476  ppl    3.151
  epoch   30  loss/token 0.8296  ppl    2.292
  epoch   40  loss/token 0.1080  ppl    1.114
  epoch   50  loss/token 0.0001  ppl    1.000

Held-out exact match
  greedy          100.0%
  beam (B=5)      100.0%
```

```
source          red house red cat new stone small house
reference       la maison rouge le chat rouge la pierre neuf la maison petit
predicted       la maison rouge le chat rouge la pierre neuf la maison petit

source          red bird red river red stone new tree big house
reference       oiseau rouge la riviere rouge la pierre rouge arbre neuf la maison grand
predicted       oiseau rouge la riviere rouge la pierre rouge arbre neuf la maison grand

source          red road small cat green field cold bird
reference       route rouge le chat petit le champ vert oiseau froid
predicted       route rouge le chat petit le champ vert oiseau froid
```

```
Alignment (rows: target, columns: source)
           red house   red   cat   new stone small house <eos>
la        0.26  0.59  0.04  0.03  0.02  0.02  0.02  0.02  0.00
maison    0.35  0.53  0.04  0.02  0.01  0.01  0.01  0.01  0.00
rouge     0.52  0.39  0.04  0.02  0.01  0.01  0.01  0.01  0.01
le        0.04  0.03  0.59  0.28  0.02  0.01  0.01  0.01  0.00
chat      0.05  0.03  0.58  0.28  0.02  0.02  0.01  0.01  0.00
rouge     0.05  0.04  0.57  0.29  0.02  0.01  0.01  0.01  0.00
la        0.01  0.00  0.00  0.00  0.52  0.42  0.02  0.02  0.00
pierre    0.01  0.00  0.00  0.00  0.17  0.75  0.03  0.02  0.00
neuf      0.00  0.00  0.00  0.00  0.52  0.41  0.03  0.03  0.00
la        0.00  0.00  0.00  0.00  0.01  0.02  0.50  0.45  0.00
maison    0.00  0.00  0.00  0.00  0.01  0.02  0.46  0.49  0.00
petit     0.00  0.00  0.00  0.00  0.02  0.02  0.50  0.46  0.00
<eos>     0.01  0.01  0.01  0.01  0.04  0.06  0.59  0.26  0.01
```

## Analysis

### Run 1 stopped mid-descent, not at a plateau

At 80 pairs per batch, one epoch over 6000 pairs is 75 updates, so 30 epochs is only about 2250 updates. The overfit check needed around 750 updates to memorise just 20 pairs, because Adadelta's update accumulator starts at zero and early steps are tiny. The loss was still falling steadily at epoch 30, so the model was undertrained, not stuck.

The run 1 outputs show what a partly trained attention model looks like. The translations are grammatical: articles match their nouns and adjectives follow nouns. But the content words are often wrong (`la pierre` for "house"). In the alignment table, the first six rows are about 0.11 everywhere, which is $1/9$, still exactly uniform. In the later rows attention has found the right region of the source and some nouns have found their source word (`pierre` puts 0.40 on "stone"), but the adjectives have not.

Exact match of 0% hides this progress. Sentences have up to 15 target tokens and one wrong word fails the whole sentence.

### Attention clicks suddenly

Between epochs 30 and 45 the loss falls from 0.83 to 0.001, far faster than anywhere before. Once the model learns where to look, every word becomes easy at once, so the loss collapses rather than declining smoothly. After epoch 50 the loss is essentially zero, and epochs 55 to 100 in run 2 change nothing, which is why the final run stops at 55.

### The model learned alignment with no alignment supervision

In run 3 every row concentrates on its own source phrase, the phrases progress through the sentence in order, and weights outside the current phrase are near zero:

```
la maison rouge   ->  red house
le chat rouge     ->  red cat
la pierre neuf    ->  new stone
la maison petit   ->  small house
```

This is the roughly monotonic alignment of the paper's Figure 3, learned from the translation loss alone.

Inside a phrase, the noun-adjective crossing is partly visible. `pierre` puts 0.75 on "stone" while `neuf` puts 0.52 on "new". `maison` leans to "house" while `rouge` leans to "red". In the last phrase, `maison` peaks on "house" and `petit` on "small".

### Phrase-level, not word-level: annotations are windows

The alignment is not perfect word by word. In `le chat rouge`, all three rows peak on "red", including `chat`. The model still translates correctly because each annotation comes from a bidirectional encoder: the annotation at "red" already contains information about "cat" from the backward GRU. Attending one position off can still retrieve the right word.

This is the caveat the paper's heatmaps carry. An annotation is a window centred on its word, not the word itself, so attention weights show where the model looks, not exactly what information it uses.

### Comparison with the seq2seq implementation

The same synthetic task defeated a Sutskever et al. (2014) seq2seq implementation, whose loss plateaued at about 1.44 per token, the level of a language model that ignores the source. RNNsearch passes that level by epoch 20 (1.15) and goes on to solve the task. This is consistent with the paper's argument that removing the fixed-length bottleneck makes the source easier to use. It is not a controlled comparison, though: the two models also differ in cell type, depth, size, initialisation and optimiser.

### Training loss near zero, but no overfitting

Training to near-zero loss raises the question of memorisation. More epochs can lead to overfitting, but the number of epochs is not the cause on its own. Overfitting happens when a model fits noise or quirks of the training set instead of the underlying pattern, and whether that happens depends on model capacity relative to the amount of data and how noisy it is. The way to detect it is to track loss on a held-out set: if training loss keeps falling while held-out loss rises, the model is overfitting, and the best stopping point is the held-out minimum. This is early stopping, which is what the paper used for RNNsearch-50*.

Here, held-out exact match is 100% on 300 sentences never seen in training, so the model generalised rather than memorised. The task is noise-free and 6000 examples cover the rule well, so there is nothing to fit beyond the rule itself, and no regularisation was needed. On real translation data, with many valid outputs, rare words and noisy references, this would not hold, and a validation set with early stopping would be necessary.

Near-zero training loss is not bad in itself. Large networks often reach it and still generalise. What matters is held-out performance. Other ways to prevent overfitting when it does occur include more data, regularisation such as dropout or weight decay, smaller models and data augmentation.

## Next steps

These are the planned ablations. Runs should be matched either by updates or by wall-clock time, and the choice reported.

1. **Attention on or off, by unidirectional or bidirectional encoder.** Every RNNsearch result in the paper changes both at once. The 2x2 separates their effects.
2. **Accuracy against source length.** The plot that tests the paper's bottleneck conjecture directly. This needs longer sentences than the current 2 to 5 phrases.
3. **Two training length limits,** the analogues of the paper's -30 and -50 models.
4. **Beam width sweep,** since the paper does not state its value.
5. **Alignment hidden size $n'$.**
6. **GRU against LSTM,** to confirm the cell choice is a control variable.
7. **Paper-style GRU against PyTorch-style GRU** (reset gate before or after the matrix).
8. **Adadelta against Adam,** to measure how much of the slow start comes from the optimiser.
9. **Held-out loss and attention entropy logged during training,** to show the moment attention starts to sharpen.

## Lessons

- **Implementing a method and reproducing a result are different things.** The architecture and training procedure follow the paper. The numbers are from a toy task and say nothing about BLEU on WMT.
- **Sanity checks earn their place.** Uniform attention at initialisation confirmed the initialisation, and the overfit check confirmed that every component learns, before any long run was started.
- **A metric of 0% can hide real progress.** In run 1 the alignment table showed attention forming while exact match was still zero.
- **Training curves are not always smooth.** The loss sat on a slow slope for 30 epochs and then collapsed within 15. Stopping early would have suggested the model could not learn the task.
- **Look at the attention, not just the outputs.** The heatmaps showed both what the model learned (phrase-level alignment) and its limits (word-level weights that depend on the encoder's context windows).

## Citations

```bibtex
@misc{bahdanau2014neural,
    title   = {Neural Machine Translation by Jointly Learning to Align and Translate},
    author  = {Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio},
    year    = {2014},
    eprint  = {1409.0473},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/1409.0473}
}
```