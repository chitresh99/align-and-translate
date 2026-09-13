# align-and-translate

From-scratch implementation of Bahdanau, Cho and Bengio (2014), "Neural Machine Translation by Jointly Learning to Align and Translate" ([arXiv:1409.0473](https://arxiv.org/abs/1409.0473)).

This is the paper that introduced attention. It is the direct successor to the fixed-vector encoder-decoder models of Sutskever et al. (2014) and Cho et al. (2014), and the direct ancestor of the Transformer.

## First-pass summary

### The problem

Neural machine translation in 2014 meant a single network trained end to end by one loss, in contrast to statistical MT, which was a pipeline of separately fit components (phrase table, language model, reordering model, distortion model) combined with weights tuned on a dev set. The neural approach was attractive because every parameter, embeddings included, is adjusted by the same gradient signal from the translation objective.

The dominant neural architecture was the encoder-decoder. An RNN reads the source sentence and its final hidden state, a single fixed-size vector of roughly 1000 numbers, is taken as a summary of the whole sentence. A second RNN is initialised from that vector and generates the target sentence word by word.

The entire source sentence therefore passes through one fixed-size vector, regardless of whether the sentence is 5 words or 50.

### The conjecture

Bahdanau et al. conjecture that this fixed-length vector is the bottleneck limiting encoder-decoder performance. The supporting observation, already reported by Cho et al., is that BLEU falls off sharply as source sentence length grows, which is exactly the failure mode the bottleneck theory predicts.

The word "conjecture" is doing real work here. The paper states a hypothesis about the cause of the degradation, builds a model that removes the hypothesised cause, and shows the degradation goes away. That is the evidence, not a direct measurement of the bottleneck itself.

### The proposal

Do not compress the source into one vector. Keep one vector per source position, and at every decoding step compute a fresh weighted average of them, with the weights depending on what the decoder is currently trying to generate.

Producing the target word for "agreement" puts most of the weight on the source position holding "agreement". A different target word produces different weights. The context vector is recomputed at every output step instead of being fixed for the whole sentence.

### Why "soft"

Classic alignment models in statistical MT made discrete decisions: source word 4 aligns to target word 6, full stop. Discrete choices are not differentiable, so alignment had to be a separate training stage.

Soft alignment means the weights are continuous and sum to 1, a distribution over source positions rather than a pick. That is differentiable, so alignment is learned by the same gradient as everything else. Hence the title: alignment stops being preprocessing and becomes part of the model.

Two reasons the blend is preferable to a hard selection:

1. Trainability. Taking the argmax is a discrete operation with zero gradient almost everywhere, so the alignment could never be learned from the translation loss.
2. Expressiveness. Some target words genuinely depend on several source words (compounds, idioms, agreement across distance). A blend can represent that; a pick cannot.

Soft selection includes hard selection as a limiting case: when the model wants a single source word it learns to put a weight of 0.99 on it. Nothing is lost by blending.

### Results claimed

Translation performance on English-to-French comparable to the state-of-the-art phrase-based system of the time. In 2014 this was a contested claim and it is the result that moved the field toward NMT.

The paper also plots the attention weight matrix as a heatmap, source positions on one axis and target positions on the other. The result is roughly diagonal with off-diagonal structure where grammar predicts it, for example the crossing pattern where English and French adjective-noun order differs. None of this was supervised.

## Architecture

### Encoder

Bidirectional RNN. One GRU reads the source left to right, a second reads it right to left, and the annotation for position `j` is the concatenation of the two states:

```
h_j = [forward_h_j ; backward_h_j]
```

With a hidden size of 1000 per direction, each annotation is 2000-dimensional. Each annotation is centred on its own word but carries context from both sides.

### Alignment model

A single hidden layer MLP scoring a decoder state against a source annotation:

```
e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)
```

with `W_a` of shape `n x n`, `U_a` of shape `n x 2n`, and `v_a` of shape `n`, where `n = 1000`.

This is additive attention: the two inputs are combined by summing their projections, in contrast to the dot product used later by the Transformer.

### Context vector

```
alpha_ij = softmax_j(e_ij)
c_i = sum_j alpha_ij h_j
```

The weights are non-negative and sum to 1. They are used to average, never to sample. The paper does interpret `alpha_ij` as the probability that target word `i` is aligned to source word `j`, which makes `c_i` an expected annotation, and that reading is the link back to the IBM alignment models. Mechanically it remains a weighted average.

### Decoder

```
s_i = f(s_{i-1}, y_{i-1}, c_i)
p(y_i | y_1..y_{i-1}, X) = softmax(g(y_{i-1}, s_i, c_i))
```

The context vector feeds both the state update and the output layer. Because `s_{i-1}` is an input to the scoring function and `s_{i-1}` changes every step, the attention weights change every step. That is the whole mechanism.

## Which choices are argued for, and which are held fixed

Worth separating, because only the first group is what the paper is actually defending.

**Argued for:**

- Per-position annotations instead of a single fixed vector. This is the contribution.
- Soft, differentiable alignment rather than hard segmentation.
- Joint training of alignment and translation under one loss.

**Held fixed as control variables or set by compute constraints:**

- GRU rather than LSTM. Chosen for consistency with the Cho et al. baseline, not argued to be better.
- Single hidden layer for the alignment MLP. A depth choice under a compute budget, given the MLP runs source-length times target-length times per sentence.
- Maxout output layer. Inherited from contemporary practice.
- Vocabulary size, embedding size, beam width. Experimental settings.

## Caveats to carry into the implementation

**The bidirectionality confound.** The paper changes two things relative to the baseline: it adds attention and it makes the encoder bidirectional. The reported gain is attributable to the pair, not cleanly to attention alone. Separating them is a natural ablation and is on the list below.

**Limits of the interpretability claim.** The heatmaps look like alignments and agree with intuition, which is genuine evidence that something meaningful is learned. It is not evidence that the weights are a faithful account of what the model uses to make each prediction. Attention weights being interpretable and attention weights being an explanation are different claims, and only the weaker one is supported here.

**Cost.** Scoring is O(source length times target length) MLP evaluations per sentence. Acceptable at sentence scale, and this is the same quadratic cost the Transformer later inherits.

## Planned ablations

- Attention on or off, holding the bidirectional encoder fixed. Isolates the attention contribution from the bidirectionality confound.
- Unidirectional against bidirectional encoder, holding attention fixed. The other half of the same separation.
- BLEU against source sentence length. The central prediction of the bottleneck conjecture, and the plot the paper leans on.
- Beam width sweep. A search-side variable over a fixed model, so nothing about the network changes.
- Alignment MLP hidden size.
- GRU against LSTM, to check that the cell choice really is a control variable and not load-bearing.

## Repository layout

```
align-and-translate/
  notes/            Reading notes, one file per pass
  src/              Model, data, training, evaluation
  experiments/      Configs and logged runs
  README.md
  WORKFLOW.md       Step-by-step forward and backward pass
```

## References

- Bahdanau, Cho and Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473
- Sutskever, Vinyals and Le (2014). Sequence to Sequence Learning with Neural Networks. arXiv:1409.3215
- Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078