# Bahdanau, Cho and Bengio (2014): Final Pass

"Neural Machine Translation by Jointly Learning to Align and Translate" ([arXiv:1409.0473](https://arxiv.org/abs/1409.0473)), published at ICLR 2015.

The complete reading of the paper: the argument, every equation with notation, the appendix details that the implementation depends on, the experiments, how to read the results, and where the paper sits in the line from seq2seq to the Transformer. Cross-checked line by line against the paper text (v7).

## One-line summaries

**The paper in one line:** Bahdanau contributed the addressing mechanism.

**What attention is, in plain words:** the decoder looks at a different part of the source for every word it writes.

**What happens mechanically:** the word being produced looks at the source annotations, all of them, weighted by relevance, recomputed for every output word.

**Where attention is in the paper:** Section 3.1, equations 5 to 7. The last paragraph of Section 3.1 names it directly: the authors write that this "implements a mechanism of attention in the decoder". The paper's main framing is alignment and (soft-)search; "attention" became the standard name for the mechanism through later work (Xu et al. 2015, Luong et al. 2015).

**What it contributed to attention:** content-based (weights come from comparing the decoder state to each source vector, not from position), soft (differentiable, so it trains from translation loss with no alignment supervision), and unconstrained (free to jump anywhere in the source). It also fixed the four-step template everything since follows: score, softmax, weighted sum, use the result.

**The line to the Transformer:** the Transformer changed only the scoring function and added heads, then removed the RNN.

## 1. The argument

### The abstract, sentence by sentence

**"A single neural network that can be jointly tuned."** Statistical MT in 2014 was a pipeline of separately trained parts: a phrase table extracted from word alignments, an n-gram language model, a reordering model, a distortion model, all combined with weights tuned on a dev set. Nobody optimised the whole system end to end. Neural MT replaces the pipeline with one network trained by one loss, so every parameter, embeddings included, is adjusted by the same gradient.

**"Encode a source sentence into a fixed-length vector."** The encoder-decoder of Cho et al. (2014) and Sutskever et al. (2014). An RNN reads the source, its final hidden state is taken as a summary of the whole sentence, and a second RNN generates the translation from that summary. The whole source passes through one vector of about 1000 numbers, whether the sentence is 5 words or 50.

**"We conjecture that the use of a fixed-length vector is a bottleneck."** The diagnosis. A long sentence has to be squeezed into the same budget as a short one. Cho et al. (2014b) had already shown performance deteriorating rapidly with sentence length, which is what a bottleneck predicts. It is stated as a conjecture because the bottleneck cannot be measured directly. The test is to build a model without it and see whether the degradation goes away.

**"Automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word."** The fix. Keep one vector per source position instead of one summary, and at every output step compute a fresh weighted average of them, weighted toward whatever matters for the word being produced now.

**"Without having to form these parts as a hard segment explicitly."** Aimed at phrase-based SMT, which first chops the source into phrase segments as a separate stage. This model never commits to a segmentation. Any "segment" is a pattern visible afterward in the attention heatmap, not something the model built.

**"Comparable to the existing state-of-the-art phrase-based system."** True on sentences without unknown words. See Section 6.

**"The (soft-)alignments found by the model agree well with our intuition."** The attention weights, plotted as a source by target heatmap, come out roughly diagonal, with crossings where word order differs between English and French. None of it was supervised.

### What "soft" means and why it matters

A hard search returns a subset: positions 3 and 4, nothing else. A soft search returns a weight for every position, all non-negative, summing to 1. Nothing is excluded; irrelevant positions are pushed toward zero rather than removed.

The paper states this directly: the alignment is not treated as a latent variable; the model computes a soft alignment so the gradient of the cost can be backpropagated through it.

It has to be soft for two reasons.

1. **Trainability.** A hard pick is an argmax, and argmax has zero gradient almost everywhere. You cannot backpropagate through "I picked position 3". A hard search would need its own training stage, exactly as alignment did in SMT. Soft weights are continuous, so translation loss flows back through them into the scoring network.
2. **Expressiveness.** Some target words depend on several source words. The paper's own example (Figure 3d): "the man" becomes "l' homme". A hard alignment maps "the" to "l'" and "man" to "homme", but choosing between le, la, les and l' requires seeing the next word. A soft alignment looks at both.

Soft selection contains hard selection as a limiting case. When the model wants a single source word, it learns to put 0.99 on it. The paper also notes a side benefit: source and target phrases of different lengths are handled naturally, with no need to align words to or from nowhere (the NULL alignment of SMT).

## 2. The baseline framework (Section 2.1)

### Encoder

$$
h_t = f(x_t, h_{t-1}) \qquad (1)
$$

$$
c = q(\lbrace h_1, \ldots, h_{T_x} \rbrace)
$$

| symbol | meaning |
| --- | --- |
| $x_t$ | source word at position $t$ |
| $h_t \in \mathbb{R}^n$ | encoder hidden state at time $t$ |
| $f$ | the recurrent cell (vanilla RNN, GRU, or LSTM) |
| $c$ | the summary vector the decoder starts from |
| $q$ | a function collapsing all hidden states into $c$ |

Sutskever et al. use $q(\lbrace h_1, \ldots, h_T \rbrace) = h_T$, so in practice $c$ is the last hidden state.

Writing $q$ as a general function of the whole set is a deliberate setup. Footnote 2 says it outright: encoding into a fixed-length vector is not necessary, and a variable-length representation may be better. Applying $q$ is exactly where the variable-length source becomes fixed-length: the bottleneck, located precisely.

Note: $c$ here is the context vector. In LSTM notation $c_t$ is the cell state. Same letter, unrelated meaning.

### Decoder

$$
p(\mathbf{y}) = \prod_{t=1}^{T} p(y_t \mid \lbrace y_1, \ldots, y_{t-1} \rbrace, c) \qquad (2)
$$

$$
p(y_t \mid \lbrace y_1, \ldots, y_{t-1} \rbrace, c) = g(y_{t-1}, s_t, c) \qquad (3)
$$

**Equation 2 in plain words:** the probability of a whole translation is the probability of the first word, times the probability of the second given the first, times the probability of the third given the first two, and so on.

**Equation 3 in plain words:** each of those probabilities is computed by a neural network that looks at the word just produced, the decoder's hidden state, and the source summary.

Worked out for "le chat noir":

$$
p(\text{le chat noir}) = p(\text{le} \mid c) \cdot p(\text{chat} \mid \text{le}, c) \cdot p(\text{noir} \mid \text{le}, \text{chat}, c)
$$

Each factor is the probability of that step's word, not of the earlier words; those are only in the conditioning.

**Equation 2 is exact.** It is the chain rule of probability, an identity that holds for any joint distribution. It turns one intractable problem (a distribution over all sentences) into $T$ tractable ones (a distribution over the next word). It does not assume independence; the conditioning is the point.

**Equation 3 is an approximation.** Equation 2 conditions on the full history, but $g$ receives only $y_{t-1}$ explicitly. The rest of the history lives in $s_t$, which was built recursively and is a fixed-size stand-in for an arbitrarily long history. Same compression problem as $c$, on the target side.

**At training time nothing is multiplied.** Taking logs turns the product into a sum,

$$
\log p(\mathbf{y}) = \sum_t \log p(y_t \mid y_{<t}, c)
$$

which avoids underflow and is exactly the per-position cross-entropy loss. Beam search scores in the same log space.

## 3. The proposed model (Section 3)

### 3.1 Decoder

**Intuition:** to pick the next word, look at the word just written, where you are in the sentence, and a fresh summary of the source built for this word specifically. The state advances by combining where it was, what it just wrote, and that same fresh summary.

$$
p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i) \qquad (4)
$$

$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$

| symbol | meaning | shape |
| --- | --- | --- |
| $\mathbf{x}$ | the full source sentence | $T_x$ words |
| $y_i$ | target word at output position $i$ | vocabulary index |
| $s_i$ | decoder hidden state at step $i$ | $\mathbb{R}^n$, $n = 1000$ |
| $c_i$ | context vector for this output step | $\mathbb{R}^{2n}$ |
| $g$ | output function, emits a distribution over the vocabulary | $\mathbb{R}^{K_y}$ |
| $f$ | recurrent cell (gated hidden unit, i.e. GRU) | $\mathbb{R}^n$ |

Compare with equation 3: the only change is $c \to c_i$. That subscript is the paper's contribution. The context used to be computed once and reused for every output word; now it is recomputed per word.

$c_i$ enters twice, through $f$ into the state and directly into $g$. The direct path lets the current context influence the current word without having to be stored in the state first.

The left side now conditions on $\mathbf{x}$ rather than $c$, because the decoder keeps access to every source position rather than one summary.

**The paper is internally inconsistent about which state predicts $y_i$.** Equation 4 writes $s_i$. Appendix A.2.2 says, in prose, that the prediction uses $s_{i-1}$, then writes the conditional as $p(y_i \mid s_i, y_{i-1}, c_i)$, then gives the actual formula $\tilde{t}_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i$. The formula is the only unambiguous statement, and it uses $s_{i-1}$. So the order of operations actually run is: score with $s_{i-1}$, build $c_i$, predict $y_i$ from $(s_{i-1}, y_{i-1}, c_i)$, then update to $s_i$.

### Equations 5 to 7: the attention mechanism

**Intuition, in execution order (bottom up):**

- **Eq. 7:** score how relevant each source word is to the target word about to be written.
- **Eq. 6:** turn those scores into weights that sum to 1.
- **Eq. 5:** blend all the source vectors together in those proportions.

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j \qquad (5)
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \qquad (6)
$$

$$
e_{ij} = a(s_{i-1}, h_j) \qquad (7)
$$

| symbol | meaning | shape |
| --- | --- | --- |
| $i$ | target position, the word being generated | |
| $j$ | source position, the word being scored | |
| $k$ | dummy index over source positions | |
| $T_x$ | source length | |
| $h_j$ | annotation for source position $j$ | $\mathbb{R}^{2n}$ |
| $s_{i-1}$ | decoder state before producing word $i$ | $\mathbb{R}^n$ |
| $e_{ij}$ | raw relevance score ("energy" in the paper) | scalar |
| $\alpha_{ij}$ | normalised weight | scalar in $[0, 1]$ |
| $c_i$ | context vector | $\mathbb{R}^{2n}$ |
| $a$ | alignment model, a small MLP | scalar output |

Every $\alpha_{ij}$ is one cell of a $T_y \times T_x$ matrix, which is the heatmap the paper plots. The paper reads it row by row: each row holds the weights for one target word.

**Eq. 7.** The alignment model runs $T_x$ times per output step. It takes $s_{i-1}$ rather than $s_i$ because $s_i$ depends on $c_i$, which depends on these scores; using $s_i$ would be circular. $a$ is the only learned component in these three equations. Its parametrisation is in Appendix A.1.2 and A.2.2:

$$
a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)
$$

with $W_a \in \mathbb{R}^{n' \times n}$, $U_a \in \mathbb{R}^{n' \times 2n}$, $v_a \in \mathbb{R}^{n'}$, where $n'$ is the alignment model's hidden size ($n' = 1000 = n$ in the paper, which is why A.1.2 writes the shapes with $n$). The two inputs are combined by adding their projections, which is why this is called additive attention. Because $U_a h_j$ does not depend on $i$, the paper precomputes it once per sentence.

**Eq. 6.** Softmax over source positions for a fixed target step. Weights sum to 1 over $j$:

$$
\sum_{j} \alpha_{ij} = \frac{\sum_j \exp(e_{ij})}{\sum_k \exp(e_{ik})} = 1
$$

The paper never writes this out; it follows from the formula. Rows of the alignment matrix sum to 1, columns do not, so nothing forces every source word to be used. Softmax provides positivity, so vectors are averaged rather than cancelled, and differentiability, which is what lets gradient reach $a$.

**Eq. 5.** A convex combination of the annotations. $c_i$ has the size of a single annotation regardless of sentence length. The compression still happens, but it is now per word and content-dependent instead of once per sentence.

The paper reads $\alpha_{ij}$ as the probability that target word $i$ is aligned to, or translated from, source word $j$, making $c_i$ an expected annotation over possible alignments. That is the link back to the IBM alignment models. Mechanically it remains a weighted average; nothing is sampled.

**Together.** Gradient flows from the translation loss through $c_i$, the weights, the softmax, the scores, and into $v_a$, $W_a$, $U_a$. The alignment is learned with no alignment supervision. That is the "jointly" in the title.

**Cost.** $a$ is evaluated $T_x \times T_y$ times per sentence pair. The paper acknowledges this in Section 6.1: acceptable for translation, where sentences are mostly 15 to 40 words, but a possible limit for other tasks. It is the same quadratic the Transformer inherits.

### 3.2 Encoder: bidirectional RNN

$$
h_j = \left[ \overrightarrow{h}_j^\top ; \overleftarrow{h}_j^\top \right]^\top
$$

**Intuition, with "the bank was steep":**

- A forward RNN reaching "bank" has seen only "the bank". It cannot tell a riverbank from a financial bank, because "steep" has not been read yet. If the decoder attends to that position, it retrieves an ambiguous vector.
- A backward RNN reads "steep was bank the". Reaching "bank", it has seen "bank was steep".
- Concatenating both gives position 2 a vector that knows both sides, so the decoder retrieves a disambiguated "bank".
- Because RNNs remember recent input more strongly, both halves peak on "bank" itself and fade outward. The vector is a window centred on its own word, not a flat sentence summary.

That last point is what makes attention work. If every annotation encoded the whole sentence equally, they would all look alike and there would be nothing to choose between. Each annotation needs to be position-specific enough to attend to and context-rich enough to be worth retrieving.

| symbol | meaning | shape |
| --- | --- | --- |
| $\overrightarrow{h}_j$ | forward state, has seen $x_1 \ldots x_j$ | $\mathbb{R}^n$ |
| $\overleftarrow{h}_j$ | backward state, has seen $x_j \ldots x_{T_x}$ | $\mathbb{R}^n$ |
| $h_j$ | annotation | $\mathbb{R}^{2n}$ |

Two separate RNNs with separate weight matrices, but a **shared word embedding matrix** (A.2.1). The transposes are bookkeeping for stacking.

**Why it is needed here.** In the baseline only $h_{T_x}$ was used, and it has read the whole sentence, so direction did not matter. Once every $h_j$ is attended to, a unidirectional $h_j$ is blind to everything after position $j$. Attention is what makes bidirectionality necessary.

The recency claim is asserted, not measured. The paper does not quantify how wide the effective window is.

## 4. Appendix details (A and B)

The body describes the framework; the appendix describes what was actually trained. An implementation depends on these.

### The recurrent cell (A.1.1)

The paper's "gated hidden unit" is the GRU of Cho et al. (2014a). For the decoder:

$$
s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i
$$

$$
\tilde{s}_i = \tanh\left(W E y_{i-1} + U[r_i \circ s_{i-1}] + C c_i\right)
$$

$$
z_i = \sigma(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i), \qquad r_i = \sigma(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)
$$

The context enters all three computations, through $C, C_z, C_r \in \mathbb{R}^{n \times 2n}$. The encoder uses the same equations without the $C$ terms.

The reset gate is applied **before** the matrix: $U[r \circ s]$. PyTorch's `nn.GRU` and cuDNN apply it after: $r \circ (U s + b)$. These are two different GRU variants.

### Initial decoder state (A.2.2)

$$
s_0 = \tanh(W_s \overleftarrow{h}_1)
$$

The decoder starts from the backward RNN's state at position 1, which is the backward pass's summary of the whole sentence.

### Output layer (A.2.2)

A deep output with one maxout hidden layer:

$$
\tilde{t}_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i \in \mathbb{R}^{2l}
$$

$$
t_i = \left[\max(\tilde{t}_{i,2j-1}, \tilde{t}_{i,2j})\right]_{j=1}^{l}
$$

$$
p(y_i \mid \ldots) \propto \exp(y_i^\top W_o t_i)
$$

with $W_o \in \mathbb{R}^{K_y \times l}$. Maxout takes adjacent pairs and keeps the larger, halving $2l$ to $l$.

### Model size (A.2.3)

| quantity | symbol | value |
| --- | --- | --- |
| hidden units | $n$ | 1000 |
| word embedding | $m$ | 620 |
| maxout hidden layer | $l$ | 500 |
| alignment hidden | $n'$ | 1000 |

### Initialisation (B.1)

- Recurrent matrices ($U, U_z, U_r$ for the decoder and both encoder directions): random orthogonal.
- $W_a$, $U_a$: Gaussian, mean 0, variance $0.001^2$.
- $v_a$ and all bias vectors: zero.
- Every other weight matrix: Gaussian, mean 0, variance $0.01^2$.

Consequence: with $v_a = 0$, every score starts at zero, so **attention begins uniform** and has to learn to sharpen.

### Training (B.2)

- Adadelta with $\epsilon = 10^{-6}$, $\rho = 0.95$.
- Gradient L2 norm rescaled to at most 1.
- Minibatch of 80 sentences.
- Training data shuffled once, then traversed sequentially. Before every 20th update, 1600 pairs are taken, sorted by length, and split into 20 minibatches. The paper does not say whether those 20 are then shuffled among themselves.

### Training statistics (Table 2)

| model | updates ($\times 10^5$) | epochs | hours |
| --- | --- | --- | --- |
| RNNenc-30 | 8.46 | 6.4 | 109 |
| RNNenc-50 | 6.00 | 4.5 | 108 |
| RNNsearch-30 | 4.71 | 3.6 | 113 |
| RNNsearch-50 | 2.88 | 2.2 | 111 |
| RNNsearch-50\* | 6.67 | 5.0 | 252 |

The models were matched by **wall-clock time**, not by updates. RNNenc-30 received nearly three times as many updates as RNNsearch-50, which saw the data just over twice. The comparison therefore favours the baseline on updates, which makes the attention result stronger, not weaker.

### The baseline as a special case

A.2.2 states that the model becomes the RNN Encoder-Decoder if $c_i$ is fixed to $\overrightarrow{h}_{T_x}$. This is true conceptually but does not type-check literally: $\overrightarrow{h}_{T_x}$ is $n$-dimensional while $C, C_z, C_r, C_o$ expect $2n$. An implementation has to decide what "attention off" means (for example $\overrightarrow{h}_{T_x}$ with a unidirectional encoder, or $[\overrightarrow{h}_{T_x}; \overleftarrow{h}_1]$ with a bidirectional one) and how the baseline's $s_0$ is initialised, since $\overleftarrow{h}_1$ does not exist without a backward RNN.

## 5. The full workflow

### Encoder, once per sentence

1. **Embed** each source word into an $m$-dimensional vector (620). One embedding matrix, shared by both directions.
2. **Forward GRU** reads left to right, giving $\overrightarrow{h}_1 \ldots \overrightarrow{h}_{T_x}$.
3. **Backward GRU** reads right to left, giving $\overleftarrow{h}_1 \ldots \overleftarrow{h}_{T_x}$.
4. **Concatenate** into annotations $h_j$ of size $2n$ (2000). Frozen for the whole decoding run.
5. **Precompute** $U_a h_j$ for every $j$.
6. **Initialise** $s_0 = \tanh(W_s \overleftarrow{h}_1)$.

### Decoder, once per output word (order as in Appendix A.2.2)

7. **Score** each annotation against $s_{i-1}$: $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$.
8. **Softmax over source positions** to get $\alpha_{ij}$.
9. **Blend** into $c_i = \sum_j \alpha_{ij} h_j$.
10. **Predict:** $\tilde{t}_i$ from $(s_{i-1}, y_{i-1}, c_i)$, maxout, project to $K_y$ logits (30,000), softmax over the vocabulary.
11. **Select** the output word $y_i$ (greedy or beam).
12. **Update state:** $s_i = \text{GRU}(s_{i-1}, y_{i-1}, c_i)$.
13. **Loop** until the end-of-sentence token.

### Two softmaxes, two jobs

| | step 8 | step 10 |
| --- | --- | --- |
| over | source positions | vocabulary entries |
| size | source length | 30,000 |
| question | where should I look | what should I write |
| used for | weighted averaging | selecting the word |

Attention never picks anything. The only hard choice in the pipeline is step 11.

### Training differences

Steps 1 to 10 and 12 are unchanged. Step 11 does not happen. The loss is the negative log of the probability given to the correct reference word. The next input is the reference word, not the model's own prediction (teacher forcing). Then backprop through everything, alignment MLP included.

## 6. Argmax, softmax, and decoding

### Argmax

`argmax` returns the input that maximises a function; `max` returns the value. Over the distribution {accord 0.71, convention 0.12, ...}, `max` is 0.71 and `argmax` is "accord". "arg" is short for argument.

**Two uses in papers.**

- **Over parameters, it states an objective.** $\theta^{\ast} = \arg\max_\theta \sum \log p(y \mid x; \theta)$ describes what training aims for. Nobody computes it; gradient descent approximates it.
- **Over outputs, it is an action.** $y_i = \arg\max_y p(y \mid \ldots)$ is greedy decoding. It is evaluated literally, at inference only.

Section 2 of this paper uses the first sense: translation is finding $\arg\max_{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x})$, and generation is searching for it.

**Why it matters.** Argmax is not differentiable. Nudge a score and the answer either stays put or jumps discontinuously, so the gradient is zero almost everywhere. Gradient descent needs a ramp; argmax is a staircase. This explains why attention blends instead of picking, why training uses cross-entropy on the full distribution, and why softmax exists.

If a paper puts argmax inside the training loop, it owes you a gradient estimator: straight-through, Gumbel-softmax, or REINFORCE.

### Softmax

$$
\text{softmax}(z)_j = \frac{\exp(z_j / \tau)}{\sum_k \exp(z_k / \tau)}
$$

A smooth approximation to argmax, not to max; the name is a misnomer. As $\tau \to 0$ it approaches a one-hot argmax, and at any $\tau > 0$ it is differentiable. The smooth version of max is logsumexp.

### Greedy versus beam search

Taking the best word at every step does not give the best sentence. The target is

$$
y^{\ast} = \arg\max_y \prod_i p(y_i \mid y_{<i}, x)
$$

over $V^T$ candidates, which is intractable. Greedy maximises each factor on its own, but each choice changes what comes next, so a locally worse word can lead to a better sentence.

Minimal example:

| path | step 1 | step 2 | sequence |
| --- | --- | --- | --- |
| A | 0.6 | 0.4 | 0.24 |
| B | 0.4 | 0.9 | 0.36 |

Greedy takes A and ends at 0.24. B was better.

Beam search keeps the top $k$ partial sentences, scored by summed log probability, extends each, and prunes back to $k$. Cost $O(k \cdot V \cdot T)$. Still a heuristic, but it scores whole prefixes instead of single steps. Implementations normalise by length because log probabilities accumulate and penalise longer outputs. The paper uses beam search but **does not state the beam width**.

## 7. Experiments (Section 4)

### Setup

- English to French, WMT '14.
- Baseline: Cho et al.'s RNN Encoder-Decoder, same training procedure, same data.
- 850M words of parallel data reduced to 348M with the Axelrod et al. (2011) data selection method, following Cho et al.
- Dev set: news-test-2012 plus news-test-2013. Test set: news-test-2014, 3003 sentences.
- Tokenised with the Moses tokenisation script.
- 30,000-word shortlist per language; anything else becomes `[UNK]`.
- No lowercasing, no stemming, no monolingual data.
- Training took about 5 days per model.

### Naming

- **RNNencdec** (RNNenc in the figures): baseline, fixed vector. 1000 hidden units in encoder and decoder.
- **RNNsearch**: proposed model. Forward and backward encoder RNNs of 1000 units each, decoder of 1000 units.
- **-30 / -50**: maximum training sentence length.

### Figure 2: BLEU against sentence length

The test of the conjecture. A bottleneck predicts a specific failure: quality dropping as sentences get longer. The paper reports RNNencdec dropping dramatically with length, while RNNsearch-50 shows no deterioration even at 50 words or more. Averages alone could not separate "better model" from "bottleneck removed"; the length curve does.

### Table 1

| Model | All | No UNK |
| --- | --- | --- |
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50\* | 28.45 | 36.15 |
| Moses | 33.30 | 35.63 |

**All** includes every test sentence. **No UNK** keeps only sentences with no unknown words in source or reference, and the models were barred from emitting `[UNK]` in that column. RNNsearch-50\* was trained much longer, until dev performance stopped improving.

**Read the pairs.** Attention gains +7.57 at length 30 and +8.93 at length 50. RNNsearch-30 (21.50) beats RNNencdec-50 (17.82), which the paper itself points to: the attention model trained on short sentences outperforms the baseline trained on long ones.

**The Moses row.** The paper's claim (Section 5.1) is that RNNsearch matches Moses when only sentences of known words are considered, and it notes that Moses additionally used a separate 418M-word monolingual corpus. On the No UNK column, RNNsearch-50\* leads by 0.52 (36.15 against 35.63). On the All column, Moses leads by 4.85 (33.30 against 28.45). The abstract's "comparable" rests on the No UNK column.

**Reading beyond the paper.** Moses gains 2.33 from excluding unknown words; RNNsearch-50\* gains 7.70. That asymmetry suggests the neural model's shortfall on the full set comes mainly from the 30k vocabulary limit rather than the architecture. This is an inference from the table, not a claim the paper makes, though the conclusion does name rare and unknown words as the main open challenge.

### Moses

The open-source phrase-based SMT toolkit released by Koehn et al. in 2007, and the standard baseline for about a decade. It word-aligns a parallel corpus with GIZA++ and the IBM models, extracts a phrase table, trains an n-gram language model, searches over segmentations and reorderings at decode time, and tunes the component weights with MERT. It is the pipeline the abstract contrasts against, and in 2014 it was the bar NMT had to clear.

### What the table cannot tell you

- It does not test the bottleneck conjecture; averages hide the length effect. Figure 2 does that.
- The 50\* row was trained much longer (252 hours against about 110), so its +1.70 is a training-budget effect, and whether the baseline would gain similarly is not reported.
- Single runs, no variance. The 8-point gains are far beyond noise; the 0.5-point comparison with Moses is not.
- Every RNNsearch row differs from RNNencdec in two ways: attention and a bidirectional encoder.

## 8. Qualitative analysis (Section 5.2)

**Alignment (Figure 3).** Alignments are largely monotonic, with strong diagonal weights, plus non-monotonic cases where word order differs. Figure 3(a): "European Economic Area" becomes "zone économique européenne"; the model aligns "zone" with "Area", jumping over two words, then steps back one word at a time. Figure 3(d): the "the man" to "l' homme" case described in Section 1.

**Long sentences.** The paper shows test sentences over 30 words where RNNencdec-50 translates correctly for a while, then drifts from the source meaning (roughly after 30 words in one example), while RNNsearch-50 keeps the full meaning. Appendix C adds more examples alongside Google Translate output from August 2014.

## 9. Where attention is, and where it leads

### Pointing to it

Section 3.1, equations 5 to 7. Equation 7 scores the decoder state against each annotation, equation 6 softmaxes the scores, equation 5 takes the weighted sum. The final paragraph of Section 3.1 names it: the authors describe this as implementing a mechanism of attention in the decoder, which lets the decoder decide which parts of the source to pay attention to and relieves the encoder of compressing everything into one vector. The scoring function's parametrisation is in Appendix A.1.2 and A.2.2.

The paper's primary framing is alignment and soft-search, organised around "$c_i$ now depends on $i$". Attention appears here as a modification to an existing architecture, which is what it was at the time. Later work (Xu et al. 2015 on image captioning, Luong et al. 2015 on MT) made "attention" the standard name.

### What was new

The pieces existed. Graves (2013) had differentiable alignment for handwriting synthesis, using a mixture of Gaussian kernels whose location could only move forward monotonically. The paper contrasts itself with exactly this: translation needs long-distance reordering, so a one-directional alignment is a severe limitation. Bahdanau's alignment is content-based, soft, and unconstrained, and that combination is what "attention" came to mean.

It sits alongside the memory work of the same period (Neural Turing Machines, Memory Networks), all asking how a differentiable model looks something up. Content-based soft addressing was the answer that stuck.

### Cross-attention versus self-attention

In Bahdanau, the target word being produced looks at the source. One direction, across two sequences. This is cross-attention, and it is the only attention in the paper.

The Transformer keeps cross-attention and adds self-attention, where a sequence attends to itself. "Every word looks at every other word" describes self-attention specifically, which does not exist in Bahdanau.

| | how a word gets its context | how output reads input |
| --- | --- | --- |
| Bahdanau | bidirectional RNN, sequential | attention |
| Transformer | attention, parallel | attention |

Bahdanau used attention for retrieval and kept the RNN to build representations. The Transformer used the same mechanism for both, and the RNN had nothing left to do.

### Mapping to the Transformer

| Bahdanau 2014 | Transformer 2017 |
| --- | --- |
| $s_{i-1}$ scored against annotations | query scored against keys |
| $h_j$ is both what is scored and what is averaged | separate key and value projections |
| additive MLP score | scaled dot product |
| one attention distribution | multiple heads |
| decoder attends to encoder | same, plus self-attention |

Equations 5 and 6 survive unchanged. Equation 7 becomes the scaled dot product. Separating keys from values is the one genuinely new conceptual piece.

### What survived and what did not

**Survived:** content-based soft addressing as a trainable primitive; variable-length representations; alignment as a learned by-product.

**Did not:** the additive score; attention as an addition to recurrence rather than a replacement; single-head attention; one vector for both matching and retrieval.

## 10. Claims versus evidence

**Argued for:**

- Per-position annotations instead of a single fixed vector.
- Soft, differentiable alignment instead of hard segmentation.
- Joint training of alignment and translation.

**Held fixed, by convention or compute:**

- GRU rather than LSTM. A.1.1 states LSTM units could be used instead; the choice is not argued.
- Single-layer alignment MLP, chosen explicitly to reduce computation since it runs $T_x \times T_y$ times per sentence.
- Maxout deep output, following Pascanu et al. (2014).
- Vocabulary size, embedding size, beam width.

**Caveats:**

- **Bidirectionality confound.** Two changes at once, so the gain belongs to the pair.
- **Interpretability.** The heatmaps are evidence something meaningful is learned. They are not evidence that the weights explain each prediction. An annotation is a window around its word, not the word itself, so a model can attend one position off and still retrieve the right information.
- **Headline claim.** Parity with Moses holds on the No UNK column only.
- **Internal inconsistency.** Equation 4 and Appendix A.2.2 disagree on whether $s_i$ or $s_{i-1}$ feeds the output.
- **Recency assumption.** The claim that annotations are centred on their word is asserted, not measured.
- **Training budget.** Models matched on wall-clock, not updates; RNNsearch-50 saw the data only 2.2 times.
- **Data subset.** Results are on an Axelrod-selected 348M-word subset, not full WMT.

## 11. Planned ablations

- The 2x2 of attention (on, off) by encoder (unidirectional, bidirectional). Separates the confound.
- BLEU or accuracy against source length. The plot that tests the conjecture.
- Two training length budgets (the -30 and -50 analogues).
- Beam width sweep, since the paper does not state its value.
- Alignment hidden size $n'$.
- GRU against LSTM, to confirm the cell is a control variable.
- Paper-style GRU (reset before the matrix) against PyTorch-style (reset after).
- Adadelta against Adam.

Decide in advance whether runs are matched by updates or by wall-clock time, and report which.

## 12. Implementation notes

- Precompute $U_a h_j$ once per sentence; the annotations never change during decoding.
- Mask padded positions to negative infinity before the attention softmax.
- Run the backward RNN over the unpadded sequence, or it starts from padding.
- The decoder is necessarily unidirectional; only the encoder is bidirectional.
- `torch.optim.Adadelta` takes an `lr` argument that Zeiler's original does not have; `lr=1.0` recovers the original.
- Log attention entropy during training: it starts at its maximum (uniform, from $v_a = 0$) and should fall as the model learns to align.

| tensor | shape |
| --- | --- |
| source embeddings | (batch, src_len, m) |
| annotations | (batch, src_len, 2n) |
| precomputed $U_a h$ | (batch, src_len, n') |
| decoder state | (batch, n) |
| scores, weights | (batch, src_len) |
| context | (batch, 2n) |
| $\tilde{t}$ before maxout | (batch, 2l) |
| $t$ after maxout | (batch, l) |
| logits | (batch, K_y) |
| full attention matrix | (batch, tgt_len, src_len) |

Save the full attention matrix to plot alignment heatmaps.

## References

- Bahdanau, Cho and Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473
- Sutskever, Vinyals and Le (2014). Sequence to Sequence Learning with Neural Networks. arXiv:1409.3215
- Cho et al. (2014a). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078
- Cho et al. (2014b). On the Properties of Neural Machine Translation: Encoder-Decoder Approaches.
- Graves (2013). Generating Sequences With Recurrent Neural Networks. arXiv:1308.0850
- Pascanu, Gulcehre, Cho and Bengio (2014). How to Construct Deep Recurrent Neural Networks.
- Goodfellow et al. (2013). Maxout Networks.
- Zeiler (2012). ADADELTA: An Adaptive Learning Rate Method. arXiv:1212.5701
- Xu et al. (2015). Show, Attend and Tell. arXiv:1502.03044
- Luong, Pham and Manning (2015). Effective Approaches to Attention-based Neural Machine Translation. arXiv:1508.04025
- Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762
- Koehn et al. (2007). Moses: Open Source Toolkit for Statistical Machine Translation. ACL 2007