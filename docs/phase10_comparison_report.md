# Transformer: held-out Model A / Model B comparison

**Completed 20 November 2026 (Asia/Kolkata). Phase 10 acceptance criteria passed.**
Both actual validation-selected checkpoints were evaluated on the M2/MPS. Results
are mixed: B predicts SimpleStories better and TinyStories worse in this pair.
Neither model produces coherent stories in the fixed greedy samples. There is no
evidence here for a general improvement in generation quality.

## Protocol and integrity

The [pre-scoring protocol](phase10_evaluation_protocol.md) preserves Phase 8's packed
primary test metric and predeclares a separate document-isolated analysis. The
[configuration](../configs/evaluation/phase10-comparison.json) and
[pre-evaluation checks](../artifacts/phase10-comparison-v1/pre_evaluation.json) record
identities before scoring. Both checkpoints loaded with verified checksums; the
78-ID tokenizer, model architecture, initialization seed 42 and original training
controls match. Both selected update **1,000** using the unchanged lowest full
token-weighted validation NLL, earliest exact tie. Validation selection weights
were 37.60% TinyStories /62.40% SimpleStories. No checkpoint was selected using test.

All original A/B files remain byte-for-byte unchanged. Recorded training histories
and summaries attest that test scoring was deferred; this verifies local records,
not unknown activity outside the project. Joint normalized-content audit found
zero exact train/validation/test overlaps. Near-duplicates are not ruled out.

Test data consists of **99 TinyStories** stories derived from the official
validation subset and **200 SimpleStories** official test stories. Revisions,
cleaning, source ownership and frozen encodings were reverified; no downloads,
repartitioning, vocabulary fitting or training occurred.

## Measured held-out results

Loss is token-weighted cross-entropy in **nats/character-or-special-token**;
perplexity is **exp(mean loss)**. Smaller values mean better next-token prediction
on this metric. EOS is included; UNK would be included, but **zero scored UNK
tokens** occurred. These perplexities are not directly comparable to unrelated
subword-token or word perplexities.

The primary metric uses 1,262 original context-256 windows and **323,072 targets**.
Each target contributes once. Windows may cross stories/sources through EOS;
source-specific scores retain that mixed context. Each window warms up from one
preceding token, with positional IDs reset. The first stream token is context only;
121 tail target pairs are omitted, including the final TinyStories EOS.

| Primary test group | Scored targets | A loss | B loss | A perplexity | B perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyStories | 74,616 | **1.950437** | 2.031649 | **7.0318** | 7.6267 |
| SimpleStories | 248,456 | 2.150583 | **2.070372** | 8.5899 | **7.9278** |
| Target-weighted combined | 323,072 | 2.104358 | **2.061428** | 8.2018 | **7.8572** |
| Secondary equal-source weighting | 50/50 source NLL | 2.050510 | 2.051010 | 7.7719 | 7.7758 |

The primary combined weighting is **23.0958% TinyStories /76.9042% SimpleStories**.
B's loss changes are **+0.081212** on TinyStories and **−0.080211** on SimpleStories.
Its lower combined loss (**−0.042929**) reflects the larger SimpleStories weight.
With 50/50 source weighting, B's primary loss is **0.000501 higher**, a very small
observed difference, not evidence of a statistically established advantage for A.
Combined perplexity is computed from weighted loss, never averaged from perplexities.

The secondary analysis keeps stories isolated: 128-target chunks use overlapping
context up to 256; only new targets are scored, with padding/context-only logits
masked out. All tails and all 299 EOS tokens are included. The first character of
each story is omitted because no BOS token exists. Early targets warm up naturally;
no extra burn-in is discarded. This yields 2,670 windows /**322,895 targets**.

| Isolated-document secondary metric | A loss | B loss | A perplexity | B perplexity |
| --- | ---: | ---: | ---: | ---: |
| TinyStories, 74,638 targets | 1.981389 | 2.050706 | 7.2528 | 7.7734 |
| SimpleStories, 248,257 targets | 2.158262 | 2.082183 | 8.6561 | 8.0220 |
| Target-weighted combined | 2.117377 | 2.074907 | 8.3093 | 7.9638 |
| Equal-source secondary | 2.069825 | 2.066444 | 7.9234 | 7.8967 |

The source tradeoff persists. The tiny equal-source difference changes direction
under this context policy, reinforcing that there is no robust overall winner.
These context policies score different target inventories and must not be substituted
for one another. Raw per-story sums/counts and full-precision results are saved.

## Training exposure and measured cost

| Measurement | Model A | Model B |
| --- | ---: | ---: |
| Trainable parameters, shared weights counted once | 3,245,056 | 3,245,056 |
| Actual training target tokens | 2,048,000 | 2,048,000 |
| TinyStories exposure | 2,048,000 | 932,922 |
| SimpleStories exposure | 0 | 1,115,078 |
| Unique /repeated window draws | 3,682 /4,318 | 8,000 /0 |
| Core training, batching/transfers included | 139.060 s | 137.104 s |
| Training sessions, validation/I/O/audits included | 168.582 s | 174.899 s |
| Generation throughput, pooled | **177.84 tokens/s** | **176.95 tokens/s** |
| Individual generation trial range | 175.34–181.43 tokens/s | 172.59–179.46 tokens/s |

Exposure was replayed and checked against committed sampler state. Equal total
tokens do not equalize source exposure, unique windows, repetition or wall time.
B's sessions include extra source diagnostics (8.592 s in timed audit sections,
plus their loading/output overhead); their wall-time difference is not an intrinsic
mixture cost. Session times exclude diagnostic sampling between sessions.

Generation used actual MacBook Air M2 /16 GiB, macOS 27.0.1 arm64, Python 3.13.7,
PyTorch 2.15.0, float32 MPS, two CPU threads, batch one, no KV cache/compilation,
CPU fallback disabled, on battery power. Each of three prompts warmed up once
per model, followed by three measured repetitions (nine trials /1,152 output
tokens per model), alternating A/B order. MPS was synchronized around timing.
Prompt encoding, transfers, model calls, argmax and decoding are included; model
loading/warmup are excluded. EOS stopping was disabled **only for fixed-work timing**
to guarantee 128 output tokens. The observed ~0.5% pooled difference falls within
overlapping trial ranges; it does not establish a speed advantage. Long-run load
and thermal effects were not measured.

## Qualitative inspection and unscored human rubric

All [six unedited continuations](../artifacts/phase10-comparison-v1/samples.md) and
[token IDs/settings](../artifacts/phase10-comparison-v1/generations.json) are retained.
They use the same original Lily, Tom/box/tree, and bird/friend prompts, greedy
temperature zero, no top-k/top-p, 128-token cap, EOS stopping and context 256.
All six hit the cap without EOS. No human ratings were collected and no external
language-model API was used. The following is descriptive inspection of these
six samples, not an objective automatic quality score or human evaluation study.

| Dimension | Proposed human rubric question (unscored) | Model A observation | Model B observation |
| --- | --- | --- | --- |
| Prompt relevance | Does the continuation develop the prompt's situation? | No coherent Lily, box, or bird development | No coherent development of those situations |
| Local grammar | Are clauses grammatical and readable? | Mostly repeated function words without complete clauses | Same; fragments such as “The she the” |
| Story continuity | Do events connect and progress? | No discernible event progression | No discernible event progression |
| Repetition | Does avoidable repetition dominate? | Dominated by “the”, occasionally “to” | Dominated by the same loops |
| Ending | Does it reach a natural conclusion? | All three capped; some end mid-word | All three capped; no natural conclusion |
| Entity consistency | Are introduced names/pronouns tracked consistently? | Named entities are not sustained; “They” lacks development | Named entities are not sustained; Tom prompt introduces “she” without explanation |

A future human study could apply this rubric with blinded model labels and multiple
raters; no numerical ratings are invented here. Cap-triggered truncation alone does
not prove a model would never produce EOS with a larger budget.

## Figures and reproducibility

Six visually checked plots are exported as PNG and SVG with units, full run IDs and
legends. Training curves show every raw update (no smoothing); their losses concern
different training streams, so a lower training curve is not a common-test advantage.
Validation curves show the original selection metric, not test-tuned checkpoints.

- [Training loss](../artifacts/phase10-comparison-v1/charts/training_loss.png)
  · [SVG](../artifacts/phase10-comparison-v1/charts/training_loss.svg)
- [Validation loss](../artifacts/phase10-comparison-v1/charts/validation_loss.png)
  · [SVG](../artifacts/phase10-comparison-v1/charts/validation_loss.svg)
- [Held-out loss](../artifacts/phase10-comparison-v1/charts/held_out_loss.png)
  · [SVG](../artifacts/phase10-comparison-v1/charts/held_out_loss.svg)
- [Held-out perplexity](../artifacts/phase10-comparison-v1/charts/held_out_perplexity.png)
  · [SVG](../artifacts/phase10-comparison-v1/charts/held_out_perplexity.svg)
- [Training duration](../artifacts/phase10-comparison-v1/charts/training_duration.png)
  · [SVG](../artifacts/phase10-comparison-v1/charts/training_duration.svg)
- [Generation throughput](../artifacts/phase10-comparison-v1/charts/generation_throughput.png)
  · [SVG](../artifacts/phase10-comparison-v1/charts/generation_throughput.svg)

**329 tests passed in 32.97 s**, no failures/skips, including five real MPS cases.
Tests cover exactly-once targets, independent single-target NLL, original packed
evaluator agreement, padding/batch invariance, UNK/EOS/short/empty inputs, logarithm
convention, selection integrity, immutable runs and output protection. An independent
post-run replay with the existing loader/evaluator reproduced both primary losses
within 1e-6; totals/perplexities/checksums and archived generation IDs were verified.

Executed evaluation command:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer evaluate-comparison --config configs/evaluation/phase10-comparison.json
```

See [README](../README.md#held-out-evaluation-and-comparison) for installation,
tests and reproduction into a fresh output directory. Matplotlib 3.11.2 is an optional
locked evaluation dependency; the existing ML runtime pins did not change.

Raw records: [results and copied training logs](../artifacts/phase10-comparison-v1/results.json),
[timing trials](../artifacts/phase10-comparison-v1/generation_benchmark.json),
[artifact checksums](../artifacts/phase10-comparison-v1/manifest.json),
[prelaunch environment/tests](../artifacts/phase10-prerequisite-verification.json),
[post-evaluation verification](../artifacts/phase10-post-evaluation-verification.json),
and [test XML](../artifacts/phase10-full-suite.xml).

## Interpretation and limits

**Measured:** B's SimpleStories test loss is lower and TinyStories loss higher;
the SimpleStories-heavy combined score favors B. Equal weighting offers almost no
separation. Both models fail the fixed greedy story-generation checks. Decode
throughputs are similar in this bounded measurement.

**Plausible explanation, not proven:** B reallocates over half its training exposure
to SimpleStories and sees fewer TinyStories targets; this may explain the domain
tradeoff. It also sees more unique windows and fewer repetitions, so this experiment
does not isolate domain composition from diversity/repetition effects.

**Uncertainty:** one initialization/training seed supplies no between-run variance or
statistical significance. Prefix-selected subsets are not representative of entire
datasets; exact deduplication misses near-duplicates. Validation-source weighting,
context/target policy, limited training, greedy-only prompts, and MPS variability
limit conclusions. Test results are now observed and must not be reused as an unseen
selection set for later tuning. No broad improvement is claimed. Stop after Phase 10;
no frontend work or further training has begun.
