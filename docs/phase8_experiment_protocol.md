# Transformer: frozen primary comparison protocol

Protocol version 1, written before launching Model A on 2026-11-19. The executable
controls are `configs/experiments/phase8-comparison.json`. This document and that
configuration are copied and hashed in the run before its first optimizer update.
Model B is specified here for a later phase; it must not be trained in Phase 8.

## Question and experimental controls

With a fixed architecture, shared character vocabulary, initialization, and target
token budget, how does training on TinyStories plus SimpleStories change held-out
next-token loss and qualitative continuations relative to TinyStories alone?
Adding a source is not presumed to improve quality. This is a single-seed,
small-subset comparison, not a broad benchmark or causal claim about all data.

| Control | Both Model A and later Model B |
| --- | --- |
| Architecture | Custom pre-normalization decoder; 4 layers, 4 heads, embedding/context 256, FF expansion 4, GELU, dropout 0.1, custom layer norm epsilon 1e-5 |
| Parameters / precision | 3,245,056 trainable parameters; tied token/output weights; float32 |
| Initialization | Fresh random weights, seed **42**; Normal(0, 0.02) embeddings/linear weights, zero biases, unit norm scale; tied weights initialized once |
| Tokenizer | Frozen Phase 3 shared training-only vocabulary, 78 IDs; no refitting and no validation/test vocabulary discovery |
| Maximum budget | **2,048,000 target-token predictions**, exactly 8,000 windows / **1,000 optimizer updates** |
| Batch / context | Microbatch 4 × accumulation 2 = eight windows/update; 256-token context; 2,048 targets/update |
| Optimizer | AdamW, beta (0.9, 0.999), epsilon 1e-8; weight decay 0.01 for matrix parameters, zero for other parameters; norm clipping 1.0 after accumulation |
| Learning rate | Peak 0.0003; 100-update linear warmup, then existing cosine schedule to 0.00003; LR set before each update, total schedule fixed at 1,000 |
| Sampling of training windows | Seeded shuffled nonoverlapping windows, without replacement per epoch; repeat epochs only when needed; zero loader workers, two CPU threads |
| Execution | MPS on the measured M2; CPU fallback disabled; deterministic-algorithms flag false; seed recorded, no cross-device/bitwise MPS promise |
| Validation / checkpoint | Every 100 updates and final update; validation in inference mode, loss weighted by target tokens |

Model A uses `train_tinystories` only: 1,000 prepared stories, 942,640 stored tokens,
3,682 context-256 windows. Its budget revisits that window set about 2.17 times.
Model B uses the combined `train` stream: 1,894 prepared stories, 2,069,638 stored
tokens, 8,084 windows. Its matched budget covers about 0.99 passes. Window sampling
uses the mixture's natural token proportions (approximately 45.5% TinyStories,
54.5% SimpleStories), not a forced equal-source mixture. Thus equal total token
exposure does not equalize per-source exposure or number of unique stories.

Both must start independently; **Model B must never initialize from Model A**.
The only training configuration differences are run ID and training stream. The
initialization weight fingerprint will be recorded for comparison. EOS-packed
windows may cross story boundaries within a stream; neither experiment crosses a
training/validation/test split boundary. The short tail that cannot form a full
window is omitted identically by the existing loader policy.

## Frozen data and checkpoint selection

Use the existing pinned Phase 2 subsets and Phase 3 token artifacts, with no new
download. TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` and official
English SimpleStories revision `e63b8adc3b1a1bdc7cac5b500d150b71346b0628` are documented
in [source verification](dataset_sources.md). Prefix acquisition has ordering bias.
Exact deduplication does not remove all near-duplicates.

- Tokenizer fingerprint: `9795b7a09fd860a748ed7bd5df3054ed84e6bf5efd3420899cdd65120866d480`.
- Prepared dataset fingerprint: `00fa2a2993f66da7d04f6de09944c61dc2c075b012f0bbf02b5a3723a5d1903c`.
- Token-data fingerprint: `19531b63399d0ca59036c51ec566b803b9d314921a89b56d06471b344c1c9687`.

Checkpoint selection uses **all 838 full windows of the shared validation stream**:
214,528 target tokens from 207 held-out stories, 210 batches at microbatch four.
The last batch contains two windows. There are 105 unused tail target pairs.
Select the **lowest token-weighted mean cross-entropy in nats**, keeping the earlier
checkpoint on an exact tie. No patience-based stopping, hyperparameter changes, or
extension beyond the maximum budget is allowed after observing losses or samples.
Preserve latest and best checkpoints separately through their atomic index.

The final evaluation dataset, reserved for a later phase after selection, is the
common combined `test` stream: 299 held-out stories, 1,262 full windows / 323,072
target tokens, 121 omitted tail target pairs. Primary metrics will be token-weighted
NLL and its exponential (character/special-token perplexity, **not word
perplexity**). Source-specific held-out summaries may be secondary analyses using
recorded provenance, never grounds for choosing checkpoints. These splits derive
from official held-out sources; their original designation and deterministic
subdivision are retained in manifests. **Do not evaluate test loss in Phase 8**.

## Fixed qualitative checks

At updates **100, 500, and 1,000**, archive the full checkpoint and generate one
continuation for each hand-written prompt, without cherry-picking:

1. `Once upon a time, there was a little girl named Lily.`
2. `Tom found a small box under a tree.`
3. `The little bird wanted to help its friend.`

Use evaluation/inference mode, **greedy argmax**, temperature recorded as zero,
top-k/top-p disabled, at most **128 new character/special tokens**, EOS stopping,
and crop to the latest 256 context tokens. Greedy decoding uses no sampling seed;
record it as null with that explanation, plus initialization seed 42. Every sample
records its prompt, generated IDs/text, update, checkpoint SHA-256, and settings.
Also sample the validation-selected best checkpoint if it is a different update.
Apply the same prompt/settings/update policy to Model B later.

Planned stops at 100 and 500 permit checkpoint archival and diagnostics; resume
the same run and original schedule afterward. Record those as planned pauses and
recovery sessions, not failures. Sampling occurs between sessions and cannot alter
saved optimizer/sampler/RNG state. Unexpected interruption is logged and requires
explicit resumption with the frozen protocol. Generated text is diagnostic,
may repeat or be nonsensical, and **is not factually reliable**.

## Budget, hardware, and records

Run ID: `phase8-model-a-tinystories-20261119-v1`. Later Model B reserves
`phase9-model-b-combined-20261119-v1`, currently unexecuted. Existing directories
must never be overwritten. Keep full frozen configuration/environment, source
identities, raw training/validation logs, recovery events, and checkpoint hashes.

Phase 7 measured MPS microbatch four / accumulation two at approximately 14,950
compute-only and 11,932 pipeline target tokens/s. Its short validation prefix is
replaced here by full validation. Linear projection suggests roughly three minutes
of training/validation, plus setup, recovery, and diagnostic overhead; this is not
a duration guarantee. Record core training, session wall time, and diagnostic time
separately. Equal token exposure does not guarantee equal wall-clock cost.

At most four diagnostic checkpoint copies (three milestones plus a distinct best)
and up to three active/publication copies suggest under 300 MiB of checkpoint
storage, plus logs. Free memory before launch must remain ≥20% on the macOS tool's
indicator. Observe memory at validation boundaries and request a safe pause on
process RSS or sampled MPS driver allocation above 4 GiB, or free indicator below
20%. These counters overlap and are not total system memory. No temperature claim
will be made without direct measurement.

Phase 8 ends after the actual Model A artifacts and summary are verified. It does
not train Model B, tune on test results, or implement a frontend.
