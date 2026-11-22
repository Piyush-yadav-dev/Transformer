# Transformer: held-out comparison protocol

Frozen before any Phase 10 model test scoring. Evaluate the existing validation-selected
best checkpoints from `configs/experiments/phase8-comparison.json`; never reselect,
tune, extend training, or refit vocabulary using these results. Both runs completed
2,048,000 targets at seed 42 with matching architecture/tokenizer/training controls.
The selector is lowest full packed-validation token-weighted NLL, earliest exact tie,
with implicit TinyStories/SimpleStories weights 37.5984%/62.4016%. Both selected update
1,000. Preserve every original run file and record its checksum before/after evaluation.

## Preflight and held-out identity

Load checkpoints with checksum verification; compare actual configuration, tokenizer,
model code, random initialization identity, budgets, validation history, and the original
protocol hashes. Audit existing prepared/token files, story offsets, text encodings,
source ownership and zero exact normalized-story overlaps across train/validation/test.
Record pinned source revisions from the run. No new acquisition, partitioning or fitting.
TinyStories test contains 99 stories derived from the official validation subset;
SimpleStories test contains 200 official test stories. Previous run records explicitly
state no test scoring, and the recorded selection history uses only validation. This is
auditable local evidence, not proof against unrecorded activity outside the project.

## Primary: original packed test metric

Honor Phase 8's specified test stream: 1,262 nonoverlapping input windows of 256 tokens,
323,072 scored shifted targets. Window k inputs positions kC through (k+1)C-1 and scores
targets kC+1 through (k+1)C. Each eligible target is counted once. The first stream token
is context only; the final 121 target pairs are omitted. No overlapping target weighting.
Position embeddings reset at every window; context length grows from one to 256 within
each window, with no additional warmup exclusion. Windows can cross story/source
boundaries separated by EOS, exactly as training did. First characters after EOS can be
scored; EOS belongs to the ending story. Attribute every target to its own source,
including mixed-source windows. Separate source metrics therefore retain mixed context.

Report total negative log likelihood, target count and their ratio in **nats/token**.
Perplexity is **exp(mean NLL)**, not exp(sum NLL), not base-10 exponentiation, and not
the arithmetic average of batch or source perplexities. The primary combined loss is
weighted by actual scored source target counts; print these weights. Also report a
predeclared secondary 50/50 source mean NLL and exp of that mean, clearly distinguished.

## Secondary: isolated-document overlapping-context metric

Read the same frozen test token mmap and per-story offsets. Never concatenate stories
for this analysis. For each document of length L including EOS, target chunks start
at s=1,1+128,...; e=min(s+128,L). Supply inputs [max(0,e-1-C),e-1), and score only
targets [s,e) at their correctly aligned logit positions. This includes all targets
except each story's first character (no BOS token exists), including all EOS and short
tails. Input contexts overlap; scored targets never overlap. Early targets warm up
from one preceding character; later full blocks have 129–256 preceding tokens. No
extra burn-in targets are discarded. Position IDs restart with each supplied window.
For a short final block, more previous context can be included. This is block-strided
context, not a claim of maximum-context single-target likelihood at every position.

Batch up to four windows, right-pad with EOS only for computation, and mask all padding
and context-only positions out of the metric. Causal attention prevents future padding
from changing scored logits. Empty test data is an error; documents with no target
pairs contribute no loss and are counted explicitly. Save per-story sums/counts and
per-source summaries. Both models must score identical target inventories.

This secondary window policy differs from training/primary validation, so its numbers
must not be substituted for the original primary result after seeing outcomes.

## Token interpretation

The frozen 78-ID tokenizer has one ID per known Unicode character, plus UNK=0 and
EOS=1. Existing unseen test characters map individually to UNK; report their scored
counts and include them in NLL. EOS is scored; no special token is silently removed.
Report character/special-token perplexity, not word perplexity. It cannot be compared
directly to unrelated subword-token perplexities. Decode UNK/EOS as visible labels.

## Generation and timing

Use the three original handwritten prompts unchanged, same selected checkpoints,
eval/inference mode, greedy argmax (temperature 0, top-k/top-p disabled, seed null),
128-token cap, EOS stopping and last-256 context cropping. Save every continuation,
including failures, with IDs/checkpoint hash/settings. No external model API or invented
human ratings. Provide a **proposed human rubric**, unscored, for prompt relevance,
local grammar, continuity, repetition, endings and entity consistency. Any descriptive
inspection is labelled as such and supported by displayed text.

Measure generation on the actual M2/MPS with float32, batch one, two CPU threads,
no KV cache/compilation. Warm up each model/prompt once. Then run three repetitions
per prompt/model, alternating A/B order by prompt/repetition. Use a fixed 128-token
decode workload with EOS stopping disabled **only for timing**, ensuring equal output
counts. The benchmark's post-EOS output is not a qualitative sample. Synchronize MPS
before/after timing. Include prompt encoding, context creation/transfers, model calls,
argmax/host synchronization and final decoding; exclude model loading and warmup.
Save every elapsed time and actual output count, pooled tokens/s and repeat range.
These are bounded autoregressive generation rates, not training/prefill-only rates;
different generated inputs and laptop load can affect timing. No temperature inference.

## Reporting and interpretation

Save configuration/protocol hashes, environment, raw per-source/per-document metrics,
generations, raw timing repetitions, source exposure and training timing from actual
logs. Plot all raw training losses versus processed targets, recorded validation
loss versus targets, primary held-out source NLL/perplexity, core/session training
seconds, and generation tokens/s with units, legends and full run IDs. Preserve raw
curves; any smoothing must be explicitly labelled. Export PNG and SVG using Matplotlib.

Separate measured findings, plausible mechanisms and uncertainty. One paired seed,
bounded prefix subsets, exact-only deduplication, different source exposure/repetition,
unequal validation weighting and MPS numerical variability limit conclusions. No
statistical significance or broad quality improvement is established by this pair.
No test-driven model selection, training or frontend work follows in this phase.
