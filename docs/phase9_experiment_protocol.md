# Transformer: frozen combined-data supplement

Written on 2026-11-19, before Model B training or source-specific model scoring.
The original [Phase 8 protocol](phase8_experiment_protocol.md) and
`configs/experiments/phase8-comparison.json` remain unchanged. This supplement
specifies source accounting and additional validation diagnostics; it does not
replace the original training controls or checkpoint selector.

## Mixture and exact exposure

Train the existing combined `train` stream, ordered by the preparation pipeline's
story hashes and packed with EOS. Use the original seed-42 shuffled window sampler:
uniform permutations of all **8,084** full context-256 windows, without replacement
within an epoch. Begin a new permutation only if the current epoch is exhausted.
The 2,048,000-target-token budget consumes **8,000 windows**, so this run will not
repeat a window. It leaves 84 windows unsampled. Source proportions are induced by
the fixed stream and sampling policy, not by document-level mixing or oversampling.

The source ratio is defined by **predicted target tokens, including EOS**. Assign
EOS to the story that ends there; assign the first character of the next story to
that next story's source. A mixed-source window contributes to both counts. Input
tokens are not added again to the training-token exposure count. Story-boundary
packing and input/target shifting are identical to Model A.

| Quantity | TinyStories | SimpleStories |
| --- | ---: | ---: |
| Prepared training documents | 1,000 | 894 |
| Stored tokens including EOS | 942,640 | 1,126,998 |
| Eligible shifted targets across all full windows | 942,639 | 1,126,865 |
| Eligible target-token mixture | **45.5490301%** | **54.4509699%** |
| Exact seed-42 planned exposure at 8,000 windows | **932,922** | **1,115,078** |
| Exact planned exposure percentage | **45.5528320%** | **54.4471680%** |

The document percentages differ and are not called token percentages. Uniform
random window sampling has expected exposure equal to budget × eligible-token
fraction; the exact precomputed seeded permutation differs slightly from that
expectation. Of all eligible windows, 919 contain targets from both sources; the
planned 8,000-window draw contains 909 such windows. The draw-order SHA-256 is
`e9b1e9471ae36a4d93748a046a56217fdbddb83565044ec0eef38804cfc674f6`.

During execution, reconstruct exposure from the deterministic sampler and verify
its full permutation, generator state, epoch, and cursor against the actual saved
checkpoint. Record cumulative source-token counts at validation checkpoints and
the final actual count. These are not inferred from a nominal document ratio.
Model A's actual exposure is reconstructed by the same method from its retained
checkpoint: it repeats TinyStories windows to reach the same total budget.

## Integrity and matched controls

Recheck prepared-file checksums, normalized story hashes, unique source ownership,
training-view membership, story offsets, EOS, and frozen-token equality. Verify
that no exact cleaned story occurs across train/validation/test, across either
source; no official held-out example may appear in training. Test text is inspected
only for identity/overlap integrity, never scored, fitted, or used to tune settings.
This check cannot rule out near-duplicates or overlap outside the acquired subsets.

No acquisition, cleaning, deduplication, repartitioning, or vocabulary refit occurs.
Both runs use the original dataset fingerprint
`00fa2a2993f66da7d04f6de09944c61dc2c075b012f0bbf02b5a3723a5d1903c`,
tokenizer fingerprint
`9795b7a09fd860a748ed7bd5df3054ed84e6bf5efd3420899cdd65120866d480`,
and token-data fingerprint
`19531b63399d0ca59036c51ec566b803b9d314921a89b56d06471b344c1c9687`.

Model B must start freshly from seed 42. Before any optimizer update, compare its
random-weight fingerprint with Model A's recorded initialization—not its trained
weights. Architecture, float32/MPS, tokenizer, dropout, effective batch eight,
context 256, AdamW, clipping, 100-update warmup/cosine schedule, and maximum
**1,000 updates /2,048,000 target tokens** remain identical. Only training stream
and run ID differ. Run ID: `phase9-model-b-combined-20261119-v1`.

Verify actual Model A configuration/environment and hashes of model, optimizer,
training loop, sampler, runtime RNG handling, aggregate validation, tokenization,
cleaning, and loading code. Refuse a changed primary control pending explicit
comparability review. Observation/orchestration changes are tested against an
uninstrumented CPU run with dropout; they must leave final weights/RNG unchanged.
Original Model A run files and checkpoints must remain untouched.

## Validation and checkpoint selection

**Keep exactly the Phase 8 selector:** full original mixed validation stream,
838 windows /214,528 targets every 100 updates; token-weighted mean NLL, lowest
score wins, earliest exact tie. Its target-source weights are now made explicit:

- TinyStories: **80,659 /214,528 =37.5983555%**.
- SimpleStories: **133,869 /214,528 =62.4016445%**.

These weights were already implicit in Model A's recorded token-weighted score.
They are not 50/50 source weights; the selector emphasizes SimpleStories targets
more heavily. Preserve this rule rather than silently rebalance Model B. There is
no criterion change requiring Model A to be retrained or reselected.

Additionally, evaluate each source **separately**, with the other source's stories
removed, keeping original within-source story order, EOS, context length and shifted
targets. Use read-only virtual views of the existing frozen validation tokens:

| Source-only diagnostic | Stories | Full windows | Scored targets | Omitted tail target pairs |
| --- | ---: | ---: | ---: | ---: |
| TinyStories | 101 | 315 | 80,640 | 18 |
| SimpleStories | 106 | 523 | 133,888 | 86 |

Evaluate both source-only views at updates **100, 500 and 1,000**, plus a distinct
selected best checkpoint if needed. Re-evaluate the same retained Model A
milestones/best retrospectively, without changing its historical selector. No
source-only history is invented for deleted intermediate Model A checkpoints.

The source-only diagnostics have different context windows and tail omissions
from the original mixed selector. **Their weighted average is not the selection
score.** They reveal domain-specific behavior, but neither score is used to tune,
stop, extend, or select this run. Both sources' validation sets remain disjoint
from training. Formal test-set comparison remains Phase 10.

## Diagnostics, timing, and stopping

Reuse the three original prompts and greedy settings, at the same updates:
temperature zero, no top-k/top-p, at most 128 new tokens, EOS stopping, context
cropped to 256. Record checkpoint/hash, prompt, generated IDs/text, parameters,
null sampling seed (greedy), and initialization seed 42. Preserve all outputs,
including repetition. They are not factually reliable.

Use the same planned save/sample/resume boundaries at 100 and 500. Preserve
latest/best and archived milestone checkpoints, full configuration/environment,
source-exposure and validation records, and interruption events. Extra source
diagnostic work restores training RNG and is timed separately. Record core
training and session wall time separately; the added diagnostics mean wall times
are not directly interchangeable with Model A's session times.

The orchestration now handles SIGINT arriving inside the final observer callback
without starting the next planned session. This is an interruption-handling fix,
not an update-math change; Model A had no unexpected interruption. Together with
unchanged compute hashes and the controlled CPU test, no repeat of A is required.
Do not change the fixed budget in response to qualitative or loss results.

Retain the Phase 8 memory headroom checks and bounded run. A single seed and small,
prefix-selected datasets limit inference; do not claim a statistically established
benefit from adding SimpleStories. Stop after Model B and its records. Do not
perform formal held-out/test comparison or frontend work.
