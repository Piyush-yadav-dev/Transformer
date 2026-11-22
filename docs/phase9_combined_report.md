# Phase 9: actual TinyStories plus SimpleStories run

**Completed 2026-11-19; Phase 9 acceptance criteria passed.** Model B completed
1,000 optimizer updates and exactly **2,048,000 target-token predictions** on the
actual MacBook Air M2, 16 GiB, using MPS. The fixed source-exposure plan was met.
No test-set scoring, formal held-out comparison, or frontend work was performed.

The [predeclared supplement](phase9_experiment_protocol.md) preserves the original
Phase 8 protocol and adds accounting and source-only validation diagnostics. Model
A's configuration, checkpoints, logs, samples, and all other run files are unchanged.

## Matched controls and integrity

Both runs used the same shared 78-ID character tokenizer, architecture (4 layers,
4 heads, width/context 256, FF expansion 4, dropout 0.1, tied weights), float32,
3,245,056 trainable parameters, seed 42, microbatch four /accumulation two, AdamW,
gradient clipping, 100-update warmup and cosine schedule, and total training-token
budget. Only run ID and training stream differ in their training configurations.
The actual Model A configuration and environment were checked, not just an example.

Before Model B's first update, its fresh random-weight fingerprint matched Model
A's original initialization exactly:
`2e2ac2b593334c3a2fd367c19db7d2742cf030500398f391df394c28af4b855c`.
Optimizer history was empty. **Model B was not initialized from trained Model A.**

| Shared identity | Fingerprint |
| --- | --- |
| Prepared dataset | `00fa2a2993f66da7d04f6de09944c61dc2c075b012f0bbf02b5a3723a5d1903c` |
| Frozen tokenizer | `9795b7a09fd860a748ed7bd5df3054ed84e6bf5efd3420899cdd65120866d480` |
| Token data | `19531b63399d0ca59036c51ec566b803b9d314921a89b56d06471b344c1c9687` |
| Phase 9 preparation | `b7bc640ab983929427d5423bca9051df971513821b90bc4c9381f0d790b00811` |

Preparation rechecked hashes, cleaning policy, source ownership, exact token
encodings, offsets, EOS boundaries, and source-view membership. It found **zero
exact normalized-story overlaps across train/validation/test**, considering both
sources jointly. No official held-out story entered training. No acquisition,
recleaning, repartitioning, or vocabulary fitting occurred. Test content was used
only for integrity/overlap checks, never model scoring or fitting. Near-duplicates
and overlap outside the acquired subsets remain outside this guarantee.

## Mixture: expected and actual exposure

The sampler draws complete windows uniformly from a seeded permutation of the
existing combined stream. The ratio is measured in **predicted target tokens,
including EOS**, not documents. Mixed-source windows are apportioned by each
target's story ownership; an EOS belongs to the story that ends there.

| Measure | TinyStories | SimpleStories |
| --- | ---: | ---: |
| Prepared training documents | 1,000 | 894 |
| Stored tokens including EOS | 942,640 | 1,126,998 |
| Eligible shifted targets in all 8,084 full windows | 942,639 | 1,126,865 |
| Eligible target-token mixture | 45.5490301% | 54.4509699% |
| Expected exposure over uniform random 8,000-window draws | 932,844.14 | 1,115,155.86 |
| Exact predeclared seed-42 exposure | **932,922** | **1,115,078** |
| Actual exposure verified from checkpoints | **932,922** | **1,115,078** |
| Actual target-token percentage | **45.5528320%** | **54.4471680%** |

All **8,000 sampled window positions were distinct**; 84 eligible windows were
unused, no epoch repetition occurred, and 909 sampled windows contained targets
from both sources. The final sampler was epoch zero /cursor 8,000. Its order hash
matched the frozen plan:
`e9b1e9471ae36a4d93748a046a56217fdbddb83565044ec0eef38804cfc674f6`.

Cumulative exposure was audited at every validation checkpoint against all sampler
state fields. Post-training verification independently intersected target-window
intervals from the actual checkpoint permutation with story-source intervals and
recovered the same totals. Model A's verified exposure was 2,048,000 TinyStories
targets, zero SimpleStories targets, 3,682 unique windows and 4,318 repeated window
draws. **Total exposure is matched; source exposure and repetition are not.**

## Validation and checkpoint selection

The selector was unchanged: full original packed validation, **838 windows
/214,528 targets** every 100 updates; minimum token-weighted NLL, earliest exact
tie. Its implicit target weights are **37.5983555% TinyStories /62.4016445%
SimpleStories**. This is not an equal-source average and places more weight on
SimpleStories validation targets.

| Update | Model B aggregate validation NLL |
| ---: | ---: |
| 100 | 2.454414 |
| 200 | 2.363608 |
| 300 | 2.316642 |
| 400 | 2.284885 |
| 500 | 2.251322 |
| 600 | 2.212474 |
| 700 | 2.158249 |
| 800 | 2.108117 |
| 900 | 2.089503 |
| **1,000** | **2.058485** |

Best and latest both reference update **1,000**. First training-update loss was
4.414072, final training-update loss 2.153969, and the last 100 updates' mean was
2.139819. These training losses use different sampled batches and training dropout.

Separate diagnostics remove the other source's stories, then pack the remaining
validation stories with EOS in original order. TinyStories contributes 101 stories,
315 windows /80,640 scored targets; SimpleStories contributes 106 stories, 523
windows /133,888 targets. Both models were evaluated on exactly these same views.
Model A's preserved milestones were scored retrospectively outside its run directory.

| Checkpoint | A TinyStories | B TinyStories | A SimpleStories | B SimpleStories |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 2.441538 | 2.442919 | 2.500645 | 2.461763 |
| 500 | 2.208201 | 2.236974 | 2.327538 | 2.260162 |
| 1,000 /selected best | **1.958923** | **2.034429** | **2.156060** | **2.072718** |

These are **validation observations from one paired seed**, not formal test results.
In this pair, Model B had lower SimpleStories source-only loss and higher TinyStories
source-only loss. The aggregate was 2.058485 for B versus A's recorded 2.081407,
under the original SimpleStories-heavy validation weighting. This does not establish
a general or statistically significant improvement.

The source-only views have different contexts and tail omissions from the original
mixed stream. **Their weighted average is not the aggregate selector.** They did
not change checkpoint selection, hyperparameters, sampling settings, or the budget.
No intermediate Model A source history was invented where checkpoints were not retained.

## Timing, recovery, and generations

| Measurement | Model B |
| --- | ---: |
| Training-core time, including batching/transfers | **137.104 seconds** |
| Core training throughput | **14,937.60 target tokens/s** |
| Three training sessions, including validation/I/O/auditing | **174.899 seconds** |
| Session throughput | **11,709.64 training targets/s** |
| Timed extra source-audit sections within sessions | **8.592 seconds** |
| Diagnostic generation time between sessions | **6.907 seconds** |
| Orchestration through training/sampling | **182.290 seconds**, about 3 min 2 s |

The last measure excludes initial preparation/preflight, interpreter startup, and
final report/verification work. Extra audit-section timers exclude their checkpoint
loading/output overhead. A's recorded core/session times were 139.060/168.582 seconds;
B's session time additionally includes source diagnostics. Do not interpret the
wall-time difference as an intrinsic cost of the dataset mixture. Equal token
budgets do not imply identical wall time or diagnostic overhead.

Planned stops/resumes at 100 and 500 succeeded. There were **no unexpected
interruptions**, resource pauses, or OOMs. The launch snapshot reported M2/Mac14,2,
16 GiB, macOS 27.0.1 arm64, Python 3.13.7, PyTorch 2.15.0, battery power, MPS with
CPU fallback disabled. Temperatures were not measured. Peak process RSS was
494.38 MiB; maximum MPS driver allocation sampled at validation boundaries was
216.66 MiB. These overlapping counters must not be summed or treated as total
system memory. The free indicator was at least 34% at validation boundaries and
33% at completion, above the 20% pause threshold. Run artifacts occupied about 150 MiB.

All nine fixed greedy continuations are retained, labelled with checkpoint/update,
prompt, hash, temperature zero, disabled top-k/top-p, maximum 128 new tokens, EOS
stopping, null sampling seed, and initialization seed 42. They still mostly repeat
`the`, with occasional fragments such as `she` or `to`. All reached the 128-token
cap. They do not demonstrate coherent generation and are not factually reliable.
No samples were edited, cherry-picked, or used to tune the experiment.

## Changes, verification, and artifacts

New source provenance/accounting lives in `data/loading.py`; isolated
validation views in `evaluation/metrics.py`; paired preparation/training
in `training/experiments.py`. The existing experiment orchestrator was
generalized to select the frozen comparison member and run observational callbacks.
CLI entry point: `train-combined`. The model, optimizer, loader, sampler, schedule,
RNG runtime, and aggregate-selection implementations match A's recorded hashes.

Tests exposed a missing parent-directory creation in new preparation code, fixed
before production. An interruption regression also verifies SIGINT during the final
observer callback cannot silently launch the next planned session. That orchestration
fix changes no update math; A had no unexpected interruption. Controlled CPU tests
with dropout confirmed instrumented/resumed B has exactly the same final weights
and RNG as uninstrumented B. **No comparability-affecting change required repeating A.**

**309 tests passed in 34.88 seconds**, zero failures/skips, including the new
source-overlap, shift/EOS attribution, repetition, source-view, frozen-control,
paired-weight/RNG, archive, and interruption cases, plus all four real MPS tests.
Installation, lint/format, lock and dependency checks passed; no dependencies changed.
Post-training replay reproduced aggregate and both source-only validation losses,
and all final sample token IDs. Best/latest and all milestone checkpoint hashes
passed, initial weights matched A's random initialization, all update counts and
finite metrics were verified, and every original Model A file remained unchanged.

Best/latest checkpoint SHA-256:
`d374977df3c3a367baddc0973d1fbe2fe764af1ce5279b845ea190158af66113`.

- [Trained Model B checkpoint](../runs/phase9-model-b-combined-20261119-v1/checkpoints/update-000001000-419df80348d4463283240daee5a02d04.pt)
  and [checkpoint index](../runs/phase9-model-b-combined-20261119-v1/checkpoints.json).
- [Frozen configuration](../runs/phase9-model-b-combined-20261119-v1/configuration.json),
  [environment/data identity](../runs/phase9-model-b-combined-20261119-v1/run.json),
  [mixture plan](../runs/phase9-model-b-combined-20261119-v1/mixture_plan.json).
- [Training logs](../runs/phase9-model-b-combined-20261119-v1/metrics.jsonl),
  [aggregate validation history](../runs/phase9-model-b-combined-20261119-v1/validation_history.json),
  [source-only history](../runs/phase9-model-b-combined-20261119-v1/source_validation_history.json),
  [actual exposure](../runs/phase9-model-b-combined-20261119-v1/source_exposure.json).
- [All diagnostic text](../runs/phase9-model-b-combined-20261119-v1/diagnostics/samples.md),
  [training summary](../runs/phase9-model-b-combined-20261119-v1/training_summary.json),
  [independent verification](../runs/phase9-model-b-combined-20261119-v1/post_training_verification.json).
- [Read-only baseline diagnostics and preparation](../artifacts/phase9-preparation-v1/manifest.json),
  including `model_a_source_validation.json`, `model_a_exposure.json`, and compatibility evidence.

## Commands executed

```sh
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training/test_combined_experiment.py tests/training/test_baseline_experiment.py --junitxml=artifacts/phase9-experiment-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase9-full-suite.xml
.venv/bin/transformer train --config configs/training/phase9-model-b.json --dry-run > artifacts/phase9-dry-run.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train-combined --protocol configs/experiments/phase8-comparison.json --plan docs/phase9_experiment_protocol.md --output artifacts/phase9-preparation-v1 --prepare-only > artifacts/phase9-preparation-cli.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train-combined --protocol configs/experiments/phase8-comparison.json --plan docs/phase9_experiment_protocol.md --output artifacts/phase9-preparation-v1 > artifacts/phase9-combined-cli.json 2> artifacts/phase9-combined-console.log
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

The focused JUnit file records an intermediate fixture-setup failure; the final
full-suite JUnit records all fixes passing. Local Python checks additionally
audited integrity, froze the actual configuration, hashed A's entire run, independently
counted actual B exposure, replayed validation/generations, and rendered unedited
sample Markdown. No optimizer updates were performed during verification.

This phase stops with a completed combined-data run and its records. A single seed,
bounded upstream prefixes, exact-only deduplication, MPS reproducibility limits,
validation-based selection, and repetitive generations limit conclusions. Formal
held-out/test comparison remains **Phase 10**, not executed here.
