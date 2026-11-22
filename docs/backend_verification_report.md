# Transformer Phase 12: backend release gate

**PASS — completed 20 November 2026 (Asia/Kolkata).** Both the required backend
implementation and actual trained Model A/Model B artifacts are available and
validated. There are no outstanding blockers for this local backend gate.
No frontend work or additional real-model training was started.

The machine is the actual target **MacBook Air M2, Mac14,2, 16 GiB unified memory**,
macOS 27.0.1 arm64, Python 3.13.7 and PyTorch 2.15.0. MPS build/runtime support is
available. Synthetic training and exact recovery use **CPU**; real checkpoint
evaluation and generation use **MPS with CPU fallback disabled**.

## Evidence and checks

The machine-readable [gate record](../artifacts/phase12-backend-gate-v1/verification.json)
contains source hashes, hardware, checkpoint identities, per-stage duration,
failure messages, replayed scores and all generated sample text/IDs.
The [console record](../artifacts/phase12-gate-console.log) and
[full-suite JUnit](../artifacts/phase12-full-suite.xml) retain executed results.

| Check executed | Status | Device / actual evidence |
| --- | --- | --- |
| Complete test suite | PASS | **403 passed in 38.63 s; 0 failed, 0 skipped**; includes six real MPS tests |
| Synthetic preparation → tokenization → train/validate/save/resume → evaluation → service/generation | PASS | CPU; **2.116 s** integrated stage |
| Resumed versus uninterrupted training | PASS | CPU; all final model tensors, scheduler and CPU RNG match exactly |
| Eight required failure cases | PASS | CPU; expected exceptions/messages, **0.037 s** stage |
| Separate real-artifact verification | PASS | MPS; **30.058 s**, both registered selected checkpoints |
| Frozen metadata, train budgets, validation selection, partition integrity | PASS | Original configs/logs/checkpoints and prepared/token manifests audited |
| Replayed packed and document-isolated held-out scores | PASS | Both models; all source and combined NLL values match saved results exactly; tolerance 1e-6 |
| Fixed generation | PASS | Three prompts × two models × 128 new tokens; all six ID sequences match saved samples |
| Original run/evaluation preservation | PASS | **83 files unchanged**, including all **19** manifest-listed evaluation outputs |
| Live upstream acquisition/download | SKIP | Intentionally not repeated; local fixture transport, pinned cached real artifacts and acquisition unit tests used |

The live-download skip is outside this offline gate; it is not a passed network
availability check. MPS execution was available and actually tested, not simulated
for the positive device checks. Simulated unavailable-device policy tests remain
separate. No fixture checkpoint is counted as a trained research model.

## Review coverage

| Area inspected | Evidence / conclusion |
| --- | --- |
| Acquisition and cleaning | Pinned source/config/revision/license metadata, bounded record acquisition/cache, conservative NFC cleaning, rejection reasons and atomic manifests; offline acquisition tests |
| Split integrity | Global normalized-content hashes, official held-out priority, deterministic document splits, and real prepared/token provenance audit; no exact story overlap across acquired splits |
| Tokenization/loading | Vocabulary from training union only, frozen fingerprint, explicit UNK/EOS, compact dtype and memmap checks, shifted targets, split-contained windows and short-data handling |
| Architecture | Causality, independent scaled/masked attention reference, custom normalization versus PyTorch, loss alignment, numerical gradients, shapes, dropout, serialization and CPU/MPS checks |
| Training/validation | Actual token-weighted accumulation, clipping, fixed warmup/decay budget, inference-mode validation, finite checks and source-exposure accounting |
| Recovery/logging | Atomic latest/best references, full optimizer/scheduler/RNG/sampler state, safe update boundaries, structured metrics recovery, rejection of incompatible state; controlled CPU resume reproduces uninterrupted execution |
| Evaluation | Validation-only checkpoint selection reconstructed from logs; packed targets and document-overlap masking tested; NLL/perplexity replay from both real checkpoints; saved charts/results checksummed |
| Inference | Trusted ID allowlist, artifact and tokenizer/vocabulary validation, frozen/eval model, greedy/seeded sampling, rolling learned positions, EOS/UNK, limits, one-model cache and no gradients |
| Configuration/errors | Validated immutable settings, portable paths, serialized snapshots; malformed data/state/device/generation requests fail meaningfully |
| Documentation/dependencies | Current statuses and commands corrected; locked package installation/import, CLI, lint, formatting, dependency and wheel checks; no frontend dependency introduced |

Exact deduplication does not prove absence of near-duplicates or overlap in unseen
upstream data. Existing focused tests cover cross-source duplicates; the integrated
fixture specifically exercises duplicates within a source and across train/held-out
partitions. The real audit checks 1,894 train, 207 validation and 299 test documents.

## Isolated synthetic workflow

The [fixture project](../artifacts/phase12-backend-gate-v1/fixture_project/fixture_verification.json)
is entirely inside the gate output. Its provider supplies invented local records;
it does not call the Hub or impersonate downloaded dataset evidence. All subsequent
preparation, tokenization, training, evaluation and inference use the existing
backend APIs.

Of **26 raw records**, two whitespace-only and two non-string records are rejected;
four duplicate occurrences are removed. The final partitions contain **10 train,
4 validation and 4 test stories**, with no cross-split exact overlap. A held-out
duplicate's training copy is removed. Preparation/tokenization reruns reuse checked
artifacts. A test-only `Ω` is absent from the **23-ID** training vocabulary.

Two tiny fixture models use one layer, two heads, width/context 16, dropout 0.1,
microbatch two and accumulation two. Each trains for **four optimizer updates /
256 target tokens**, stopping/saving at update two and resuming to four. A third
uninterrupted CPU reference verifies exact final-state recovery. Full fixture
validation selects checkpoints before held-out evaluation. The isolated registry
then loads each fixture checkpoint and produces a continuation capped at eight IDs.
These fixtures establish integration, not model quality or real-model availability.

The rejection stage checks missing token data, zero microbatch size, tokenizer
fingerprint mismatch, unsupported `cuda`, an existing run directory, zero sampling
temperature, corrupted checkpoint bytes, and a checksum-valid checkpoint missing
optimizer state. Corruption is confined to disposable copies of synthetic runs.
The complete errors are in the gate record. Tests also verify that missing real
artifacts force `incomplete`, and that existing/outside/symlink output paths are refused.

## Actual trained-model availability and result provenance

Both real models have **3,245,056 parameters**, the same **78-ID** tokenizer, matching
four-layer/four-head/width-256/context-256 architecture and seed 42. Each completed
**1,000 updates /2,048,000 target tokens**. The selected update is 1,000 for both;
the rule remains lowest full-validation token-weighted NLL, earliest exact tie.
Model A's exposure is exclusively TinyStories; Model B has **932,922 TinyStories /
1,115,078 SimpleStories** targets. Sampler state and logs support these counts.

| Model / stable service ID | Selected checkpoint SHA-256 | Packed NLL (nats/target) | Packed perplexity |
| --- | --- | ---: | ---: |
| A / `transformer-tinystories-v1` | `b39c15d30b3abf420dca0f94a104e5d22044a23fe1debd3bf3223ef2f78474ad` | 2.1043575029318404 | 8.20183166356758 |
| B / `transformer-combined-v1` | `d374977df3c3a367baddc0973d1fbe2fe764af1ce5279b845ea190158af66113` | 2.0614283108758396 | 7.857184301313232 |

Packed scoring uses **323,072 targets** per model; document-isolated scoring uses
**322,895**. Replayed document NLL is **2.117376990956518 A /2.074907105853958 B**.
Every replayed combined and source-specific NLL has zero observed difference from
the saved Phase 10 values on this machine. Perplexity uses exp(mean NLL in nats)
over character-token targets including scored EOS/UNK; it is not directly comparable
with unrelated subword perplexities. Source weights and special-target counts are
in the original [evaluation protocol](phase10_evaluation_protocol.md) and raw records.

Both actual models generate all fixed prompts through the public service without
gradients. Continuations remain repetitive. This gate verifies artifact provenance
and execution; it does not strengthen the mixed, single-seed quality conclusions of
the [held-out comparison](phase10_comparison_report.md).

## Defect fixed and cleanup

An extreme finite loss caused `math.exp` to overflow in the **equal-source secondary
perplexity** despite the primary metric already handling this condition. A new
regression first failed with `OverflowError` ([before-fix evidence](../artifacts/phase12-overflow-before.xml)).
The secondary summary now reuses the existing overflow-safe conversion, returning
JSON `null` perplexity with `perplexity_overflow: true`. The regression and full
suite pass; normal real-data metrics remain unchanged. Model/training mathematics,
weights, selected checkpoints and saved Phase 10 results were not modified.

Stale unqualified training-status text, the obsolete unimplemented-directory claim,
and misleading future/archived launch wording were corrected. Generated `build/`
source copies were removed after packaging verification: **64 files /323,999 bytes**,
each verified byte-identical to its authoritative `src/` file before deletion. Functional historical
sampling helpers remain: frozen experiment diagnostics and controlled evaluation
timing still call them, so merging them would change reproducibility or measurement
semantics. No dead executable placeholder path was found. All declared dependencies
have active responsibilities; no dependency removal or new dependency was needed.
Valid checkpoints and useful experiment records are preserved.

## Reproduce and interpret the gate

Run from the project root, in the locked isolated environment. The commands actually
executed for this phase include:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m pytest tests/verification tests/training/test_held_out_evaluation.py::test_equal_source_perplexity_overflow_is_serializable --junitxml=artifacts/phase12-focused.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase12-full-suite.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/maintenance.py verify-backend --device mps --test-report artifacts/phase12-full-suite.xml --output artifacts/phase12-backend-gate-v1 > artifacts/phase12-gate-console.log 2>&1
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m build --wheel --outdir artifacts/phase12-wheel
```

For a repeat, use a new output such as `artifacts/phase12-backend-gate-v2`. The gate
requires the evaluation extra and recent full-suite JUnit evidence. It records
installed source hashes; test evidence is supplied by the trusted local operator,
not a cryptographic assertion that no external activity occurred. Exit zero means
all required stages passed. Failures persist an `incomplete` report; dependent
checks are skipped rather than marked passed. Interrupted execution retains the
last incomplete report. Use ordinary Python, without `-O`.

The installed `transformer diagnose --device mps --check` command passed its real
tensor check; output is retained in `artifacts/phase12-diagnostics.txt`.
The focused run passed **11 tests in 3.73 s**. Early gate development also caught
a relative fixture-root handling error; paths now resolve before artifact-relative
paths are formed. The failed development fixture remains labelled under
`artifacts/phase12-development-fixture-v1`, separate from the passing release record.

Remaining limits: only this locked macOS CPU/MPS environment was verified; exact
resume across different releases/hardware is not promised. Historical Phase 8
continuation requires its original recorded training implementation because strict
code-identity checks refuse later orchestration code; completed weights remain
fully loadable. Recovery guarantees stop at completed atomic checkpoint boundaries.
The service assumes a trusted local root/catalog and one cached model per service
instance. No long-running server/load/security review or live Hub availability test
was performed. One seed, small subsets, exact-only deduplication and repetitive
output limit research conclusions. None is an outstanding blocker for this scoped
local backend gate. Stop here; frontend development remains a separate phase.
