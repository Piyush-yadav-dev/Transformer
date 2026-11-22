# Phase 7: training diagnostics and hardware report

Executed on **2026-11-19**. Phase 7 acceptance criteria passed. No production
baseline, full-corpus training, new tokenizer, or frontend was started.

## Environment and reproducibility

This execution was on the target **MacBook Air M2, Mac14,2, 16 GiB unified memory**,
arm64, macOS 27.0.1, Python 3.13.7, PyTorch 2.15.0. MPS build/runtime support was
available and real MPS operations succeeded with CPU fallback disabled. These are
actual M2 measurements; M2 verification is not pending.

Initial inspection reported AC power, but **every benchmark worker recorded
battery power**. The results therefore reflect the conditions recorded during
measurement, not an assumption of AC operation. Power mode and temperatures were
not measured. Software versions remain governed by the unchanged `uv.lock`.

All runs used local prepared artifacts with these exact identities:

| Artifact | SHA-256 fingerprint |
| --- | --- |
| Shared tokenizer | `9795b7a09fd860a748ed7bd5df3054ed84e6bf5efd3420899cdd65120866d480` |
| Prepared dataset | `00fa2a2993f66da7d04f6de09944c61dc2c075b012f0bbf02b5a3723a5d1903c` |
| Token data | `19531b63399d0ca59036c51ec566b803b9d314921a89b56d06471b344c1c9687` |

Corpus training views were `train_tinystories`; validation used the common
validation stream. Model A retains the shared training-only character vocabulary.
No held-out test text was used for learning, validation, or hardware decisions.

## Three independent outcomes

| Check | Executed work | Result and scope |
| --- | --- | --- |
| Pipeline smoke | Small model, real prepared tokens, MPS, six updates; stop/load at three, resume to six, validate, save/reload, greedy diagnostic | Passed; final validation loss 4.234095 over 128 tokens. Exercises execution and recovery only. |
| Tiny-data overfit | One fixed synthetic 32-target-token window, CPU, dropout zero, ≤150 updates / 30 seconds | Passed in 21 updates; loss **4.383251 → 0.088682**, ratio 0.020232. Demonstrates learning this one example. |
| Hardware benchmark | Six fresh-process cases plus one bounded sustained case, fixed primary architecture | All completed without OOM/resource-limit stops. Measures this pipeline's throughput and memory. |

The overfit success threshold was fixed before execution: final loss ≤0.1 and
≤10% of initial loss. The update/evaluation loop took 0.068 seconds, excluding
setup. No threshold was loosened and no unsuccessful actual overfit run was hidden.
An offline unit test deliberately uses an unreachable threshold and verifies that
failure is recorded, exits through the failure status, and preserves artifacts.

The smoke prompt `Once upon a time` produced 48 spaces. The unmodified continuation
is saved with **DIAGNOSTIC ONLY** labelling. It confirms bounded sampling executes;
it is plainly not a useful story or evidence of language quality. Neither tiny
overfit nor hardware timing changes that interpretation.

## Fixed-architecture batch measurements

All cases: four layers, four heads, embedding/context 256, feed-forward expansion
four, dropout 0.1, float32, tied weights, **3,245,056 trainable parameters**. Effective
batch is always eight windows / **2,048 target tokens per update**. Four warmup
updates precede eight timed updates in each matrix case. Each case has a 45-second
boundary-checked budget and ran in a fresh process. No architecture search occurred.

| Microbatch | Accumulation | Compute-only target tokens/s | Pipeline target tokens/s | Largest worker lifetime peak RSS, MiB | Maximum sampled MPS driver allocation, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 11,824 | 9,302 | 453.36 | 88.67 |
| 2 | 4 | 13,442 | 10,399 | 455.53 | 152.67 |
| **4** | **2** | **14,950** | **11,932** | **454.56** | **216.67** |

The RSS column takes the larger OS lifetime peak across that batch's compute and
pipeline workers. Compute-only timing uses resident cached inputs and the actual
training optimizer code, including finite checks/clipping/AdamW. Pipeline timing
also includes mmap sampling, batching/transfers, structured training logs,
validation on 16 windows, and atomic checkpoints every four updates. It excludes
cold setup and benchmark observer I/O. MPS synchronization brackets optimizer
timing. See the [guide](benchmarking.md) for exact timing and memory definitions.

These are single short measurements, not statistical confidence intervals.
Microbatch four is selected by the predeclared rule: smallest batch within 90% of
the best measured pipeline rate. Batch two achieved 87.15% and did not meet that
threshold. Larger batches were not tried; no maximum-capacity claim is made.

## Bounded sustained check

Microbatch four / accumulation two completed **four warmups +120 measured
updates**, within the preset 60-second cap. Measured work: 245,760 target tokens
over 20.641 seconds = **11,906 tokens/s**. Case wall time including setup and
observer overhead was 23.288 seconds; full subprocess wall time including Python
startup and metadata collection was 24.510 seconds. All seven hardware subprocesses
together took 50.460 seconds and performed 196 optimizer updates in total.

For a fair short comparison, the following analysis takes the first and last
**eight** measured updates. Each section has exactly two validations and two
checkpoints:

| Section | Pipeline tokens/s | Training core tokens/s |
| --- | ---: | ---: |
| Early eight | 12,330.58 | 15,321.64 |
| Late eight | 12,240.38 | 15,302.80 |

Pipeline rate declined **0.73%**; training core rate declined **0.12%**. This is a
small observed change, not a thermal diagnosis. The automatic raw summary also
contains first/last-five-update rates (12,867.63 and 10,911.44); those sections
contain different numbers of periodic I/O events, so their apparent 15.2% drop
must not be interpreted as sustained compute slowdown. The equal-cadence analysis
above is recorded in `artifacts/phase7-verification.json`.

This run lasted seconds, not hours. It cannot establish long-duration thermal,
battery, or production stability.

## Memory and headroom

Across all hardware workers, maximum OS process lifetime peak RSS was **455.53
MiB**. Maximum sampled MPS live tensor allocation was **172.69 MiB**, and maximum
sampled MPS driver allocation was **216.67 MiB**. During the sustained run, sampled
driver allocations remained within that same maximum. These are different,
potentially overlapping unified-memory counters: **do not sum them or call them
total system memory**. Sampling does not guarantee capture of every transient peak.

Metal reported a recommended working-set size of 12,713,115,648 bytes; that is a
driver recommendation, not available RAM. The sustained run's sampled macOS
`memory_pressure -Q` free indicator was at least **32%**, above the configured 20%
stop threshold. It is an OS indicator, not a byte count or a reservation. Separate
4-GiB RSS/driver guards left substantial margin relative to observed allocations.
The hardware artifact directory occupied 186.64 MiB at completion, below 512 MiB.
No OOM was provoked; larger batches were not explored simply to consume memory.

## Proposed settings and runtime projection

Saved, **unexecuted** settings:
[`configs/training/phase7-m2-proposed.json`](../configs/training/phase7-m2-proposed.json).
Its duplicate is stored with the raw benchmark evidence.

| Setting | Proposal |
| --- | --- |
| Device / precision | MPS / float32 |
| Architecture | Existing primary 4-layer, 4-head, 256-dimensional, context-256 model |
| Microbatch / accumulation | **4 / 2**, effective eight windows |
| Loader workers / CPU threads | 0 / 2 |
| Optimizer / LR / decay | Existing AdamW policy; peak LR 0.0003; weight decay 0.01 |
| LR schedule / clipping | 100-update warmup, cosine decay to 10%; gradient norm cap 1.0 |
| Proposed example budget | 1,000 updates / 2,048,000 target tokens |
| Validation / checkpoint cadence | Every 100 updates, including final update |
| Validation budget | Four batches of four windows = 4,096 target tokens per event |
| Run ID | `phase7-proposed-model-a-unexecuted` |

Only batch/accumulation and device settings are informed by this benchmark. LR,
architecture, and the 1,000-update example budget are not quality-tuned findings.
The small validation prefix has ordering bias and is not a final evaluation protocol.

Measured training plus recurring housekeeping cost was 0.137065 seconds/update.
Mean validation and checkpoint costs were 0.057163 and 0.081136 seconds per event.
The saved projection uses:

```text
0.595647 setup + 0.189886 extra cold-warmup
+ 1,000 × 0.137065
+ 10 × 0.057163 validation + 10 × 0.081136 checkpoint
= 139.23 seconds, approximately 2.3 minutes
```

This excludes command-launch/interpreter startup; add a small startup allowance.
At the benchmark's more frequent four-update validation/checkpoint cadence, direct
steady-state projection is **171.64 seconds (2.9 minutes)** before startup. The
sustained rate gives a similar 172.01-second projection at that cadence. These
are different operating schedules, not confidence bounds.

Actual long-run duration may differ with power mode, temperature, competing apps,
memory pressure, filesystem behavior, or a larger validation set. No direct
temperature measurements were made. Preserve macOS/application headroom and repeat
the bounded command if working conditions or architecture change materially.

## Verification and artifacts

- **290 tests passed in 32.76 seconds**, no failures or skips, including all four
  real MPS tests with fallback disabled. New benchmark/diagnostic suite: **22
  passed in 6.92 seconds**. All routine data are labelled synthetic and offline.
- Tests cover bounded configuration, warmup exclusion/token-weighted rates,
  cached versus streamed accumulation math, refusal of incomplete cycles,
  observer boundary ordering including SIGINT, diagnostic inference context,
  overfit failure reporting, memory guards, compatible artifact checks, protected
  outputs, fresh subprocess orchestration, and an unexecuted recommendation.
- Regular-wheel install, Ruff lint/format, lock consistency, and dependency checks
  passed. No new dependency was added. The proposed config passed `train --dry-run`;
  its run directory does not exist.

Important source files are `benchmarking/{diagnostics,memory,worker,hardware_benchmark}.py`
and `configuration/benchmark_config.py` under `src/transformer/`. Training now
exposes complete-update observation and reuses exactly the same optimizer math for
resident and streamed inputs. Existing numerical/recovery tests still pass.

Raw artifacts, excluded from version control by existing ignore rules:

- [Hardware summary](../artifacts/phase7-hardware-v1/summary.json), configuration,
  implementation hashes, `*-mb*/iterations.jsonl`, worker logs, and checkpoints
  under `artifacts/phase7-hardware-v1/`.
- [Smoke summary](../artifacts/phase7-smoke-v1/summary.json) and labelled continuation;
  recoverable smoke checkpoints under `runs/phase7-smoke-v1/`.
- [Overfit summary](../artifacts/phase7-overfit-v1/summary.json) and complete loss trace.
- [Verification audit](../artifacts/phase7-verification.json), JUnit XML files, and
  [proposed-configuration dry-run](../artifacts/phase7-proposed-dry-run.json).

The [guide](benchmarking.md) and README contain exact commands and rerun naming
rules. The next work is the user's next phase prompt; baseline training remains
unstarted. Passing these checks establishes execution, tiny-example learning, and
bounded hardware measurements—not final model quality.
