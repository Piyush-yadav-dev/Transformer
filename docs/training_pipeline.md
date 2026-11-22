# Configurable training and recovery

This pipeline implements local float32 training and recovery for Transformer.
Phase 6 verified it with synthetic fixtures. The bounded real Model A and Model B
runs subsequently completed in Phases 8 and 9; their records are described in
[the baseline protocol](phase8_experiment_protocol.md) and
[the combined-model report](phase9_combined_report.md).
The separate `primary-example.json` below remains an unexecuted configuration example.

## Configuration and commands

`configs/training/primary-example.json` is a reviewable **unexecuted example** for
Model A. It embeds the existing primary architecture, pins tokenizer/corpus/token
fingerprints, and selects `train_tinystories`. Model B must use a different run ID,
`train_stream="train"`, the same architecture/tokenizer, and an explicitly comparable
budget. A new run always initializes fresh weights; it cannot silently continue a
different experiment's checkpoint.

From the project root after the README's locked installation:

```sh
.venv/bin/transformer train --config configs/training/primary-example.json --dry-run
```

Dry-run checks source artifacts, vocabulary compatibility, device selection, window
availability, and the configured budget. It reports parameter count and a rough
checkpoint disk estimate. It performs no optimizer updates and creates no run
directory. It is not a throughput, activation-memory, or training-quality benchmark.

To launch a new experiment after choosing its run ID, device, budget, and disk allowance:

```sh
transformer train --config configs/training/primary-example.json
transformer train --config configs/training/primary-example.json --resume
```

Those two production commands were **not executed in Phase 6**. `--project-root`
resolves the configuration's project-relative token and tokenizer paths.
`--project-config` supplies the existing `paths.runs_dir` convention. The model
configuration is embedded in the training snapshot so later edits to a separate
model JSON file cannot change an existing run.

For bounded sessions, `--stop-after-updates N` stops at the absolute optimizer
update N without changing the original scheduled budget. Resume with the same
configuration and `--resume`; omit that stop flag to finish the original budget.
It does not permit extending or shortening the original LR history.

The training configuration validates:

- Portable run ID and relative paths; all three expected SHA-256 identities.
- Model settings, device (`cpu`, `mps`, `auto`), seed, microbatch size, accumulation
  count, positive gradient clipping norm, learning rate and nonnegative weight decay.
- Exactly one of `max_updates` or `max_training_tokens`. Token budgets must be a
  positive multiple of context length; no context truncation is silently applied.
- Warmup length, minimum LR ratio, evaluation/checkpoint/console intervals, and
  positive validation-batch limit.
- Float32 training, zero data workers, configured CPU thread count, and an explicit
  deterministic-algorithms setting. Worker prefetch, mixed precision, and distributed
  state are unsupported and are not silently ignored.

Defaults in the example are microbatch 2, accumulation 4, context 256, CPU with
2 threads, 1,000 updates, warmup 100, LR 0.0003, weight decay 0.01, clipping norm 1,
evaluation/checkpoint every 100 updates, and at most 8 validation microbatches.
This means 2,048 target tokens per full update and 2,048,000 scheduled training
tokens. These are explicit example settings, not measured hardware recommendations.

## Optimizer updates and stopping boundaries

The CPU window sampler uses an independent seeded generator. Each shuffled epoch
visits every complete training window once. Microbatches may cross epoch boundaries
to fill a cycle; the epoch, permutation, cursor, and generator state are saved.
There are no background workers consuming samples beyond the recorded position.
Permutation memory scales with the number of windows, while the token corpus stays
memory-mapped on CPU. Only the current batch moves to the selected device.

Each optimizer cycle:

1. Sets training mode and clears gradients with `zero_grad(set_to_none=True)`.
2. Loads up to the configured number of microbatches. Each mean loss is weighted
   by its actual target-token count divided by the cycle's target-token count.
3. Backpropagates all microbatches; rejects missing/non-finite losses or gradients.
4. Clips the accumulated gradient once and records the norm **before clipping**.
5. Sets this update's LR, executes exactly one AdamW step, checks parameter and
   optimizer-state finiteness, advances the schedule, then clears gradients.

The normal update budget always uses complete accumulation cycles. A token budget
can end with fewer windows or a smaller final microbatch. That final cycle receives
one update with correct weighting, and the processed-token count reaches the exact
configured budget. No partially accumulated gradients are checkpointed. Validation
and checkpoints occur only after optimizer-update boundaries.

Training-token counts mean target positions used by **committed optimizer updates**,
including EOS/unknown targets and repeated windows across epochs. They exclude
validation and any discarded work after an unexpected failure.

## AdamW and the learning-rate schedule

AdamW supplies adaptive moments with decoupled weight decay. The implementation
uses its conventional betas `(0.9, 0.999)` and epsilon `1e-8`, with non-fused,
non-foreach execution for explicit CPU/MPS behavior. Matrix parameters, including
embeddings, receive configured decay; one-dimensional biases/normalization parameters
do not. Tied weights appear in the optimizer once. No dataset-specific optimizer
tuning or performance improvement is claimed.

The schedule is defined by **one-based optimizer update k**, not microbatch count.
Let U be the original total update budget, W warmup updates, L the peak LR, and r
the minimum LR ratio:

- For `k <= W`, use `L * k / W`.
- After warmup, use `L * [r + (1-r)*(1+cos(pi*p))/2]`.
- If `W > 0`, `p=(k-W)/(U-W)`. If `W=0`, `p=(k-1)/max(1,U-1)`.

Thus warmup ends at L, the final update reaches `r*L` when U>1, and a one-update
run without warmup uses L. LR is assigned **before** the corresponding optimizer
step; completed-update count advances only after it. The checkpoint stores schedule
settings, position, and optimizer LR. Resume rejects changes to the schedule,
optimizer options, moments, or step counters instead of restarting them.

## Validation and metrics

The trainer constructs only the selected training stream and `validation` stream.
It never evaluates the test stream. Artifact integrity checking may checksum other
stream files; their contents do not enter model decisions.

Validation runs in `torch.inference_mode()` and evaluation mode, then restores the
previous model mode even on an error. It aggregates loss by the number of evaluated
target tokens, including a smaller final batch. Validation visits a fixed sequential
prefix of complete windows, bounded by `max_validation_batches`; this has ordering
bias and is not full-corpus validation unless the limit covers all windows.

Validation runs at the configured interval and at the **original final budget**.
A temporary session stop or keyboard interrupt does not introduce an extra
off-schedule validation decision. The test set remains reserved for later evaluation.

`metrics.jsonl` records one update row per completed update, plus lifecycle events:

- Optimizer update, total processed tokens, cycle tokens, token-weighted training
  loss, LR used, pre-clipping gradient norm, and newly computed validation loss/count.
  Unscheduled validation fields are `null`/zero, not stale carried-forward scores.
- Update seconds and tokens/second: measured from synchronous batch loading through
  forward/backward, clipping, optimizer step, and finite checks, with MPS synchronized.
  These exclude validation, logging, and checkpoint I/O.
- Cumulative training throughput uses total committed tokens divided by cumulative
  update seconds. Elapsed time starts after preflight, model/optimizer construction,
  and resume setup; it includes loop overhead up to the sampled boundary and excludes
  downtime. A checkpoint records elapsed time before its own write; its final write
  duration and time spent on discarded post-checkpoint work are not retained on resume.

Console progress goes to stderr at the configured interval and after validation;
the CLI final summary is JSON on stdout. No hosted tracker is required.

## Checkpoint publication and recovery

Each run contains `configuration.json`, `run.json`, `metrics.jsonl`, a `checkpoints/`
directory, and `checkpoints.json`. The index defines **latest** and **best**; use
`CheckpointManager(run_directory).load("latest")` or `.load("best")` to resolve them.
Resume uses latest. Best is the lowest finite scheduled validation loss; ties retain
the earlier best. Best is absent until validation has executed.

Checkpoints contain model weights/configuration, optimizer moments/groups, schedule,
update/token position, last/best validation records, time counters, data/tokenizer
identities, source metadata, complete configuration, implementation/environment
identity, RNG state, sampler state, and the committed metrics byte count/hash.
Python, NumPy, CPU PyTorch, and (for MPS runs) MPS generator states are saved.

A new checkpoint is written to a temporary file in the destination directory,
flushed/fsynced, and renamed. A second atomic write publishes one index containing
both latest and best references with hashes. Only after that commit are superseded
referenced checkpoints removed. Normal retention is latest plus best; publication
can temporarily require a third copy. Failed publications can leave unreferenced
orphan files, which are not automatically deleted.

A checkpoint at update zero exists before the first optimizer cycle. Further saves
occur at checkpoint intervals, every scheduled validation, and session/end/interrupt
boundaries. Resume checks hashes, exact configuration/data/code/environment identity,
model dtype, scheduler/token/sampler positions, and complete AdamW moment history.
Unsupported or missing saved state fails explicitly. Strict matching includes
device, CPU threads, deterministic setting, library versions, and MPS fallback policy;
there is no silent cross-device or version-migration resume mode.

If a crash left log rows beyond the last published checkpoint, resume first verifies
the committed prefix, archives the uncommitted tail to
`metrics.uncommitted-<id>.jsonl`, and truncates the active log to its checkpoint
boundary. The resume event records that archive. Missing or altered committed log
bytes cause an error. Failed/replayed work is not reported as newly committed tokens.

New runs refuse an existing directory. Resume requires an explicit `--resume` and
matching configuration. An advisory file lock prevents concurrent writers and is
released by the OS when a process exits; its small lock file may remain. macOS/Linux
locking is supported; a Windows backend has not been implemented.

## Interruption and reproducibility limits

In the main thread, SIGINT/Ctrl-C requests a stop instead of raising inside the
optimizer. Initialization or the current accumulation/update/validation finishes,
then a safe checkpoint is published. An interrupted run returns CLI status 130.
Repeated SIGINTs
remain deferred. Large configured cycles can therefore delay a graceful stop.

A nonstandard `KeyboardInterrupt` exception outside that deferred path may leave
in-memory optimizer state partially changed. The trainer does not publish it and
reports only the last committed disk checkpoint as recoverable. Other failures,
including non-finite computations, also preserve that checkpoint. Abrupt process
termination or power loss can recover no further than the last completed atomic
publication; unsaved updates are not promised to survive.

Controlled CPU tests reproduce uninterrupted weights, optimizer/scheduler/RNG/sampler
state, losses, and LR history **exactly**, with dropout enabled and an epoch crossing.
Timing metrics naturally differ. MPS RNG state is saved/restored and a tiny real MPS
resume is tested, but bitwise accelerator reproducibility is not promised. Hardware,
library kernels, and floating-point execution can differ; unsupported deterministic
operations fail rather than silently changing settings. Training stays float32.

## Verification

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest
```

Tests build tiny labeled synthetic story/token fixtures and execute at most four
updates per run. They cover parameter changes; accumulation against a large-batch reference;
partial token budgets; clipping/step counts; no-gradient weighted validation; exact
CPU resume; actual SIGINT; unexpected interruption; invalid state/settings; atomic
publication failure; latest/best retention; log recovery; locks; CLI stop/resume;
explicit dtype; and an actual MPS checkpoint/resume when available. That MPS test is
skipped on unavailable hosts. See `project_progress.md` for executed results.

References: [PyTorch AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html),
[MPS RNG capture](https://docs.pytorch.org/docs/2.15/generated/torch.mps.get_rng_state.html),
and [MPS RNG restoration](https://docs.pytorch.org/docs/2.15/generated/torch.mps.set_rng_state.html).
