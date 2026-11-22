# Bounded training diagnostics and hardware measurements

Phase 7 answers three separate questions. A **smoke check** exercises the pipeline;
an **overfit check** asks whether one deliberately tiny example can be learned;
a **hardware benchmark** measures this implementation's throughput and resource
use. None establishes language-generation quality. All use the frozen Phase 3
tokenizer. Corpus checks load existing local tokens without network acquisition;
test text is never selected for training or validation.

## Commands and limits

From the project root, after the regular-wheel installation described in README:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer training-smoke --config configs/benchmarking/smoke.json --output artifacts/phase7-smoke-v1
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer overfit-check --config configs/benchmarking/overfit.json --output artifacts/phase7-overfit-v1 --max-seconds 30
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer benchmark-training --config configs/benchmarking/m2-conservative.json
```

The smoke model uses two layers, two heads, width/context 32, and six updates. It
stops at update three, loads the saved checkpoint, resumes the original schedule
to six, validates, reloads final weights, and emits at most 48 greedy tokens after
`Once upon a time`. The output is labelled **DIAGNOSTIC ONLY**. Cropping to the
latest context and stopping at EOS are internal diagnostic support, not the public
inference interface. Unknown prompt characters fail explicitly.

The overfit configuration uses CPU, one layer, two heads, width/context 32,
dropout zero, microbatch/accumulation one, and AdamW with constant LR 0.01 and no
weight decay. One synthetic `red blue.\n` repetition is truncated to 32 characters
then terminated by EOS. It is a single fixed next-token training window. Its loss
is measured before updates and after each update on **that same training window**.
Stop after reaching loss ≤0.1 **and** ≤10% of initial loss, 150 updates, or 30
seconds. Time is checked at complete update boundaries. A failed threshold exits
nonzero and preserves the loss trace; it is not relabelled as success.

Both diagnostics save configuration and JSON summaries. Benchmark cases save raw
per-update JSONL, hardware/configuration snapshots, timing summaries, process logs,
and pipeline checkpoints. Outputs are exclusive: choose new names for a rerun.
For smoke, change both `run_id` and `--output`; for overfit change `--output`; for
benchmark change `output_dir` in a copied configuration. Existing evidence remains
intact. The benchmark saves hashes of installed Python implementation files.

## Controlled benchmark matrix

All cases retain the primary architecture: four layers, four heads, embedding and
context 256, feed-forward expansion four, dropout 0.1, float32, tied output weights,
and 3,245,056 trainable parameters. Only microbatch size changes: **1, 2, 4**.
Accumulation is **8, 4, 2**, respectively, keeping eight windows / 2,048 target
tokens per update. No architecture or precision tuning is mixed into this search.
Each case starts in a fresh process with the same seed and no loader workers.
Microbatch grouping changes dropout draws; this is a timing comparison, not an
exact numerical equivalence claim between stochastic runs.

Hardware warmup excludes four complete updates from steady-state measurements.
Eight subsequent updates are measured for each case. LR is held constant during
benchmark cases to keep schedule effects separate; the proposed training config
retains Phase 6's warmup/cosine schedule and original total budget. Hardware warmup
and learning-rate warmup are different concepts.

- **Compute-only:** batch tensors are already resident on the selected device.
  Repeated cached training windows exercise the exact shared forward/backward,
  accumulation, finite checks, clipping, AdamW update, and gradient-clearing code.
  Timing excludes data loading, transfers, validation, and checkpoint/log I/O.
  It is not an isolated matrix-multiply or theoretical GPU peak benchmark.
- **Pipeline:** actual `run_training` draws shuffled memory-mapped training
  windows, batches and transfers them, executes the same optimizer code, logs,
  validates, and atomically saves checkpoints. Validation uses 16 windows for
  every candidate. Evaluation and checkpoint intervals are four updates. This
  steady-state end-to-end rate includes those recurring costs, but excludes
  initial setup/initial checkpoint and observer instrumentation. Whole-worker
  wall time and setup-to-ready time are also reported separately.

MPS is used only when available; explicit `mps` still errors when unavailable.
Workers disable MPS CPU fallback. The shared optimizer synchronizes MPS immediately
before and after timed updates. Validation materializes scalar losses and
checkpoints copy tensors to CPU, so boundary timestamps include completed work.
Memory counters sampled by a lightweight model-forward hook are inside measured
compute; process/system sampling and benchmark JSONL writes between updates are
excluded from the boundary rate, but included in whole-worker wall time.

The selection rule chooses the **smallest safe microbatch within 90% of the best
measured pipeline throughput**, requiring both its compute and pipeline cases to
complete. There is one run per case, so small timing differences are not
statistically established wins. The selected candidate gets a separate pipeline
run capped at 120 measured updates plus four warmups, or 60 seconds. Early and late
rates use up to five measured updates from each end. This is a short sustained
check; background load, initialization, and checkpoint timing can affect it. No
temperature sensors are measured, so changes are not diagnosed as thermal throttling.

## Memory and resource limits

Each matrix case has a 45-second wall budget checked after complete updates;
subprocess startup/cleanup has a separate parent timeout of budget +60 seconds.
The sustained case uses 60 seconds. A boundary check may exceed its time budget by
one in-progress update and measurement. The parent kills a hung worker at the hard
timeout. No recovery promise is made beyond that worker's last atomic checkpoint.

Guards cap lifetime peak process RSS and sampled MPS driver allocations separately
at 4 GiB, require macOS `memory_pressure -Q` free percentage ≥20% when available,
and cap the artifact directory at 512 MiB between cases. These are conservative
stop signals, not reservations or guarantees that OOM cannot occur. The search
stops on a failed/resource-limited case; larger batches are not retried. An actual
OOM is recorded and is not followed by escalating candidates. Unsupported counters
are `null`; target-M2 conclusions remain pending on other hardware.

Counter names deliberately describe their scope:

| Measurement | Meaning |
| --- | --- |
| `process_current_rss_bytes` | Current process resident set from `ps` (KiB converted to bytes) |
| `process_lifetime_peak_rss_bytes` | OS process lifetime high-water RSS; bytes on Darwin, KiB converted on Linux |
| `mps_live_tensor_bytes` | Current PyTorch MPS tensor allocations; excludes allocator cache |
| `mps_driver_allocated_bytes` | Metal driver allocations, including cached pools and MPS/MPSGraph allocations |
| `maximum_sampled_mps_*` | Maximum observed at model-forward/update boundaries; not a guaranteed instantaneous peak |
| `mps_recommended_working_set_bytes` | Metal's recommended working-set limit; not physical or free system RAM |
| `system_memory_pressure_free_percent` | macOS tool's system-wide indicator; not a byte count of allocatable RAM |

Process and accelerator counters can overlap in unified memory: **do not add them**.
They do not measure total machine use. See PyTorch's definitions of
[current tensor memory](https://docs.pytorch.org/docs/2.15/generated/torch.mps.current_allocated_memory.html),
[driver allocations](https://docs.pytorch.org/docs/2.15/generated/torch.mps.driver_allocated_memory.html),
and [recommended working set](https://docs.pytorch.org/docs/2.15/generated/torch.mps.recommended_max_memory.html).
Leave capacity for macOS and other applications; repeat the bounded check if normal
working conditions differ materially.

## Runtime estimate and next steps

`summary.json` records measured throughput, chosen batch, a separate sustained
result, and a runtime projection. `proposed_training.json` is saved but **never
executed**. The estimate charges mean training/housekeeping cost per update,
observed validation/checkpoint event costs at the proposed intervals, measured
setup, and estimated extra cold-warmup overhead. It also reports a direct projection
at the benchmark's more frequent save/evaluation cadence. This estimate is tied to
the exact token, architecture, and validation budget in the saved configuration.

Longer runs may differ because of power mode, other processes, memory pressure,
thermal behavior, filesystem costs, and validation policy. A minute-scale run
cannot establish hours-long stability. MPS reproducibility limits from the
[training guide](training_pipeline.md) still apply. Benchmarking uses nondeterministic
MPS settings; it does not promise bitwise resume across devices or releases.
Do not begin baseline training until the next phase is explicitly requested.
