# Transformer Phase 11: reusable inference verification

**Completed 20 November 2026 (Asia/Kolkata). Acceptance criteria passed.**
The actual baseline and combined-data checkpoints load and generate through a
reusable Python service and installed CLI. No frontend, new training, dataset
download, test-based checkpoint reselection or dependency change occurred.

## Implemented interface

- [Generation settings](../configs/settings.py):
  distinct greedy/sample modes, validated temperature/top-k/seed, explicit unknown
  handling, 0–1,024 output tokens and a 16,384-character prompt bound.
- [Registry](../configs/inference/model_registry.json) and
  [registry loader](../inference/service.py): stable IDs
  `transformer-tinystories-v1` and `transformer-combined-v1`, with pinned checkpoint,
  tokenizer, configuration and actual metrics associations. Generation never accepts
  an arbitrary checkpoint path or discovers files by scanning user input.
- [Checkpoint loading](../inference/service.py):
  checksum-verified local file handles, restricted deserialization, architecture
  reconstruction, tokenizer/vocabulary/state/metrics compatibility, frozen parameters
  and evaluation mode. CPU and explicit/automatic MPS follow the existing device policy.
- [Sampling](../inference/service.py) and
  [service](../inference/service.py): private CPU RNG,
  stable exact top-k, EOS stopping, rolling learned positions, inference mode,
  one-model cache, serialized requests, switching/cleanup and separate load/generation
  timing. Results carry identity, settings, raw IDs/count, text, stop reason and runtime.
- Installed commands `list-models` and `generate`; no frontend libraries required.

The [inference guide](inference_backend.md) supplies executable Python/CLI examples,
the complete edge-case table, trusted-root/catalog boundary and resource ownership.
It documents that EOS contributes to the generated count but not visible continuation,
whereas UNK remains visible. Empty input errors because there is no BOS token. Long
prompts retain their last context-length IDs; learned positions reset to 0..L−1 on
every recomputed window. Zero output performs no forward pass.

## Tests and genuine defect fixed

The first focused run had **60 passes /1 MPS failure**: converting MPS logits directly
to CPU float64 raised `Undefined type Double`. The fix transfers logits to CPU in
their original dtype first, then converts them for probability calculations. The
focused rerun passed **61 tests in 2.78 s**. Two further targeted checks covered the
maximum output length/inference mode and verified-file restricted deserialization.

The final full suite passed **392 tests in 34.68 s**, no failures/skips, including
**63 inference tests** and **six real MPS tests** across the project. Coverage includes:

- Missing/corrupt artifacts, incompatible config/tokenizer/vocabulary/state/tied weights,
  finite float32 weights, path traversal, symlink files/parents and unknown model IDs.
- Temperature probabilities against an independent numerical reference, exact top-k
  tie handling, seeded reproducibility, greedy stability and top-k-one equivalence.
- Empty/unknown/invalid Unicode/oversized prompts; zero/max/excess output; immediate EOS
  and visible UNK; actual rolling-position hooks and bounded context windows.
- No gradients, frozen weights, restored RNG/thread/dtype settings, concurrent calls,
  one-entry cache eviction, changed-file rejection, CLI errors and explicit MPS failure.

No training/model math was modified, and no test outcome was addressed by changing
trained weights or the previously recorded evaluation results.

## Actual trained-model verification

The [bounded verification script](../tools/maintenance.py) ran on the actual
MacBook Air M2 /16 GiB, macOS 27.0.1 arm64, Python 3.13.7 /PyTorch 2.15.0, on battery,
MPS with CPU fallback disabled. The [full record](../artifacts/phase11-inference-verification.json)
contains all output IDs/text/settings, checkpoint identities and memory samples.

| Check | Actual outcome |
| --- | --- |
| Load both selected checkpoints | Passed; 3,245,056 parameters each, frozen/eval |
| Three archived greedy prompts per model | All six 128-token ID sequences matched exactly |
| Temperature 0.8 /top-k 20 /seed 1234, up to 64 tokens | Repeat calls matched for each model |
| Long prompt versus explicit last-256-character suffix | Identical output IDs for both models |
| Zero requested output | Empty IDs/text; zero generated tokens |
| Parameter gradients and global CPU/MPS RNG | No gradients; caller RNG states unchanged |
| Original trained-run and Phase 10 evaluation files | All **83 files unchanged** |
| CPU and MPS CLI examples | Executed successfully; text and JSON outputs saved |

These are execution/reproducibility checks, not new quality claims. The sampled
continuations still contain repetition and malformed words. The full raw text is
retained rather than selected for fluency. Original held-out comparison conclusions
and checkpoint selection remain unchanged.

## Bounded switching measurements

After warming both models through generation, the script alternated A/B three times
(six load/generate observations), keeping only one model cached:

| Measurement | Observed result |
| --- | ---: |
| Cached models after every switch | 1 |
| MPS live-tensor bytes at every switch | **13,242,368 bytes /12.629 MiB**, identical across all six |
| MPS driver allocation at every switch | **17,432,576 bytes /16.625 MiB**, identical across all six |
| Process current RSS at every switch | **435,568,640 bytes /415.391 MiB**, identical across all six |
| Process lifetime peak RSS in verification | **436,404,224 bytes /416.188 MiB** |
| After explicit unload | 0 cached models, **0 MPS live-tensor bytes** |
| MPS driver allocation after unload | **655,360 bytes /0.625 MiB** |

This bounded sequence showed no growth at the sampled switch boundaries. It is not
a long-running leak proof or an instantaneous accelerator peak measurement. Process
RSS, live tensors and driver allocation overlap on unified memory and must not be
summed or called total system memory. CPU allocator/runtime memory remains resident
after unload; freeing the model does not promise zero process RSS.

Cold model-load measurements in that run were approximately 0.111 s for A and
0.131 s for B. Individual 128-token greedy generation calls took 0.710–0.894 s;
these diagnostic timings are not a controlled benchmark. Cache hits reported zero
load time. Generation timing excludes loading, queue/lock wait and JSON serialization.

## Executed commands and records

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/inference --junitxml=artifacts/phase11-inference-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase11-full-suite.xml
.venv/bin/transformer list-models > artifacts/phase11-model-list.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer generate --model transformer-tinystories-v1 --device mps --prompt 'Once upon a time, there was a little girl named Lily.' --max-new-tokens 16 --json > artifacts/phase11-cli-greedy.json
.venv/bin/transformer generate --model transformer-tinystories-v1 --device cpu --mode greedy --prompt 'Tom found a small box under a tree.' --max-new-tokens 64 > artifacts/phase11-cli-cpu.txt
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer generate --model transformer-combined-v1 --device mps --mode sample --temperature 0.8 --top-k 20 --seed 42 --prompt 'The little bird wanted to help its friend.' --max-new-tokens 128 --json > artifacts/phase11-cli-sampled.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/maintenance.py verify-inference --device mps --output artifacts/phase11-inference-verification.json
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

The [full-suite XML](../artifacts/phase11-full-suite.xml),
[focused-suite XML](../artifacts/phase11-inference-tests.xml),
[model listing](../artifacts/phase11-model-list.json),
[CPU CLI text](../artifacts/phase11-cli-cpu.txt), and
[MPS sampled CLI result](../artifacts/phase11-cli-sampled.json) preserve actual outputs.
Lock/dependency/lint/format checks passed; no dependencies changed.

The service is local and synchronous, with one cache entry per instance. A trusted
operator owns the root/catalog; it is not an upload or remote-checkpoint interface.
Same-seed output is not guaranteed bitwise across devices or PyTorch releases.
The remaining frontend is outside this phase. Stop here and await the next prompt.
