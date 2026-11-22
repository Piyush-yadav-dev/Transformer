# Checkpoint format and recovery

The authoritative implementation is
[checkpoint_manager.py](../training/engine.py) and
[training_loop.py](../training/engine.py).
Checkpoints contain model/configuration, optimizer/scheduler, training position,
tokens, best validation, tokenizer/data identity, RNG and sampler state, plus
committed metrics-journal state. Treat project checkpoints as trusted local inputs.

A checkpoint is written at an optimizer-update boundary into a temporary file in
the destination directory, flushed/fsynced and atomically replaced. The checkpoint
index identifies latest and best with checksums. Run locks prevent competing writers.
Recovery checks committed log length/hash and archives uncommitted trailing records
before restoring the committed boundary. It does not recover unsaved work after
abrupt power loss.

For a new generic run, use the README train example. Resume with the exact original
configuration and --resume; --stop-after-updates means an absolute stopping point
without changing the schedule. Do not alter budget, scheduler, tokenizer or model
settings to force acceptance. Existing directories are protected.

Historical Phase 8/9 orchestration has strict source identity checks. Current code
may reject historical continuation because implementation changed after those runs.
Restore the recorded implementation/environment for that operation; do not bypass
checks. Completed checkpoints remain valid for current inference. The repository
does not promise an automatically downloadable historical source checkout.

CPU controlled resume tests pass. Exact accelerator reproduction across hardware,
OS or PyTorch releases is not guaranteed. See [training details](training_pipeline.md).
