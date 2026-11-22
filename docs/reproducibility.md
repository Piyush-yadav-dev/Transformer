# Reproducibility and artifact ownership

Track source, tests with tiny synthetic fixtures, scripts, pyproject.toml, uv.lock,
.python-version, configs/, docs/, and the explicit model registry. Do not track
virtual environments, credentials, downloaded datasets, token binaries, large
checkpoints or routine logs. .gitignore currently excludes all root artifacts/,
data/ and runs/: a source checkout alone does not contain their evidence.

For archival reproducibility, separately preserve immutable run directories,
tokenizer/token manifests, prepared manifests, checkpoint indices and selected
weights, environment/source hashes, evaluation outputs and raw measurements.
Keep small evidence manifests in a deliberate release archive if needed; do not
silently assume ignored local files are version-controlled. No checkpoint download
service or public release bundle is configured.

On a fresh machine:
1. Install the locked environment and optional web/evaluation extras.
2. Run prepare-data with configs/data/phase2-small.json. First acquisition requires
   network access and the pinned upstream sources; --offline needs populated cache.
3. Run prepare-tokens with configs/tokenization/phase3-small.json.
4. Restore a trusted complete experiment archive, or perform new budgeted training.
   New runs need new IDs and provenance; do not overwrite reported experiments.
5. For new model versions, intentionally create registry entries with actual
   configuration/tokenizer/checkpoint/metric paths and hashes after verification.
   Never change hashes merely to suppress an integrity error.

Reported settings: configs/experiments/phase8-comparison.json, frozen Phase 8/9
protocols, and configs/evaluation/phase10-comparison.json. Actual snapshots live in
the run directories named by configs/inference/model_registry.json. Generic
configs/training/primary-example.json did not produce those results.

Fresh training can reproduce the methodology without guaranteeing byte-identical
MPS weights. CPU/MPS kernels, OS, library versions and stochastic state affect
numerics/timings. Preserve the lock; log actual device/environment; treat unavailable
MPS checks as skipped. See [recovery constraints](checkpoint_recovery.md).
