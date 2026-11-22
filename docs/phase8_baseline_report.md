# Phase 8: actual TinyStories-only baseline

**Completed on 2026-11-19. Phase 8 acceptance criteria passed.** Model A was trained
from random initialization on the prepared TinyStories-only view, on the actual
MacBook Air M2 with 16 GiB memory and MPS. No Model B training or test-set evaluation
was performed. The fixed greedy samples remain highly repetitive; this completed
baseline is not a fluent or factually reliable story generator.

## Protocol and prerequisites

The [experiment protocol](phase8_experiment_protocol.md) and
[executable comparison settings](../configs/experiments/phase8-comparison.json)
were written before launch and copied/hashed before the first optimizer update.
The only planned Model A/B training-setting differences are run ID and source
stream. Both budgets are **2,048,000 target-token predictions**, not matched epochs
or a wall-clock cutoff. Model B remains an unexecuted plan and must start from
fresh initialization later.

Verified prerequisites included architecture/reference tests, CPU exact recovery,
real MPS recovery, unchanged benchmarked compute implementation, frozen tokenizer
fingerprint, prepared-data/token checksums, and a successful bounded-config
dry-run. All **299 regression tests passed** before launch. The actual target M2
measurements from Phase 7 remain applicable: model/optimizer/data-loading compute
code was unchanged. Hardware at launch reported macOS 27.0.1 arm64, Python 3.13.7,
PyTorch 2.15.0, working MPS, and battery power. Temperatures were not measured.

Model: four layers, four heads, embedding/context 256, FF expansion four, dropout
0.1, tied embeddings/output, float32, 3,245,056 trainable parameters. Shared frozen
78-ID character vocabulary; the weights learned from `train_tinystories` only.
Initialization seed 42, microbatch four, accumulation two, zero loader workers,
two CPU threads. AdamW peak LR 0.0003, 100-update warmup, cosine decay to 0.00003,
weight decay 0.01 on matrices, norm clipping 1.0. Hyperparameters were not changed
after examining losses or samples.

## Actual run and selection

Run directory: `runs/phase8-model-a-tinystories-20261119-v1/`.

| Item | Measured result |
| --- | --- |
| Optimizer updates | **1,000 of 1,000**, contiguous with no missing/duplicated updates |
| Training exposure | **2,048,000 target tokens**, exactly the declared budget |
| Data | 1,000 TinyStories training stories; 3,682 full windows, revisited about 2.17 times |
| First training-update loss | 4.424727, before the first optimizer step |
| Final training-update loss | 2.040301 |
| Last 100 updates, mean training loss | 2.065175 |
| Best validation checkpoint | **Update 1,000**, validation NLL **2.081407 nats/target token** |
| Validation coverage per event | All **838 windows / 214,528 targets**, 210 batches, last batch size two |
| Validation token perplexity | 8.015740, character/special-token perplexity; not word perplexity |
| Unexpected interruptions | **0** |
| Planned stop/resume boundaries | **100 and 500** for checkpoint archival and fixed diagnostics |

Training losses come from different sampled batches and include training dropout;
they should not be treated as evaluation on a fixed dataset. No untrained-model
validation loss was measured. The first full-validation observation was at 100.

| Update | Training target tokens processed | Full validation NLL |
| ---: | ---: | ---: |
| 100 | 204,800 | 2.478317 |
| 200 | 409,600 | 2.382418 |
| 300 | 614,400 | 2.347681 |
| 400 | 819,200 | 2.313076 |
| 500 | 1,024,000 | 2.283199 |
| 600 | 1,228,800 | 2.245303 |
| 700 | 1,433,600 | 2.180574 |
| 800 | 1,638,400 | 2.135084 |
| 900 | 1,843,200 | 2.104299 |
| 1,000 | 2,048,000 | **2.081407** |

The predeclared rule selected the minimum token-weighted full-validation NLL,
with the earlier checkpoint winning an exact tie. The selected checkpoint was
also the final checkpoint, so `latest` and `best` point to the same verified file.
This is expected; the two references remain explicit in the atomic index. Test
performance was not computed or used for selection.

## Duration, throughput, and resources

| Measurement | Result | Scope |
| --- | ---: | --- |
| Training-core time | **139.060 s** | Batching/transfers, forward/backward, accumulation, finite checks, clipping, optimizer updates |
| Core training throughput | **14,727.48 targets/s** | Total training target tokens divided by core time; not a pure GPU benchmark |
| Training-session wall time | **168.582 s** | Three sessions including setup/resume, validation, logs, checkpoints, and observation |
| Session throughput | **12,148.37 training targets/s** | Includes validation/I/O time in the denominator; validation tokens are not added to the numerator |
| Full validation time | 26.979 s | Ten events, already included in session time |
| Diagnostic sampling time | 8.095 s | Nine fixed continuations, between training sessions |
| Orchestration elapsed through training/sampling | **177.212 s**, about 2 min 57 s | Also includes archive/diagnostic overhead; excludes initial preflight, interpreter startup, and final report writing |

The sessions took 16.753 s (updates 1–100), 66.975 s (101–500), and 84.854 s
(501–1,000). Model B must record its own compute and wall time even with the same
token budget. These are observed local timings, not guarantees for later runs.

OS process lifetime peak RSS was **437.33 MiB**. Maximum MPS driver allocation
sampled at validation boundaries was **224.66 MiB**; those samples do not establish
the instantaneous forward/backward peak. These unified-memory counters overlap
and must not be added or described as total system memory. The sampled macOS
system free indicator at validation boundaries stayed between 34% and 36%, above
the 20% pause threshold. No memory guard fired, and no OOM occurred. The run
occupied approximately **149.53 MiB** including archived checkpoints and logs.

## Qualitative results and limitations

All **nine** predefined samples are preserved: three prompts at each of updates
100, 500, and 1,000. Each generated 128 tokens under greedy argmax, temperature zero,
top-k/top-p disabled, EOS stopping, and a 256-token context. None encountered EOS
before the length cap. Sampling seed is explicitly null because argmax consumes
no random sampling; initialization seed 42 is recorded separately. Every sample
has its prompt, checkpoint update/hash, generated IDs/text, and sampling settings.

The samples repeatedly produce variants of `the` and `they`. At update 100 all
three prompts led to repeated `the`; the later checkpoints remain strongly
repetitive. **Validation loss improved, but these diagnostics do not demonstrate
coherent story generation.** Outputs were not edited, cherry-picked, regenerated
under different sampling settings, or used to extend the fixed training budget.
The verification replay below used the identical fixed settings only.

This is a single-seed baseline on bounded, prefix-selected data, with known
ordering bias and exact-only deduplication. It cannot establish broad language
quality, factual accuracy, or that adding SimpleStories will improve results.
The later comparison needs the same frozen controls; final test evaluation remains
reserved until checkpoint selection is complete for the experiments being compared.

## Checkpoint verification and artifacts

Fresh-process verification passed after training:

- Initial weight fingerprint exactly matched a newly constructed seed-42 model;
  initialization had zero optimizer history. Trained weights differ from it.
- All 1,000 update indices, accumulated tokens, finite losses/gradient norms/LRs,
  scheduler position, and planned resume events agree with the configured budget.
- Loaded best/latest checkpoints and all three milestone copies passed SHA-256
  verification. They contain model, optimizer, scheduler, sampler, RNG, configuration,
  environment/data identity, and training position.
- Re-evaluating the selected checkpoint on the same full validation set reproduced
  **2.081407102024925** exactly on this host. All three final sample token sequences
  were regenerated identically. This does not imply cross-device MPS reproducibility.
- Frozen protocol/configuration/source hashes remained unchanged during training.
  Model B's run directory does not exist, and no test-loss evaluation was performed.

Best/latest checkpoint SHA-256:
`b39c15d30b3abf420dca0f94a104e5d22044a23fe1debd3bf3223ef2f78474ad`.
Initial random-weight fingerprint:
`2e2ac2b593334c3a2fd367c19db7d2742cf030500398f391df394c28af4b855c`.

Actual local artifacts (generated files remain ignored by existing rules):

- [Trained best/latest checkpoint](../runs/phase8-model-a-tinystories-20261119-v1/checkpoints/update-000001000-7f3a90ed13f642e9a699a73d5343b056.pt)
  and [atomic checkpoint index](../runs/phase8-model-a-tinystories-20261119-v1/checkpoints.json).
- [Frozen run configuration](../runs/phase8-model-a-tinystories-20261119-v1/configuration.json),
  [run/environment identity](../runs/phase8-model-a-tinystories-20261119-v1/run.json),
  [frozen comparison protocol](../runs/phase8-model-a-tinystories-20261119-v1/comparison_protocol.json).
- [Training metrics](../runs/phase8-model-a-tinystories-20261119-v1/metrics.jsonl),
  [validation history](../runs/phase8-model-a-tinystories-20261119-v1/validation_history.json),
  [session/recovery observations](../runs/phase8-model-a-tinystories-20261119-v1/experiment_events.jsonl).
- [All nine diagnostic continuations](../runs/phase8-model-a-tinystories-20261119-v1/diagnostics/samples.md),
  with raw sample JSON and full milestone checkpoints in the same directory.
- [Machine-readable training summary](../runs/phase8-model-a-tinystories-20261119-v1/training_summary.json)
  and [independent verification](../runs/phase8-model-a-tinystories-20261119-v1/post_training_verification.json).

## Commands actually executed

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/model_verification tests/training/test_recovery.py --junitxml=artifacts/phase8-prerequisites.xml
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training/test_baseline_experiment.py --junitxml=artifacts/phase8-experiment-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase8-full-suite.xml
.venv/bin/transformer train --config configs/training/phase8-model-a.json --dry-run > artifacts/phase8-dry-run.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train-baseline --protocol configs/experiments/phase8-comparison.json > artifacts/phase8-baseline-cli.json 2> artifacts/phase8-baseline-console.log
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

The initial prerequisite suite passed 58 tests in 5.50 s. The early focused runner
suite passed eight tests in 1.40 s; the final complete suite included the added
interruption test: **299 passed in 32.42 s**, no skips/failures, including four
real MPS tests. Tests also establish exact CPU equivalence of uninterrupted versus
sampled/paused/resumed training, refusal of unmatched controls/partial validation,
frozen-protocol protection, archive integrity, and recovery between sessions.
Regular-wheel installation, lint/format, lock, and dependency checks passed;
dependencies remain unchanged (60 locked /39 installed).

Local Python verification commands additionally checked source/data hashes,
generated configuration snapshots, verified and re-evaluated checkpoints,
regenerated final samples, and rendered the unedited sample Markdown from JSON.
Evidence is in `artifacts/phase8-prerequisite-verification.json`,
`artifacts/phase8-post-training-verification.json`, JUnit files, and the run itself.

The documented command refuses an existing run unless `--resume` is supplied.
That flag preserves the frozen budget; a completed run receives no more updates.
Phase 8 stops here. The next expected work is the user's Model B phase, using the
frozen controls and a separate random initialization—not continuation of Model A.
