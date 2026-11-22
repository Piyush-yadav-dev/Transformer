# Transformer

Transformer is a university and portfolio project for a custom decoder-only,
autoregressive language model built from PyTorch primitives and trained from random
initialization. The primary comparison uses character-level tokenization with
TinyStories alone versus TinyStories plus the verified SimpleStories dataset.

**Current status: Phases 1–20 implemented and verified within their recorded scope; Phase 21 documentation complete. Accessibility verification remains partial.**
Configuration, device selection, diagnostics, pinned dataset acquisition,
conservative cleaning, joint exact deduplication, document splits, and manifests
are implemented, along with a shared character tokenizer and CPU memory-mapped
next-token batches and a custom pre-normalization decoder-only neural network.
Architecture verification now covers causality, independent numerical references,
gradients, serialization, and CPU/MPS execution. Budgeted training, validation, and
checkpoint recovery are implemented and tested on tiny synthetic fixtures.
Phase 7 adds separately labelled smoke, synthetic overfit, and hardware checks.
Phase 8 runs a bounded TinyStories-only baseline under a frozen comparison protocol.
Phase 9 adds the controlled combined-data run with exact source-exposure accounting.
Phase 10 adds reproducible held-out metrics, paired samples, timing, and charts.
Phase 11 adds registry-backed public Python/CLI generation. Phase 12 verifies the
integrated backend and both actual trained models. Phases 13–14 add the local editorial
interface, prompt editing and session history. Phase 15 connects Generate to the actual
registered models through the existing inference service.
Generated datasets are local
ignored artifacts; no model checkpoints are bundled.

## Open the website

```sh
cd /Users/piyushyadav/Desktop/project
.venv/bin/python main.py --device auto
```

Open http://127.0.0.1:8000. Training scripts are in `training/`, story fine-tuning
scripts in `fine_tuning/`, website files directly in `frontend/`, and API
code in `backend/`. Data stays in `data/`; trained weights stay in
`runs/`. See the [folder guide](docs/project_structure.md) for the complete map.

## Start here

The local application has Generate, Model insights and Architecture pages.
It currently uses the registered story-continuation adapter and its local pretrained
base model. Historical custom Model A/B checkpoints remain as experiment records. A fresh source
checkout does not include generated datasets or checkpoints; follow the
[reproducibility guide](docs/reproducibility.md) before expecting generation to work.

- Developers: [project structure](docs/project_structure.md), then installation below.
- Evaluators: [experiment methodology](docs/experiment_methodology.md) and
  [model/data notes](docs/model_data_notes.md).
- Returning users: [checkpoint recovery](docs/checkpoint_recovery.md) and
  [troubleshooting](docs/troubleshooting.md).
- Complete guide index: [documentation map](docs/README.md).

Reported runs used a MacBook Air M2 (Mac14,2), 16 GB unified memory, macOS arm64,
Python 3.13.7 and PyTorch 2.15.0. CPU is supported; explicit MPS requires actual
MPS availability. CUDA and public hosting are not implemented deployment targets.

## Installation

Run commands from the project root in a macOS terminal. Use native Apple Silicon
Python **3.13**; the development interpreter is **3.13.7**, recorded in
`.python-version`. The initial supported Python range is intentionally limited to
3.13; other versions require a compatibility check before widening it.

The following installs the pinned `uv` tool into a local tooling environment, then
installs the package and development dependencies into a separate `.venv`. Neither
environment uses global site-packages or changes the global Python installation.

```sh
python3.13 -m venv .tools/uv
.tools/uv/bin/python -m pip install 'uv==0.13.9'
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
source .venv/bin/activate
python -c 'import transformer; print(transformer.__version__)'
```

`pyproject.toml` separates runtime dependencies (`torch`, `numpy`, `datasets`,
`pyarrow`, and `huggingface-hub`) from the `dev` group
(`pytest`, `ruff`, and `build`). `uv.lock` is the single resolved dependency lock;
keep it tracked. `uv sync --locked` checks that the lock agrees with project
metadata. Do not substitute an unpinned `pip install` for the documented setup.
The build backend and bootstrap tool are pinned as well. NumPy is included for
PyTorch's NumPy integration, which otherwise emits an initialization warning.

Runtime-only installation uses the same sync command with `--no-dev`. To restore
development tools, run the original sync command again. To leave the activated
environment, use `deactivate`.

The installation uses a regular wheel. Rerun the sync command after editing Python
source so the installed package reflects those changes. `--reinstall-package`
refreshes this project's wheel without upgrading locked dependencies. This avoids
a host issue observed during Phase 2: the macOS hidden flag was repeatedly applied
to the editable installation's `.pth` file, which Python 3.13 then skipped. No
global Python settings were changed.

## Diagnostics

With `.venv` activated:

```sh
transformer diagnose
transformer diagnose --device cpu --check
transformer diagnose --device mps --check
python -m transformer diagnose --device auto --json
```

The report includes Python and PyTorch versions, OS, architecture, interpreter,
environment isolation, MPS build support, MPS runtime availability, selected device,
and the selection reason. `--check` additionally executes a tiny matrix operation
and checks its gradient on the selected device. It is a smoke check, not a model
compatibility test or a performance benchmark. Without `--check`, tensor execution
is explicitly reported as `not_run`.

Device policies:

- `auto`: use MPS when available; otherwise use CPU and explain why.
- `cpu`: always select CPU, including on an MPS-capable machine.
- `mps`: require MPS; return an actionable error and exit status 2 if unavailable.

An unsupported device name also exits with status 2. CPU tests and simulated MPS
availability tests do not establish actual MPS execution. The real MPS test is
skipped when the local accelerator is unavailable; validation on the target
MacBook then remains pending.

## Configuration

`configs/default.json` contains the Phase 1 settings: schema version, device
preference, and project-relative data/run directories. Unknown keys, invalid types,
unsupported devices, absolute paths, and parent-directory traversal are rejected.
Use `/` in serialized paths. Loading configuration creates no data or run directories.

```sh
transformer diagnose --config configs/default.json --device cpu --save-config artifacts/phase1-config.json
```

`--device` overrides the configured preference. `--save-config` writes the effective
settings as readable JSON and **refuses to overwrite an existing file**; choose a
new filename when repeating the command. Snapshots preserve the requested device
policy, while the diagnostic report records the device actually selected.

Python callers can use `ProjectConfig.load(path)`, `config.to_dict()`,
`config.to_json()`, and `config.save(path)` from
`transformer.configuration.settings`. Resolve data/run locations with
`config.paths.resolve(project_root)` and an explicit root; paths do not depend on
the installation location or a hard-coded personal directory. Relative CLI input
and output filenames are resolved from the current working directory.

## Initial checks

```sh
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m build
```

Tests cover configuration errors and round trips, snapshot overwrite protection,
device policy branches, CLI execution, CPU tensor operations, and a separate real
MPS tensor check. Policy tests use simulated availability where needed; the real
MPS test uses the actual backend and is skipped if unavailable. Phase 4 adds tiny
model forward/backward checks on CPU and a separate real MPS model check, plus
architecture/input validation and initialization/weight-tying checks.

## Prepare bounded story data

```sh
.venv/bin/transformer prepare-data --config configs/data/phase2-small.json
.venv/bin/transformer prepare-data --config configs/data/phase2-small.json --offline
```

The configuration pins the authoritative TinyStories and English SimpleStories
repositories to exact revisions. It caps each source at 1,000 training-source and
200 official held-out records (2,400 total before cleaning). Selection is a prefix,
with documented ordering bias. Preparation preserves document boundaries, performs
global exact deduplication, and prevents acquired held-out stories from entering
training. Reruns verify checksums; compatible artifacts are reused and incompatible
or corrupt outputs produce an error without overwriting.

Outputs live in `data/prepared/phase2-stories-small-v2/`. Read
[source verification](docs/dataset_sources.md) and the
[preparation guide](docs/data_preparation.md) before expanding the subset. The
guide explains the training views, common evaluation files, Unicode policy, cache
recovery, and manifests. Test text is reserved for final evaluation and must never
be used for tokenizer fitting or training decisions.

Data tests use tiny labeled synthetic fixtures and mocked metadata. Verify the
suite without Hub access using:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
```

Build artifacts go into ignored `dist/`. Dependency updates are deliberate: edit
constraints if needed, run `.tools/uv/bin/uv lock --upgrade`, sync, rerun checks,
and record the new validation results. Do not upgrade the lock merely to run the
project.

## Shared tokenizer and token batches

```sh
.venv/bin/transformer prepare-tokens --config configs/tokenization/phase3-small.json
.venv/bin/transformer check-token-data --tokens data/tokens/phase3-shared-characters-v1 --loader-config configs/loading/default.json --stream train_tinystories --batches 2
.venv/bin/transformer check-token-data --tokens data/tokens/phase3-shared-characters-v1 --loader-config configs/loading/default.json --stream train --batches 2
```

One frozen vocabulary uses the union of allowed TinyStories and SimpleStories
**training** characters. Model A shares that vocabulary and loads only
`train_tinystories`; Model B loads the combined `train` stream. Validation and
test text never contribute vocabulary characters. Both experiments use the same
EOS boundary policy and common evaluation streams.

Token files and the frozen tokenizer live under
`data/tokens/phase3-shared-characters-v1/`. Batches are CPU `torch.int64` tensors;
the default context is 256, batch size 8, and worker count 0. The check command
only reads bounded batches. It does not create or train a model. Read the
[tokenization and loading guide](docs/tokenization_and_loading.md) for special
token semantics, unknown handling, packed story boundaries, short-stream policy,
fingerprints, and Python usage.

## Custom decoder-only model

```sh
.venv/bin/transformer check-model --config configs/model/tiny.json --device cpu
.venv/bin/transformer check-model --config configs/model/tiny.json --device mps
```

Each command runs one synthetic forward/backward check with no optimizer updates.
The common candidate in `configs/model/primary.json` has 4 layers, 4 heads,
256 embedding dimensions, a 256-token context, and tied embedding/output weights.
It has 3,245,056 trainable parameters with the current 78-ID shared vocabulary.
Model A and Model B must use the same architecture and initialization policy.
The model consumes the loader's shifted targets directly. Read the
[architecture guide](docs/model_architecture.md) for component responsibilities,
configuration, initialization, interface, and the scope of executed checks.

Phase 5's [correctness report](docs/model_correctness_report.md) explains the
independent references, numerical tolerances, results, and limitations. Run the
focused suite with:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/model_verification
```

Unavailable MPS checks are skipped. Architecture checks use random weights;
separate tiny training fixtures verify optimizer and recovery behavior. Neither
establishes language-generation quality.

## Configurable training

The [training guide](docs/training_pipeline.md) defines accumulation, scheduling,
validation, atomic checkpoints, and resume limitations. The example configuration
has not been used for a production run. Inspect it safely with:

```sh
.venv/bin/transformer train --config configs/training/primary-example.json --dry-run
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training
```

To start and later resume a **new example run** (potentially long on CPU; not
executed during documentation verification):

```sh
.venv/bin/transformer train --config configs/training/primary-example.json --stop-after-updates 100
.venv/bin/transformer train --config configs/training/primary-example.json --resume
```

The first command requires the example run ID to be unused and prepared token files
to exist. The second resumes the original 1,000-update schedule; it does not add a
new budget. This example is not the configuration that produced the reported runs.

Actual training requires an explicit configured budget and a new run ID; resuming
requires `--resume` with identical settings. Checkpoints store complete optimizer,
schedule, RNG, sampler, tokenizer, and dataset state. Only tiny synthetic training
checks were executed in Phase 6; no language-quality result is claimed.

## Bounded training diagnostics and benchmarking

Read the [benchmarking guide](docs/benchmarking.md) for timing definitions, resource
limits, and safe reruns. The [Phase 7 report](docs/phase7_benchmark_report.md) records
actual measurements and proposed settings. These commands do not launch baseline
training or evaluate language quality:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer training-smoke --config configs/benchmarking/smoke.json --output artifacts/phase7-smoke-v1
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer overfit-check --config configs/benchmarking/overfit.json --output artifacts/phase7-overfit-v1 --max-seconds 30
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer benchmark-training --config configs/benchmarking/m2-conservative.json
```

The documented artifact/run names are protected against reuse. For another run,
copy the configuration and select a new `output_dir` (benchmark) or `run_id` plus
`--output` (smoke). The synthetic overfit check needs only a new `--output`.
The benchmark produces an **unexecuted** proposed training configuration. The actual
paired runs use the separately frozen protocol below. Internal benchmark sampling
is limited to diagnostic text; public generation is documented under inference.

## TinyStories-only baseline

The [frozen experiment protocol](docs/phase8_experiment_protocol.md) fixes both
models' architecture, seed, shared tokenizer, 2,048,000-token budget, complete
validation set, and diagnostic prompts. The
[Model A report](docs/phase8_baseline_report.md) records actual results and artifacts.
Both paired runs are complete. This is the archived Model A launch command; its
existing run ID is protected and cannot be reused for a new experiment:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train-baseline --protocol configs/experiments/phase8-comparison.json
```

Recovery requires `--resume` with the unchanged protocol, document, code, data,
and environment. The saved Phase 8 training implementation predates later experiment
orchestration changes; current code intentionally refuses incompatible recovery.
Use the recorded implementation for historical continuation, rather than bypassing
that check. Completed artifacts load through the current inference service.
Completed runs are not extended. Planned stops at updates 100 and 500 archive
checkpoints and fixed greedy samples, then resume the original schedule to 1,000.
Latest/best checkpoints, configuration, hardware, validation history, samples,
and session timing are saved under `runs/phase8-model-a-tinystories-20261119-v1/`.
The wrapper requires the full prepared validation set and never evaluates test
loss. Diagnostic text may be repetitive or incorrect and is not factually reliable.

## TinyStories plus SimpleStories experiment

The [Phase 9 supplement](docs/phase9_experiment_protocol.md) preserves the original
paired controls and aggregate validation selector. It defines the target-token
mixture and separate source-only validation diagnostics. The
[combined-run report](docs/phase9_combined_report.md) contains actual exposure,
checkpoints, logs, and limitations. These are the archived preparation/launch
commands; both their output and Model B run already exist and are protected.
Preparation verifies provenance and scores retained Model A validation milestones:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train-combined --protocol configs/experiments/phase8-comparison.json --plan docs/phase9_experiment_protocol.md --output artifacts/phase9-preparation-v1 --prepare-only
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train-combined --protocol configs/experiments/phase8-comparison.json --plan docs/phase9_experiment_protocol.md --output artifacts/phase9-preparation-v1
```

The second command trains **only Model B**, from fresh weights matching Model A's
original initialization. Existing runs are protected; use `--resume` only with the
unchanged frozen settings/preparation. A completed run receives no extra updates.
The 8,000-window draw has no repetition and its exact target counts are verified
against saved sampler state. The aggregate selector remains 37.60% TinyStories /
62.40% SimpleStories by validation targets. Separate source-only losses have
different contexts and are not averaged to change selection. Formal held-out
comparison is documented below; one seed cannot establish a general improvement.

## Held-out evaluation and comparison

The [evaluation protocol](docs/phase10_evaluation_protocol.md) fixes the scoring
policy before test evaluation. The [comparison report](docs/phase10_comparison_report.md)
contains actual results, limitations and links to all raw outputs and figures.
Install the optional plotting dependencies and run from the project root:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer evaluate-comparison --config configs/evaluation/phase10-comparison.json
```

An existing evaluation output directory is protected. To reproduce scoring, copy
the JSON configuration and change only `output_directory` to a new relative path.
The saved primary configuration uses MPS; an explicit `cpu` configuration allows
CPU reproduction with expected numerical/timing differences. Evaluation loads
the existing **validation-selected best** checkpoints; it never trains or reselects.
Both run directories remain unchanged. Tiny synthetic tests require no network;
chart integration needs the optional `evaluation` extra (otherwise that test skips).

The primary metric preserves the original packed test-window policy and attributes
each target to its dataset. A separate document-isolated analysis uses overlapping
context while counting targets exactly once, including short tails and EOS. Mean
cross-entropy is token-weighted in nats; perplexity is its exponential and applies
to **character/special tokens**, not unrelated subword tokenizations. Fixed greedy
samples and synchronized generation timing use identical settings for both models.
Output includes raw metrics, per-story sums/counts, frozen identities, all samples,
raw timing trials, and PNG/SVG plots. Test results do not change checkpoint selection.

## Reusable inference

Use the [inference guide](docs/inference_backend.md) for the Python service, trusted
model registry, device/cache policy, sampling controls and edge cases. The
[Phase 11 report](docs/phase11_inference_report.md) records actual verification.
Only registered IDs are accepted; the selected checkpoint versions remain frozen.

```sh
.venv/bin/transformer list-models
.venv/bin/transformer generate --model transformer-story-continuation-v2 --device cpu --mode greedy --prompt "Tom found a small box under a tree." --max-new-tokens 64
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer generate --model transformer-story-continuation-v2 --device mps --mode sample --temperature 0.8 --top-k 20 --seed 42 --prompt "The little bird wanted to help its friend." --max-new-tokens 128 --json
```

The service caches one model, switches explicitly, runs without gradients, and
reports generation time separately from loading. These models still produce
repetitive text; backend functionality does not imply generation quality.

## Backend release gate

The [backend verification report](docs/backend_verification_report.md) records
403 passing tests, the isolated synthetic workflow, expected failure cases, and
separate verification of both real checkpoints on MPS. Fixtures cannot satisfy
the real-model requirement. Missing or incomplete training leaves the gate incomplete.

To repeat the gate from the project root, install the evaluation extra, run the
full suite, then choose a new output directory:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase12-full-suite.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/maintenance.py verify-backend --device mps --test-report artifacts/phase12-full-suite.xml --output artifacts/phase12-backend-gate-v2
```

The completed evidence uses `phase12-backend-gate-v1`; output directories cannot
be reused or placed outside `artifacts/`. The JUnit path is an operator-supplied
full-suite record: rerun tests against the installed revision before every gate.
Use `--device cpu` deliberately on a host without MPS; unavailable MPS tests remain
skipped. Gate exit code is zero only when all four required stages pass. A partial
JSON report is retained on failure. The fixture project and its three tiny training
runs stay under the selected output directory, separate from real experiments.

## Local generation interface

The Generate page uses one local Qwen2.5-based model fine-tuned on TinyStories and SimpleStories openings. See [story continuation training](docs/story_continuation_training.md).
Write an opening and choose
Greedy or Sampling (with temperature), then **Generate continuation**.
Output remains plain selectable text. Expand **Generation details** for checkpoint
identity, token count, sampling settings, stop reason and measured timing. The original
prompt remains separate.
Model insights reads saved run and evaluation evidence; Architecture shows the implemented decoder flow.

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --extra story --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m transformer.web --device auto --port 8000
```

Open `http://127.0.0.1:8000`. Stop with Ctrl-C. Use `--device mps` to require MPS,
`--device cpu` for CPU, or `--project-root /path/to/project` when launching elsewhere.
The launcher binds to loopback with one worker. Models load on the first generation;
only one model is cached. Another concurrent job receives a busy response. Closing
or navigating away does not cancel accelerator work; there is no streaming or Stop
control. Refreshing clears tab-local drafts/history and does not resubmit a request.

A missing/corrupt artifact gives a recovery message. Restore registered files rather
than replacing their hashes casually. Technical events/errors go to the rotating
`logs/web/inference.jsonl` log; the interface receives no raw tracebacks or checkpoint
paths. Normal completions log identifiers/counts/timings, not story text. No hosted
service, account, or Node runtime is needed. Backend-only installs can omit `web`.

Default API access is same-origin. An optional exact development origin can be set
with `--development-origin http://127.0.0.1:5173` or `http://localhost:5173`; other
browser origins cannot submit generation. This is a local application, not a public
hosted deployment. Requests accept registered model IDs only.

See [frontend integration](docs/frontend_integration.md) for the API, concurrency,
validation and executed end-to-end checks, and [design foundations](docs/frontend_design.md)
for the visual system.

```sh
.venv/bin/python -m pytest tests/web
node --test tests/web/*.test.mjs
```

Node 24.14.0 is needed only for developer JavaScript tests. Synthetic UI fixtures
remain isolated: `.venv/bin/python tools/maintenance.py preview-ui-fixtures --port 8001`, then
open `http://127.0.0.1:8001`. The fixture harness is excluded from the installed wheel
and never substitutes for normal generation.

## Structure

```text
main.py          Start the website
main/            Python CLI entry points
frontend/        HTML, CSS and JavaScript
backend/         Local web API
training/        Custom-model training code and scripts
fine_tuning/     Story-model fine-tuning scripts
data/            Data preparation code, stories, tokens and base models
model/           Neural network layers
inference/       Model loading and text generation
tokenization/    Character tokenizer
evaluation/      Evaluation calculations
configs/         Validated settings and experiment configurations
runs/            Trained checkpoints and saved run state
runtime/         Device selection and diagnostics
benchmarking/    Bounded hardware measurements
verification/    Integrated backend checks
tools/           Maintenance scripts
tests/           Automated tests
docs/            Guides and recorded results
artifacts/       Saved experiment evidence
```

Generated contents of `data/`, `runs/`, checkpoints, local logs, environments, and
caches are ignored. Python data-processing code remains trackable. Configuration under `configs/`, documentation, and `uv.lock`
remain trackable.

## Hardware and reproducibility

The target is a fanless MacBook Air M2 with 16 GB of shared system/GPU memory. A
successful tiny MPS check does not validate a complete Transformer or its memory
requirements. Later training must set an explicit run ID, device, step/token
budget, evaluation/checkpoint frequencies, and disk estimate before execution.
Measure throughput on the actual machine before estimating training time; allow
memory headroom for other applications and sustained thermal slowdown.

Dependency locks improve environment repeatability; they do not promise identical
numerical results across hardware. See [project progress](docs/project_progress.md)
for actual executed checks and limitations. Both bounded paired runs and their
held-out comparison, reusable inference backend, and local release gate are complete.
Phase 16 adds full generation controls, bounded session history and sequential model comparison.
One seed and small subsets limit conclusions;
work stops until the next phase prompt.

Compatibility references checked during setup:
[PyTorch installation](https://pytorch.org/get-started/locally/),
[PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html),
[PyTorch package metadata](https://pypi.org/pypi/torch/2.15.0/json), and
[uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

See [Phase 16 controls and verification](docs/generation_experience.md) for control semantics, session behavior and actual checks.


## Model insights and architecture (Phase 17)

Open **Model insights** for source-backed model settings, raw training and validation
curves, primary packed-test loss and character-token perplexity, measured training
duration, generation throughput and actual source exposure. **Architecture** shows
the configured pre-normalization decoder, including both residual branches.

Charts have hover values and keyboard-accessible raw tables. There is no smoothing.
Missing or malformed evidence is explained rather than replaced with zero. Paired
metrics require compatible saved artifacts. These single-seed experiments do not
establish general language quality.

```sh
.venv/bin/python tools/maintenance.py verify-analytics
```

This read-only audit checks the projections against real artifacts and writes
`artifacts/phase17-source-verification.json`. See
[analytics verification](docs/model_insights.md) for source mapping and limitations.

Phase 20 maintenance fixes malformed-log edge cases and avoids duplicate verified
JSON parsing within an analytics request. The measured projection median improved
from 23.03 ms to 16.44 ms; this is not a generation-speed claim.
See [maintenance evidence](docs/phase20_maintenance.md).


## Current limitations

The two small, single-seed runs demonstrate a functioning pipeline, not reliable
story quality. Repetition, malformed words and unfinished continuations are observed.
A 256-character-token context is much shorter than 256 words. Generated statements
are not guaranteed factual. Prefix subsets and exact-only deduplication limit
generalization; near-duplicates may remain.

There is no streaming, cancellation, public deployment, persistent user database or
multi-accelerator serving. History is tab-local and capped at ten entries. Browser
zoom, comprehensive keyboard/screen-reader and measured contrast verification remain
pending; the interface is not certified WCAG-conformant. Short resource tests do
not prove long-term leak freedom. Accelerator results can differ by runtime/device.
