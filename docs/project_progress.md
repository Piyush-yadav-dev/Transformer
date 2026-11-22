# Transformer project progress

## Phase 1 — project setup and environment configuration

Status: **complete — Phase 1 acceptance criteria passed**, verified 2026-11-18
(Asia/Kolkata). Scope is the installable backend foundation, validated configuration,
CPU/MPS selection, and environment diagnostics only.

### Workspace and host inspection

- The workspace was empty on inspection; no existing project or repository-specific
  `AGENTS.md` was present in the project or its parent directories.
- Host reports macOS 27.0.1, Darwin 27.0.0, arm64, model identifier Mac14,2,
  Apple M2, and 17,179,869,184 bytes (16 GiB) memory. This matches the target hardware.
- Base Python is 3.13.7 with no active virtual/Conda environment. Python framework
  and system Python executables are present; `uv`, Conda, pyenv, and Poetry were
  not found on PATH. No environments were found in the inspected common locations.
- Global PyTorch 2.14.1 reports MPS build support and runtime availability. These
  preliminary facts are not substituted for checks in the isolated project environment.
- No global packages were installed, upgraded, or removed.

### Implementation

- `pyproject.toml`: installable `Transformer` distribution, `transformer` package,
  src-layout, Python 3.13, runtime PyTorch/NumPy, separate development dependency group.
- `uv.lock`: the single resolved dependency record; `.tools/uv` holds the
  pinned local package manager and `.venv` holds project dependencies.
- Configuration: frozen validated objects, versioned JSON, strict unknown-key
  rejection, project-relative paths, and snapshots that refuse to overwrite files.
- Runtime: explicit CPU/MPS and automatic selection, selection reasons, and useful
  errors when required MPS is unavailable.
- Diagnostics: environment facts and optional tiny matrix/gradient verification.
- Future data, tokenization, model, training, evaluation, and inference directories
  contain responsibility notes only. No placeholder functions or classes were added.
- Tests cover configuration, CLI, device policies, and actual CPU/MPS operations.
- README, ignore rules, default configuration, and source-distribution manifest.

### Compatibility decisions and inspection commands

Executed: workspace `pwd`, `ls -la`, `rg --files --hidden`, checks for ancestor
`AGENTS.md`; `uname -srm`, `sw_vers`, `sysctl -n hw.model hw.memsize machdep.cpu.brand_string`,
Python/pip version checks, `whence -a` for Python/environment tools, Python
`platform`, `sys`, package metadata and environment-directory inspection.

The initial `command -v -a` form was not accepted by zsh; tool discovery was rerun
with `whence -a`. Python standard-library HTTPS metadata retrieval failed because
its default certificate store could not verify the issuer. Metadata inspection was
successfully retried with system `curl` using certificate verification; no TLS
verification was disabled or global certificates changed.

PyPI metadata was retrieved for torch, numpy, uv, pytest, ruff, setuptools, and build.
PyTorch 2.15.0 declares Python >=3.11 and provides
`torch-2.15.0-cp313-cp313-macosx_14_0_arm64.whl`. The other selected packages provide
compatible universal or Apple Silicon wheels. The project initially supports only
Python 3.13; widening that range requires further verification.

Sources: [PyTorch installation](https://pytorch.org/get-started/locally/),
[MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html),
[torch metadata](https://pypi.org/pypi/torch/2.15.0/json),
[uv lock/sync semantics](https://docs.astral.sh/uv/concepts/projects/sync/).

### Verification

Initial `.venv/bin/python -m pytest`: **56 passed, 1 warning**. The warning was
PyTorch's missing NumPy initialization warning; NumPy 2.5.5 was subsequently added
after verifying its Python >=3.12 requirement and CPython 3.13 Apple Silicon wheels.
Initial lint/format checks found import style and long-line/formatting issues;
these were corrected before the final pass.

Commands executed for installation and verification, from the project root:

```sh
python3.13 -m venv .tools/uv
.tools/uv/bin/python -m pip install 'uv==0.13.9'
.tools/uv/bin/uv lock
.tools/uv/bin/uv sync --locked --python python3.13
source .venv/bin/activate
python -c 'import transformer; print(transformer.__version__)'
transformer diagnose
transformer diagnose --device cpu --check
transformer diagnose --device mps --check
python -m transformer diagnose --device auto --json
transformer diagnose --config configs/default.json --device cpu --save-config artifacts/phase1-config.json
transformer diagnose --config configs/default.json --device cpu --save-config artifacts/phase1-verified-config.json
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m pytest
.venv/bin/ruff format .
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/ruff check --show-files
.venv/bin/python -m build
```

The lock and sync were run again after adding NumPy. Both snapshot filenames above
were new when written; repeating them intentionally fails rather than overwriting.
An additional Python snippet loaded `artifacts/phase1-verified-config.json` and
asserted its effective runtime setting was CPU. The tests execute additional CLI
subprocesses, including invalid devices, malformed/missing configuration, and
existing snapshot errors; simulated MPS availability is used only for policy tests.
Both environment `pyvenv.cfg` files were inspected and have
`include-system-site-packages = false`.

Final results:

| Check | Observed result |
| --- | --- |
| Isolated installation and import | Passed; Transformer 0.1.0 imports in `.venv` |
| Dependency resolution | Lock current; 40 packages recorded across platform markers, 20 installed on this Mac |
| Dependency compatibility | `uv pip check`: all installed packages compatible |
| Base / project PyTorch | Global 2.14.1 untouched; isolated project uses 2.15.0 |
| Other primary resolved versions | NumPy 2.5.5, pytest 9.2.0, Ruff 0.17.5, build 1.6.1, setuptools 84.2.1; bootstrap uv 0.13.9 |
| Automatic selection | MPS selected; build support and runtime availability both true |
| Explicit CPU | CPU selected; real tensor and gradient check passed |
| Explicit MPS | MPS selected; real tensor and gradient check passed on the detected M2 Mac |
| Unavailable MPS / invalid device | Policy and CLI tests passed; explicit MPS raises rather than falling back |
| Configuration | Invalid inputs rejected; round trip, overrides, path resolution and overwrite protection passed |
| Final test suite | **56 passed in 9.39 seconds, zero warnings, zero skips** |
| Ruff lint and format | Both passed after fixes |
| Packaging | Wheel and source distribution built successfully, with wheel built from the source distribution |
| README commands | Installation, activation, diagnostics, snapshot writing, and listed initial checks executed successfully |

The built files are `dist/transformer-0.1.0-py3-none-any.whl` and
`dist/transformer-0.1.0.tar.gz`. They are local ignored build outputs. The source
distribution includes the lock, configuration, tests, documentation, and future
component notes; the wheel contains the implemented Python package only.

### Remaining limitations

- No unresolved Phase 1 test or installation failure remains.
- Python 3.13.7 / macOS arm64 is the executed verification environment. Other
  Python versions, operating systems, and Intel Macs have not been validated.
- MPS build/runtime availability and the tiny tensor/autograd check passed on
  this host. The future complete model and training workloads are **not** MPS
  validated by these checks. No throughput, memory-capacity, or thermal benchmark
  was run.
- The base Python standard-library HTTPS certificate-store issue described above
  remains a host condition. It did not block pip/uv installation or isolated builds;
  it was not repaired by changing the global environment.
- This directory was not an existing Git repository, and no repository was
  initialized. Source/configuration/documentation and lock files are ready to track;
  `.gitignore` defines the intended artifact exclusions.

### Boundaries and next phase

No dataset was downloaded, no neural network or frontend was implemented, and no
training was run. A tensor smoke check is not model validation, a training run, or
a throughput measurement. Full model MPS support, memory sizing, and training speed
remain unmeasured. No performance or quality claim is made.

Next expected work: the user's next phase prompt, likely bounded dataset preparation
and character-level tokenization. SimpleStories identity must be verified before
use. Do not begin another phase automatically.

## Phase 2 — dataset acquisition and preprocessing

Status: **complete — Phase 2 acceptance criteria passed**, verified 2026-11-18
(Asia/Kolkata). Phase 1 implementations and records were preserved. The master
specification, project files, and ancestor/project instruction locations were
inspected before this phase; no additional repository instructions were found.

### Implemented files and decisions

- `configuration/data_config.py` and `configs/data/phase2-small.json`: validated
  source IDs, full commit revisions, configurations, text fields, official split
  limits, source seeds, derived split fractions, and a raw-story size limit.
- `data/preparation.py`: metadata verification before acquisition,
  deterministic bounded prefixes, synchronous column-only Parquet batches,
  explicit early-stop closure, and reusable checksummed raw subsets.
- `data/preparation.py`: LF/NFC normalization and outer trimming, preserving
  case, punctuation, paragraphs, and meaningful Unicode; explicit reject reasons.
- `data/preparation.py`: disk-backed global duplicate/provenance index,
  story-level splits, held-out precedence, training views, counts, lengths, source
  cards/revisions, checksums, implementation identity, and environment manifests.
- `data/preparation.py`: atomic publication, locks, integrity checking,
  corruption errors, and protection against accidental output replacement.
- `transformer prepare-data`: uses Phase 1 project-relative path conventions;
  supports verified cache reuse and explicit offline operation.
- `tests/test_data_preparation.py` and `tests/fixtures/stories/`: synthetic offline
  fixtures and metadata mocks, including empty/malformed records and cross-source
  duplicates. These fixtures are not real training data or model outputs.
- README, `docs/dataset_sources.md`, `docs/data_preparation.md`, package metadata,
  lockfile, and source-distribution fixture inclusion were updated.

SimpleStories identity is resolved to the official English
`SimpleStories/SimpleStories`. The author's older namespace redirects to the same
repository/revision. The paper/card and namespace evidence distinguish it from
the Japanese dataset and similarly named derivatives. Exact sources, configurations,
licenses, revisions, available splits, and verification links are recorded in
[dataset sources](dataset_sources.md). No dataset substitution was made.

### Executed preparation and results

The configured budget is 1,000 training-source records plus 200 official held-out
records from each dataset, seed 42. Records are a pinned prefix; rejects/duplicates
are not replenished. TinyStories' official validation is divided by content hash
into validation/test; SimpleStories' official test stays test-only and validation
is derived from its training subset. Global conflict priority is test, validation,
then training. The common evaluation files are reserved for both later experiments.

Final artifacts: `data/prepared/phase2-stories-small-v2/`.

| Final file / check | Actual result |
| --- | --- |
| `train.jsonl` | 1,894 unique stories |
| `train_tinystories.jsonl` | 1,000 TinyStories training members |
| `train_simplestories.jsonl` | 894 SimpleStories training members |
| `validation.jsonl` | 207 stories (101 TinyStories, 106 SimpleStories) |
| `test.jsonl` | 299 stories (99 TinyStories, 200 SimpleStories) |
| Raw records acquired | 2,400, within all configured limits |
| Actual rejected records | 0 in this bounded subset |
| Actual exact duplicate removals | 0 in this bounded subset |
| Prepared directory logical size | 5,696,431 bytes, including training views and manifest |
| Corpus fingerprint | `00fa2a2993f66da7d04f6de09944c61dc2c075b012f0bbf02b5a3723a5d1903c` |
| Offline rerun | `reused: true`, same fingerprint |
| Final unit/integration suite | **87 passed in 9.15 seconds; zero warnings, zero skips**, with Hub/datasets offline flags |
| Lint, formatting, lock and installed dependencies | All checks passed |
| Phase 1 diagnostic regression | Explicit CPU tensor/gradient check passed; real MPS test also passed in the suite |
| Packaging | Phase 2 wheel and source distribution built; archived code, configuration, fixtures, and documentation checked against current files |

An independent audit recomputed all story IDs and file SHA-256 checksums, checked
counts/limits, asserted train/validation/test disjointness, verified no held-out
origin appears in training, and checked that the union of source training views
matches the combined training file. The final version 2 corpus has exactly the
same data fingerprint as the initial audited version 1 output.

The synthetic fixture tests actually exercise three duplicate removals, including
two cross-source occurrences, and four malformed/empty-record rejections. Zero
removals in the real subset is not evidence that the upstream datasets are globally
duplicate-free or clean.

### Commands and verification history

Source inspection used web source cards/papers and certificate-verified `curl`
requests to Hub and dataset-server APIs, followed by pinned metadata checks through
the installed dataset library. PyPI metadata confirmed `datasets` 5.2.0 supports
Python >=3.10; dependency resolution installed compatible Python 3.13 packages.
The lock now contains 60 packages across platform markers, with 39 installed here.
Relevant resolved additions are datasets 5.2.0, PyArrow 26.0.1, and
huggingface-hub 2.4.0.

Executed commands included:

```sh
.tools/uv/bin/uv lock
.tools/uv/bin/uv sync --locked --python python3.13
.venv/bin/python -m pytest tests/test_data_preparation.py
.venv/bin/ruff format src/transformer tests
.venv/bin/ruff check .
.venv/bin/python -m pytest
/usr/bin/time -l .venv/bin/transformer prepare-data --config configs/data/phase2-small.json
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/transformer prepare-data --config artifacts/phase2-reader-check.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/transformer prepare-data --config configs/data/phase2-small.json --offline
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/transformer diagnose --device cpu --check
.venv/bin/python -m build --outdir dist/phase2
```

The offline final preparation command was executed twice: first it created version
2 from verified raw caches; the second reused the prepared artifact. Additional
Python audit snippets checked JSONL counts/hashes/provenance and compared manifests.
No held-out text was used to select modeling/tokenization settings.

Issues found and resolved during this phase:

1. TinyStories' library metadata exposes split dictionaries without upstream row
   counts, whereas SimpleStories provides split objects. The verifier now handles
   both and records unknown counts as `null`; mocked regression tests cover this.
2. The initial `datasets` streaming reader completed all data files but hung during
   Arrow thread-pool shutdown. A macOS `sample` of the process located the wait in
   `arrow::internal::ThreadPool::Shutdown`. The process was terminated after its
   outputs were independently verified; that initial command is **not** recorded
   as a successful exit. The final reader uses synchronous `ParquetFile` batches
   with explicit closure. A fresh additional bounded check (16 records per each
   of four official source splits, 64 total) exited successfully in 35.14 seconds
   under a 60-second subprocess timeout. Its artifacts are separately labeled
   `data/prepared/phase2-reader-check/`; they are not the final corpus. This was a
   functional acquisition check, not a training or throughput benchmark.
3. The editable package's `.pth` repeatedly acquired macOS `UF_HIDDEN`, causing
   Python 3.13 to skip it and report an import failure. Clearing the flag with
   `chflags nohidden` restored import only temporarily. Installation now uses a
   regular local wheel with the explicit sync command above; imports, CLI checks,
   real acquisition, and all tests then passed. README explains that source edits
   require resync. No global Python environment was modified.

The initial `phase2-stories-small-v1` output is preserved for audit. The final
configuration uses `phase2-stories-small-v2` because implementation identity changed;
no existing prepared artifact was overwritten. Local logs, source caches, and
process-sampling evidence are under ignored `artifacts/` and `data/` directories.

### Limitations and next expected phase

No unresolved Phase 2 verification failure remains. Prefix selection has ordering
bias, exact hashes do not detect near-duplicates, and leakage checks cover acquired
subsets rather than the unacquired full repositories. Parquet pages/range buffers
can transfer more bytes than the retained record count implies. Declared dataset
licenses and full cards are recorded; neither pinned source includes a separate
license file. Anonymous Hub access worked but emitted a rate-limit advisory.

No tokenizer, token windows, model, training, generation, or frontend was implemented.
Test stories remain reserved for final evaluation. Next expected work is the user's
next phase prompt, likely character-level tokenizer fitting and token preparation
using training data only. Do not start that phase automatically.

## Phase 3 — character tokenization and efficient token loading

Completed on 2026-11-18 on the existing Apple M2 macOS host, using the isolated
Python 3.13.7 environment. Scope: freeze the common character tokenizer, encode
the selected Phase 2 corpus, and provide validated CPU next-token batches. No
model, training loop, new dataset acquisition, or frontend was implemented.
Existing Phase 1/2 files and prepared artifacts were preserved.

### Implementation and policy

- `tokenization/character_tokenizer.py`: immutable deterministic Unicode-code-point
  vocabulary, encode/decode, explicit unknown and EOS IDs, versioned JSON save/load,
  and SHA-256 fingerprint. Unknown characters are replaced individually or rejected
  in strict mode, never silently dropped. Literal special-token labels remain text.
- `tokenization/token_preparation.py`: fit only the two allowed source training
  views, stream stories into compact token files, write story offsets and manifests,
  validate dtype capacity, and publish atomically with verified reuse. The selected
  corpus fits in uint8; larger vocabularies can use uint16/uint32.
- `data/loading.py`: read-only CPU memory maps, independent
  int64 input/target tensors, fixed context windows, integrity/fingerprint checks,
  seeded batching, and worker-safe reopening. Default workers: zero; supported
  nonzero workers use spawn with one prefetched batch and no persistent workers.
- `configuration/token_data_config.py`, `configs/tokenization/phase3-small.json`,
  and `configs/loading/default.json`: validated preparation and loader settings,
  with separate identities so batch changes do not invalidate encoded artifacts.
- CLI commands `prepare-tokens` and `check-token-data`; offline synthetic tests in
  `tests/test_character_tokenizer.py` and `tests/test_token_data.py`; README and
  `docs/tokenization_and_loading.md` explain usage, semantics, and limitations.

The shared vocabulary uses the union of TinyStories and SimpleStories **training**
characters. Model A shares that definition but will consume only TinyStories
training text; Model B will consume the combined training stream. Validation and
test records were never used to discover characters. Source file checksums,
dataset revision/license metadata, tokenizer identity, and corpus identity are
recorded with the tokens. No dependencies or lock versions changed this phase.

Both experiments use packed stories with one EOS ID after each story. A window
may cross an EOS within its stream; it never crosses a train/validation/test file
boundary. EOS does not reset future model attention. With context C, each item
reads C+1 tokens and returns the first C as input and the last C as target. Short
tails are dropped; streams shorter than C+1 fail with a clear error, without padding
or concatenating splits. Empty held-out streams can be prepared but not loaded as
training examples. Hinglish comments explain shifting and boundary decisions near
the relevant logic.

### Actual prepared artifacts and checks

Input: `data/prepared/phase2-stories-small-v2/`.
Output: `data/tokens/phase3-shared-characters-v1/`.

| Stream | Stories | Tokens including EOS | Context-256 windows | Dropped tail pairs |
| --- | ---: | ---: | ---: | ---: |
| Combined training | 1,894 | 2,069,638 | 8,084 | 133 |
| TinyStories training | 1,000 | 942,640 | 3,682 | 47 |
| SimpleStories training | 894 | 1,126,998 | 4,402 | 85 |
| Validation | 207 | 214,634 | 838 | 105 |
| Test | 299 | 323,194 | 1,262 | 121 |

- Vocabulary: **78 IDs**: 76 characters, unknown ID 0, EOS ID 1. Actual unknown
  counts are zero in these bounded streams; synthetic tests exercise unknowns.
- Storage: uint8 tokens, little-endian uint64 story offsets; 4,719,319 logical bytes
  for the complete directory, including the duplicate training views and metadata.
- Tokenizer fingerprint:
  `9795b7a09fd860a748ed7bd5df3054ed84e6bf5efd3420899cdd65120866d480`.
- Dataset fingerprint:
  `00fa2a2993f66da7d04f6de09944c61dc2c075b012f0bbf02b5a3723a5d1903c`.
- Token-data fingerprint:
  `19531b63399d0ca59036c51ec566b803b9d314921a89b56d06471b344c1c9687`.
- Initial preparation succeeded; the identical rerun returned `reused: true` with
  unchanged fingerprints. Phase 2's offline preparation also reused successfully.
- CLI checks inspected two batches each from TinyStories-only and combined training:
  input and target shapes `[8, 256]`, CPU device, `torch.int64`.
- An independent Python audit reconstructed the training-only character union,
  decoded every stored story against its prepared text, checked offsets and EOS,
  and checked shifted and reproducible seeded batches for all five streams. Audit
  results are retained in ignored `artifacts/phase3-audit.json`.
- Full offline suite: **128 passed in 20.09 seconds**, zero warnings and zero skips.
  This includes earlier phase regressions, held-out-only character exclusion,
  tokenizer round-trips/order/save-load, special and unknown behavior, split
  isolation, exact shifting, incompatible fingerprints, empty/short streams, dtype
  overflow, corrupted artifacts, reproducible reuse, and an actual spawned worker.
- Ruff lint/format checks, dependency lock check, and installed dependency check
  passed. The lock has 60 resolved packages; 39 are installed on this host.
- Phase 3 wheel and source distribution built successfully under `dist/phase3/`.
  Archive contents were checked against source modules, configurations, documents,
  and tests. The final documentation update was included in a subsequent rebuild.

### Commands actually executed

```sh
.venv/bin/ruff format src/transformer tests
.venv/bin/ruff check .
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/transformer prepare-data --config configs/data/phase2-small.json --offline
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/transformer prepare-tokens --config configs/tokenization/phase3-small.json
.venv/bin/transformer check-token-data --tokens data/tokens/phase3-shared-characters-v1 --loader-config configs/loading/default.json --stream train_tinystories --batches 2
.venv/bin/transformer check-token-data --tokens data/tokens/phase3-shared-characters-v1 --loader-config configs/loading/default.json --stream train --batches 2
.venv/bin/python -m build --outdir dist/phase3
```

`prepare-tokens` was executed twice. Additional local Python snippets inspected
the installed loader signature and independently audited the resulting files;
they did not tune settings using held-out content. Local command results are in
ignored `artifacts/phase3-*.json`. Installation remains a regular wheel because of
the host's editable-install issue documented in Phase 2.

### Limitations and next expected phase

No unresolved Phase 3 functional check remains. Results describe only the selected
bounded corpus. Unknown characters in future text cannot be losslessly decoded.
Packed windows permit attention across stories, and trailing pairs are excluded;
future evaluation must report the actual evaluated-token denominator. Each dataset
construction verifies checksums with bounded reads. Shuffled loading holds a CPU
permutation of window indices; sequential loading avoids that allocation. No
throughput tuning or training-memory benchmark was performed.

Phase 3 acceptance criteria passed. Token loading stays on CPU; this is not a
validation of a complete model on MPS. Next expected work is the user's next phase
prompt for custom model components. Stop here until that prompt arrives.

## Phase 4 — custom decoder-only neural network

Completed on 2026-11-18 in the existing macOS 27.0.1 arm64 / Apple M2 environment,
using isolated Python 3.13.7 and PyTorch 2.15.0. MPS build support and runtime
availability were both confirmed. The master specification and earlier progress
records were read first; no repository-specific AGENTS.md was found. The workspace
is not a Git checkout. Existing data/token artifacts and dependency versions were
preserved.

### Implemented files and decisions

- `configuration/transformer_config.py`: frozen architecture settings, strict input
  validation, versioned JSON loading/snapshots, and overwrite protection.
- `model/transformer.py`, `causal_self_attention.py`,
  `layer_normalization.py`, `feed_forward.py`, `transformer_block.py`, and
  `language_model.py`: learned token/position embeddings, explicit query/key/value
  projections, scaled and future-masked multi-head attention, output projection,
  custom population-variance normalization, GELU feed-forward layers, dropout,
  pre-normalization residual blocks, final normalization, and vocabulary logits.
- `TransformerOutput` always contains logits `[B,T,V]` and optional scalar mean
  cross-entropy. The model consumes already-shifted loader targets directly, with
  no additional shift. Empty/overlong inputs, invalid IDs/dtypes/devices, and
  mismatched targets fail clearly. No padding/ignored target policy was introduced.
- `configs/model/primary.json`: 4 layers, 4 heads, embedding/context 256,
  feed-forward expansion 4, dropout 0.1, vocabulary 78, epsilon 1e-5.
  `configs/model/tiny.json`: 2 layers, 2 heads, embedding 16, context 8, expansion 2.
- Both configs tie token embeddings to the bias-free vocabulary projection.
  Every unique embedding/linear weight receives final normal initialization with
  mean 0 and standard deviation 0.02; linear biases are zero, normalization scales
  one and biases zero. Shared weights receive that initialization once. Tying is
  configurable; parameter reporting counts unique trainable parameters only.
- `runtime/device.py` and the `check-model` CLI run one synthetic batch with
  backward propagation, but no optimizer or parameter update. They report device,
  seed, architecture, shapes, finite-gradient status, parameter counts, and tying.
- `tests/test_language_model.py`, README, the model responsibility note, and
  `docs/model_architecture.md` cover the interface, initialization, policies, and
  phase boundaries. Hinglish comments explain reshaping, scaling, masking,
  residual shapes, and avoiding a second target shift.

No ready-made Transformer, attention, or normalization module is used. The primary
experiments must use this same architecture/initialization policy and shared frozen
tokenizer, with newly initialized models. Model A retains its TinyStories-only
training stream. EOS does not reset attention or positions inside a packed window.

### Commands and actual verification

```sh
.venv/bin/ruff format src/transformer tests
.venv/bin/ruff check .
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest tests/test_language_model.py
.venv/bin/transformer check-model --config configs/model/tiny.json --device cpu
.venv/bin/transformer check-model --config configs/model/tiny.json --device mps
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/transformer prepare-tokens --config configs/tokenization/phase3-small.json
```

| Check | Actual result |
| --- | --- |
| New model tests | **52 passed in 3.54 seconds** |
| Complete offline regression suite | **180 passed in 19.34 seconds**, zero warnings/skips |
| Tiny CPU and real MPS CLI checks | Both passed; inputs `[2,8]`, logits `[2,8,78]`, finite scalar loss and present finite gradients for every parameter |
| Tiny parameter count | 5,856 trainable scalars across 36 unique parameter tensors; tied on CPU and MPS |
| Primary candidate parameter count | 3,245,056 trainable scalars, tied weights counted once |
| Existing training-loader integration | One CPU tiny-model forward/backward batch each from `train_tinystories` and `train`, both passed |
| Phase 3 artifact regression | `reused: true`, unchanged tokenizer/dataset identities |
| Installation, Ruff, lock, dependencies | Passed; 60 locked packages / 39 installed, no dependency changes |
| Optimizer updates | **Zero** |

Additional local Python checks compared both configs' vocabulary size to the frozen
tokenizer, instantiated the primary model to count parameters (no primary-model
forward pass), and connected the existing memory-mapped loader to the tiny model.
Those checks used only training streams and did not inspect held-out text. Reports
are in ignored `artifacts/phase4-model-cpu.json`, `phase4-model-mps.json`,
`phase4-data-integration.json`, and `phase4-token-regression.json`.

The synthetic seed-42 CPU/MPS losses were approximately 4.351218 and 4.343973.
These are random-weight execution diagnostics with dropout, not evaluation scores
or evidence of learned text quality; equality across devices was not required.
Initial lint found two long lines, which were corrected. Format checking also
found a pre-existing Phase 3 documentation code block requiring line breaks; only
that formatting was changed, and the final check passed.

### Limitations and next expected phase

Phase 4 acceptance criteria passed with no unresolved functional failure. Only
tiny float32 forward/backward execution has been checked on CPU/MPS. The primary
architecture's full-context batch memory, throughput, mixed precision, and sustained
training stability are unmeasured. ID validation synchronizes devices when checking
ranges; explicit attention materializes quadratic score tensors. No speculative
performance tuning was introduced.

Comprehensive mathematical validation remains Phase 5: independent attention and
normalization references, causal-independence tests, numerical gradient checks,
and broader numerical/device validation. No training loop, generation, checkpoint
workflow, or frontend was implemented. Stop after Phase 4 and wait for that prompt.

## Phase 5 — mathematical and architectural verification

Completed on 2026-11-18 after inspecting the existing model, configuration, tests,
master specification, and progress records. Environment: the same Apple M2,
macOS 27.0.1 arm64, isolated Python 3.13.7, PyTorch 2.15.0. MPS build/runtime support
were confirmed, and test processes explicitly disabled MPS-to-CPU fallback.

### Changes and findings

- Added `tests/model_verification/`: seeded deterministic CPU fixtures; an independent
  NumPy prefix/head attention reference; float32/float64 normalization and derivative
  comparisons against PyTorch; bounded component numerical gradient checks; causal
  prediction, aligned loss, dropout mode, shape, serialization, and actual MPS tests.
- Negative controls show the fixtures detect unscaled/unmasked attention, reversed
  head order, wrong epsilon, future-token leakage, and an extra target shift.
- Both tied/untied models are covered. Shared embedding gradients equal the sum of
  untied embedding/output contributions. Configuration-plus-weight reconstruction
  preserves evaluation logits exactly and retains the intended parameter tying.
- Existing Phase 4 tests continue to cover malformed inputs and configuration errors.
  The actual primary configuration additionally passed a CPU forward pass with
  `[1,256,78]` logits at full context; this was not a training benchmark.
- Created `docs/model_correctness_report.md` with the test-to-contract mapping,
  tolerances, reproducibility commands, and limitations. Updated README and model
  documentation to reflect the completed verification.

**No production-model defect was revealed by these tests.** Model Python code,
initialization, dependency versions, and existing data/token artifacts were left
unchanged. Tolerances were selected before execution and were not widened after
failures. The only initial issue was one overlong test docstring caught by Ruff;
it was shortened before executing tests.

### Executed checks

```sh
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/ruff format tests/model_verification
.venv/bin/ruff check tests/model_verification
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/model_verification --junitxml=artifacts/phase5-correctness.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase5-full-suite.xml
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

| Check | Actual result |
| --- | --- |
| Focused correctness suite | **38 passed in 3.84 seconds** (37 CPU, 1 actual MPS case) |
| Full regression suite | **218 passed in 19.90 seconds**, zero warnings/skips/failures |
| MPS evidence | All three real MPS checks in the full suite passed with fallback disabled; new coverage compares CPU/MPS logits, loss, every parameter gradient, and causal prefixes |
| Numerical gradients | Attention, custom normalization, and feed-forward inputs/parameters passed float64 central-difference checks on `[1,3,4]` inputs |
| Serialization | Exact CPU evaluation-logit equality for tied and untied synthetic weights |
| Lint, formatting, install, dependencies | Passed; 60 locked packages / 39 installed, no changes to dependencies |
| Training/optimizer updates | **None / zero** |

Additional Python commands inspected installed API signatures, recorded host/MPS
availability, and summarized JUnit results with model/configuration/test SHA-256
hashes in ignored `artifacts/phase5-verification.json`. JUnit XML evidence is kept
alongside it. Saved weight files are temporary synthetic test artifacts, not
trained checkpoints. Tests needed neither internet access nor corpus acquisition.

### Status and limits

Phase 5 acceptance criteria passed; no unresolved test failure remains. The report
states numerical tolerances and the conditional skip policy for hosts without MPS.
These checks establish the tested causal and numerical contracts, not language
quality, convergence, mixed-precision safety, sustained training stability, or a
full-model training resource budget. No training/recovery system or frontend was
started. Wait for the user's next phase prompt before proceeding.

## Phase 6 — configurable training pipeline and recovery

Completed in the existing Apple M2 / macOS 27.0.1 arm64 environment with isolated
Python 3.13.7 and PyTorch 2.15.0. The verified model, token loader, prior progress,
configuration/path conventions, and repository instructions were inspected first.
No production training was started. All executed optimizer updates belonged to
tiny synthetic verification runs, with at most four updates per run.

### Implementation and decisions

- `configuration/training_config.py`: immutable run ID, relative paths, three data
  fingerprints, embedded model configuration, device/seed, microbatch/accumulation,
  explicit update-or-token budget, optimizer/clipping/schedule, validation/checkpoint
  intervals, and strict input validation. Token budgets must fit whole context windows.
- `training/engine.py`: AdamW with conventional betas/epsilon, decay on matrix
  weights only, token-weighted accumulation, one clipped optimizer step per cycle,
  finite loss/gradient/parameter/state checks, and fixed-budget warmup/cosine LR timing.
- `data/loading.py`, `random_state.py`, and `validation.py`: resumable CPU window
  permutations; Python/NumPy/CPU/MPS RNG state; explicit float32 and CPU-thread
  settings; inference-mode, token-weighted validation with restored model mode.
- `training/engine.py`: same-directory temporary writes, flush/fsync and
  atomic replacement, one latest/best index publication, checkpoint hashes, exclusive
  run lock, and archived recovery of uncommitted JSONL metric tails.
- `training/engine.py`: fresh-run protection, strict resume identity/state
  checks, safe update boundaries, deferred SIGINT, explicit temporary session stops,
  JSONL metrics, readable console output, and source/config/environment snapshots.
- CLI `train` supports `--dry-run`, `--resume`, `--stop-after-updates`, project root,
  and the existing project path configuration. Added an unexecuted pinned example
  at `configs/training/primary-example.json` and detailed `docs/training_pipeline.md`.
  Updated README and the training responsibility note.

Training uses zero data workers and float32; unsupported prefetch/mixed-precision
state is not silently accepted. The original schedule budget cannot change on
resume. Validation uses a bounded fixed prefix of the validation stream and never
evaluates test data. CPU exact-resume evidence covers dropout and epoch crossings;
MPS generator state is retained, but bitwise MPS equivalence is not promised.

### Executed verification and findings

| Check | Actual result |
| --- | --- |
| Final focused training suite | **50 passed in 5.05 seconds**, including actual MPS checkpoint/resume |
| Final complete regression suite | **268 passed in 23.56 seconds**, zero warnings/skips/failures; all four real MPS cases passed with fallback disabled |
| CPU recovery | Exact equality of uninterrupted/resumed model weights, optimizer/scheduler/sampler/RNG state, loss/LR/gradient-norm history, and best validation records; timing fields excluded |
| Accumulation | Unequal microbatches and a partial final cycle match a large-batch reference; clipping and optimizer step happen once; intentionally incorrect equal-batch weighting differs |
| Validation | Inference/no-gradient mode observed, token-weighted aggregation includes the smaller final batch, previous mode and existing gradients preserved |
| Recovery faults | Missing/invalid optimizer state and incompatible settings rejected; simulated index-publication failure preserved the old checkpoint; uncommitted metrics archived; concurrent writers blocked |
| Interruptions | Actual SIGINT finished and saved one complete cycle, then resumed exactly; unexpected KeyboardInterrupt preserved only the last committed state |
| Installation / Ruff / lock / dependency checks | Passed; installed training source matches workspace; 60 locked / 39 installed packages, no dependency changes |
| Production corpus command | Dry-run only; **no `runs/model-a-example` directory created** |

A deliberate recovery test exposed an initial defect: PyTorch accepted an empty
optimizer history at a nonzero update, silently resetting AdamW moments. That
regression failed before the fix. Explicit validation now requires complete moment
tensors, matching options/parameter order, finite compatible dtypes/shapes, correct
step counts, and LR agreement. The regression and related corruption cases pass.
Training also fixes float32 explicitly rather than inheriting an ambient default
dtype, and restores the caller's settings afterward.

Fixture review replaced repeated identical windows with varied token sequences,
and added negative controls so incorrect equal weighting cannot pass the
accumulation/validation comparisons. Final results above use those stronger fixtures.

The production-example dry-run reported 1,000 scheduled updates / 2,048,000 target
tokens, 2,048 tokens per full update, 3,682 TinyStories training windows, and 838
validation windows. It counted 3,245,056 trainable parameters and estimated
38,970,128 bytes per checkpoint with up to three normal publication copies,
excluding overhead/logs/orphans. This was not a training-memory or throughput test.

Commands actually executed included:

```sh
.venv/bin/ruff format src/transformer tests/training
.venv/bin/ruff check .
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest tests/training/test_recovery.py::test_missing_optimizer_history_is_not_silently_reset
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training --junitxml=artifacts/phase6-training.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase6-full-suite.xml
.venv/bin/transformer train --config configs/training/primary-example.json --dry-run
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

Source edits were reinstalled before execution. Additional local Python commands
inspected installed APIs, created the unexecuted example configuration, and recorded
environment, JUnit counts, dry-run evidence, and source hashes in ignored
`artifacts/phase6-verification.json`. CLI stop/resume commands were executed only
against temporary synthetic fixtures by tests. JUnit and dry-run JSON are in
`artifacts/phase6-*.xml` and `artifacts/phase6-dry-run.json`.

### Status and limitations

Phase 6 acceptance criteria passed with no unresolved functional failure. Resume
requires the same recorded code/configuration/data/environment; no cross-device
migration or training-budget extension is silently permitted. Validation prefixes
have ordering bias. Training throughput and full-context backward memory on the
target machine still need a separately authorized budgeted measurement. Mixed
precision, multiprocessing prefetch, and distributed training are out of scope.

SIGINT is deferred to a safe boundary; abrupt termination/power loss cannot recover
beyond the last completed atomic publication. No checkpoint recovery claim is made
for unsaved updates. Tests establish infrastructure behavior, not learned language
quality or sustained production stability. Stop here and wait for the user's next
phase prompt; do not start a production run automatically.

## Phase 7 — bounded training diagnostics and hardware benchmarking (2026-11-19)

### Scope and implementation

Inspected the existing Phase 1–6 project, package/configuration conventions,
training/recovery code, prepared tokens, and progress records. Preserved existing
work and dependencies; the directory remains a non-Git workspace. Added three
separate commands: `training-smoke`, `overfit-check`, and `benchmark-training`.
No full baseline, dataset acquisition, tokenization, or frontend work was started.

New files include `configuration/benchmark_config.py`, the `benchmarking/` package,
three `configs/benchmarking/*.json` files, `tests/training/test_benchmarking.py`,
`docs/benchmarking.md`, and `docs/phase7_benchmark_report.md`. README now documents
Phase 7. The generated proposal is also recorded at
`configs/training/phase7-m2-proposed.json`, with a new, unexecuted run ID.

The training loop exposes observation after complete optimizer boundaries,
including final checkpoint I/O. Resident/cached inputs and normal mmap batching
share the same token-weighted optimizer implementation. Tests verify identical
updates with unequal microbatches and reject incomplete input cycles. Boundary
observation handles deferred SIGINT without emitting a duplicate update. Existing
CPU exact-resume and real MPS recovery checks remain passing.

The smoke command runs a small model against the prepared TinyStories view,
stops/saves/loads/resumes, validates, reloads, and produces at most 48 greedy tokens.
This is minimal internal diagnostic sampling, not a public inference layer. The
overfit command uses one explicit synthetic example, no dropout, fixed success
criteria, and update/time caps. Failures preserve their trace and return a failure
status. The benchmark uses fixed architecture/effective batch, serial fresh
processes, hardware warmup, MPS synchronization, separate compute/pipeline timing,
precisely named memory counters, conservative resource limits, and exclusive
artifact directories. Source hashes, raw JSONL, configurations, and summaries are
preserved. No OOM search or speculative architecture changes were performed.

### Environment and actual results

`sysctl` and filtered `system_profiler SPHardwareDataType -json` confirmed **MacBook
Air M2, Mac14,2, 16 GiB**, not a remote substitute. macOS 27.0.1 arm64, Python
3.13.7, PyTorch 2.15.0; MPS built and available. Initial inspection showed AC power;
all benchmark worker snapshots showed **battery power**. Temperatures and power
mode were not measured. No unavailable-device result is marked as passed.

| Check | Actual result |
| --- | --- |
| Smoke | Passed on MPS, stopped/reloaded at update three, completed at six; 384 training target tokens; final validation loss 4.234095 on 128 tokens |
| Diagnostic continuation | 48 spaces; preserved and explicitly labelled diagnostic, not a useful story |
| Synthetic overfit | Passed at update 21/150; initial loss 4.383251, final 0.088682; fixed threshold ≤0.1 and ≤10% of initial; 30-second cap |
| Matrix | Six cases, each four warmups +eight measured updates; all completed without resource/OOM failures |
| Microbatch 1 / accumulation 8 | Compute 11,824 tokens/s; pipeline 9,302 tokens/s |
| Microbatch 2 / accumulation 4 | Compute 13,442 tokens/s; pipeline 10,399 tokens/s |
| Selected microbatch 4 / accumulation 2 | Compute 14,950 tokens/s; pipeline 11,932 tokens/s |
| Sustained selected case | Four warmups +120 measured updates; 11,906 tokens/s; 23.288-second case / 24.510-second subprocess wall time, within 60-second cap |
| Equal-cadence early/late eight updates | 12,330.58 → 12,240.38 pipeline tokens/s (−0.73%); core −0.12%; no temperature attribution |
| Memory across hardware workers | Max process lifetime peak RSS 455.53 MiB; sampled MPS live tensor 172.69 MiB / driver 216.67 MiB; these overlap and must not be summed |
| Resource guard evidence | Sustained sampled system free indicator ≥32%, versus 20% stop threshold; 4-GiB separate RSS/driver caps; artifacts 186.64 MiB versus 512 MiB cap |
| Runtime projection | 1,000 updates / 2,048,000 tokens: 139.23 seconds at proposed 100-update validation/save cadence, plus command startup; long runs may differ |
| Final regression suite | **290 passed in 32.76 seconds**, no skips/failures; four actual MPS tests passed with fallback disabled |
| New focused benchmark suite | **22 passed in 6.92 seconds**, including fresh subprocess orchestration |
| Installation/lint/format/lock/dependencies | Passed; 60 locked / 39 installed packages; no dependency changes |
| Proposed training config | Dry-run passed; production run directory was not created |

The raw automatic five-update early/late sustained summary has unequal numbers
of validation/checkpoint events. It is retained unchanged; the report uses equal
eight-update sections with two events each, avoiding a misleading slowdown claim.
The recorded matrix has one trial per case, so small timing differences are not
statistical conclusions. The sustained run lasted seconds, not hours.

### Commands executed and evidence

```sh
sysctl -n hw.model hw.memsize machdep.cpu.brand_string
memory_pressure -Q
pmset -g batt
.venv/bin/ruff format src/transformer tests/training/test_benchmarking.py
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training --junitxml=artifacts/phase7-training.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training/test_benchmarking.py --junitxml=artifacts/phase7-benchmark-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer training-smoke --config configs/benchmarking/smoke.json --output artifacts/phase7-smoke-v1
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer overfit-check --config configs/benchmarking/overfit.json --output artifacts/phase7-overfit-v1 --max-seconds 30
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer benchmark-training --config configs/benchmarking/m2-conservative.json > artifacts/phase7-benchmark-cli.json
.venv/bin/transformer train --config configs/training/phase7-m2-proposed.json --dry-run > artifacts/phase7-proposed-dry-run.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase7-full-suite.xml
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

Local Python scripts filtered hardware metadata to omit device identifiers,
created configuration files, copied the unexecuted proposal, and derived the
equal-cadence comparison/audit from raw measurements. Source edits were reinstalled
before execution. An early training-only suite had 71 passing tests; the final
complete suite includes the additional coordinator integration test (290 total).

Evidence is in ignored `artifacts/phase7-hardware-v1/`, `artifacts/phase7-smoke-v1/`,
`artifacts/phase7-overfit-v1/`, `artifacts/phase7-*.xml`, and
`artifacts/phase7-verification.json`. The smoke run's atomic checkpoints remain in
`runs/phase7-smoke-v1/`. Detailed measurement definitions, configuration, output
links, calculations, and limitations are in `docs/phase7_benchmark_report.md` and
`docs/benchmarking.md`.

### Status and stopping point

Phase 7 acceptance criteria passed; no unresolved functional failure remains.
The recommendation is microbatch four / accumulation two on MPS with unchanged
primary architecture and conservative zero-worker loading. It is not a quality
claim or maximum-memory search. LR, training budget, and evaluation policy still
require experiment design; the existing 1,000-update example is merely projected,
not executed. Longer-run duration and thermal/battery stability remain unverified.
MPS bitwise reproducibility limitations from Phase 6 continue to apply. Stop here
and await the next phase prompt; do not launch long baseline training automatically.

## Phase 8 — TinyStories-only baseline (2026-11-19)

### Implementation and predeclared experiment

Read the governing specification and existing progress/configuration/training code;
preserved existing runs and artifacts. Added `configuration/comparison_protocol.py`
and `training/experiments.py`, the `train-baseline` CLI, synthetic runner
tests, `configs/experiments/phase8-comparison.json`, and
`configs/training/phase8-model-a.json`. README documents baseline execution/recovery.

Wrote `docs/phase8_experiment_protocol.md` **before launch** and froze its hash with
the executable protocol. It specifies the research question, shared architecture,
training-only tokenizer, independent seed-42 initialization, equal 2,048,000-token
budgets, optimizer/schedule, effective batch/context, full validation, selection,
future test evaluation, and fixed prompts/greedy sampling. Model A/B configurations
may differ only in run ID and training stream. Model B is a plan only.

Verified the actual M2/MPS target, architecture/recovery tests, frozen vocabulary,
prepared and token checksums, and Phase 7 hardware evidence. The benchmarked model,
optimizer, training-loop, and token-loading code remained unchanged. Compared with
the Phase 7 example, this run explicitly uses a token budget and **all 838 validation
windows**, not the benchmark's short prefix. No hyperparameters changed after
losses or samples became visible.

The runner trains only Model A, protects existing run directories, archives full
milestones, samples between planned training sessions, records recovery and timing,
checks full-validation coverage, selects by validation alone, and leaves test data
untouched. Interruptions during training or between sessions are recorded; resume
requires the frozen protocol. Memory observation can request a safe pause. Unit
tests verify sampled/paused/resumed CPU training matches uninterrupted weights.

### Actual training result

Run: **`phase8-model-a-tinystories-20261119-v1`**, using `train_tinystories` only.
Fresh random seed-42 initialization, four layers/four heads/width and context 256,
3,245,056 parameters, float32 MPS, microbatch four / accumulation two. The tokenizer
is the unchanged shared 78-ID vocabulary. Initial weights matched a separately
reconstructed random model and had zero optimizer history.

| Check | Result |
| --- | --- |
| Actual maximum budget | **1,000 updates /2,048,000 training target tokens**, exactly completed |
| First/final training-update loss | 4.424727 → 2.040301; different sampled training batches |
| Last 100 training updates, mean loss | 2.065175 |
| Full validation | Ten events, each 838 windows /214,528 target tokens |
| First measured validation (update 100) | 2.478317 nats/target token |
| Selected best validation | **2.081407 at update 1,000**, also latest; token perplexity 8.015740 |
| Core training | **139.060 seconds**, **14,727.48 target tokens/s**, including batching/transfers |
| Training-session wall | **168.582 seconds**, **12,148.37 training targets/s**, including validation/setup/logging/checkpoints |
| Training/sampling orchestration elapsed | **177.212 seconds**; excludes initial preflight/interpreter startup/final report writing |
| Diagnostic sampling | 8.095 seconds; three fixed prompts at 100/500/1,000, all nine samples preserved |
| Recovery | Planned stops/resumes at 100 and 500; no unexpected interruptions or uncommitted-log recovery |
| Memory | Process lifetime peak RSS 437.33 MiB; max sampled MPS driver at validation boundaries 224.66 MiB; counters overlap |
| Headroom / disk | Sampled system free indicator 34–36%; no resource guard/OOM; run artifacts approximately 149.53 MiB |
| Final test suite | **299 passed in 32.42 seconds**, zero skips/failures; all four real MPS tests passed |
| Independent checkpoint verification | Best/latest and milestones checked, exact validation replay and identical final greedy token IDs on this host |
| Model B / test evaluation | **Not run** |

Training used the actual MacBook Air M2, Mac14,2, 16 GiB, macOS 27.0.1 arm64,
Python 3.13.7, PyTorch 2.15.0, with MPS CPU fallback disabled. The launch snapshot
reported battery power. No temperatures were measured or thermal claims made.

The best/latest file is
`runs/phase8-model-a-tinystories-20261119-v1/checkpoints/update-000001000-7f3a90ed13f642e9a699a73d5343b056.pt`,
SHA-256 `b39c15d30b3abf420dca0f94a104e5d22044a23fe1debd3bf3223ef2f78474ad`.
Full archived checkpoints at 100, 500, and 1,000 remain under `diagnostics/`.
The index maintains explicit best/latest references even when they refer to one file.

### Verification, commands, and records

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/model_verification tests/training/test_recovery.py --junitxml=artifacts/phase8-prerequisites.xml
.venv/bin/ruff format configs/settings.py training/experiments.py src/transformer/cli.py
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

Architecture/recovery prerequisites passed 58 tests in 5.50 seconds. An early runner
suite passed eight tests; the final complete suite includes nine baseline tests.
Regular-wheel installation, lint/format, lock, and dependency checks passed; no
dependencies changed. Additional local Python commands verified source/data hashes,
created frozen configs, independently reloaded/re-evaluated trained checkpoints,
proved initial seed-42 weights, regenerated final samples, and rendered all sample
text from its raw JSON. No optimizer updates were performed during verification.

The run stores `configuration.json`, `comparison_protocol.json`,
`experiment_protocol.md`, `run.json` (environment/source identities), initialization
proof, hardware/memory snapshots, `metrics.jsonl`, `validation_history.json`,
`experiment_events.jsonl`, `training_summary.json`, `post_training_verification.json`,
and all raw diagnostic samples/checkpoints. Prelaunch evidence is copied into the
run and also stored in `artifacts/phase8-prerequisite-verification.json`. Post-training
evidence is in `artifacts/phase8-post-training-verification.json`; JUnit and console
logs are under `artifacts/phase8-*`. Detailed results and clickable artifact links
are in `docs/phase8_baseline_report.md`.

### Limitations and stopping point

**Phase 8 functional acceptance criteria passed.** Loss decreased, but greedy
continuations still repeat `the`/`they` and do not demonstrate coherent story
generation. All outputs were retained without editing or sampling-policy tuning.
This bounded, single-seed, prefix-subset baseline is not factually reliable and
does not establish final language quality. Test performance remains unknown.

The completed token budget was not extended. Model B's matched-token protocol is
frozen but its run directory does not exist. The next expected work is the user's
Model B phase with fresh independent initialization and separately measured time.
Do not begin it or frontend work automatically.

## Phase 9 — TinyStories plus SimpleStories (2026-11-19)

### Protocol, implementation, and compatibility

Read the frozen Phase 8 protocol, actual Model A configuration/logs/checkpoints,
source manifests, and sampler/token-loading code. Preserved the original protocol
and **every original Model A run file**, verified by full before/after file hashes.

Added `data/loading.py`, `evaluation/metrics.py`,
`training/experiments.py`, the `train-combined` CLI, and ten new synthetic
test cases. Generalized the existing orchestration to choose the frozen experiment
member and run observational callbacks. Added `configs/training/phase9-model-b.json`,
`docs/phase9_experiment_protocol.md`, and `docs/phase9_combined_report.md`; updated
README commands. The Phase 9 supplement and preparation manifest were frozen before
Model B training or source-specific model scoring.

The source audit revalidated cleaned-story hashes, immutable dataset/token files,
source ownership, original training-view membership, exact token content and EOS
offsets. It found **zero exact normalized-story overlap across train/validation/test**
jointly across sources, with no official held-out story in training. Near-duplicate
and unacquired-upstream overlap remain unverified. No data download, recleaning,
vocabulary refit, or test scoring occurred.

The mixture follows the Phase 8 plan: uniform seed-42 shuffled complete windows
from the combined stream, without replacement until an epoch ends. Its ratio is
defined by **shifted target tokens including EOS**, apportioned inside mixed-source
windows. Of 8,084 eligible windows, 8,000 are sampled once; no repetition is needed.
Natural eligible-target proportions are 45.5490301% TinyStories /54.4509699%
SimpleStories. The exact precomputed seeded plan and actual counts agree:

| Source | Uniform-draw expected targets | Seeded planned targets | Actual targets | Actual share |
| --- | ---: | ---: | ---: | ---: |
| TinyStories | 932,844.14 | **932,922** | **932,922** | **45.5528320%** |
| SimpleStories | 1,115,155.86 | **1,115,078** | **1,115,078** | **54.4471680%** |

The document counts (1,000/894) are recorded separately and are not called token
ratios. There are 909 mixed-source sampled windows; final sampler epoch/cursor is
0/8,000. Actual exposure was checked against all saved sampler fields and independently
recounted from checkpoint window intervals. Model A's same 2,048,000-target budget
contains only TinyStories, with 3,682 unique windows and 4,318 repeated draws.

Both runs match architecture, frozen 78-ID tokenizer, source fingerprints, seed 42,
optimizer/schedule, microbatch four /accumulation two, context 256, and total budget.
Model B's update-zero weight fingerprint exactly matches A's original random model,
with empty optimizer history. **B did not start from trained A.** Core compute,
sampler, data, and aggregate-validation source hashes remain unchanged from A.

The primary selector is unchanged: lowest full mixed-validation token-weighted NLL,
838 windows /214,528 targets every 100 updates; earliest exact tie. Its now-explicit
weights are 37.5983555% TinyStories /62.4016445% SimpleStories. Separate source-only
validation views are diagnostics at 100, 500, 1,000 and a distinct best if needed.
They use different context windows/tails and are **not averaged for selection**.
A's retained milestones were scored retrospectively outside A's directory; its
historical selection was not changed. No rerun was required for these diagnostics.

### Actual run and outcomes

Run: **`phase9-model-b-combined-20261119-v1`**, actual M2/MPS execution.

| Check | Actual result |
| --- | --- |
| Budget | **1,000 updates /2,048,000 target tokens**, exactly completed |
| Selected best/latest | **Update 1,000**, original aggregate validation NLL **2.058485** |
| First/final training-update loss | 4.414072 → 2.153969; different sampled batches |
| Last 100 training updates, mean loss | 2.139819 |
| B source-only validation, best checkpoint | TinyStories **2.034429** on 80,640 targets; SimpleStories **2.072718** on 133,888 targets |
| A retrospective source-only validation, best | TinyStories **1.958923**; SimpleStories **2.156060**, same diagnostic views |
| Core training time/rate | **137.104 s /14,937.60 targets/s**, includes batching/transfers |
| Training-session wall time/rate | **174.899 s /11,709.64 training targets/s**, includes extra diagnostics |
| Additional timed source-audit sections | **8.592 s** inside sessions, excluding their checkpoint loading/output overhead |
| Sampling /orchestration elapsed | **6.907 s /182.290 s** through training/sampling; preflight/final reports excluded |
| Recovery | Planned stops/resumes at 100 and 500; **zero unexpected interruptions** |
| Memory | Peak process RSS **494.38 MiB**; MPS driver max sampled at validation boundaries **216.66 MiB**; counters overlap |
| Headroom | Free indicator ≥34% at validation boundaries, 33% at completion; no resource pause/OOM |
| Tests | **309 passed in 34.88 s**, no skips/failures; four real MPS cases passed |
| Independent verification | Checkpoint hashes, exact token budget, independent source counts, both validation modes, regenerated sample IDs, and unchanged A files all passed |
| Formal test comparison /frontend | **Not performed** |

The launch snapshot recorded MacBook Air M2, Mac14,2, 16 GiB, macOS 27.0.1 arm64,
Python 3.13.7, PyTorch 2.15.0, battery power, MPS available with CPU fallback disabled.
No temperatures were measured. Extra source diagnostics mean B's wall time is not
directly interchangeable with A's original session time; compute time is separate.

The selected checkpoint is
`runs/phase9-model-b-combined-20261119-v1/checkpoints/update-000001000-419df80348d4463283240daee5a02d04.pt`,
SHA-256 `d374977df3c3a367baddc0973d1fbe2fe764af1ce5279b845ea190158af66113`.
Best/latest point explicitly to it; full milestones at 100/500/1,000 also remain.
Run artifacts occupy about 150 MiB.

### Executed checks, fixes, and artifacts

The preparation integration test exposed missing creation of a new output parent;
this was fixed before actual preparation. The runner now also stops correctly if
SIGINT arrives inside its final observer callback, rather than starting the next
planned session. This affects interruption handling only; A had no such interruption.
CPU tests with dropout prove observed/paused/resumed B matches uninstrumented B's
weights and RNG exactly. No numerical/configuration change invalidated the pair or
required repeating A. The final full suite passed; an intermediate focused JUnit
retains a fixture-setup failure that was corrected before that final run.

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

Local Python commands audited all partitions, froze the actual configuration,
hashed every A file, independently intersected sampled target/story intervals,
reloaded checkpoints, replayed aggregate/source validation and final generation,
and rendered the nine unedited samples. No training updates occurred in verification.
Installation/lint/format/lock/dependency checks passed; no dependencies changed.

Preparation, original-A diagnostics and compatibility evidence live under
`artifacts/phase9-preparation-v1/`. The B run stores frozen configuration/protocol,
data/environment identity, `metrics.jsonl`, `validation_history.json`,
`source_validation_history.json`, cumulative `source_diagnostics/` records,
`source_exposure.json`, recovery events, samples/checkpoints, and
`training_summary.json`. Independent proof is copied into
`post_training_verification.json` and `artifacts/phase9-post-training-verification.json`.
JUnit, dry-run, prelaunch and original-A preservation hashes are under
`artifacts/phase9-*`. The report provides clickable artifact links.

### Limitations and stopping point

**Phase 9 acceptance criteria passed.** The source-only validation observations
show a tradeoff in this pair: higher TinyStories loss, lower SimpleStories loss
for B; the original aggregate gives more weight to SimpleStories. This is not a
statistically established general improvement. Both runs use one seed and bounded,
prefix-selected data; exact deduplication does not remove all near-duplicates.

All nine B generations remain highly repetitive and are not factually reliable.
No generation setting, selection rule, or budget was changed after seeing results.
Formal held-out/test comparison is reserved for **Phase 10**, which has not begun.
Stop here and await the next phase prompt.

## Phase 10 — held-out evaluation and comparison

Completed **20 November 2026, Asia/Kolkata** (raw artifact timestamps are UTC).
Scope: evaluate the actual selected A/B checkpoints, preserve their records, measure
held-out loss/perplexity and generation throughput, present all fixed-prompt samples,
and save plots and a factual comparison. No training or frontend work occurred.

### Implementation and frozen protocol

Added `configuration/evaluation_config.py`, the bounded
`configs/evaluation/phase10-comparison.json`, and CLI `evaluate-comparison`.
`evaluation/metrics.py` scores exact target positions using memory-mapped
tokens, token-weighted NLL and exp(mean NLL), with per-story/source accounting.
`generation_comparison.py` supplies paired greedy outputs and synchronized fixed-work
timing. `comparison.py` verifies actual selection/budgets/provenance, protects existing
outputs, hashes both runs and atomically publishes read-only results.
`comparison_charts.py` produces six PNG/SVG charts from actual logs and measurements.

The [Phase 10 protocol](phase10_evaluation_protocol.md) was written before real test
scoring. Inspection found that Phase 8 already specified packed test windows, so that
policy remains primary. A separate predeclared document-isolated analysis uses
overlapping input context with no duplicated scored targets. The original training
protocols, tokenizer, data, checkpoints and all run files were preserved.

Primary inventory: 1,262 windows /323,072 shifted targets, one initial context token
and 121 tail target pairs excluded. EOS is included (298 scored; the last lies in
the omitted tail). Secondary inventory: 2,670 windows /322,895 targets, all 299 EOS
included and one first character per document excluded (no BOS). No scored UNK
occurred. Both use the same 99 TinyStories /200 SimpleStories held-out documents.
Zero exact normalized-content train/validation/test overlap was verified jointly.

Selection was reconstructed from actual validation logs: both best checkpoints are
update 1,000, lowest full-validation token-weighted NLL with earliest ties. Recorded
training histories explicitly deferred test scoring. No test result was used to
reselect, tune or extend either run. This is evidence from local project records,
not a guarantee against unrecorded external activity.

### Actual results

| Primary metric, nats/character-or-special token | Model A | Model B |
| --- | ---: | ---: |
| TinyStories loss, 74,616 targets | 1.950436588 | 2.031649037 |
| TinyStories perplexity | 7.031756894 | 7.626652625 |
| SimpleStories loss, 248,456 targets | 2.150582842 | 2.070371586 |
| SimpleStories perplexity | 8.589863475 | 7.927768417 |
| Target-weighted combined loss | 2.104357503 | 2.061428311 |
| Target-weighted combined perplexity | 8.201831664 | 7.857184301 |
| Equal-source secondary loss | 2.050509715 | 2.051010311 |
| Document-isolated combined secondary loss | 2.117376991 | 2.074907106 |
| Document-isolated equal-source secondary loss | 2.069825270 | 2.066444412 |
| Parameters, shared weights counted once | 3,245,056 | 3,245,056 |
| Actual training target exposure | 2,048,000 TinyStories | 932,922 TinyStories +1,115,078 SimpleStories |
| Core /session training duration | 139.060 /168.582 s | 137.104 /174.899 s |
| Pooled generation throughput | 177.842 tokens/s | 176.954 tokens/s |

Combined primary weights are **23.0958% TinyStories /76.9042% SimpleStories**.
B's SimpleStories loss is lower and TinyStories loss higher. Its aggregate advantage
depends on source weighting; the equal-source difference is tiny and changes direction
under the secondary context policy. This does not establish a broad improvement.
Character/special-token perplexity is not directly comparable to unrelated subword
perplexity. All six greedy generations remain dominated by repeated function words;
all hit the 128-token cap without EOS. A descriptive six-dimension comparison and an
unscored proposed human rubric are provided; no human ratings or external model API.

Generation ran on the actual M2 MacBook Air /16 GiB, macOS 27.0.1 arm64,
Python 3.13.7, PyTorch 2.15.0, MPS float32, two CPU threads, batch one, battery power,
CPU fallback disabled. One warmup per prompt/model, then three repetitions across
three prompts: nine measured trials /1,152 tokens per model, interleaved A/B order.
Fixed-work timing disabled EOS stopping, synchronized MPS, included encoding/transfers/
argmax/decoding, and excluded model loading/warmup. Raw ranges overlap; no speed
advantage or long-run thermal conclusion is claimed. B's historical session time
includes extra source diagnostics, so wall-time differences are not mixture cost.

### Executed commands and verification

```sh
.tools/uv/bin/uv lock
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable --reinstall-package transformer
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/training/test_held_out_evaluation.py --junitxml=artifacts/phase10-evaluation-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase10-full-suite.xml
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/transformer evaluate-comparison --help
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer evaluate-comparison --config configs/evaluation/phase10-comparison.json > artifacts/phase10-comparison-cli.json 2> artifacts/phase10-comparison-console.log
```

Focused evaluation tests initially **19 passed in 7.51 s**; an additional short/UNK/EOS
case was added before the final full suite: **329 passed in 32.97 s**, no skips/failures,
including five real MPS tests. Lint, formatting (161 files in the final check) and all 46 installed-package
compatibility checks passed. Installed package source files matched the workspace.
No model, tokenizer, sampler, optimizer or training mathematics changed.

Local Python verification independently replayed the primary metric using the existing
token loader/training evaluator: absolute NLL differences were **1.50e-9 A /4.79e-9 B**.
Per-document/source sums, exp(loss), raw timing arithmetic, every published checksum,
and original A/B inventories were checked. New greedy token IDs exactly matched each
selected checkpoint's previously archived diagnostic samples. All six PNGs were
visually inspected for readable axes, units, values, legends and run IDs.

Final `uv lock --check` passed (67 resolved packages). All 19 published artifact
checksums, 54 frozen package-source hashes, the pre-scoring protocol hash, 39 local
documentation links, and unchanged original A/B file inventories were verified.

Matplotlib **3.11.2** was added as the optional `evaluation` extra, resolved consistently
in `uv.lock`; all previous ML runtime pins were retained. Compatibility was checked
against [official package metadata](https://pypi.org/pypi/matplotlib/3.11.2/json)
(Python >=3.11, NumPy >=1.25) and the actual installation. Initial standard-library
HTTPS lookup hit the local certificate-store issue; verified HTTPS via system curl
succeeded without disabling verification. Metadata is saved in
`artifacts/phase10-dependency-compatibility.json`.

### Deliverables, limitations and stopping point

The [comparison report](phase10_comparison_report.md) links configuration, frozen
protocol, quantitative results, six unedited samples, six PNG/SVG plots, and raw timing.
Full artifacts live in `artifacts/phase10-comparison-v1/` with a checksummed manifest,
pre-evaluation record, results JSON containing copied actual training/validation logs,
per-story/source loss sums, exposure, environment and checkpoint identities.
Prelaunch evidence is `artifacts/phase10-prerequisite-verification.json`; independent
post-run evidence is `artifacts/phase10-post-evaluation-verification.json`.

**Phase 10 acceptance criteria passed.** One seed, small prefix-selected data,
exact-only deduplication, different source exposure/repetition, mixed validation weights,
context policy, greedy-only prompts and MPS numerical variability limit conclusions.
The test results are now observed and cannot serve as an unseen tuning/selection set.
No general quality improvement or statistical significance is claimed. No further
training or frontend work has begun; stop and wait for the next phase prompt.

## Phase 11 — reusable inference backend

Completed **20 November 2026, Asia/Kolkata**. Scope: safe local checkpoint loading,
stable model registry, reusable Python service, greedy/temperature/top-k generation,
explicit edge cases, one-model resource ownership and CLI. No frontend, model training,
dataset download, checkpoint reselection or dependency change.

### Implementation and policies

Added `configuration/generation_config.py` and the functional `inference/` package:
`model_registry.py`, `checkpoint_loading.py`, `sampling.py`, `generation_service.py`,
and public imports in `__init__.py`. Stable registered versions are
**`transformer-tinystories-v1`** (baseline A) and **`transformer-combined-v1`** (B),
both original validation-selected update 1,000 checkpoints. The explicit
`configs/inference/model_registry.json` associates checkpoint, tokenizer, configuration
and actual Phase 10 metrics with SHA-256 hashes. It is trusted operator configuration,
not generation-request input. No arbitrary-file discovery or checkpoint-path request
is provided; only registered IDs are accepted.

Loader checks relative approved locations, symlink files/parents, bounded regular
files, hashes, format/version, saved architecture/configuration, tokenizer fingerprint
and vocabulary size, model kind/run/update, metrics association, strict state shapes/
keys, finite float32 values, tied weights and parameter count. Restricted
`torch.load(weights_only=True, map_location="cpu")` uses the same open verified file.
It reconstructs on CPU, discards optimizer payload state, freezes parameters, sets
eval mode and moves one model to the selected CPU/MPS device. Explicit unavailable
MPS fails; automatic selection may choose CPU with an explanation. Trust is grounded
in the operator-owned catalog/project, not claimed authentication against a party
who can rewrite both files and registry.

`GenerationService` provides `list_models`, `load`, `generate`, `cache_info`, `unload`
and context-manager cleanup. CLI commands are `list-models` and `generate`. Inference
requires no plotting/frontend dependency. The service caches one model per instance,
releases it before switching, clears unused MPS cache, serializes requests, detects
artifact file changes, and restores PyTorch thread/dtype settings. Parameters are frozen
and forwards run under inference mode; no mutable model is returned by the public API.

Generation rules are documented in [the guide](inference_backend.md):

- Explicit greedy argmax with lowest-ID ties; temperature omitted/zero, no top-k/seed.
- Sample mode requires finite positive temperature; top-k is omitted or 1..vocab size,
  exactly k IDs with stable lower-ID boundary ties. A private CPU generator supports
  an explicit seed or reports a generated effective seed. Global CPU/MPS RNG is preserved.
- Empty prompt rejected (no BOS); unknown defaults to an actionable error, explicit
  replacement reports UNK count; invalid Unicode surrogates rejected; nonempty whitespace allowed.
- Prompt at most 16,384 characters; retain last context-capacity IDs. Each rolling
  learned-position window restarts at 0..L−1; no position wrapping or retained KV cache.
- Output 0–1,024 tokens, with zero performing no forward. Stop immediately on EOS;
  count/store EOS ID but omit its visible marker. UNK remains visible as `<unk>`.
- JSON results contain identities, prompt/continuation/IDs/count, settings/effective seed,
  unknown/truncation counts, device/runtime, stop reason, cache hit and separate timings.
  Generation includes encoding/forward/transfers/sampling/decoding/synchronization and
  excludes loading, lock wait and JSON serialization. Cache-miss load timing includes eviction.

### Verification and defect fixed

Initial focused suite: **60 passed /1 failed**. The real MPS test exposed
`Undefined type Double` when transferring and converting logits directly to CPU float64.
Fixed by transferring in the original dtype first, then converting on CPU. The
focused rerun passed **61 tests in 2.78 s**. Two extra cases were then added for the
1,024-token bound/inference mode and restricted verified-handle deserialization.

Final complete suite: **392 passed in 34.68 s**, zero failures/skips, comprising
**63 inference tests** plus all existing checks, including **six real MPS tests**.
Offline synthetic fixtures cover corruption/identity/vocabulary/config errors,
traversal/symlinks, sampling probabilities/top-k ties, repeatable sampling and greedy
outputs, caller RNG preservation, unknown/empty/long prompts, zero/max/excess limits,
EOS, actual learned-position hooks, gradients, concurrent requests, cache eviction,
artifact changes, installed CLI commands and explicit unavailable-MPS errors.

The reusable bounded `tools/maintenance.py verify-inference` then exercised both actual trained
models on the **MacBook Air M2 /16 GiB, macOS 27.0.1 arm64, Python 3.13.7, PyTorch
2.15.0, MPS, battery power, CPU fallback disabled**:

| Real-artifact check | Result |
| --- | --- |
| Both models load frozen/eval | Passed; 3,245,056 parameters each |
| Three original greedy prompts per model | All six 128-token sequences matched archived IDs |
| Sampled repeats, temperature 0.8/top-k 20/seed 1234 | Same output IDs for each model, 64-token cap |
| Long prompt vs explicit final 256 characters | Identical generated IDs for both models |
| Zero output, no gradients, preserved global RNG | Passed |
| Original runs and Phase 10 artifacts | All **83 files unchanged** |
| CPU greedy CLI /MPS sampled CLI | Both executed successfully with actual checkpoints |

Sampling introduced text variation but did not establish coherent or reliable generation;
all actual raw text is retained. Cross-device/release bitwise identity is not promised.
No model or training mathematics changed to address an inference test.

### Measured memory and timing

Six alternating A/B load/generate observations (three cycles) each retained exactly
one model and reported **13,242,368 MPS live-tensor bytes (12.629 MiB)**,
**17,432,576 MPS driver bytes (16.625 MiB)** and **435,568,640 process RSS bytes
(415.391 MiB)**. Those sampled values were identical across all six switches. After
unload, the cache held zero models and MPS live-tensor bytes returned to **zero**;
driver allocation remained **655,360 bytes (0.625 MiB)**. Lifetime peak process RSS
was **436,404,224 bytes (416.188 MiB)**. These overlapping counters must not be summed
or described as total system memory. CPU runtime/allocator residency persists after
unload. The bounded observations do not prove absence of every long-running leak.

First-load durations in that script were approximately **0.111 s A /0.131 s B**,
with **0.710–0.894 s** per 128-token greedy continuation. Cache hits reported zero
load time. These are diagnostic measurements, not a new controlled performance study.

### Commands executed and deliverables

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

Lock/dependency/lint/format checks passed; the 67-package lock and 46 installed
packages remained compatible with no new dependencies. The public
[guide](inference_backend.md), [Phase 11 report](phase11_inference_report.md), README,
registry, tests and verification script document reproducible use. Actual evidence
is under `artifacts/phase11-*`: full/focused JUnit, listing, CPU/MPS CLI output and
the complete real-model/memory verification JSON.

**Phase 11 acceptance criteria passed.** Both registered model versions are usable
from Python and CLI; generated text remains repetitive/unreliable, same-seed behavior
is runtime/device dependent, and the service assumes a trusted operator-owned root
and catalog. Cache ownership is per service instance; use one long-lived instance
per application process. No web interface was built. Stop and await the next prompt.

## Phase 12 — backend release gate (20 November 2026, Asia/Kolkata)

**Completed; backend gate PASSED on the actual M2 MacBook Air /16 GiB.**
The [backend verification report](backend_verification_report.md) and
[raw release record](../artifacts/phase12-backend-gate-v1/verification.json) contain
the review, executed checks, pass/fail/skip statuses, actual devices, limitations
and artifact identities. No frontend or further real-model training was started.

### Implementation and review

Inspected preparation, splitting, vocabulary/token loading, architecture, training,
recovery, evaluation, inference, configuration, logs, error paths and documentation.
Added `verification/checks.py`, `failure_cases.py`, `backend_gate.py`,
`tools/maintenance.py verify-backend` and ten offline gate/integration tests. The gate requires
full-suite evidence plus successful synthetic, negative and separate real-artifact
stages. It returns a nonzero exit and retains an incomplete JSON report on failure.
Fixtures cannot replace actual Model A/B availability. New outputs must stay under
`artifacts/`, and existing directories/symlink destinations are refused.

A new extreme-logit regression reproduced an **OverflowError** in equal-source
secondary perplexity. Reusing the existing overflow-safe NLL conversion fixed it:
JSON null plus `perplexity_overflow: true`, while normal metrics are unchanged.
The failed-before XML is preserved. No training or model math changed. A relative
fixture-root error found while developing the gate was fixed by resolving its root.
Its failed development artifacts remain separate and are not passing evidence.

Corrected stale current-status and launch-command wording in README/training docs.
Removed the obsolete unimplemented-directory claim. Removed only generated `build/`
copies after wheel verification: **64 files, 323,999 bytes**, checked against source.
Active historical diagnostic/evaluation sampling paths remain because frozen
experiments/timing still depend on their semantics. No unused dependency or dead
executable placeholder was found; all runtime/dev/optional dependencies were retained.

### Executed verification

The target environment is macOS 27.0.1 arm64, Python 3.13.7, PyTorch 2.15.0 with real
MPS availability. Locked regular-wheel installation is used because the host hides
editable `.pth` paths. Final results:

- Focused suite: **11 passed in 3.73 s**.
- Full suite: **403 passed in 38.63 s**, no failures/skips, including six real MPS tests.
- Synthetic CPU integration: **2.116 s**. Twenty-six records become ten train/four
  validation/four test stories; four rejects and four duplicates are recorded. The
  held-out-only character does not enter the 23-ID vocabulary. Two fixture runs
  train four updates/256 targets each, save and resume at update two, evaluate and
  generate through their isolated registry. A third uninterrupted run matches every
  final tensor plus scheduler and CPU RNG exactly.
- Negative checks: **eight passed**, covering missing data, invalid config, tokenizer
  mismatch, unsupported device, existing run, invalid sampling, corrupted bytes and
  incomplete checkpoint state. All destructive cases use copied synthetic artifacts.
- Real artifacts: **30.058 s**, MPS with fallback disabled. Both 3,245,056-parameter
  selected checkpoints load. Both completed budgets remain 1,000 updates/2,048,000
  targets, with matched tokenizer/architecture and validation-only selection.
- Real packed and document-isolated source/combined held-out NLL values reproduce
  Phase 10 **exactly** on this machine (acceptance tolerance 1e-6). Six fixed
  128-token generations reproduce archived token IDs. All **83** original run and
  evaluation files remain unchanged; all **19** evaluation-manifest entries verify.
- Locked dependency checks, Ruff checks, wheel packaging and installed MPS diagnostic
  pass. No live Hub download was rerun; that optional online check is skipped.

Actual commands included:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m pytest tests/training/test_held_out_evaluation.py::test_equal_source_perplexity_overflow_is_serializable --junitxml=artifacts/phase12-overflow-before.xml
.venv/bin/python -m pytest tests/verification tests/training/test_held_out_evaluation.py::test_equal_source_perplexity_overflow_is_serializable --junitxml=artifacts/phase12-focused.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase12-full-suite.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/maintenance.py verify-backend --device mps --test-report artifacts/phase12-full-suite.xml --output artifacts/phase12-backend-gate-v1 > artifacts/phase12-gate-console.log 2>&1
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m build --wheel --outdir artifacts/phase12-wheel
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer diagnose --device mps --check
```

The before-fix test failed as intended; its standalone rerun and subsequent suites
passed. An initial manual `diagnostics` CLI spelling was rejected; the documented
`diagnose` command then passed. Its retained output is `artifacts/phase12-diagnostics.txt`.
Wheel output and build log are under `artifacts/phase12-wheel*`. Final consistency
evidence is `artifacts/phase12-final-verification.json`.

### Limitations and stopping point

No blockers remain for this scoped **local backend** release gate. Full upstream
network availability, long-running deployment/load/security behavior and other
hardware/software environments were not certified. Exact resume is controlled by
recorded implementation/environment; historical Phase 8 continuation needs its
original code, while inference loads completed artifacts with current code. Recovery
does not exceed the last completed atomic checkpoint after abrupt power loss.
Single-seed/small-subset results, exact-only deduplication and repetitive generated
text remain limitations. The final report makes no stronger language-quality claim.
**Phase 12 complete. Stop; await the next phase prompt before frontend development.**

## Phase 13 — frontend direction and application shell (20 November 2026)

**Completed; Phase 13 acceptance criteria passed.** The backend gate was confirmed
before frontend work: all four Phase 12 stages passed, all 64 recorded backend Python
source hashes matched, both real models' registered artifacts verified, and the
original 83 run/evaluation files were unchanged. No missing prerequisite was hidden
by the frontend. The existing backend remains usable; web generation is deliberately
not connected until Phase 14.

### Decision and implemented structure

Selected a **buildless HTML/CSS/JavaScript frontend with a local FastAPI/Uvicorn host**.
Direct control of editorial typography, reading columns, native forms, keyboard flow
and responsive comparison justified this approach over Streamlit. No JavaScript
package manager, Node runtime or browser model is required to run the application.
FastAPI/Uvicorn are isolated in the optional `web` extra; the single `uv.lock` records
resolved versions. PyPI metadata confirmed Python 3.13 compatibility. FastAPI 0.165.0
and Uvicorn 0.54.0 installed successfully; 76 locked /55 installed packages pass
compatibility checks. The original backend dependency versions were not upgraded.

Implemented:

- `backend/application.py` and `__main__.py`: one-worker loopback launch,
  packaged static assets and an honest generation-disabled status endpoint. Default
  same-origin access; one exact localhost/127.0.0.1 development origin can be opted in.
  No arbitrary project files or future inference routes are exposed.
- `web/static/index.html`: Generate, Compare, Model insights and Architecture shells.
  The writing desk includes a local draft, preferences and disabled continuation action.
- `web/static/tokens.css` and `application.css`: warm off-white/graphite/deep-teal
  palette, serif editorial headings, system interface typography, spacing, restrained
  rules, forms/buttons/focus states and responsive layouts.
- `web/static/application.js` and `components.js`: hash navigation, tab-memory draft
  retention, code-point counting, variation reveal, clear/error feedback, status retry
  and labelled loading/empty/success/error specimens at `#/foundations`.
- `web/contracts/inference.openapi.json`: versioned request/result/error schemas and
  implemented-versus-planned operation markers. Phase 14 adapters will delegate to
  the existing `GenerationService`; no inference logic was copied into the frontend.
- `docs/frontend_design.md`: framework decision, page structures, interaction/state
  plan, responsive rules, contract semantics, local origin policy and Phase 14 handoff.
  README includes actual install/start/check commands.

The Sites skill's local-only workflow was used for repository/preview guidance.
No site was registered, hosted or published. No response samples or training charts
were invented. Models were not loaded by the web shell, and no production training
or dataset acquisition was initiated.

### Checks actually executed

```sh
.tools/uv/bin/uv lock
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
node --check frontend/application.js
node --check frontend/components.js
.venv/bin/python -m transformer.web --port 8000
.venv/bin/python -m pytest tests/web --junitxml=artifacts/phase13-web-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase13-full-suite.xml
.tools/uv/bin/uv pip check --python .venv/bin/python
.tools/uv/bin/uv lock --check
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python -m build --wheel --outdir artifacts/phase13-wheel
```

- **17 web checks passed in 0.99 s**. Actual loopback HTTP requests verified public
  assets, capability status, planned-route/private-path 404s, host rejection,
  default-denied and exact opt-in CORS. Subprocess checks confirmed no PyTorch/inference
  import at startup and useful invalid-port errors. Contract references and backend
  token/input limits agree. No external service is needed by these tests.
- **420 full-suite tests passed in 39.71 s**, zero failures/skips; the existing six
  actual MPS tests remain included. Only synthetic test training was performed.
- Browser verification in the Codex in-app browser covered all four navigation areas,
  retained draft/preferences, variation reveal, Unicode count (`a😀` → 2), over-limit
  error, clear feedback/focus, native radio keyboard order, keyboard feedback actions
  and unknown-route recovery. Generation remained disabled. No console errors/warnings
  were observed. These are focused manual checks, not a complete accessibility audit.
- Responsive checks: **1440 ×1000**, **768 ×1024**, **320 ×740** on Generate and
  **390 ×844** on Compare. Document width matched viewport width at every check.
  Mobile inspection prompted a real improvement: preferences now precede the action
  and empty result, with a full-width button and two-row navigation on narrow phones.
  A joined mobile sentence was corrected and rechecked.
- Wheel build, syntax, Ruff, lock and dependency checks passed. The installed package
  and wheel include the shell's assets and API contract. The development server is
  stopped after verification; the README start command opens it again locally.

Evidence: `artifacts/phase13-web-tests.xml`, `phase13-full-suite.xml`,
`phase13-browser-verification.json`, `phase13-final-verification.json`,
`phase13-wheel-build.log`, package metadata JSON and the seven actual screenshots
under `artifacts/phase13-preview/`. Final consistency checks verify source/wheel
agreement and preservation of the recorded backend/artifact identities.

### Limits and next expected phase

This release is a polished **design shell**. Functional generation, paired-result
handling, live insights/chart construction and detailed architecture interaction are
not implemented. Planned API operations return 404 and their contracts say Phase 14;
the status endpoint is not a real-model readiness signal. Drafts are memory-only and
are lost on refresh. Localhost access is not a public-deployment authentication design.
No blocker remains for the requested shell/design deliverables. The next phase is
**Phase 14: functional page construction and existing-service integration**.
Stop here and wait for that phase prompt.

## Phase 14 — core Generate interface (20 November 2026)

**UI implementation complete; scoped acceptance passed with documented verification limits.**
The latest phase prompt explicitly leaves inference wiring to Phase 15, superseding
Phase 13's earlier integration roadmap. Current README, design plan and API contract
reflect that boundary. Existing experiment files and backend implementation were preserved.

### Implemented

- Generate page with a visible opening label, three authored example prompts, Unicode
  character count, validation, model/style/length controls, disabled Generate action,
  a readable continuation region, local status/retry and bounded session history.
- One-step undo for clearing/replacing openings; save/restore of draft and settings;
  newest ten items in tab memory only. Navigation preserves state; refresh clears it.
- Separate modules for routing, page binding, state/validation, history, status and
  safe output rendering. Approved visual tokens and shared components remain intact.
- Empty, ready, loading, success, validation-error, unavailable-backend and missing-
  checkpoint UI states. Successful output preserves paragraphs, separates the original
  prompt, provides copy feedback and names the stop reason. Immediate EOS is handled.
- Keyboard controls, focus rings, labelled errors, disabled/busy semantics and stacked
  mobile order. Long words wrap; generated markup is rendered as literal text.
- Separate synthetic fixture server, with no model loading or generation simulation.
  Phase 13's design specimens moved out of the normal app into this harness as well.
  Normal application routes/assets and the installed wheel exclude all UI fixtures.
- README updates and `docs/frontend_interface.md` with behavior, testing, limitations
  and Phase 15 handoff. Contract version 1.1.0 reports Phase 14 `core_ui`; generation
  remains disabled and planned inference routes still return 404.

### Commands actually executed

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
node --test tests/web/frontend_state.test.mjs
.venv/bin/python -m transformer.web --port 8000
.venv/bin/python tools/maintenance.py preview-ui-fixtures --port 8001
.venv/bin/python -m pytest tests/web --junitxml=artifacts/phase14-web-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase14-full-suite.xml
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/ruff check tools/maintenance.py preview-ui-fixtures --fix
.venv/bin/ruff format backend/__main__.py tools/maintenance.py preview-ui-fixtures
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m build --wheel --outdir artifacts/phase14-wheel
node --version
lsof -nP -iTCP:8000 -iTCP:8001 -sTCP:LISTEN
```

Additional bounded Python verification invoked `node --check` for all seven production
JavaScript modules and two isolated fixture modules, checked six fixture-server HTTP
resources on an allocated loopback port, and compared source/installed/wheel bytes,
registry references and the original backend/experiment hashes. Reinstallation and
wheel build were repeated after packaged-asset changes. Full-suite stdout is retained
in `artifacts/phase14-full-suite.log`; JavaScript test output is in
`artifacts/phase14-frontend-tests.log`.

### Results

- **10 JavaScript state tests passed**, no failures/skips; Node **24.14.0**.
- **429 full Python tests passed in 38.88 s**, no failures/skips, including the existing
  six MPS tests on the actual local Mac. After the final fixture-route isolation check
  was added, the relevant **27 web HTTP tests passed in 0.93 s**. The full-suite record
  intentionally remains the earlier 429-test run rather than claiming an unexecuted
  430-test full run.
- Nine JavaScript syntax checks, six fixture-server HTTP checks, Ruff, format, lock,
  installed dependency compatibility and final wheel build passed. Initial Ruff issues
  were an import ordering issue and one long line; both were fixed and rechecked.
- Browser checks covered default/ready/error states, long prompt, all isolated fixture
  states, history, undo, preferences, navigation, keyboard focus/disclosure/copy,
  immediate EOS and safe text rendering. Viewports: **1440×1000**, **768×1024**,
  **320×740** and long output at **390×844**. No horizontal overflow was observed.
- Stopping/restarting the real local server exercised unavailable/retry behavior while
  retaining draft/history. Keyboard review revealed a blank-page reload at the skip-link
  hash; routing was fixed and browser reload then displayed the editor correctly.
- Clipboard writes resolved and success feedback appeared. The automation virtual
  clipboard did not contain native Clipboard API data, so exact native paste and
  permission-denied behavior are **manual checks pending**, not recorded as passed.
- Final consistency proof: **64 original backend source files unchanged**, **83 original
  experiment/evaluation files unchanged**, **8 registered artifact references verified**,
  production web source/installed/wheel agreement, unchanged `uv.lock`, no UI fixture
  files in the wheel. The first one-off wheel assertion was overbroad and matched the
  existing backend `verification/checks.py`; the check was corrected to the
  web/test boundary, preserving that valid backend module.
- All owned preview servers were stopped; the final port inventory had no listeners.
  No production training, model changes, dataset downloads or inference wiring occurred.

Evidence: `artifacts/phase14-final-verification.json`, `phase14-browser-verification.json`,
`phase14-fixture-checks.json`, web/full-suite XML, frontend test log, wheel/build log and
eight actual screenshots in `artifacts/phase14-preview/`. Focused browser checks do not
constitute a full accessibility audit or cross-browser certification.

### Next expected phase

**Phase 15: connect the Generate interface to the existing inference service.**
The current interface can edit/save openings but cannot generate stories. Loading,
completed-output and missing-checkpoint views were verified with explicitly labelled,
isolated UI fixtures only. Compare, insights and architecture retain their earlier
shells. Stop after Phase 14; await the next phase prompt.

## Phase 15 — frontend and inference integration (20 November 2026)

**Complete; Phase 15 acceptance passed.** Generate now uses the existing inference
service and actual registered Model A/B checkpoints. The backend gate and prior
experiment artifacts were verified before integration and remain intact.

### Implementation and decisions

- Added strict schemas, bounded JSON receipt, safe error envelopes and an application-
  owned inference adapter in `web/schemas.py`, `http_boundary.py` and `inference_adapter.py`.
  `application.py` now implements status, registry and generation routes; the launcher
  accepts operator-owned project root/device settings and retains one loopback worker.
- One existing `GenerationService` lives for the application lifespan. Its one-model
  cache and loader validation are unchanged. An immediate busy response prevents
  competing jobs; the worker retains its lock after an HTTP disconnect until work ends.
  No queue, streaming, cancellation or authentication infrastructure was added.
- Added `generation_client.js`, `model_picker.js` and `generation_metadata.js`; connected
  the existing page, session/history and output components. Submitting is explicit and
  guarded, independent of rendering. Prior output stays readable while controls lock.
  Registry/response identities are checked, errors retain input, and each tab owns its
  own state. Results and metadata use plain text rendering.
- Prompt/code-point and token limits reuse backend constants. The UI/API reject blank
  openings; the standalone service's documented whitespace-only behavior is unchanged.
  JSON bodies are bounded to 256 KiB, unknown fields/invalid types are rejected, and
  64-bit seeds cross JSON as decimal strings. No checkpoint paths are accepted.
- Technical JSONL logs live under ignored `logs/web/`, with a 2 MiB rotation threshold
  and three backups. Normal completions log identities/counts/timings, not story text;
  technical failure tracebacks stay local. Error responses include request IDs.
- Contract 2.0.0 is generated from implemented schemas/routes. Pydantic 2.14.4 is now
  declared directly in the optional `web` extra; it was already installed transitively.
  Other package versions were retained (76 locked /55 installed). Removed the obsolete
  Phase 14 status module and its stale derived build copy. UI fixtures remain isolated.
- Updated README and added `docs/frontend_integration.md`; earlier UI/design documents
  link to the current integration guide while retaining historical phase context.

### Commands and checks executed

```sh
.tools/uv/bin/uv lock
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
node --test tests/web/*.test.mjs
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/web --junitxml=artifacts/phase15-web-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase15-full-suite.xml
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m transformer.web --device mps --port 8000
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m build --wheel --outdir artifacts/phase15-wheel
```

Python verification also ran `node --check` over production and fixture modules,
exported/compared OpenAPI, matched browser metadata to real registry/completion logs,
compared source/installed/wheel file sets and bytes, and checked preserved artifacts.
Initial lint findings in newly written code were corrected before clean checks. The
JavaScript suite was rerun after tightening catalog-ID validation and clearing stale
request errors on Undo; backend source was unchanged after the full Python run.

### Results and real browser evidence

- **57 web/API tests passed in 8.58 s**, zero failures/skips. Synthetic local checkpoints
  exercise actual inference, model switching, seed precision, zero output, malformed
  requests, missing/corrupt checkpoints/tokenizers, HTTP limits/origins, failure logs,
  recovery, busy rejection and continued lock ownership after a real socket disconnect.
- **460 full Python tests passed in 47.61 s**, zero failures/skips, including the existing
  real MPS checks. **15 JavaScript tests passed**, including duplicate-submit prevention,
  request/result identity, no submit on rerenders, retained output, recovery and tab isolation.
- Actual in-app-browser generation used **MPS with fallback disabled** on this Mac.
  Displayed Model A greedy, Model A sampled recovery and Model B greedy results each
  contained 128 generated IDs. Metadata matched the original registered checkpoint
  hashes and update 1,000. Observed generation durations were 1.347 s, 1.112 s and
  1.087 s respectively; B's switch/load took 0.243 s. These individual diagnostic
  observations are not a hardware benchmark or quality comparison.
- An unsupported snowman character U+2603 produced a safe 422 recovery message while
  retaining the draft and prior result. Editing it allowed sampled generation to finish.
  Loading visibly disabled inputs and kept the previous continuation readable.
- Navigation during/after generation, status refresh, history restoration and a second
  browser tab caused **zero extra generation starts**. The second tab had empty prompt,
  history and output. Metadata wrapped without overflow at 390 px; no application
  console warnings/errors were observed.
- The first temporary browser tab disappeared after submission. Its bounded request
  completed server-side; a fresh tab repeated generation and displayed its real result.
  Accordingly, the retained operational log contains **five starts, four completions
  and one deliberate invalid-character failure**, rather than claiming only three runs.
- Final checks: all **64 original backend source files** and **83 experiment/evaluation
  files** unchanged; all **8 registered artifact references** verified; exact production
  web source/installed/wheel agreement with no UI fixtures or obsolete status asset;
  exported OpenAPI agrees with implementation. The preview server shut down cleanly.

Evidence: `artifacts/phase15-final-verification.json`, `phase15-browser-verification.json`,
three actual browser result JSON files, `phase15-browser-events.jsonl`, rerender baseline,
web/full-suite XML and logs, frontend test log, wheel/build log and five actual screenshots
under `artifacts/phase15-preview/`.

### Limitations and stopping point

The existing small models still produce repetitive and malformed text. This phase
establishes working integration, not improved language quality. Locks cover the one
application process, not unrelated CLI/server processes. Refresh/closing a tab loses
its memory-only history and cannot retrieve an already-disconnected response. Streaming,
cancellation and automatic generation retry are absent. Focused browser checks are not
a full accessibility/multi-browser audit; the Phase 14 native clipboard paste and
permission-denial manual checks remain pending.

**Stop after Phase 15. Next expected phase: Phase 16 advanced controls and comparisons.**
No production training, dataset download, checkpoint modification or Phase 16 feature
construction was performed.

## Phase 16 — controls, session history and comparison (21 November 2026)

**Complete.** Added maximum-token, temperature, top-k, greedy and optional seed
controls, shared sequential comparison, bounded history clearing/reopening and
explicit model-identity preservation. Kept existing registry, API validation,
one-model cache, accelerator lock and plain-text rendering. No training artifacts
were changed. Details: [generation experience](generation_experience.md).

Commands actually executed:
```sh
node --version
node --test tests/web/*.test.mjs
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m pytest -q
.venv/bin/python -m transformer.web --device auto --port 8002
```

Results: 21 JavaScript tests passed; 460 Python tests passed in 54.10 seconds.
The final JavaScript checks include the final registry-refresh protection.
Locked installation succeeded. Actual browser comparison ran both saved models on
MPS with identical 32-token, temperature 0.8, top-k 20, seed 42 settings. Verified
history reopening/clearing, unchanged old-result identity after switching, copy
success feedback, rerender/navigation preservation and desktop/mobile columns.
The browser-exposed input-reset defect was fixed and verified.

Raw actual events and summary: `artifacts/phase16-verification.json`; screenshot:
`artifacts/phase16-comparison.png`. No quality or speed ranking is inferred from
these samples. Native clipboard paste into another application remains untested.
No streaming/cancellation. The verification server was stopped. No outstanding
Phase 16 implementation blockers; next phase requires the user's next prompt.

## Phase 17 — model insights and experiment analytics (21 November 2026)

**Complete.** Added a bounded read-only analytics endpoint; source-backed model
overview; unsmoothed training/validation curves and exact raw tables; primary
held-out metrics, generation throughput, training duration and source exposure;
dataset/evidence disclosures; and a configuration-driven architecture view.
See [model insights](model_insights.md) for source mapping, handling policies and QA.

Commands actually executed:
```sh
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m pytest tests/web -q
.venv/bin/python -m pytest -q
node --test tests/web/*.test.mjs
.venv/bin/python tools/maintenance.py verify-analytics
.venv/bin/python -m transformer.web --device auto --port 8002
```

Full suite: 475 Python tests passed in 46.25 s; 23 JavaScript tests passed.
Actual source audit: two models, 11 files, 2,000 training points and 20 validation
points, all displayed metric/overview projections, parameter count and six model
implementation hashes passed. Browser verified real metrics, raw pagination,
model architecture switching and desktop/mobile layouts. Missing/partial/malformed
data cases use synthetic offline evidence; real run files were not modified.

Reports: docs/model_insights.md and artifacts/phase17-source-verification.json.
Screenshots: artifacts/phase17-insights.png and phase17-training-chart.png.
No new dependencies, generation, training or held-out scoring. Single-seed limitations
remain explicit. No Phase 17 blockers. Stop here and await the next phase prompt.

## Phase 18 — visual refinement and usability review

Implemented minimum control text/target sizing, stronger keyboard focus, information
page spacing, tablet comparison stacking and clearer disabled/loading feedback.
Inspected all four loaded pages at 360/390/768/1280/1440 px with no page overflow,
plus isolated mobile loading/error/missing-model/long-output fixtures. Copy feedback
and one keyboard focus transition verified. 23 JavaScript tests passed; locked
wheel installation succeeded. See docs/phase18_usability.md and recorded browser
evidence. Browser zoom, full keyboard journey, assistive speech and quantitative
contrast audit remain pending; verification is partial, not certification.

## Phase 19 — end-to-end verification

Completed real-checkpoint browser workflows: registry, prompt, A greedy, B sampling,
history restore, shared comparison, charts, architecture, copy confirmation, history
clear and unknown-character error recovery. Full regression: 475 Python tests passed
in 48.40 s; 23 JavaScript tests passed. Failure simulations used isolated artifacts.

Added tools/maintenance.py verify-web-smoke: ten bounded sequential requests verified identity,
parameters, limits, warm-cache reuse and switching. Sampled RSS 380,289,024–380,502,016
bytes; no obvious growth beyond that small range. No long-term leak claim.
Report: docs/phase19_end_to_end.md; raw resources: artifacts/phase19-resources.json.
No new concrete defects found. Earlier accessibility verification gaps remain explicit.
Stop here; broader performance work belongs to Phase 20.

## Phase 20 — measured maintenance

Complete. Reviewed backend gate, hardware benchmarks, Phase 19 end-to-end/resources
and known usability-verification gaps. Reproduced and fixed two analytics log
defects: overflow on extremely large integers and false truncation warnings at
exactly 10,000 updates. No saved experiment was affected.

Profiling demonstrated repeated parsing of shared registered JSON. Added only
request-local sharing keyed by path and expected SHA-256; each new request reads
and verifies again. The previous implementation is preserved as a benchmark input.

Eleven paired alternating trials after two warmups per path: median analytics
projection 23.0287 ms → 16.4358 ms (28.63%, 6.59 ms saved). Complete valid responses
were equal and eleven source hashes unchanged. No generation-speed claim.

Checks executed: two intentional pre-fix regression failures; 15 focused tests
passed; full suite **478 passed in 45.75 s**; **23 JavaScript tests passed**; real
analytics source audit passed for both models. Locked wheel reinstall succeeded.
Report and reproduction commands: docs/phase20_maintenance.md.
Raw benchmark, profile, baseline source and regression output: artifacts/phase20/.

Scientific artifacts, tokenizer, architecture, interpretation, evaluation weighting
and budgets remain unchanged; no reruns required. Existing accessibility audit and
long-duration soak limits remain explicit. Supported improvements complete; stop.

## Phase 21 — complete project documentation

Complete. Updated README to reflect implemented application and actual limitations,
added the documentation map, structure, methodology, model/data notes,
reproducibility, checkpoint recovery and troubleshooting guides. Retained focused
architecture/data/tokenizer/training/evaluation guides and historical reports.

Verified CPU diagnostics, dry-run training configuration, model listing, web help,
two token batches, eight-token real CPU generation, fourteen CLI help commands,
README configuration paths and local Markdown file links (none missing).
Evidence: artifacts/phase21/; details: docs/documentation_verification.md.

No dataset download, training/resume, full evaluation or hardware benchmark executed.
Historical code-identity resume limitations, missing-on-fresh-checkout artifacts,
accelerator variation and partial accessibility verification are explicit.
No scientific artifacts or implementation changed. Stop; university report and
viva guide belong to later phases.

## Extended English story training

Completed the separate 8,192-update MPS run from 25,000 TinyStories training
documents. Registered the validation-selected update 8,000 checkpoint and verified
generation on both reported openings through the actual app endpoint. Output
includes readable English sentences, with uneven grammar and story relevance.
Results and reproduction are recorded in [extended English training](extended_english_training.md).

Generate now uses the original registered TinyStories + SimpleStories checkpoint
directly. Removed the model picker and verified the live generation endpoint
with that checkpoint. An additional MPS run using the expanded combined token
stream was started under configs/training/english-stories-combined.json.

Completed the 33.6-million-target joint TinyStories + SimpleStories MPS run.
The active app catalog now serves only its validation-selected update 8,192.
Model insights shows that run and its real samples. English is more readable,
with prompt relevance still inconsistent. See [record](extended_english_training.md).

Started an expanded TinyStories + SimpleStories run on MPS with the
`english-stories-expanded-v1` prepared data and a 67,108,864-target budget.
See [expanded training](expanded_story_training.md) for the data selection,
commands and status.

Completed two local story-opening fine-tuning runs with Qwen2.5 instruction
weights and 96 opening/continuation pairs from TinyStories and SimpleStories.
The app serves the 1.5B adapter at update 96. The older custom-model runs
remain research artifacts. See [story continuation training](story_continuation_training.md).

## 22 November 2026 — Folder simplification

Organized operator scripts into training/, fine_tuning/ and tools/. Separated
packaged frontend assets into frontend/. Added main.py and updated
folder/command documentation. Removed only regenerable caches/build output.
Preserved all 854 existing configuration/data/run/artifact files and verified active
registry hashes. Reinstalled package; 140 Python and 25 JavaScript checks passed;
20 HTTP assets and real MPS generation passed. Website restarted on port 8000.
See [cleanup verification](folder_cleanup.md) for commands, moves and limitations.

### Further folder flattening

Moved source areas directly to top-level purpose folders and preserved installed
transformer imports using explicit setuptools mappings. Combined training code
with its scripts. No model/generation logic or frontend bytes changed. All 480
Python and 25 JavaScript tests passed. Reinstalled and restarted the website;
20 assets and API routes passed, and a bounded MPS greedy generation matched
before/after exactly. Data and model evidence retained. See folder_cleanup.md.

### Two-to-three source files per folder

Consolidated Python modules and frontend assets without changing model settings,
weights, HTML body or CSS rules. Full suite: 467 Python and 25 JavaScript checks
passed. Verified real MPS greedy and seeded continuations against before-change
results, matching exactly. Inspected Generate, Insights, Architecture and real UI
generation in browser. Historical implementation evidence retained; current
commands documented in project_structure.md and folder_cleanup.md.
