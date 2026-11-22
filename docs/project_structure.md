# Simple project structure

Each main code folder has at most three source files. `__init__.py` is Python's
small package entry file; the implementation is in the other named files.

| Folder | Main files and responsibility |
| --- | --- |
| `frontend/` | `index.html` — page; `styles.css` — all styling; `application.js` — interactions and chart data |
| `backend/` | `application.py` — API, validation and analytics; `__main__.py` — server launcher; `__init__.py` |
| `training/` | `engine.py` — training, validation and checkpoints; `experiments.py` — prepare/check/register and experiment orchestration; `__init__.py` |
| `fine_tuning/` | `train_story_continuation.py`, `review_story_continuation.py`, `register_story_continuation.py` |
| `data/` | `preparation.py` — acquisition and cleaning; `loading.py` — token batches and source attribution; `__init__.py` |
| `model/` | `transformer.py` — custom network layers; `__init__.py` |
| `inference/` | `service.py` — registry and generation; `continuation_model.py` — story adapter; `__init__.py` |
| `configs/` | `settings.py` — validated settings; `__init__.py`; saved JSON settings remain separate |
| `tokenization/` | `character_tokenizer.py`, `token_preparation.py`, `__init__.py` |
| `evaluation/` | `metrics.py`, `comparison.py`, `__init__.py` |
| `runtime/` | `device.py` — device support and diagnostics; `__init__.py` |
| `benchmarking/` | `worker.py`, `hardware_benchmark.py`, `__init__.py` |
| `verification/` | `checks.py`, `__init__.py` |
| `main/` | `cli.py`, `__main__.py`, `__init__.py` |
| `tools/` | `maintenance.py` — maintenance subcommands; README |

`main.py` at the project root starts the website:

```sh
.venv/bin/python main.py --device auto
```

Open http://127.0.0.1:8000. Run commands from the project root. The existing
`.venv/bin/python -m transformer.web --device auto` command also works.

Custom-training helper commands are now:

```sh
.venv/bin/python training/experiments.py prepare --help
.venv/bin/python training/experiments.py check --help
.venv/bin/python training/experiments.py register --help
.venv/bin/python tools/maintenance.py --help
```

Examples: `tools/maintenance.py export-story-charts` refreshes the measured-data
section inside `frontend/application.js`; `preview-ui-fixtures` serves synthetic
UI states separately. These commands do not need to run to open the website.

## Data and records

The three-file limit applies to source code. Downloaded data, base models,
tokenizers, trained checkpoints in `runs/`, JSON configurations, reports in
`artifacts/`, tests, documentation and logs keep their individual files. Do not
combine or delete these: saved records and registry checksums depend on them.

## Editing and installation

`pyproject.toml` maps the source folders to the installed `transformer` package.
The implementation uses normal Python imports and readable JavaScript sections;
there is no code archive or runtime source extraction. Internal module paths have
changed with consolidation; current guides and tests use the new paths.

After editing Python or frontend code, reinstall the package, then restart the
server. This checkout uses a regular installation, not editable installation:

```sh
.tools/uv/bin/uv pip install --python .venv/bin/python --no-deps --no-build-isolation --reinstall .
```

Historical training records retain their original implementation hashes. A
source-integrity check against that older layout requires review; those checks
are not bypassed or relabelled as passing. Model weights, dataset content and
published experiment results have not been changed by this reorganization.
