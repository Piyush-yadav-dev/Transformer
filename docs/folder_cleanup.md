# Folder cleanup verification — 22 November 2026

## Changes

Grouped custom-training scripts in `training/`, story adapter scripts in
`fine_tuning/`, and maintenance scripts in `tools/`. Moved website assets from
`backend/static/` to `frontend/` and updated serving,
package-data rules, chart export, tests and documented commands. Added `main.py`
and a concise folder guide. Browser `/assets/` URLs remain unchanged.

Removed regenerable build/dist output and Python, pytest and Ruff caches. Retained
all datasets, model weights, experiment records, logs, environments and dependency
locks. Historical reports are retained as evidence, not treated as unused files.

## Checks executed

- Before and after: `.venv/bin/python -m pytest -q tests/web tests/inference --disable-warnings`
  — 140 passed on each run.
- `node --test tests/web/*.test.mjs` — 25 passed.
- `.tools/uv/bin/uv pip install --python .venv/bin/python --no-deps --no-build-isolation --reinstall .`
  — built and installed successfully with packaged frontend assets.
- `.venv/bin/python main.py --help` — passed.
- `.venv/bin/python fine_tuning/train_story_continuation.py --help` — passed.
- `.venv/bin/python training/experiments.py prepare --help` — passed.
- `.venv/bin/python -m compileall -q training fine_tuning tools main.py` — passed.
- Restarted existing local website using `.venv/bin/python main.py --device auto --port 8000`.
- HTTP checks: homepage, model catalog, analytics and all 20 frontend assets passed;
  served asset bytes matched source files.
- Real HTTP generation: existing `transformer-story-continuation-v2`, MPS,
  greedy mode, 32-token limit — nonempty continuation, correct identity and limit.
- All 854 pre-existing files in configs/data/runs/artifacts retained identical sizes
  and modification times. Active checkpoint/configuration/metrics/tokenizer SHA-256
  hashes verified against the unchanged model registry.

No retraining or experiment changes. This was a bounded functional check, not a
new assessment of model quality. No new visual/browser inspection was performed;
frontend content and styling were not redesigned. Full training was not rerun.

## Script moves

- `scripts/prepare_english_run.py` → `training/experiments.py prepare`
- `scripts/check_english_run.py` → `training/experiments.py check`
- `scripts/register_english_run.py` → `training/experiments.py register`
- `scripts/train_story_continuation.py` → `fine_tuning/train_story_continuation.py`
- `scripts/register_story_continuation.py` → `fine_tuning/register_story_continuation.py`
- `scripts/review_story_continuation.py` → `fine_tuning/review_story_continuation.py`
- `scripts/verify_backend.py` → `tools/maintenance.py verify-backend`
- `scripts/export_story_charts.py` → `tools/maintenance.py export-story-charts`
- `scripts/benchmark_analytics.py` → `tools/maintenance.py benchmark-analytics`
- `scripts/verify_analytics.py` → `tools/maintenance.py verify-analytics`
- `scripts/preview_ui_fixtures.py` → `tools/maintenance.py preview-ui-fixtures`
- `scripts/verify_web_smoke.py` → `tools/maintenance.py verify-web-smoke`
- `scripts/verify_inference.py` → `tools/maintenance.py verify-inference`

## Further simplification — top-level source folders

Removed the src/transformer source nesting. The pyproject package-directory mapping
keeps installed Python imports and package-relative artifact paths unchanged.
Training implementation and scripts now share training/. Frontend assets are
directly in frontend/, API code in backend/, CLI code in main/, and configuration
code joins the existing configs/. Data preparation code joins data/; ignore rules
preserve tracking for its Python code while excluding downloaded data.

No model algorithms, generation settings, weights or frontend assets changed.
Internal dataset/checkpoint folders remain because saved records refer to them.
All pre-existing configs/data/runs/artifacts files retain sizes and modification
times; registered checkpoint, tokenizer, configuration and metrics hashes verified.

Checks actually executed after the layout change:

- Package reinstall with uv pip, no dependency changes: passed.
- `.venv/bin/python -m pytest -q --disable-warnings`: **480 passed**, 50.26 seconds.
- `node --test tests/web/*.test.mjs`: **25 passed**.
- `.venv/bin/python main.py --help`: passed.
- Reopened the website on port 8000 using main.py with device auto.
- HTTP homepage, status, model catalog and analytics: passed.
- All 20 HTTP frontend assets match source bytes; original frontend hashes unchanged.
- Same prompt and greedy 32-token request before and after restart on MPS:
  identical continuation, token count, stop reason, settings and checkpoint identity.
  Timing/request IDs naturally differ. Stochastic generations are not promised to
  repeat without controlled settings; the sampling algorithm is unchanged.
- Removed regenerated build/test caches after verification.

No visual redesign or new browser visual inspection. No training performed.

Source-to-package mappings:

- `main/` → `transformer`
- `backend/` → `transformer.web`
- `configs/` → `transformer.configuration`
- `frontend/` → `transformer.frontend`
- `training/` → `transformer.training`
- `runtime/` → `transformer.runtime`
- `tokenization/` → `transformer.tokenization`
- `verification/` → `transformer.verification`
- `model/` → `transformer.model`
- `inference/` → `transformer.inference`
- `evaluation/` → `transformer.evaluation`
- `benchmarking/` → `transformer.benchmarking`
- `data/` → `transformer.data`
- `data/token_loading/` → `transformer.data.token_loading`

## Code consolidation — 22 November 2026

Each primary code folder now contains at most three source files, counting Python
package initializers. Combined 54 Python modules into 12 implementation files;
maintenance commands share tools/maintenance.py. Frontend has exactly index.html,
styles.css and application.js. Documentation is in docs/ rather than extra source
folder READMEs. Generated data/checkpoints, JSON settings, tests and historical
reports remain separate.

Verification executed:

- Clean package rebuild/reinstall with uv pip: passed.
- Full Python suite: 467 passed in 47.75 seconds. The earlier 480 cases included
  13 additional per-asset URL cases that are obsolete after bundling. Functional
  assertions remain; served asset tests now target the combined files and favicon.
- JavaScript suite: 25 passed after updating imports to named application sections.
- The first consolidation test run exposed one obsolete training source path;
  updated it and reran the complete suite successfully.
- All three frontend assets served with bytes matching source; favicon unchanged.
- HTML body is identical to the previous page. CSS contains the exact original
  rules in the same order. JavaScript functions retain their bodies in named,
  isolated sections, with one shared module for tests and the application.
- Real MPS requests before and after restarting the server, same prompt and
  32-token budget: greedy mode and seeded sampling (temperature 0.65, top-k 10,
  seed 42) returned identical continuations, token counts, stop reasons, checkpoint,
  tokenizer and registry identities. Request IDs and timing are not equality criteria.
- Active registry checkpoint/tokenizer/configuration/metrics SHA-256 hashes passed.
- Browser inspection: Generate layout, Model insights values/charts, Architecture
  layout, loading state and successful real continuation rendered. No new responsive
  breakpoint sweep was performed.
- Training prepare and fine-tuning --help, maintenance --help: passed.

No retraining or edits to dataset/checkpoint contents. Internal Python imports and
maintenance command paths changed; current docs and tests were updated. Historical
experiment implementation hashes are preserved, not rewritten to endorse the new
source layout. Historical source-integrity checks still require review when those
hashes differ; this does not prevent existing checkpoint inference. Controlled
sample equality is evidence for these cases, not a cross-hardware determinism claim.
