> Historical Phase 15 report. Phase 16 controls and comparison are now implemented; see [generation experience](generation_experience.md).

# Transformer: Phase 15 frontend and inference integration

**Complete: Generate uses actual registered, trained checkpoints.** The existing
Python inference service owns model loading, tokenization and sampling. The web
adapter adds strict HTTP types, bounded requests, immediate busy responses and
private technical logs. The browser owns only its tab-local draft, history and
presentation state. Compare and advanced controls remain Phase 16 work.

## Run locally

From the project directory:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m transformer.web --device auto --port 8000
```

Open `http://127.0.0.1:8000`. Select a registered model, enter an opening and click
**Generate continuation**. Consistent uses greedy next-token selection; Varied uses
the temperature slider. Length presets request 128, 256 or 512 new character tokens.
EOS can stop sooner. Expand **Generation details** for the actual checkpoint SHA-256,
update, model ID, vocabulary identity, counts, device, sampling seed and timing.

`--device mps` requires MPS without silent fallback; `--device cpu` explicitly uses
CPU. `--project-root /path/to/project` selects the trusted operator-owned project
when launching elsewhere. The launcher binds to 127.0.0.1 with one worker. Stop with
Ctrl-C. Reinstall after editing packaged files, then restart/reload; this setup uses
a regular wheel installation because this host hides editable package paths.

The selected registry must exist at `configs/inference/model_registry.json`. Its
large local artifacts are not bundled in the wheel. Missing artifacts remain errors;
no synthetic checkpoint or generated text is substituted. The real versions are
`transformer-tinystories-v1` and `transformer-combined-v1`, both selected at update
1,000. Their recorded scientific limitations remain unchanged.

## Implemented API

The exported OpenAPI contract is version **2.0.0**, generated from the implemented
schemas/routes, replacing the earlier design-only contract. `/openapi.json` describes
all three implemented operations. The packaged copy is
`backend/contracts/inference.openapi.json`.

| Operation | Meaning |
| --- | --- |
| `GET /api/v1/status` | Interface version, selected device, service availability and whether its generation slot is busy |
| `GET /api/v1/models` | Explicit registry entries, display names, kind, pinned checkpoint/tokenizer identity and file availability |
| `POST /api/v1/generations` | One prompt/settings snapshot → complete continuation, stop reason and verified metadata |

`generation_enabled` indicates that the service initialized; it is not a claim that
all models are loaded or verified. Catalog `registered` means the required local
files exist and are accessible. Full checksum, tokenizer, configuration, weight and
metrics checks remain in the existing loader on first use or file change. A registered
but corrupt artifact is rejected during loading. No weights are loaded by catalog refresh.

Example request:

```json
{
  "model_id": "transformer-tinystories-v1",
  "prompt": "Tom found a small box under a tree.",
  "settings": {"mode": "greedy", "max_new_tokens": 128, "unknown_policy": "error"}
}
```

Strict Pydantic schemas reject unknown fields, coercions such as boolean/string token
limits, malformed IDs and invalid sampling values. Limits reuse the backend's
`MAX_PROMPT_CHARACTERS` (16,384 Unicode code points) and `MAX_NEW_TOKENS` (1,024).
The web UI/API additionally require a non-whitespace opening; the standalone Python
service's documented whitespace-only behavior is unchanged. Invalid Unicode surrogates
are rejected. The 256 KiB JSON body limit is enforced while receiving bytes, including
bodies without Content-Length; non-JSON requests receive 415.

Greedy requests omit temperature, top-k and seed. Sample requests require finite
positive temperature; optional top-k must fit the selected vocabulary. Optional seeds
use decimal strings bounded by 2^63−1 so JavaScript cannot round them. The backend
converts them to integers and the response returns requested/effective seeds as strings.
The current UI exposes only the existing simple controls; the larger API settings range
reuses the earlier backend contract. Zero requested tokens is valid through the API
and returns an honest empty continuation. The UI presets always request positive output.

Unknown-character handling defaults to `error` in the interface. The service identifies
the code point and zero-based position; the user can edit and retry. The API also
supports the backend's explicit `replace` policy. There is no implicit text normalization
or silently fitted vocabulary. The response omits checkpoint paths and raw token arrays.

Errors have the envelope `{ "error": { "code", "message", "retryable", "field", "request_id" } }`.
HTTP statuses distinguish invalid input (422), unknown model (404), busy service (409),
oversized body (413), wrong content type (415), disallowed browser origin (403),
missing/unverifiable artifact or unavailable service (503), and unexpected failure (500).
The interface renders safe recovery text and preserves the opening and any previous result.

## Resource ownership and session behavior

One `GenerationService` is created at application lifespan startup and closed at
shutdown. It retains its existing one-model cache, evicts before switching models,
freezes parameters and runs generation in inference mode. An application-level,
nonblocking lock rejects competing requests immediately. The synchronous worker
owns that lock until generation ends, even if the HTTP client disconnects. A disconnected
request is not cancelled and is not automatically retried. There is no generation queue.

The existing service's runtime lock and model lock remain unchanged. The HTTP worker
runs outside the event loop, so status/asset requests remain responsive while inference
runs. Use the documented single-worker launcher; these locks do not coordinate unrelated
CLI processes, extra server processes or other applications on the same accelerator.

Browser submission is an explicit action outside the renderer. An in-flight guard
prevents duplicate submits, while the API independently guards all tabs. Repeated
rendering, navigation, history restoration and status refresh send no generation request.
Input controls lock during the request, while prior output stays readable. The interface
waits for complete text and offers neither streaming nor cancellation. A network failure
warns that work may still be running and asks the user to check the service before retrying.

Draft/history state lives in each tab's own module instance. No server-side session
stores prompts or results, and one tab cannot overwrite another tab's state. Refresh
clears tab memory and cannot recover an in-flight result after the old tab is gone.
Completed history restores its original prompt and settings; changing the current model
never relabels a previous result. The client rejects response identities/settings that
do not match its captured request and registry snapshot.

Generated text, registry labels, errors and metadata use `textContent`, never generated
HTML. Paragraphs and long words wrap. Prompt and continuation are separate; EOS/length
stop reasons remain explicit. The API does not accept checkpoint paths, URLs or uploads.
Local access is the default. Same-origin generation is enforced; an exact optional
localhost development origin can be enabled through the existing launcher option.
No authentication infrastructure or public deployment was added.

## Technical logs

`logs/web/inference.jsonl` records startup/shutdown, request IDs, model identity,
completion counts/timings/cache hits, validation categories and failure tracebacks.
Rotation uses a 2 MiB threshold and three backups. Normal request/completion events
omit prompt and continuation text. Tracebacks stay local; browser responses contain
safe messages and request IDs. These ignored operational logs do not modify experiment
records. Restore files or correct device/registry settings before restarting a service
that failed initialization; a catalog refresh alone cannot recreate that service.

## Files and dependencies

- `web/schemas.py`: strict request/result/catalog/status/error contracts.
- `web/http_boundary.py`: bounded JSON receipt, origin checks and safe error envelopes.
- `web/inference_adapter.py`: service lifecycle, job admission, result projection and logs.
- `web/application.py`, `__main__.py`: local routes, lifespan and operator configuration.
- `frontend/generation_client.js`: catalog/status refresh, explicit submit and identity checks.
- `frontend/model_picker.js`, `generation_metadata.js`: focused registry/metadata views.
- Existing session/page/output modules now handle real responses, safe errors and history.
  The obsolete Phase 14 `service_status.js` was removed, including its derived build copy.

Pydantic **2.14.4**, already installed transitively, is now declared directly in the
optional `web` extra because the adapter imports it. FastAPI **0.165.0**, Uvicorn
**0.54.0** and all other installed versions were retained. The existing `uv.lock`
remains the sole dependency record: 76 resolved /55 installed packages. No JavaScript
framework or additional test package was introduced. Node **24.14.0** is used only for
native developer tests.

Primary implementation references: [strict schema validation](https://pydantic.dev/docs/validation/latest/concepts/strict_mode/),
[unknown-field rejection](https://pydantic.dev/docs/validation/latest/api/pydantic/config/),
[FastAPI validation errors](https://fastapi.tiangolo.com/tutorial/handling-errors/), and
[Starlette's worker-thread execution](https://www.starlette.io/threadpool/).

## Executed verification

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

- **57 web/API tests passed in 8.58 s.** Offline synthetic checkpoints verify actual
  service inference, catalog identity, switching/cache bounds, strict invalid input,
  seed roundtrip, unknown characters, zero output, missing/corrupt files, safe error
  logging, bounded bodies, origin rejection, busy admission and recovery. A real socket
  disconnect while a worker is held proves the slot remains occupied until completion.
- **460 full Python tests passed in 47.61 s**, zero failures/skips, including the existing
  actual MPS checks. No production training was launched. **15 JavaScript tests passed**,
  including duplicate-submit prevention, request/result identity, no generation on
  rerenders, retained output, corrected-input retry, independent sessions and timeout.
- Actual browser verification ran on the Mac's **MPS**, with fallback disabled. Three
  displayed completions were cross-checked against the registry and JSONL completion
  events: Model A greedy (128 tokens, 1.347 s generation), Model A sampled recovery
  (128, 1.112 s), and Model B greedy after switching (128, 1.087 s plus 0.243 s loading).
  These are individual diagnostic observations, not a throughput benchmark.
- The first temporary browser tab disappeared after submitting. Its bounded request
  completed on the server; a fresh tab repeated the operation and displayed the actual
  result. Logs therefore contain **five starts: four completions and one deliberate
  unknown-character failure**. No extra starts appeared after navigation, status refresh,
  history restoration or opening another tab. That second tab had no prior draft/output.
- The invalid U+2603 request showed the character/position, preserved text and previous
  output, and recovered after editing. Prior output remained readable during loading.
  Metadata wrapped at a 390 px viewport without horizontal overflow. No application
  console warnings/errors were observed. The owned preview server was stopped afterward.

Raw evidence is under `artifacts/phase15-*`: web/full-suite XML and logs, frontend test
log, actual browser result JSON, browser event log, verification JSON, wheel/build log
and five screenshots under `artifacts/phase15-preview/`. The final consistency check
compares production source, installed files and wheel, preserves all 64 original
backend source hashes and 83 experiment files, and verifies the eight registry references.

Acceptance passed for the connected local Generate page. These small checkpoints still
produce repetition and malformed sentences; integration checks establish no language-
quality improvement. Streaming, cancellation, advanced controls and comparisons were
not added. The focused browser inspection is not a complete accessibility or multi-browser
audit. The Phase 14 native clipboard paste/permission-denial manual-check limitation remains.
