# Transformer: Phase 14 Generate interface

**Historical Phase 14 record.** Inference was intentionally disconnected in that phase.
The current connected implementation and commands are documented in
[Phase 15 frontend integration](frontend_integration.md).
The existing backend gate passed before frontend work. This phase adds no model
training, inference endpoint, frontend dependency or fabricated application result.
The approved off-white, graphite and deep-teal design system is retained.

## Use the local interface

Run from the project directory:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m transformer.web --port 8000
```

Open `http://127.0.0.1:8000`. Stop the server with Ctrl-C. The regular wheel install
is intentional: this host does not reliably expose editable installations. Reinstall
after changing packaged files, then reload the browser; there is no development
hot reload. No Node runtime is needed to use the application.

Write an opening or choose one of three authored story starters. These examples
are prompts, not generated responses. Choose a model, writing style and approximate
continuation length. **Save opening** stores the draft and preferences in this tab's
session history. **Clear** and example replacement offer one-step **Undo**. Typing or
changing settings starts a new edit and clears that undo snapshot. History keeps the
newest ten saved items, deduplicates identical openings/settings and restores them
on selection. Navigation preserves state; refresh/closing the tab clears it. Nothing
is written to local storage or sent to a model.

The disabled **Generate continuation** action has an adjacent explanation. Model
choices identify the existing research versions; they do not claim live checkpoint
availability. **Check interface** checks the local UI server only. A successful check
still says generation is not connected. An unavailable server keeps the draft and
history, gives recovery instructions, and offers a real retry.

## Components and state

| File under `frontend/` | Responsibility |
| --- | --- |
| `index.html` | Page templates, visible labels, semantic form and landmarks |
| `tokens.css`, `application.css` | Approved shared design foundations |
| `generate.css` | Generate layout, examples, history and reading surface |
| `application.js` | Navigation and page lifecycle; skip-link reload recovery |
| `generate_page.js` | Form event binding and state-to-view updates |
| `session_state.js` | Validated draft, bounded tab history, undo and output state |
| `session_history.js` | Safe history rendering and restore actions |
| `story_output.js` | Feedback, prompt/continuation separation and copying |
| `service_status.js` | Four-second local status check; no inference transport |
| `components.js` | Shared text-only feedback component |

The state interface accepts explicit completed output for the future service adapter
and isolated component tests. The normal application never constructs a completed
result or loading-model state. Session state is private to the application module;
there is no test query parameter, global fixture switch or public fixture route.

| State | Behavior and verification |
| --- | --- |
| Initial empty | Guidance with no story; unavailable action explained |
| Ready | Valid opening can be saved; generation remains unavailable |
| Loading | Inputs/actions locked, output marked busy, no invented progress, cancellation or streaming; isolated fixture only |
| Success | Original prompt separately disclosed; selectable continuation, model label, copy action and stop reason; isolated synthetic fixture only |
| Validation error | Empty/whitespace-only, over-limit or malformed Unicode input receives a recovery message; text is preserved |
| Backend unavailable | Real status request failure and isolated fixture; retry preserves session state |
| Missing checkpoint | Helpful choose/restore-model explanation; isolated fixture only until registry integration |

The prompt limit is 16,384 Unicode code points, matching the backend's size unit.
The UI requires a non-whitespace opening even though the backend can handle empty
input. Unknown vocabulary characters remain the service's responsibility in Phase 15.
No arbitrary checkpoint paths are accepted. The presentation state is not a replacement
for the backend's validated request/result types.

Completed output remains associated with its original prompt when the current draft
changes. A later failure preserves that output as a previous result. Immediate EOS
shows an honest explanation with copying disabled. Text is rendered using
`textContent`; literal HTML stays text. Paragraphs use `white-space: pre-wrap`, long
words wrap, and the serif reading area is capped at 66 characters. The prompt is in
its own keyboard-operable disclosure; the continuation is focusable/selectable.
Copy writes only the continuation and reports success after the Clipboard API promise
resolves. Failure preserves the text and offers manual selection/copy instructions.
See the primary references for [paragraph wrapping](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/white-space)
and [Clipboard API behavior](https://developer.mozilla.org/en-US/docs/Web/API/Clipboard/writeText).

Desktop/tablet use a writing column and preference/history column. At 700 px and
below, the reading order is opening, preferences, actions, continuation, history.
At 380 px and below, navigation occupies two rows. Native buttons/selects/radios,
explicit labels, error descriptions, visible focus rings, status/alert semantics,
skip navigation and non-color-only disabled states are retained.

## Isolated UI testing

The fixture harness imports the same installed templates/styles/components as the
normal app. Only the separate test server serves its scenario selector and synthetic
text. Its banner and output explicitly identify test fixtures. It loads no model,
uses no network service, and performs no timed simulation of generation.

```sh
.venv/bin/python tools/maintenance.py preview-ui-fixtures --port 8001
```

Open `http://127.0.0.1:8001`. Select empty, ready, static loading, synthetic long output,
validation error, unavailable backend, missing checkpoint, immediate EOS, or prompt
length boundaries. Test files remain under `tests/web/ui_fixtures/` and are excluded
from the wheel. The normal application's fixture URLs return 404. The earlier Phase 13
`#/foundations` design specimens were moved into this harness at
`http://127.0.0.1:8001/fixtures/foundations.html`; they are no longer an application route.

Executed automated checks:

```sh
node --test tests/web/frontend_state.test.mjs
.venv/bin/python -m pytest tests/web --junitxml=artifacts/phase14-web-tests.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase14-full-suite.xml
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python -m build --wheel --outdir artifacts/phase14-wheel
```

Node 24.14.0 is only needed for the developer state tests, which use its built-in
test runner; there are no JavaScript test dependencies. Nine JavaScript files also
passed `node --check`. The first Ruff run found an import order issue and one long
line in new/edited web code; both were fixed before the final clean check.

## Verification results and limits

- **10 JavaScript state tests passed**, covering Unicode validation, undo, history
  bounds/deduplication, snapshots/subscriptions, loading locks, preserved results,
  EOS, HTTP/network/JSON/version failures and request timeouts.
- **27 final web HTTP tests passed in 0.93 s**; **429 full Python tests passed in 38.88 s**,
  zero failures/skips. The full run preceded the final additional fixture-isolation
  case; all 27 web cases were rerun afterward. The existing six actual MPS tests
  are included. Tests use
  bounded fixtures; no new production training run was started.
- Browser inspection used the local Codex in-app browser on this Mac. Generate was
  checked at 1440 ×1000, 768 ×1024 and 320 ×740; long fixture output at 390 ×844.
  Document width matched each viewport. All required states, long prompts/output,
  session restoration, disclosure/copy keyboard controls and focus rings were checked.
- A real server stop/restart verified the unavailable state and retry recovery while
  preserving the draft/history. Reload clears history as documented. Keyboard testing
  exposed a skip-link hash reload that could leave the page blank; the router was
  corrected and reloading `#main-content` then displayed the editor successfully.
- Long output retained nine paragraphs; literal `<script>`/`<b>` text created no
  injected elements. Immediate EOS disabled copying. No application console
  errors/warnings were observed. Both owned preview servers were stopped afterward.
- Clipboard writes resolved and success feedback appeared, but the automation virtual
  clipboard did not receive native Clipboard API data. **Exact native paste and
  permission-denied copy behavior remain manual checks**, not asserted passes.
  Keyboard/contrast checks are focused inspection, not a full accessibility audit
  or certification across browsers and assistive technologies.

Evidence: `artifacts/phase14-{frontend-tests.log,web-tests.xml,full-suite.xml,full-suite.log,
browser-verification.json,fixture-checks.json,final-verification.json}`,
`artifacts/phase14-wheel-build.log`, and actual screenshots in `artifacts/phase14-preview/`.


## Phase 15 handoff

Implement the planned `GET /api/v1/models` and `POST /api/v1/generations` adapter
through the existing Python inference service, then enable Generate only when ready.
Map real results/errors into the focused UI state and preserve the original prompt
and settings snapshot. Keep one model cached and serialize service access. The
design contract is now version 1.1.0; only status is implemented. Do not substitute
the fixture server for inference, expose arbitrary files, or claim streaming or
cancellation without implementing them. Compare/insights/architecture remain their
Phase 13 shells pending their own construction scope.
