# Transformer: frontend direction and application structure

**Phase 15 update:** Generate now uses the existing inference service and actual
registered checkpoints. See [frontend integration](frontend_integration.md) for the
implemented API and verification. This document records the original Phase 13 design
and subsequent UI direction; its planned adapter notes are superseded by that guide.

## Phase boundary and prerequisite

Phase 13 implements a local application **shell**, not working generation pages.
Before starting, the Phase 12 verification JSON was checked: all four required
stages passed, its 64 backend source hashes matched, both registered models' eight
artifact references verified, and all 83 original run/evaluation files were unchanged.
See the [backend gate report](backend_verification_report.md).

Implemented: four-area navigation, shared visual foundations, a local draft editor,
preference previews, accessible feedback components, responsive layouts, local
serving and a documented future inference contract. Generate is visibly disabled.
Compare, insights and architecture contain clearly identified page structure, with
no fabricated continuations, training scores or charts. The backend remains usable
through its existing Python service and CLI.

## Framework decision

Use **buildless HTML, CSS and JavaScript modules, served by FastAPI/Uvicorn**.
The four-area editorial interface needs direct control over typography, responsive
reading columns, keyboard flow and paired results. A small native browser shell
gives that control without a React/Vite/Node installation or build step. Streamlit
would reduce initial Python UI work, but its layout/rerun conventions provide less
control over this particular reading and comparison experience.

FastAPI serves the local shell now and will provide a thin Python inference adapter
in Phase 15. All tokenization, model loading, caching and sampling stay inside the
existing `GenerationService`; no browser model, duplicate inference code or hosted
service is introduced. Start a single worker and eventually own one service instance.

The optional `web` extra contains FastAPI **0.165.0** and Uvicorn **0.54.0**, resolved
with their dependencies in the existing `uv.lock`. Their package metadata supports
Python 3.13 (minimums 3.11 and 3.10 respectively); the locked environment passed
dependency checks. The backend's original runtime requirements remain separate.
No frontend dependency is required for ordinary CLI inference/training.

Primary references checked during implementation: [FastAPI package metadata](https://pypi.org/project/fastapi/0.165.0/),
[Uvicorn package metadata](https://pypi.org/project/uvicorn/0.54.0/),
[static serving](https://fastapi.tiangolo.com/tutorial/static-files/),
[explicit CORS origins](https://fastapi.tiangolo.com/tutorial/cors/), and
[Uvicorn settings](https://uvicorn.dev/settings/).

## Visual system

The visual direction is an editorial writing desk: warm paper, graphite, a restrained
deep teal, ruled sections and readable text. There are no decorative metrics, gradients,
large hero illustrations or repeated rounded cards. The small square controls have
3 px corners. State feedback uses words and structure as well as color.

| Foundation | Implementation |
| --- | --- |
| Page / writing surface | `#f5f3ed` / `#fcfbf7` |
| Main / secondary text | `#282f2e` / `#626862` |
| Accent / hover | `#245f56` / `#194b43` |
| Rules / control outlines | `#d6d8cf` / `#8b958d` |
| Error meaning | Muted brick `#8b3930`, reserved for errors |
| Editorial font | Georgia, with Times New Roman/serif fallback |
| Interface font | Local system sans-serif; no remote font request |
| Typography | 46 px maximum page title, 24 px subheading, 18 px reading/editor text, 16 px body, 14 px controls/help, 13 px secondary metadata |
| Spacing | 4 px base: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64 px |
| Layout | 1,160 px content maximum; reading measure 68 characters |
| Interaction | 120 ms color/border feedback; visible focus ring; reduced-motion preference disables transitions; no decorative animation |

All values live in [tokens.css](../frontend/styles.css).
[application.css](../frontend/styles.css) implements shared
buttons, fields, separators, typography, navigation, responsive composition and states.
Controls have labels and at least 44 px button/select targets. The radio choice's
whole label is clickable. A skip link, landmarks, current-page indication, native
form semantics and focus transfer to the main region support keyboard navigation.

## Page structure and interaction plan

| Area | Current shell | Later integration |
| --- | --- | --- |
| Generate | Opening editor, character count, clear feedback, model/style/length choices, disabled continue action, honest empty continuation | Validate the request, run the selected real model, display continuation separately from prompt, copy actual text, show stop reason and concise warnings |
| Compare | Shared-prompt introduction and equal-weight Model A/B reading regions | One prompt/settings snapshot; run A then B sequentially through one service, preserve each result/error, label identical settings and any effective seed |
| Model insights | Sections for identity, training record, held-out comparison and control explanations | Read verified registry/run/evaluation artifacts; show actual units, IDs, budget, source exposure, selection rule, per-dataset metrics and real-log charts |
| Architecture | Reading path from characters to context to next-character prediction | Explain the actual saved model configuration, attention mask, blocks, shared vocabulary, learned positions and special-token boundaries |

Generate is the default hash route `#/generate`. Other routes are `#/compare`,
`#/insights` and `#/architecture`; hash routes work on refresh without a server-side
router fallback. Unknown routes offer a recovery action. The page title and current
navigation state update with the route. Navigation retains a draft and control
choices in JavaScript memory; refresh/closing the tab discards them. There is no
local storage, cloud sync, prompt logging or prompt submission in Phases 13–14.

The generation button stays disabled regardless of input. The capability endpoint
also reports `generation_enabled: false`; it does not claim models are loaded.
The radio choices describe the two verified research versions, not live availability.

### Plain-language controls

- **Choose a model:** training collection names first; Model A/B as supporting labels.
- **Consistent:** greedy next-character selection. Temperature/top-k/seed have no role.
- **Varied:** sampling. Reveal a Variation slider only in this mode; lower follows
  likely choices more closely, higher permits more unexpected turns.
- **Continuation length:** short/medium/longer previews map to 128/256/512 new IDs.
  Visible character lengths are approximate: EOS may stop early and a rendered UNK
  marker may occupy multiple characters. The service/API limit remains 1,024 IDs.
- Seed, exact top-k, tokenizer fingerprints, context mechanics and timing definitions
  belong in model information/advanced explanations, not the main writing task.

The character count uses Unicode code points to agree with the backend prompt limit,
not JavaScript's UTF-16 length. Input beyond 16,384 characters shows an error without
discarding the draft. Unknown-character handling remains the service's responsibility
when connected; it is never silently replaced by this shell.

### States and accessibility

The reusable `renderState` helper in [components.js](../frontend/application.js)
creates text-only DOM content, never interpolated HTML from a prompt or response.
Loading/empty/success states use status semantics; errors use alert semantics.

| State | Intended behavior |
| --- | --- |
| Loading | Explain the active operation. No invented progress percentage or animated prose; preserve the prompt and disable duplicate submission |
| Empty | Say what belongs in the space and how to begin; show no sample model response |
| Success | Confirm a real completed action. In this phase, clearing a draft is a real success; generation-success styling is only a labelled specimen |
| Error | Plain-English reason, retained input, useful recovery action; avoid raw traces/filesystem paths |

The Phase 13 `#/foundations` reference moved in Phase 14 to the separate fixture server
at `http://127.0.0.1:8001/fixtures/foundations.html`. It is no longer served by the
normal application. Its
loading/empty/success/error specimens explicitly identify themselves as examples;
they are not model activity. It also demonstrates enabled, disabled, focus and invalid
form styles. In the live shell, the initial status fetch has a four-second timeout
and a retry action if the local preview is unavailable.

### Responsive behavior

- Above 900 px: editor/result column with a 304 px preference column and 64 px gutter.
- 701–900 px: 270 px preference column and 32 px gutter; secondary heading caption hides.
- At 700 px and below: one-column document order is opening → preferences → action →
  continuation. Compare panels stack A then B. Controls expand to the available width.
- At 380 px and below: navigation uses two rows rather than shrinking labels or scrolling
  horizontally. Typography uses relative units and the main title scales down to 32 px.
- No sticky panel, forced horizontal reading, or decorative animation competes with text.

## Local server and API boundary

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m transformer.web
```

Open `http://127.0.0.1:8000`. `--port 8765` changes only the port. The launcher binds
to **127.0.0.1**, uses **one worker**, and disables access logs. It has no public-host
flag. Packaged assets are the only static directory; project data, runs, checkpoints,
Python source and contract files are not public routes. No model or PyTorch import
occurs at shell startup.

Default serving is same-origin with **no allowed cross-origin callers**. If a separate
local frontend server is deliberately used, select one exact development origin:

```sh
.venv/bin/python -m transformer.web --development-origin http://127.0.0.1:5173
```

The only options are `http://127.0.0.1:5173` and `http://localhost:5173`. There are no
wildcards or credentialed CORS requests. Allowed methods are GET/POST and the declared
request header is Content-Type. Host headers must identify localhost/127.0.0.1.
CORS is a browser origin policy, not authentication; this is a local application,
not a reviewed public deployment. A self-only CSP, no-sniff and no-referrer headers
support the browser boundary. No external assets, cookies, telemetry SDK or accounts
are configured.

### Request and response contracts

[inference.openapi.json](../backend/contracts/inference.openapi.json)
is the versioned OpenAPI 3.1 **design contract**, version 1.1.0. Each operation records
whether it is implemented or planned. `/openapi.json` served by FastAPI documents
only the actual status endpoint; planned endpoints are not mounted and return 404.

| Method / path | Status | Contract |
| --- | --- | --- |
| `GET /api/v1/status` | Implemented | Application/phase/interface/contract version and generation-disabled capability; no inference readiness claim |
| `GET /api/v1/models` | Phase 15 | Trusted registry summaries with stable ID, display name, baseline/combined kind, registered/unavailable state and registry fingerprint |
| `POST /api/v1/generations` | Phase 15 | `{model_id, prompt, settings}` → projection of the existing service result |

The request excludes project paths, checkpoint uploads, device selection and arbitrary
URLs. Device and trusted project root are operator configuration. Greedy/sample
settings are separate schema branches; greedy excludes sampling-only settings.
The adapter must reject unknown keys, invalid types, non-finite numbers, invalid
Unicode, oversized bodies, unknown IDs, vocabulary-incompatible top-k and unavailable
artifacts with useful errors, using existing backend validation as the authority.

The response contains prompt, continuation, model ID, checkpoint SHA/update, effective
sampling settings, generated ID count, stop reason, tokenizer identity, device,
truncation/unknown counts and separate generation/load durations. It intentionally
omits personal checkpoint paths. EOS counts as an ID but is not rendered. UNK remains
visible. Immediate EOS is an honest successful empty continuation. Seeds travel as
decimal strings (or null) to avoid JavaScript's 53-bit integer limit; the adapter
converts and validates the backend's 0..2^63−1 range.

Errors use `{error: {code, message, retryable, field?}}`, with 422 invalid input,
404 unknown model, 413 oversized body, 409 busy service, 503 unavailable artifact/device
and 500 unexpected inference failure. These are planned adapter semantics, not claims
that Phase 14 has implemented generation validation or scheduling.

Phase 15 must create one `GenerationService`, serialize access, preserve its one-model
cache and run synchronous inference off the event loop. Compare issues A then B with
one frozen settings snapshot. It must not keep two models on MPS or create unbounded
request queues. No Stop/cancellation control is shown: aborting an HTTP request alone
does not safely cancel GPU computation. Insight data should be an explicit projection
of saved verified metadata; add its detailed response schema when that page is built.

## Verification and next phase

HTTP tests cover assets/capability, unavailable planned routes, private-path denial,
default/opt-in origin policy, host rejection, no model import, invalid port and contract
references/limits. Browser checks cover navigation, draft retention, variation reveal,
code-point counting, clear/error states, keyboard focus and responsive layouts.
Raw records and screenshots are under `artifacts/phase13-*`; exact outcomes are in
[project progress](project_progress.md).

Phase 14 completed the Generate UI structure. Phase 15 is expected to connect its
adapter to the existing inference service. The web app cannot yet generate stories,
even though the backend gate passed. Other page construction awaits its phase prompt.
