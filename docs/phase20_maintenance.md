# Phase 20 — evidence-led maintenance

## Reviewed evidence and scope

Reviewed the Phase 12 backend gate, Phase 7 hardware benchmark, Phase 18 known
verification gaps, Phase 19 end-to-end report and its raw resource measurements.
Phase 19 already demonstrated warm model reuse and bounded short-run RSS. No
evidence justified changing attention, device synchronization, tensor allocation,
training data loading or inference caching. Existing one-model cache, inference
mode and ten-entry history remain unchanged.

A five-call cProfile investigation of analytics found duplicate parsing of the
same registered 1.5 MiB evaluation artifact for the two models. JSON parsing was
a substantial portion of the measured projection work. The optimization is limited
to this demonstrable duplication, not a new long-lived cache.

## Fixed defects

Two new synthetic regression tests failed against the previous implementation:
- An enormous JSON integer loss caused OverflowError during finite-number validation,
  hiding the entire model's analytics. It now rejects only that log row with the
  existing line-number warning and preserves subsequent valid measurements.
- A log containing exactly 10,000 valid updates incorrectly claimed later records
  had been omitted. The truncation warning now appears only when another valid
  update is actually omitted.

Neither defect occurs in the saved real runs. No training run or evaluation needs
repeating; no existing result is invalidated or superseded.

## Supported optimization

Within one load_analytics call, registered JSON objects are shared by exact
(path, expected SHA-256). Each unique artifact is read, hash-verified and parsed once.
Each model still receives its own provenance record. Nothing is retained between
requests, so file deletion/corruption on a later request is still detected. Logs,
validation histories, summaries and manifests continue their existing verification.

This establishes one verified snapshot per shared artifact per request. Concurrent
operator replacement during a request is not promised to appear halfway through that
response; the next request reopens files. No metadata-only freshness shortcut, TTL,
global state, lock, external package, compilation or attention cache was introduced.

## Before/after measurements

Actual target MacBook Air M2, macOS arm64, Python 3.13.7. Same registry, checkpoints,
tokenizer, logs, evaluation results and request projection. The old source snapshot
remains at artifacts/phase20/analytics_before.py for reproduction.

Two warmups per implementation, then eleven paired trials with alternating order.
Timing includes synchronous reads, SHA verification, parsing and projection; excludes
HTTP serialization, network transfer and rendering. OS file cache is warm.
Every trial asserts exact equality of the complete response. All eleven unique
source artifact hashes were rechecked afterward.

| Measurement | Before | After |
| --- | ---: | ---: |
| Median analytics projection | 23.0287 ms | 16.4358 ms |
| Shared evaluation JSON reads per request | 2 | 1 |
| Shared tokenizer JSON reads per request | 2 | 1 |

Median reduction: **28.63%**, approximately **6.59 ms** per projection. This is a
small absolute saving, not a claim that the entire application is 28.63% faster.
Raw paired measurements and environment: artifacts/phase20/analytics-benchmark.json.
Initial profile: artifacts/phase20/profile-before.txt.

Model loading and generation were not benchmarked again: neither path changed.
Prompt length, generated length and sampling mode are inapplicable to this analytics
request. No inference throughput or cold-load improvement is claimed.

## Reproduction and regression

Commands executed from the project directory:
```sh
.venv/bin/python -m pytest tests/web/test_analytics.py -k 'extreme_integer or exact_log_limit' -q
.tools/uv/bin/uv sync --locked --extra evaluation --extra web --python python3.13 --no-editable --reinstall-package transformer
.venv/bin/python -m pytest tests/web/test_analytics.py -q
.venv/bin/python tools/maintenance.py benchmark-analytics --baseline artifacts/phase20/analytics_before.py --output artifacts/phase20/analytics-benchmark.json
.venv/bin/python -m pytest -q
node --test tests/web/*.test.mjs
.venv/bin/python tools/maintenance.py verify-analytics
```

The first command deliberately captured both pre-fix failures in
artifacts/phase20/regressions-before.txt. The focused post-fix suite passed 15 tests.
A new sharing test verifies one read for each shared registered artifact, no
cross-request mutable state, and detection of subsequent corruption. Full regression
results are recorded in full-tests.txt and javascript-tests.txt in the same artifact
directory. The real-artifact audit confirms numeric/source consistency.

## Integrity and limitations

No tokenizer, model implementation, checkpoint interpretation, evaluation weighting,
training budget or stored training artifacts changed. Responses for valid saved
experiments are exactly equal before and after. No production training, additional
test-set scoring, frontend feature work or persistent caching was performed.

No new browser pass was needed for identical valid API output and unchanged UI.
Earlier zoom, exhaustive keyboard, assistive speech and contrast-verification gaps
remain open, as does a long-duration memory soak. The measurement is local and
bounded; hardware load and filesystem cache affect absolute latency. Stop after
these supported fixes; no speculative optimization is justified.

Final verification: **478 Python tests passed in 45.75 s**, **23 JavaScript tests
passed**, and the real-artifact audit passed for both registered models / 11 source
files. No remaining failure in the scoped regression suite.
