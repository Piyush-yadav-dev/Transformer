# Phase 19 end-to-end verification

## Environment and commands

Actual local MacBook, macOS arm64, Python 3.13.7, PyTorch 2.15.0,
MPS inference. Existing locked environment; loopback server, no external service.
Real registered update-1000 Model A and Model B checkpoints were used.

Commands executed:
```sh
.venv/bin/python -m transformer.web --device auto --port 8002
.venv/bin/python -m pytest -q
node --test tests/web/*.test.mjs
.venv/bin/python tools/maintenance.py verify-web-smoke --port 8002 --server-pid 62388
```
Use the actual server PID when reproducing the last command.

## Passed

Browser checks used the normal application, not simulated responses:
1. Launched and viewed both registered models.
2. Entered “Tom found a small box under a tree.”
3. Generated Model A, greedy, limit 24; verified its ID, checkpoint, MPS and count.
4. Switched to B; sampled with temperature 0.8, top-k 20, seed 42, limit 24.
   Result metadata matched all settings and B's checkpoint.
5. Reopened A's history entry; its original identity and settings were restored.
6. Compared both models sequentially with the restored shared prompt/greedy settings.
7. Opened actual training curves and their raw-value table.
8. Opened the architecture view.
9. Copied continuation; copy-success feedback appeared.
10. Cleared session history.
11. Submitted an unknown snowman character; received a useful U+2603 error.
    The prompt remained editable. Replacing it with valid text generated successfully,
    and history showed one new entry.

Unit/integration regression: **475 Python tests passed in 48.40 s; 23 JavaScript
tests passed**, with zero reported failures or skips. Existing isolated tests exercise:
- missing checkpoint and tokenizer, checksum and incompatible metadata;
- invalid prompt/sampling, injected generation exception and successful retry;
- simultaneous requests, duplicate submission and disconnect lock retention;
- missing/corrupt evaluation evidence, malformed and partial log entries;
- original model identity, request parameter transmission and response limits;
- ten-item history bound, restoring and clearing; no rerender-triggered submission.

All destructive failure simulations operate on temporary synthetic artifacts.
Real checkpoints, tokenizers and evaluation artifacts were not edited.

## Loading and resources

The first browser A request reported cold load 0.134 s and generation 0.833 s.
Switching to B reported load 0.170 s and generation 0.183 s. These single observations
are not a benchmark or a comparison of model quality.

A further ten sequential HTTP requests used A×4, B×4, A×2, each bounded to 16 new
tokens. Assertions checked identity, checkpoint SHA, output type, token limit and
settings. All seven consecutive same-model requests were cache hits with zero model
load time. Both deliberate switches were cache misses. A single model remains cached.

Sampled server process RSS ranged from 380,289,024 to 380,502,016 bytes; final RSS
was 380,502,016 bytes. No obvious unbounded growth or repeated warm loading appeared
in this bounded sequence. RSS is process resident memory, not accelerator allocation
or total unified-memory use. A ten-request check cannot establish long-term leak freedom.
Raw per-request records: artifacts/phase19-resources.json.

## Failed, skipped and open items

No failed assertions or new concrete backend/UI defects were found in the executed
Phase 19 checks. Generated prose remains repetitive and malformed; this is model
quality, not a fabricated successful quality assessment.

Not performed: prolonged resource soak, separate accelerator allocation sampling,
native clipboard paste into another application, exhaustive browser combinations,
and every injected failure through the browser itself. Failure transport/loading
was exercised through isolated unit/integration tests instead. Earlier Phase 18 zoom,
full keyboard-only and quantitative contrast checks remain pending; this phase does
not claim to close them.

To reproduce the resource check, launch the server, obtain its PID, and run the
script above. To reproduce artifact failures safely, run tests/inference/test_loading.py,
tests/web/test_inference_api.py and tests/web/test_analytics.py; do not remove real files.

Browser screenshot: artifacts/phase19-browser.png. Owned verification server stopped
after testing. No training, new evaluation scoring or performance tuning was started.
Phase 19 complete within the stated test scope.
