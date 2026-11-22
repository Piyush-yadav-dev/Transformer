# Phase 16 — generation controls and comparison

Implemented and verified locally on 21 November 2026. No new frontend dependencies,
persistent tracking, database, training run, rankings or automatic quality scores.

## Controls and behavior

Generate exposes the registered model, mode and maximum new tokens (1–1,024).
Sampling also shows temperature (finite, greater than zero); advanced settings contain
optional top-k and seed. Top-k limits candidate tokens and cannot exceed the loaded
vocabulary. Blank top-k uses the whole vocabulary. Seeds are decimal strings up to
9223372036854775807, preserving precision across JSON and JavaScript. Greedy chooses
the most likely token and omits all sampling-only settings, even if previously entered.

Counts are character tokens, not words; EOS counts toward the limit but is not
displayed. Temperature changes randomness. Seeds do not guarantee identical outputs
across hardware or runtime versions. Existing API validation remains authoritative.
The Phase 15 API contract version stays 2.0.0; Phase 16 extends its client.

Switching a model preserves the prompt and previously completed output, including
its original model identity. Registry refresh does not silently replace an explicitly
selected model. Missing artifacts require restoration or explicit selection.

History retains the newest ten openings/completions in this tab's memory. Completed
entries include prompt, continuation, original model, settings, timestamp and response
metadata. Navigation and ordinary rerenders preserve it; reload clears it. Reopening
from Compare opens the result on Generate. Copy copies plain continuation text.
Clear history removes history entries while leaving the currently displayed result
and comparison readable. No data is persisted to browser storage.

Compare captures one prompt/settings snapshot and runs Model A then Model B.
An omitted sampling seed is generated once for the pair; both effective seeds are
reported. The one-model backend cache and existing accelerator lock remain in use.
A lost response or busy response stops the pair; no automatic retry or competing
second job starts. Model-specific failures stay associated with that model.
Both registered models must be available to begin. Outputs show original identity,
token count, generation duration excluding loading, stop reason and full metadata.
Desktop uses two columns; narrow screens stack. Equal settings support exploration,
but one random sample is not a definitive quality test or a throughput benchmark.

## Verification actually executed

- Locked wheel reinstall with the existing web/evaluation extras succeeded.
- `node --test tests/web/*.test.mjs`: 21 passed, including exact settings transport,
  sequential calls, one shared generated seed, missing model, lost response, model
  switching, reopening, clearing, bounded histories and registry refresh protection.
- `.venv/bin/python -m pytest -q`: 460 passed in 54.10 seconds.
- Local service: `.venv/bin/python -m transformer.web --device auto --port 8002`.
  Browser comparison used the actual Model A/B update-1000 artifacts on MPS.
  Prompt: “The little bird found a blue ribbon beside the old oak tree.”
  Both: sampling, temperature 0.8, top-k 20, seed 42, maximum 32 tokens.
- Model A: 32 tokens, max_new_tokens, generation 0.955898 s, loading 0.141675 s.
  Model B: 32 tokens, max_new_tokens, generation 0.184107 s, loading 0.175131 s.
  These are single diagnostic calls, not comparative speed conclusions.
- Browser verified transmitted settings in result metadata, history restoration
  from Compare into Generate, switching to B while A's output retained its label,
  history clearing, copy-success feedback, and comparison persistence on navigation.
- Responsive inspection: 390 px viewport had one 350 px output column and no
  horizontal page overflow; 1280 px viewport had two 564 px columns.
- A browser check exposed a token-count field resetting before blur. It now updates
  session state on input; the actual comparison used the requested 32-token limit.
- Screenshot: `artifacts/phase16-comparison.png`. Actual structured server events:
  `artifacts/phase16-verification.json`. Owned verification server stopped afterward.

## Limitations

Actual continuations contained malformed words. No model quality improvement is
claimed. MPS generation works here, but cross-device bitwise repeatability is not
promised. Clipboard success feedback was observed; native cross-application paste
was not tested. No streaming or cancellation is implemented. Model information
pages remain outside this phase. Phase 16 is complete; stop and await the next prompt.
