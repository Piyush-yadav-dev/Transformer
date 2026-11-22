# Phase 17 — model insights and experiment analytics

Implemented 21 November 2026. The existing local Python service and buildless
frontend are retained; no dependencies, hosted services or model training were added.

## Data and interface

GET /api/v1/analytics projects only the trusted project model registry's experiment
artifacts. It accepts no artifact path parameter and never deserializes checkpoints.
Each file read is capped at 16 MiB; training charts retain up to 10,000 valid updates
per run, with explicit warnings if truncated. File paths are project-relative and
provenance disclosures show SHA-256 digests. The endpoint reads on demand and the
page's Reload artifact evidence action refreshes it.

| Display | Saved source |
| --- | --- |
| Model ID, run ID, checkpoint update | configs/inference/model_registry.json |
| Layers, heads, width, context, vocabulary, weight tying, budget | Registered configuration.json |
| Character tokenizer | Registered tokenizer.json |
| Parameter count and checkpoint selection | Registered evaluation results.json / runs |
| Training curves | Each run's metrics.jsonl |
| Validation curves | Each run's validation_history.json |
| Training wall duration | Each run's training_summary.json |
| Dataset revisions, licenses, packing and vocabulary policy | token_data_dir/manifest.json |
| Test loss and perplexity | Registered results.json / scores / member / packed / sources |
| Actual total and source exposure | Registered results.json / runs / member / training_exposure |
| Generation throughput and environment | Registered results.json / generation_benchmark and environment |

Registered configuration, tokenizer and evaluation JSON must match registry hashes.
Configuration, checkpoint identity, tokenizer, run identity and absence of test-based
selection must agree with the evaluation snapshot. Validation histories and summaries
must match their hashes saved in that snapshot. The token manifest's dataset/token
fingerprints and sources must agree. No new inference is needed to display analytics.

A changed training log may still be inspected, but is labelled unverified against the
frozen snapshot. Malformed/nonfinite/nonmonotonic update lines are omitted with
line-number warnings; missing updates break the plotted line. Missing or corrupt logs
do not remove independent verified test measurements. Missing duration is null and
displayed as unavailable, not zero. Corrupt registered evaluation evidence withholds
that model's metrics entirely. Incompatible runs receive separate sections, not a paired
comparison. The current two runs pass all compatibility checks without warnings.

Charts preserve raw training loss and validation loss against actual training target
tokens. Axes use nats/target token and training target tokens; run labels identify
the full run. SVG point titles provide hover detail. Expand Inspect raw measurements
for exact, unsmoothed values, with 50-row pagination and keyboard controls. Numeric
table cells show full precision in their native title; visible metrics round to four
decimal places. Charts scroll locally on small screens to preserve readable labels.

The primary packed-test policy from Phase 10 is used consistently, not mixed with its
secondary isolated-document metric. Eligible shifted targets are counted once; EOS
and any UNK targets count. First stream token and incomplete tail are excluded.
Perplexity is exp(mean cross-entropy in nats), and is per character token. It cannot
be directly compared with unrelated subword perplexity.

Generation throughput comes from the saved synchronized repeated-trial benchmark,
not the Phase 16 UI calls. Its definition, environment and loading exclusion appear
next to the table. Training duration includes the saved session-wall scope; the
timing definition explains setup, validation, I/O and diagnostic-sampling exclusions.
Equal token budgets are not described as equal wall-clock work.

## Architecture and interpretation

The model selector reads actual configuration. The structured flow shows token IDs,
token plus learned absolute position embeddings and dropout, repeated causal blocks,
final custom affine layer normalization, bias-free vocabulary projection and next-token
selection. Each block explicitly retains the input, normalizes, computes scaled/masked
attention, adds the first residual, normalizes again, applies linear/GELU/linear/dropout
and adds the second residual. Shape, head width, feed-forward width, epsilon and tying
come from configuration. The flow was checked against the six implementation modules.

Plain-language explanations cover training loss, validation loss, overfitting,
character-token perplexity and why a mixture can help or hurt. No ranking, invented
human score or blanket quality-improvement claim is added.

## Executed verification

- Locked install using existing evaluation/web extras succeeded.
- Full Python suite: **475 passed in 46.25 seconds**.
- JavaScript suite: **23 passed**.
- Focused web suite initially passed 71 tests; a subsequent malformed-evaluation
  case was added and included in the full 475-test run.
- tools/maintenance.py verify-analytics passed for both actual registered models, 11 distinct
  source artifacts, all 2,000 training and 20 validation points, and every projected
  overview/metric field. It reconstructed small CPU model objects only to verify the
  saved 3,245,056 parameter count; it did not load weights or run training.
- The audit also matched all six model implementation file hashes to the saved
  evaluation source hashes, supporting the architecture diagram's provenance.
- Offline tests cover missing registry/log/summary, partial final log lines, malformed
  and nonfinite losses, duplicate/out-of-order rows, gaps, corrupt evaluation,
  incompatible configuration, malformed validation, mismatched manifest, and no
  arbitrary path reads. Synthetic test evidence is isolated from real experiments.
- Browser inspection on local port 8002 verified both model overviews, real scores,
  all displayed numeric rows, raw table pagination (rows 1–50 then 51–100), architecture
  switching and 390 px responsive layout. Page width stayed 390 px; charts scroll
  inside their own region. Desktop 1280 px layout and chart rendering inspected.
- Screenshot artifacts: phase17-insights.png and phase17-training-chart.png.
  Source audit: artifacts/phase17-source-verification.json.
- The owned preview server was stopped after verification.

No missing required artifacts or Phase 17 blockers were found. This is a saved,
single-seed, bounded-subset experiment, not a live training dashboard or statistical
proof of better stories. No new held-out evaluation or training was performed.
