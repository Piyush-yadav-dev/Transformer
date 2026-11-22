# Extended English story training

The original paired checkpoints learned early character patterns and often emitted
repetition or malformed words. This separate run increases both data coverage and
optimizer exposure using the existing custom decoder and character-level pipeline.

The bounded acquisition uses the pinned dataset sources in
[english-stories.json](../configs/data/english-stories.json).
Training uses 25,000 prepared TinyStories documents and 22,310,660 stored tokens.
The new 101-ID vocabulary was fitted on allowed training data. SimpleStories text
contributes to that vocabulary and the common validation stream. Model weight
updates use only the TinyStories training stream.

The [training configuration](../configs/training/english-stories-a.json) schedules
8,192 MPS updates and 33,554,432 predicted character targets. The configured
validation cap is 128 batches of eight windows, with evaluation every 500 updates
and on completion. The best checkpoint is chosen by token-weighted validation loss.
Diagnostic samples use the two reported openings and two additional prompts.
The historical Model A/B metrics use a different setup and are a separate comparison.

```sh
.venv/bin/transformer prepare-data --config configs/data/english-stories.json
.venv/bin/python training/experiments.py prepare
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train --config configs/training/english-stories-a.json
.venv/bin/python training/experiments.py check --device cpu
```

An interrupted run can resume with the exact configuration and the training CLI's
`--resume` flag. The diagnostic script verifies the indexed checkpoint checksum,
loads it locally, and prints the exact returned continuations.

## Completed run and app integration

The MPS run completed 8,192 updates and 33,554,432 character targets in 1,659
seconds. The registered checkpoint is update 8,000, with validation loss
1.195082 nats per target across 262,144 validation targets. Its registry label
is **TinyStories · Extended training**. Generate initially selects it when its
files are available and starts in Sampling mode at temperature 0.65 and top-k 10.

The actual app endpoint generated from both reported prompts on MPS. At seed 42,
the box prompt returned “They were very happy. They were happy to have fun.”
The bird prompt yielded mostly English words but included “the bird was all the
bird.” Seed 7 also returned malformed words and shifted characters. The
extended model can generate short English sentences, with inconsistent grammar
and prompt relevance. Those samples do not establish dependable story continuation.

Raw diagnostics and endpoint responses are in artifacts/english-stories-a-v1/.
The original comparison remains evaluated on the original pair. The new model
has validation and diagnostic evidence and no held-out comparison score.
479 Python and 24 JavaScript regression tests passed. The actual API reported all
three registered entries and the original two-run analytics remained compatible.

## Single active TinyStories + SimpleStories model

The app serves one combined model, `transformer-combined-extended-v1`. Its
[training configuration](../configs/training/english-stories-combined.json)
uses the joint training stream with 25,000 TinyStories documents and 4,467
SimpleStories documents. The run completed 8,192 MPS updates and 33,554,432
character targets. Update 8,192 had the best validation loss, 1.01608 nats
per character token over 262,144 targets.

The user-facing quality requirement is to continue the supplied opening in
sensible English while retaining its characters and situation. The model now
often writes recognizable English, but this requirement remains partially met.
The verified web response for the blue-ribbon prompt was “They were very happy.”
The small-box response mentioned a box but introduced unrelated details and
grammar errors. The samples are diagnostic, with no new held-out comparison score.

The original entries are preserved in
[the historical registry](../configs/inference/historical_model_registry.json).
The current catalog, Generation API and Model insights were checked on port
8000 against the sole active checkpoint. Full Python regression passed 480 tests
and JavaScript regression passed 25. Raw evidence is in
`artifacts/english-stories-combined-v1/`.
