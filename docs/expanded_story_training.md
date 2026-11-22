# Expanded TinyStories + SimpleStories training

The run uses larger bounded subsets of the two datasets in
[the acquisition configuration](../configs/data/english-stories-expanded.json).
Acquisition scanned 90,400 records including held-out records. Preparation removed
100 duplicate occurrences and rejected 12 empty texts. A fresh character tokenizer
uses only training text. Whole stories remain separated across splits.

The new custom Transformer uses the existing packed next-character objective,
with 16,384 optimizer updates and a budget of 67,108,864 character predictions.
The previous app checkpoint stays available during training. TinyStoriesInstruct,
Project Gutenberg and subword tokenization are candidates for separate experiments.
This run focuses on the two existing story sources.

Reproduce from the project root with the locked environment:

```sh
.venv/bin/transformer prepare-data --config configs/data/english-stories-expanded.json
.venv/bin/python training/experiments.py prepare --expanded
.venv/bin/transformer train --config configs/training/english-stories-expanded.json --dry-run
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer train --config configs/training/english-stories-expanded.json
```

If interrupted, use the same configuration with `--resume`. The run and metrics
live in `runs/english-stories-expanded-v1/`. The long run uses real MPS.

After checkpoint selection, inspect continuations for the original prompts and
six held-out openings. The references are saved for review and are passed
separately from each prompt. None of the six story IDs occur in training.
Judge grammar, continuity and retention of the prompt's characters and situation.

```sh
.venv/bin/python training/experiments.py check --run english-stories-expanded-v1 --held-out data/prepared/english-stories-expanded-v1/test.jsonl
```

At the start of the run, data preparation, tokenizer preparation and training
dry run passed. Model quality remains pending until a checkpoint is sampled.

The training split contains 59,888 TinyStories and 26,911 SimpleStories
documents, for 86,799 documents and 86,933,099 stored character tokens.
The first 1,000-update checkpoint reached validation loss 1.63637 nats per
token. Its four CPU diagnostic continuations still contained malformed
words. The full run continues, with a local review process set to save
two seeded samples per opening once training reports completion.

## Larger follow-up

Acquisition was started with caps of 120,000 TinyStories and 60,000
SimpleStories training-source records in
[the larger-data configuration](../configs/data/english-stories-larger.json).
These are caps before deduplication and validation splitting. The follow-up
prepares a tokenizer, checks the 100,663,296-target budget with a dry run,
and starts only after the current expanded run records successful completion.
The standard trainer initializes fresh weights for the new vocabulary and run.
The two training jobs therefore do not compete for MPS.

The follow-up saves its preparation, dry run, training log and two seeded
final reviews under `artifacts/english-stories-larger-v1/`. The app checkpoint
is selected separately after reviewing story quality.
