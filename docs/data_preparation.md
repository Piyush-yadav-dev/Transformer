# Bounded data preparation

## Run and reuse

Install/sync using the README, then run from the project root:

```sh
.venv/bin/transformer prepare-data --config configs/data/phase2-small.json
.venv/bin/transformer prepare-data --config configs/data/phase2-small.json --offline
```

`--project-root` selects the root used to resolve generated paths. An optional
`--project-config` accepts Phase 1 settings, including `paths.data_dir`. Relative
configuration filenames are resolved from the current working directory.

The default writes to `data/prepared/phase2-stories-small-v2/` and caches bounded
raw subsets in `data/cache/subsets/<request-hash>/`. Full prepared artifacts and
matching subset caches are checksum-verified before reuse. A rerun with the same
identity returns `reused: true` without network access. `--offline` can also build
a new preparation name from already-cached matching subsets.

Change the preparation name when changing settings or implementation. Existing
outputs are never silently overwritten. Corruption causes an error and preserves
the suspect files. Each artifact uses an exclusive lock and a temporary sibling
directory; publication occurs by directory rename after the manifest is complete.
Ordinary failures remove only the current temporary directory. A killed process
may leave a lock/staging directory: inspect it and confirm no process is active
before manually removing it. There is no automatic stale-lock deletion or force
overwrite flag.

## Acquisition and memory

The pipeline uses `datasets` to verify metadata and resolve configured Parquet
file order at a full commit SHA. It streams those files through `HfFileSystem`
range reads and `pyarrow.parquet.ParquetFile.iter_batches`, reading only the text
column in batches of at most 128 records. Arrow background prefetching and parallel
decoding are disabled, and readers/file handles are explicitly closed on early
termination. It stops after each configured raw-record
limit; invalid or duplicate records are **not** replaced by additional downloads.
Source exhaustion below the limit is recorded. It never materializes the full
dataset, performs a full-dataset shuffle, or scans the full training split.

Streaming Parquet can read metadata and whole row groups or range-read buffers;
the record cap is not a byte-transfer cap. Library buffers are additional to
application memory. An on-disk SQLite index handles exact deduplication and
provenance, with an 8 MiB SQLite page-cache target and disk-backed temporary work.
Only one record and its duplicate provenance group are assembled for export at a
time. The SQLite working database is removed before publication. Local raw caches
remain available for subsequent configurations. Source verification is described
in [dataset sources](dataset_sources.md).

## Cleaning and rejection

Cleaning converts CRLF/CR to LF, applies Unicode **NFC**, and strips only outer
whitespace. Internal paragraphs, tabs, capitalization, punctuation, accents, and
other valid Unicode are preserved. There is no ASCII conversion, lowercasing,
compatibility normalization, language filtering, or aggressive whitespace collapse.

Records are rejected with explicit counters for non-object records, missing text
fields, non-string text, empty/whitespace-only text, surrogate code points, unsupported
control characters, or raw story length above the configured limit. Control
characters in Unicode category `Cc` are rejected except CR, LF and TAB. Malformed
JSONL syntax fails with a file/line error; an interrupted or invalid raw artifact
is never published as complete. No text is silently repaired with replacement
characters. Manifest length statistics count Unicode code points, not model tokens.

## Splits and duplicate precedence

The normalized-content identity is SHA-256 of the cleaned UTF-8 story. All sources
and acquired official splits share one duplicate index. Exact duplicates collapse
to one canonical record with every origin retained. A seeded hash of content
determines derived membership independently of processing order:

- TinyStories: official training stays training; its bounded official validation
  subset is divided approximately 50/50 into validation and test.
- SimpleStories: official test remains test; approximately 10% of its bounded
  training subset is reserved for validation, with the remainder used for training.

Fractions are deterministic hash thresholds, not exact quotas. The same content
always gets the same derived decision for a given source seed and split policy.
No windows or tokens exist at this stage.

If a story has conflicting proposed splits, **test takes precedence over validation,
which takes precedence over training**. Ties retain the first configured source
as canonical attribution. All origins remain visible. This prevents an acquired
official held-out story from entering training, including through another source.
Within-source and cross-source duplicate counters can overlap; they describe
different relationships, not mutually exclusive removal reasons.

Exact deduplication does not remove paraphrases, near-duplicates, or differences
in internal whitespace. Coverage is limited to acquired subsets; no claim is made
about duplicates against unacquired upstream records. Future expanded preparation
must rerun joint deduplication before any tokenizer fitting or training.

## Output contract

Each JSONL line is one entire story with `id`, `text`, canonical `source`, and
`origins` (source, official split, zero-based source index, proposed split).
Newlines within stories are JSON-escaped; document boundaries remain explicit.
Records are sorted by content hash for stable output bytes.

- `train.jsonl`: globally deduplicated training union for the later combined-data run.
- `train_tinystories.jsonl`: TinyStories membership within that safe training union.
- `train_simplestories.jsonl`: SimpleStories membership within that safe training union.
- `validation.jsonl` and `test.jsonl`: common evaluation pools for both later runs.
- `manifest.json`: full configuration, source verification and cards, revisions,
  seeds, cleaning policy, input/removal/duplicate counts, per-file document counts,
  min/max/mean/total character lengths, file sizes, SHA-256 checksums, corpus
  fingerprint, implementation hash, Unicode version, and environment versions.

Do not concatenate per-source training views: a cross-source duplicate can be a
member of both views. Use the already-deduplicated `train.jsonl` for the combined
run. Source-specific views and the combined run share the same evaluation files.
Later training still needs a documented step/token budget and independent random
initialization for each run.

**Test text is reserved for final evaluation.** No tokenizer fitting, model
selection, training decisions, or prompt selection may use it. Later vocabulary
fitting must use training text only and freeze one shared tokenizer for both runs.
Preparation computes test hashes/lengths solely for integrity and leakage checks.

## Offline verification

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.tools/uv/bin/uv lock --check
.tools/uv/bin/uv pip check --python .venv/bin/python
```

Routine tests use explicitly labeled synthetic JSONL fixtures and mocked Hub
metadata. They do not download real data. See the progress record for actual
acquisition results and verification outcomes.
