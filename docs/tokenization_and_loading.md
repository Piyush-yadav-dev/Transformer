# Shared character tokenization and token loading

## Prepare the selected corpus

Sync the installed wheel after source edits using the README command. From the
project root:

```sh
.venv/bin/transformer prepare-tokens --config configs/tokenization/phase3-small.json
.venv/bin/transformer check-token-data --tokens data/tokens/phase3-shared-characters-v1 --loader-config configs/loading/default.json --stream train_tinystories --batches 2
.venv/bin/transformer check-token-data --tokens data/tokens/phase3-shared-characters-v1 --loader-config configs/loading/default.json --stream train --batches 2
```

These commands use the existing bounded Phase 2 preparation. They perform no
network acquisition, model initialization, training, or generation. `--batches`
limits the diagnostic to 1–100 batches and prints shapes/metadata, never story
text. Repeating `prepare-tokens` checks integrity and reuses compatible artifacts.

Token preparation configuration pins the prepared dataset fingerprint and selects
the output name and storage dtype. `--project-root` and `--project-config` preserve
the Phase 1 project-relative data-path conventions. Loader configuration is
separate: changing batch size, context length, seed, or workers does not require
refitting or re-encoding. CLI configuration filenames are relative to the current
working directory.

## Vocabulary policy for the primary comparison

The vocabulary is the union of characters in **only**
`train_tinystories.jsonl` and `train_simplestories.jsonl`. Characters are ordered
by Unicode scalar value, independent of story order and frequency. Fitting checks
that every origin in those views is an allowed training origin. It never reads
validation or test records to discover characters. Checksumming held-out files
for integrity is separate from vocabulary discovery.

Model A shares this vocabulary definition, including characters contributed by
SimpleStories, but its future training examples come **only** from the
`train_tinystories` token stream. Model B uses `train`, the deduplicated union.
Both later runs must start from random weights, use this same frozen tokenizer,
and share `validation` and `test` token streams under the same evaluation protocol.
No training has been implemented or executed here.

`tokenizer.json` freezes the characters, special IDs, semantics, format version,
and SHA-256 fingerprint. Tokenizer identity is independent of file formatting and
story order. Dataset identity is recorded separately: identical vocabularies can
come from different training corpora. The manifest records the exact training
files/checksums used for fitting and the source configuration/revisions/licenses.

## Character and special-token semantics

| ID range | Meaning | Default decoding |
| --- | --- | --- |
| `0` | Unknown character | Visible `<unk>` label, once per unknown character |
| `1` | End of story | Visible `<eos>` label |
| `2...vocab_size-1` | One vocabulary character per ID | The original character |

No padding or beginning-of-story token is needed for the fixed packed-window
policy. Special IDs are outside the character mappings. Literal text such as
`<eos>` is encoded character by character; it is never parsed as a control token.
Printed labels are display conventions, not a reversible serialization of special
IDs. Use IDs or the JSON tokenizer definition to preserve the distinction.

`encode(text)` adds no EOS by default. `encode(text, add_eos=True)` appends exactly
one EOS ID. Unknown characters become ID 0 individually; they are never dropped.
`unknown="error"` instead raises an error with the code point and position. The
preparation pipeline uses strict mode for training and replacement mode for
held-out text. It counts unknown IDs without extending the vocabulary.

`decode(ids)` displays both special labels. `decode(ids, skip_eos=True)` explicitly
omits EOS markers but still displays every unknown marker. Known-character
encoding/decoding is lossless; unknown-character decoding cannot recover the
original character. Invalid or out-of-range IDs fail clearly.

Empty text encodes to `[]`, or `[1]` when EOS is requested. An empty ID sequence
decodes to an empty string. Fitting with no training characters is an error.
The tokenizer performs **no additional normalization**: Phase 2 already applied
NFC and line-ending cleanup. A character means a Unicode code point, not an entire
multi-code-point grapheme. A 256-token window therefore covers approximately 256
characters, with EOS markers consuming positions too.

## Story boundaries and next-token windows

Each stream contains `story₁ characters, EOS, story₂ characters, EOS, ...` in
the Phase 2 order. Windows may span multiple stories **within one stream**.
EOS makes the boundary explicit but does not reset attention or hide the preceding
story. A later model can condition the next story's first character on the previous
story and EOS; those transitions reflect the prepared order rather than narrative
continuity. This policy is identical for both experiments.

Training, validation, and test live in separate files, so a window can never cross
a dataset split boundary. Per-source training views also remain separate from
the union; do not concatenate them to construct Model B's training stream.

For context length `C`, window index `i` starts at `i*C` and reads `C+1` tokens:

```text
read:    t0 t1 t2 ... tC
input:   t0 t1 t2 ... t(C-1)
target:  t1 t2 t3 ... tC
```

The tensors are independent CPU `torch.int64` copies with shape `[C]`; batching
produces `[batch, C]`. All complete windows are used. Adjacent windows share one
boundary token, while each retained next-token pair appears once per sequential
pass. With `N` tokens there are `floor((N-1)/C)` windows and `(N-1) % C` trailing
pairs not used. That tail is dropped, never padded or joined to another split.
If `N < C+1`, construction fails with the available/required counts. Empty held-out
streams can be encoded, but cannot form a loader. Later evaluation must report its
actual evaluated-token denominator rather than assume every corpus token was used.

## Storage, integrity, and memory

The artifact contains `tokenizer.json`, a manifest, and two files per stream:

- `<stream>.bin`: raw unsigned token IDs. `auto` chooses the smallest of uint8,
  uint16, and uint32 that can represent **every vocabulary ID**, including specials.
  Explicit insufficient dtypes are rejected before writing. Multibyte values use
  little-endian byte order.
- `<stream>.offsets.bin`: little-endian uint64 offsets, beginning at zero and
  ending at the token count; each adjacent pair encloses a complete story plus EOS.

The manifest stores document/token/unknown counts, dtype, boundary policy, source
file hashes, dataset and tokenizer fingerprints, all file checksums, implementation
identity, and a token-data fingerprint. Existing artifacts are validated and reused;
incompatible settings/code, missing files, or corruption cause errors without
overwriting. Choose a new output name for a changed preparation.

Encoding holds one story at a time in CPU memory. Loading first checks integrity
with bounded reads, then opens a read-only memory map lazily. Only one `C+1` window
is copied to native int64 for an item; no corpus is loaded onto MPS or another
accelerator. File checksumming is a sequential disk pass at dataset construction,
not per item. Story offsets and special-token placement are validated too.

The loader uses Python `mmap` with a NumPy array view to allow explicit resource
closure; it does not depend on NumPy's private memmap-close implementation. Spawned
workers reopen their own mappings instead of receiving a serialized token array.
Use the dataset as a context manager, or call `close()` after the loader iterator
has finished. Returned tensors remain valid after closure.

## Batching and seeds

`configs/loading/default.json` uses context 256, batch size 8, seed 42, shuffling,
zero workers, and `drop_last=false`. These are conservative functional defaults,
not measured performance recommendations. Worker values 0–2 are supported; nonzero
workers use macOS-compatible spawn, one prefetched batch per worker, no persistent
workers, and no pinned accelerator memory.

Fresh loaders with the same seed/configuration reproduce the same ordering in
the tested environment. Reiterating one shuffled loader advances its dedicated
generator and yields a deterministic sequence of different epoch orders. Later
training recovery must preserve generator state or reconstruct the epoch ordering;
checkpoint recovery is outside this phase. Bit-for-bit reproducibility across
different library versions or hardware is not promised.

For evaluation, use `shuffle=false` and `drop_last=false`. PyTorch's shuffled
sampler holds a CPU permutation of window indices; memory is proportional to the
number of windows, not to the entire encoded corpus. Sequential ordering avoids
that permutation. No throughput tuning or training benchmark has been performed.

## Python API

```python
from pathlib import Path
from transformer.configuration.settings import DataLoaderConfig
from transformer.data.loading import TokenWindowDataset, create_data_loader
from transformer.tokenization.character_tokenizer import CharacterTokenizer

directory = Path("data/tokens/phase3-shared-characters-v1")
tokenizer = CharacterTokenizer.load(directory / "tokenizer.json")
settings = DataLoaderConfig(context_length=256, batch_size=8, seed=42)
with TokenWindowDataset(
    directory,
    "train_tinystories",
    tokenizer.fingerprint,
    context_length=settings.context_length,
) as dataset:
    loader = create_data_loader(dataset, settings)
    inputs, targets = next(iter(loader))
```

Pass `dataset_fingerprint=` as well when binding a future experiment to a specific
prepared corpus. The tokenizer fingerprint is mandatory. The tokenizer can be
saved/loaded independently; saving refuses to overwrite an existing file.

## Checks

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Tests use synthetic examples and cover held-out-only characters, special/literal
token distinction, empty/short streams, dtype capacity, corrupt artifacts,
fingerprint mismatch, shifting, split isolation, seeded batching, pickle reopening,
and a spawned worker. Actual executed results are in the project progress record.

Implementation references: [NumPy memory mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
and [PyTorch data loading](https://docs.pytorch.org/docs/2.15/data.html).
