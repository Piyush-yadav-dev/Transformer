# Dataset source verification

Verified on 2026-11-18 before bounded acquisition. The source configuration is
[`configs/data/phase2-small.json`](../configs/data/phase2-small.json).

| Dataset | Authoritative repository | Pinned revision | Configuration | Text field | Official splits | Declared license |
| --- | --- | --- | --- | --- | --- | --- |
| TinyStories | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | `default` | `text` (string) | `train`, `validation` | `cdla-sharing-1.0` |
| SimpleStories | [SimpleStories/SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) | `e63b8adc3b1a1bdc7cac5b500d150b71346b0628` | `default` | `story` (string) | `train`, `test` | `mit` |

## Identity and attribution

TinyStories is the original English dataset by Ronen Eldan and Yuanzhi Li, described
in [their paper](https://arxiv.org/abs/2305.07759). This project uses the repository's
default Parquet configuration. Other raw-text variants and the archive in that
repository are not selected or downloaded.

SimpleStories means the official **English** dataset associated with
[Parameterized Synthetic Text Generation with SimpleStories](https://arxiv.org/abs/2504.09184)
by Lennart Finke, Chandan Sreedhara, Thomas Dooms, Mat Allen, Emerald Zhang,
Juan Diego Rodriguez, Noa Nabeshima, Thomas Marshall, and Dan Braun (2025).
The dataset card links that paper and the authors' generation repository. Its older
`lennart-finke/SimpleStories` API address resolves to `SimpleStories/SimpleStories`
at the same SHA. This is a namespace migration, not a competing dataset identity.
The Japanese dataset and the separately named SimplerStories derivative are not
selected. The canonical repository, paper, and author alias resolve the identity
without a source-choice question.

## Verification evidence

- Hub repository metadata supplied each exact commit SHA, license declaration,
  and file listing.
- Dataset-server split/configuration metadata independently confirmed the split
  names and the default configuration.
- The installed `datasets` library rechecked the pinned configurations, string
  features, and official split names before streaming any selected records.
- TinyStories' dataset-server metadata reported 2,119,719 training and 21,990
  validation records. Its pinned library metadata exposes split names without
  counts; manifests intentionally store those unavailable counts as `null`.
- SimpleStories' pinned card/library metadata reports 2,115,696 training and
  21,371 test records. The `processed.parquet` file is not part of the selected
  official split configuration.
- Neither pinned repository lists a separate license file. License identifiers
  come from the cards; the pipeline preserves the full upstream cards verbatim
  in cache and prepared manifests, including attribution and source references.
  Generated records retain source and official-split provenance. No new license
  for the combined data is asserted.

Pinned cards:
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md),
[SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories/blob/e63b8adc3b1a1bdc7cac5b500d150b71346b0628/README.md).
Metadata endpoints:
[TinyStories split information](https://datasets-server.huggingface.co/info?dataset=roneneldan%2FTinyStories&config=default),
[SimpleStories repository information](https://huggingface.co/api/datasets/SimpleStories/SimpleStories).

## Selected acquisition budget

| Source | Official training prefix | Official held-out prefix | Seed |
| --- | --- | --- | --- |
| TinyStories | At most 1,000 raw records | At most 200 validation records | 42 |
| SimpleStories | At most 1,000 raw records | At most 200 test records | 42 |

The maximum is 2,400 raw documents, including rejected and duplicate records.
These are development subsets, not representative benchmark samples. Selection
takes the first records in pinned upstream stream order, so upstream ordering bias
remains. The seed controls derived document splits, not prefix selection. No
claim is made that adding SimpleStories improves generation quality.
