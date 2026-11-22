# Model and data notes

Intended use: local education, reproducible small-model experiments and inspection
of character-level language modeling. These are not production writing assistants,
factual knowledge systems or safety-reviewed models.

| Source | Pinned revision | Recorded upstream license |
| --- | --- | --- |
| roneneldan/TinyStories | f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 | cdla-sharing-1.0 |
| SimpleStories/SimpleStories | e63b8adc3b1a1bdc7cac5b500d150b71346b0628 | mit |

These are declarations preserved from the pinned dataset cards, not a new license
grant or a legal determination for downstream redistribution. See
[source attribution and pinned cards](dataset_sources.md) for authors, papers and
upstream notices. Retain attribution and applicable upstream terms when sharing.
The combined dataset is not assigned an invented replacement license.

Both sources are synthetic English stories. Acquisition uses small ordered prefixes,
so selection bias and limited topic/language coverage remain. Conservative cleaning
and joint normalized exact hashes prevent exact split overlaps in the prepared data;
they do not prove absence of semantic or near duplicates. Test text never fitted
the vocabulary. Model A shares a vocabulary defined using both permitted training
partitions, while its weight updates use TinyStories only.

Evidence demonstrates executable training/recovery/inference and learning on a tiny
overfit fixture. Real continuations often repeat phrases, produce malformed words,
or end abruptly. No factual reliability, general reasoning or child-safety guarantee
is established. Treat output as untrusted text.

The 256-token context counts characters and special tokens, not words. Older prompt
tokens fall outside the rolling context; learned absolute positions restart at zero.
Unknown prompt characters produce useful errors under the default policy. EOS is a
special ID, distinct from literal text spelling. See [tokenizer semantics](tokenization_and_loading.md).
