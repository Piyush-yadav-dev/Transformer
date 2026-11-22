# Experiment methodology

The exact paired settings are [phase8-comparison.json](../configs/experiments/phase8-comparison.json).
Model A uses TinyStories training targets only. Model B starts independently from
the same random seed and uses the combined stream. It is not fine-tuned from A.

Both use seed 42, four layers/four heads, embedding and context 256, one shared
78-ID training-only character vocabulary, and 2,048,000 training targets. The
optimizer/schedule, effective batch, evaluation interval and checkpoint rules are
frozen in that configuration. The selected checkpoints are update 1,000.

Selection minimizes complete packed-validation token-weighted loss, earliest exact
tie. It does not use held-out test scores. The mixture is measured by sampled target
tokens, not claimed from a document ratio: A saw 2,048,000 TinyStories targets; B saw
932,922 TinyStories and 1,115,078 SimpleStories targets, including the documented EOS
treatment. Wall time is recorded separately from token exposure.

Read the [original protocol](phase8_experiment_protocol.md),
[mixture supplement](phase9_experiment_protocol.md) and
[evaluation protocol](phase10_evaluation_protocol.md) before reproducing results.
Primary packed scoring and secondary isolated-document scoring have different
contexts and must not be interchanged. Character-token perplexity is exp(mean
natural-log cross-entropy), not an average of batch perplexities.

The [actual comparison](phase10_comparison_report.md) is mixed across datasets.
One seed and bounded prefixes cannot establish statistically general improvement.
No external model API or invented human ratings support the conclusions.
