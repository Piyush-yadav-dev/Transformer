# Custom decoder-only Transformer

Phase 4 implements the neural network from basic PyTorch layers and tensor
operations. It uses no ready-made Transformer, attention module, normalization
layer, or pretrained weights. It provides forward computation, optional loss,
parameter reporting, and one-batch diagnostics. Training and generation are not
implemented in this phase.

## Architecture and configuration

`configs/model/primary.json` is the common candidate for both primary experiments:

| Setting | Starting value |
| --- | ---: |
| Vocabulary | 78 IDs, matching the frozen Phase 3 tokenizer |
| Decoder layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| Learned position capacity | 256 tokens |
| Feed-forward expansion | 4 (hidden width 1,024) |
| Dropout | 0.1 |
| Layer normalization epsilon | 0.00001 |
| Initialization standard deviation | 0.02 |
| Tied embedding/output weights | Yes |

`TransformerConfig` validates positive integer dimensions, at least two vocabulary
IDs, divisibility by the head count, finite numeric settings, dropout in `[0, 1)`,
positive epsilon/initialization scale, and strict booleans/schema version. It
supports `load`, `from_dict`, `to_dict`, `to_json`, and exclusive `save` snapshots.
Unknown keys and malformed JSON fail clearly. Vocabulary size must come from the
frozen tokenizer, including special IDs; vocabulary size alone is not a tokenizer
identity check. Future run records must bind the tokenizer and dataset fingerprints
alongside this architecture configuration.

The forward path is:

1. Token embedding plus learned absolute position embedding, then dropout.
2. Repeat the pre-normalization decoder block: normalize → causal attention →
   residual addition; normalize → feed-forward → residual addition.
3. Final custom layer normalization, then vocabulary projection to raw logits.

Attention has separate learned query, key, and value projections. Each is reshaped
from `[B, T, E]` to `[B, H, T, E/H]`. Query/key dot products are divided by
`sqrt(E/H)`, and positions strictly after the query position are replaced with
negative infinity before softmax. The current position and earlier positions stay
visible. Attention probabilities receive dropout; heads are combined, projected
back to E dimensions, and passed through residual-branch dropout.

Custom normalization uses the mean and population variance over the final
embedding dimension, with epsilon inside the square root and learned scale/bias.
Its statistics are computed in float32 for half/bfloat16 inputs. Phase 5 verifies
float32/float64 normalization against PyTorch; half/bfloat16 remain unverified.
The feed-forward branch expands E to
`E * feed_forward_expansion`, uses exact GELU, projects back to E, then applies
dropout. Linear layers have biases except for the final vocabulary projection.
Residual branches always preserve `[B, T, E]`.

Positions start at zero for each supplied window. The Phase 3 packed-story policy
still applies: EOS has its own token embedding but does not reset positions or
attention. No padding mask, sliding context, KV cache, or generation API exists.
Each attention block stores a boolean causal mask as a nonpersistent buffer;
moving the model moves that mask, while reconstruction uses the saved architecture.

## Interface and targets

```python
from pathlib import Path
import torch
from transformer.configuration.settings import TransformerConfig
from transformer.model import TransformerLanguageModel

config = TransformerConfig.load(Path("configs/model/tiny.json"))
torch.manual_seed(42)
model = TransformerLanguageModel(config)

# Explicit synthetic fixture, not real story data or model-quality evaluation.
tokens = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.int64)
output = model(tokens[:, :-1], targets=tokens[:, 1:])
print(output.logits.shape, output.loss.item())
print(model.trainable_parameter_count())
output.loss.backward()  # No optimizer step or weight update.
```

`forward(token_ids, targets=None)` always returns a `TransformerOutput` with:

- `logits`: floating-point tensor `[batch, sequence, vocabulary]`, without softmax.
- `loss`: scalar mean cross-entropy when targets are supplied, otherwise `None`.

Inputs and targets must be nonempty `torch.int64` tensors shaped `[B, T]`, on the
model's device, with IDs in `[0, vocab_size)`. Targets must match the input shape;
sequences longer than the configured context capacity fail instead of truncating.
These checks include ID range checks, which can synchronize an accelerator; no
performance optimization has been measured yet.

**The model does not shift targets.** Use the input and already-shifted target pair
from `TokenWindowDataset` directly. Every supplied position contributes to the
mean loss, including EOS or unknown target IDs. There is no ignored padding ID;
`-100` is rejected. Inputs and targets may be noncontiguous slices.

In the Phase 3 Python example, after creating `inputs, targets`, replace the toy
tensor with `output = model(inputs, targets)` and ensure the model context capacity
is at least the loader's context length. For accelerator use, move the model and
both batch tensors to the explicitly selected device. Keep the corpus on CPU.

New models default to training mode, so dropout is active. For deterministic
inference checks, use `model.eval()` and `torch.no_grad()`; evaluation mode alone
does not disable gradient recording.

## Initialization, tying, and parameter counts

The final initialization policy is identical for both primary experiments:

- Every unique token/position embedding and linear weight gets independent
  normal samples with mean zero and configurable standard deviation (default 0.02).
- Every linear bias is zero. Each normalization scale is one and bias is zero.
- There is no depth-specific residual scaling or pretrained initialization.

When `tie_embeddings=true`, the vocabulary projection and token embedding share
the **same Parameter**, not copies. Tying is established before final initialization;
the initializer visits that shared weight once. With tying disabled, the output
matrix is independently initialized. Positions are never tied to output weights.

Call `torch.manual_seed(seed)` before construction to reproduce initialization in
the same supported environment. Both future primary runs must construct fresh
models from the same configuration and initialization policy; Model B must not
continue Model A's weights. Use recorded seeds and tokenizer identities. No
cross-platform bitwise reproducibility claim is made.

`model.trainable_parameter_count()` sums only parameters requiring gradients and
counts shared parameters once. For the current vocabulary, the primary tied
configuration has **3,245,056** trainable parameters; the tiny check configuration
has **5,856**. These counts do not estimate total training memory: activations,
attention score matrices, gradients, and later optimizer state require more space.

## Checks and scope

Reinstall source changes using the README's locked regular-wheel sync command.
Then run:

```sh
.venv/bin/transformer check-model --config configs/model/tiny.json --device cpu
.venv/bin/transformer check-model --config configs/model/tiny.json --device mps
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest
```

The command seeds random initialization, generates two synthetic sequences of at
most eight input tokens, runs one forward/backward pass, and verifies output
shape, finite logits/loss, and present finite gradients for all parameters. It
reports configuration, device, seed, loss, unique parameter counts, and tying.
It performs **zero optimizer updates**, saves no weights, and reports no text
quality metric. `--seed` is configurable; `--device auto` uses the existing device
policy. Explicit unavailable MPS fails without a fallback.

The tiny configuration is an execution check, not the primary architecture or a
training recommendation. Phase 5 adds causal-independence tests, independent
attention/normalization references, small component numerical gradient checks,
weight round-trips, and CPU/MPS comparisons. The primary configuration has passed
one CPU forward check at full context; training memory/throughput are unmeasured.
See the [correctness report](model_correctness_report.md) for exact evidence and
tolerances. These checks do not establish mixed-precision or sustained training
stability, or language quality. Progress records retain phase-specific results.

API references checked for the installed PyTorch version:
[module/parameter semantics](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)
and [cross-entropy inputs](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cross_entropy.html).
