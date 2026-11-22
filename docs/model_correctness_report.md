# Transformer architecture correctness — Phase 5

**Result: all 38 new correctness cases passed; the complete suite passed 218 tests
in 19.90 seconds, with zero failures, warnings, or skips.** Executed 2026-11-18 on
the existing Apple M2 / macOS 27.0.1 arm64 host, Python 3.13.7, PyTorch 2.15.0.
Real MPS checks passed with CPU fallback explicitly disabled. No production-model
defect was exposed by these checks, so model code and initialization were preserved.

The model, configuration, previous tests, master specification, and progress record
were inspected before adding tests. All new examples are synthetic; no dataset
acquisition, training updates, or held-out story inspection was needed.

## What the tests establish

| Area | Evidence and meaning |
| --- | --- |
| Causality | In evaluation mode, changing every token after t leaves logits through t unchanged. Cases cover t = 0, 3, 6, tied/untied models, and truncated-prefix equivalence. Changed suffix logits differ, excluding a constant-output false positive. A deliberately disabled-mask control detects leakage. |
| Attention | An independent NumPy reference visits each head, batch example, query, and visible prefix directly. It uses vector dot products and explicit probability-weighted sums instead of the implementation's head reshaping and triangular mask. Agreement covers projections/biases, scaling, softmax, masking, head order, and output projection for 1/2/3 heads and lengths 1/5. Controls detect missing scaling, missing masking, and reversed heads. |
| Normalization | Forward values and input/scale/bias gradients match `torch.nn.functional.layer_norm` in float32 and float64. Inputs include shifted means, substantial variance, nearly constant and constant rows; widths 1/4, nontrivial affine parameters, and epsilon 1e-5/0.2 are covered. An altered-epsilon control differs. |
| Loss alignment | Model loss matches both direct cross-entropy and mean per-position log-sum-exp minus the target logit. The logits gradient matches `(softmax - one_hot) / number_of_targets`. Noncontiguous, already-shifted targets include EOS/unknown IDs; a second shift produces a different loss. |
| Gradients | All trainable parameters receive present finite gradients. Double-precision numerical Jacobians check inputs and every parameter of tiny attention, normalization, and GELU feed-forward components. The tied embedding gradient equals the sum of equivalent untied embedding/output gradients; other parameter gradients agree. Zero gradients are allowed where mathematically appropriate. |
| Shape/validation | New cases cover batches 1/3 and lengths 1/3/8, including the tiny context limit and independence between batch examples. The actual primary 4-layer/4-head/256-dimension model also executes a CPU forward pass at context 256 with `[1,256,78]` logits. Existing Phase 4 tests reject invalid head dimensions, IDs, ranks, dtypes, target shapes/devices, empty input, and overlong sequences. |
| Train/evaluation mode | Different seeded training dropout masks change logits; resetting the seed reproduces them. Repeated evaluation is exactly stable and consumes no CPU RNG state. No optimizer updates occur. |
| Serialization | Tied and untied models are saved as configuration JSON plus synthetic `state_dict` weights, reconstructed, and loaded strictly with `weights_only=True`. Evaluation logits are exactly equal on CPU, tying/counts survive, and nonpersistent causal masks are recreated. Temporary test weights are not trained checkpoints or a recovery system. |
| Devices | CPU runs the independent and numerical references. Actual MPS float32 forward/loss/backward results agree with CPU within explicit tolerances, including every parameter gradient; MPS also passes causal-prefix and repeated-evaluation checks. The device test is marked to skip if MPS build/runtime support is unavailable. |

Tests live in [`tests/model_verification/`](../tests/model_verification/), with
existing input-validation checks in [`test_language_model.py`](../tests/test_language_model.py).
The focused suite contains 37 CPU cases and one MPS case. The full suite includes
two additional real MPS checks from earlier phases.

## Determinism and tolerances

New tests use seed 1729 and deterministic CPU algorithms, restoring the previous
CPU RNG/algorithm settings afterward. Causality/reference/serialization tests use
evaluation mode, so configured nonzero dropout cannot introduce randomness.
The dropout-specific test uses seeds 91/92 deliberately. MPS comparison uses the
same saved weights as CPU and dropout zero; it does not assume float64 MPS support
or bitwise cross-device equality.

| Comparison | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| CPU float64 attention, causal prefixes, loss and batch independence | 1e-10 | 1e-12 |
| CPU float64 normalization values/gradients | 1e-9 | 1e-10 |
| CPU float32 normalization values | 3e-5 | 2e-5 |
| CPU float32 normalization gradients | 3e-5 | 2e-4 |
| Tied versus summed untied gradients | 1e-9 | 1e-11 |
| Component numerical gradients, central difference step 1e-6 | 1e-3 | 1e-5 |
| MPS versus CPU logits/loss; MPS prefix/repeat checks | 3e-4 | 3e-5 |
| MPS versus CPU parameter gradients | 1e-3 | 3e-5 |
| CPU repeated evaluation and weight round-trip | 0 | 0 |

Float32 tolerances account for accumulation and cancellation, particularly for
nearly constant normalized inputs. Numerical checks operate only on tiny
components (`[1,3,4]` inputs), not a full-model numerical Jacobian. The tolerances
were specified before execution and did not need widening to pass.

## Reproduction and evidence

After the README's locked regular-wheel installation:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/model_verification --junitxml=artifacts/phase5-correctness.xml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest --junitxml=artifacts/phase5-full-suite.xml
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Focused execution: **38 passed in 3.84 seconds**. Both JUnit reports and
`artifacts/phase5-verification.json` are local ignored evidence; the JSON records
environment, counts, and hashes of verified model/configuration/test source files.
Ruff, the dependency lock check, and installed dependency checks passed.

These tests support causal next-token computation and the tested numerical
contracts. They do **not** establish language quality, learning/convergence,
full training stability, optimizer/checkpoint recovery, mixed-precision safety,
or a training memory/throughput budget. Primary-config verification was a single
CPU forward pass at full context, not a full-size backward or MPS benchmark.
No full training or generation work was started.

Reference APIs: [PyTorch layer normalization](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.layer_norm.html),
[numerical gradient checking](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.gradcheck.gradcheck.html),
[functional parameter calls](https://docs.pytorch.org/docs/2.15/generated/torch.func.functional_call.html),
and [weight loading](https://docs.pytorch.org/docs/2.15/generated/torch.load.html).
