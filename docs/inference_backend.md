# Transformer inference backend

Phase 11 provides a synchronous Python service and CLI over the actual selected
checkpoints. No frontend, network server, training or checkpoint reselection is involved.

## Registered model versions

| Stable model ID | Training data | Frozen selected checkpoint |
| --- | --- | --- |
| `transformer-tinystories-v1` | TinyStories only, Model A | Phase 8, update 1,000 |
| `transformer-combined-v1` | TinyStories + SimpleStories, Model B | Phase 9, update 1,000 |

The [explicit registry](../configs/inference/model_registry.json) pins checkpoint,
tokenizer, training configuration and Phase 10 metrics file SHA-256 hashes. It also
records run ID, tokenizer fingerprint, model kind and selected update. Existing
versions must keep their identity: register a new stable ID for changed weights.
Metrics association does not reselect models using test results. Both checkpoints
remain the original validation-selected versions, with the limitations in the
[held-out comparison](phase10_comparison_report.md).

The application owner supplies the trusted project root when constructing the
service. That root and `configs/inference/model_registry.json` are operator-owned
configuration, not generation-request fields. Requests supply only a listed model
ID. There is no filename discovery, checkpoint upload, remote URL, `--checkpoint`
option or request-provided registry path. Missing local artifacts produce errors.
The registry does not bundle the large ignored checkpoints/data into the package.

Relative paths must remain inside the documented project subdirectories; traversal,
absolute paths, symlink files/parents and nonregular files are rejected. Files are
bounded and hash-checked before parsing. Checkpoint hashing and restricted
`torch.load(weights_only=True, map_location="cpu")` use the same open file handle.
Only this trusted local allowlist is supported; hashes do not authenticate artifacts
against someone who can rewrite both the catalog and project files. PyTorch also
advises against loading untrusted serialized data. See
[torch.load documentation](https://docs.pytorch.org/docs/2.15/generated/torch.load.html).

Loading reconstructs the custom architecture from saved configuration, checks the
tokenizer format/fingerprint/vocabulary, registry kind/run, checkpoint update,
associated metrics, exact state-dict keys/shapes, finite float32 weights, tied-weight
equality and parameter count. It discards training-only payload state, moves one
model to CPU/MPS, freezes all parameters and sets evaluation mode. An unavailable
explicit MPS request errors; `auto` uses the existing MPS-then-CPU policy.

## Python service

Use one long-lived service per application process; loading and generation calls
are serialized. No mutable model or accelerator tensors are returned publicly.

```python
from pathlib import Path
from transformer.inference import GenerationConfig, GenerationService

with GenerationService(Path.cwd(), device="auto") as service:
    available = service.list_models()
    # Optional prewarm: returns verified architecture, checkpoint and metrics metadata.
    model_info = service.load("transformer-tinystories-v1")
    result = service.generate(
        "transformer-tinystories-v1",
        "Tom found a small box under a tree.",
        GenerationConfig(
            mode="sample",
            temperature=0.8,
            top_k=20,
            seed=42,
            max_new_tokens=128,
        ),
    )
    print(result["continuation"])
    print(result["stop_reason"], result["generated_token_count"])
    print(service.cache_info())
```

`GenerationConfig()` defaults to greedy argmax and 128 new tokens. `generate`
returns a JSON-serializable dictionary containing prompt, continuation, raw token
IDs/count, stable model ID/kind, checkpoint path/hash/update, tokenizer/registry
fingerprints, requested/effective sampling settings, truncation/unknown counts,
device/runtime details, stop reason, cache hit and timing. `load` can prewarm; `unload`
or `close` releases the cache. The context manager releases it even after an error.

## CLI

From the project root, using the existing isolated environment:

```sh
.tools/uv/bin/uv sync --locked --python python3.13 --no-editable
.venv/bin/transformer list-models
.venv/bin/transformer generate --model transformer-tinystories-v1 --device cpu --mode greedy --prompt "Tom found a small box under a tree." --max-new-tokens 64
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/transformer generate --model transformer-combined-v1 --device mps --mode sample --temperature 0.8 --top-k 20 --seed 42 --prompt "The little bird wanted to help its friend." --max-new-tokens 128 --json
```

`--project-root` is an operator option for invoking the CLI from elsewhere. Without
`--json`, it prints the continuation and a short identity/count/stop/timing footer.
Use `--json` for all output metadata. Inference has no frontend or plotting-library
dependency. To run the full suite including optional evaluation/chart integration,
retain the evaluation extra:

```sh
.tools/uv/bin/uv sync --locked --extra evaluation --python python3.13 --no-editable
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/maintenance.py verify-inference --device mps --output artifacts/phase11-verification-repeat.json
```

The verification script uses existing actual runs; it never downloads data or
trains. Choose a new output filename for each run. On machines without MPS, use
`--device cpu`; MPS-specific tests skip rather than pass without execution.

## Sampling and edge cases

| Input or setting | Defined behavior |
| --- | --- |
| Greedy mode | Argmax; lowest ID wins an exact tie. Temperature omitted/0; top-k and seed are rejected because greedy uses neither. |
| Sample mode | Requires finite temperature >0. Scale logits, softmax and sample using a request-local CPU generator. |
| Top-k | Omitted = all IDs; integer 1 through vocabulary size. Select exactly k highest logits; equal-score boundary ties prefer lower IDs. Top-k 1 matches greedy. Zero, negative, noninteger or oversized values error. |
| Seed | Sample-mode integer 0 through 2^63−1. If omitted, generate and return `effective_seed`; replay with that seed. Greedy has no effective seed. |
| Empty prompt | Rejected, including with zero requested output; there is no trained BOS token. Whitespace-only nonempty prompts are allowed. |
| Unknown characters | Default `unknown_policy="error"` identifies the character/position. Explicit `replace` inserts one UNK per unknown character and reports the count. Invalid Unicode surrogates error under either policy. No implicit normalization. |
| Long prompt | At most 16,384 characters; larger requests error. Encode the whole bounded prompt, use only the final context-capacity IDs and report the initial truncation count. |
| Output length | Integer 0–1,024. Zero returns an empty continuation, zero IDs and `max_new_tokens`, without a model forward. Negative, boolean or excessive limits error. |
| EOS | Stop immediately after EOS, even as the first generated ID. Include it in IDs/count, omit its marker from visible continuation, and report `eos`. Immediate EOS means an empty continuation and one generated token. |
| UNK output | Preserve as visible `<unk>`; never silently discard. Special IDs remain distinguishable from literal characters through raw IDs. |
| Non-finite logits | Error; no invalid probability distribution is sampled. |

Each step retains the most recent **context_length** token IDs and recomputes the
entire window. Learned absolute positions restart at **0..L−1** for that retained
window. Position IDs do not wrap modulo context length or continue increasing past
capacity. No KV cache is retained. Tests hook actual positional embeddings and
check both short growing windows and overflow/cropping. A long prompt and its
last-256-character suffix produce the same continuation for these frozen character
models under identical generation settings.

Sampling probabilities are computed in CPU float64 after transferring float32 logits
off MPS; this avoids MPS's unsupported float64 transfer/conversion path. Stable
logit ordering defines top-k ties. CPU sampling uses a private
[multinomial generator](https://docs.pytorch.org/docs/2.15/generated/torch.multinomial.html);
it does not seed or advance application-global CPU/MPS RNG state. Model construction
also restores the caller's CPU RNG. Repeated seeds reproduce outputs with the same
weights, inputs and runtime/device. Floating-point changes across devices/releases
can change logits and sampled output; this is not a cross-platform bitwise promise.
See [PyTorch reproducibility limits](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

## Timing and resource ownership

`generation_duration_seconds` includes encoding, forward passes, transfers,
sampling, decoding and device synchronization. It excludes model loading, lock
wait, service construction and output/JSON serialization. `model_load_seconds`
separately includes cache-miss eviction, checksum/compatibility validation and
device transfer; it is zero on cache hits. The device is synchronized for timings.

Each service caches **one** model, on CPU or MPS. Switching releases the previous
model and clears its unused MPS allocator cache before loading the next. No CPU
backup of another model, optimizer or retained gradients is cached. All forward
passes use `torch.inference_mode()`. Artifact file changes invalidate a cache hit;
reload verifies the original pinned hashes rather than silently adopting new weights.
The catalog is immutable for a service instance; restart the service after an
operator registers a new version. Missing/corrupt artifacts are not silently replaced.

The service temporarily uses two CPU threads by default and float32 construction,
then restores previous PyTorch settings. Service calls are serialized across instances
while changing those process settings; unrelated application PyTorch work is outside
that lock. Cache capacity is per service, not a global limit on arbitrary Python
instances, other processes or other applications. Reuse one application-owned
instance instead of creating one per request.

Actual correctness, CLI execution, memory measurements and preserved-artifact checks
are recorded in the [Phase 11 report](phase11_inference_report.md). These trained
models remain small, repetitive and unreliable as sources of factual text; the
backend does not claim to improve their learned quality.
