# Source validation
"""Separate source-only validation views over already frozen memory-mapped tokens."""

import numpy as np
import torch

from transformer.configuration.settings import TrainingConfig
from transformer.data.preparation import fingerprint
from transformer.data.loading import SOURCES
from transformer.data.loading import TokenWindowDataset
from transformer.model import TransformerLanguageModel
from transformer.runtime.device import select_device
from transformer.training.engine import execution_settings
from transformer.training.engine import evaluate


class SourceValidationDataset(TokenWindowDataset):
    def __init__(self, provenance, source):
        if provenance.stream != "validation" or source not in SOURCES:
            raise ValueError("Source diagnostics accept only the prepared validation partition.")
        super().__init__(
            provenance.directory,
            "validation",
            provenance.tokenizer_fingerprint,
            context_length=provenance.context_length,
            dataset_fingerprint=provenance.dataset_fingerprint,
        )
        selected = np.flatnonzero(provenance.sources == SOURCES.index(source))
        self.original_starts = provenance.offsets[selected]
        self.source_offsets = np.concatenate(
            ([0], np.cumsum(np.diff(provenance.offsets)[selected], dtype=np.int64))
        )
        self.window_count = (int(self.source_offsets[-1]) - 1) // self.context_length
        self.dropped_tail_pairs = (int(self.source_offsets[-1]) - 1) % self.context_length
        if self.window_count < 1:
            raise ValueError(f"Source {source} needs at least one full validation window.")
        self.view_identity = {
            "source": source,
            "documents": len(selected),
            "stored_tokens": int(self.source_offsets[-1]),
            "windows": self.window_count,
            "scored_target_tokens": self.window_count * self.context_length,
            "dropped_tail_pairs": self.dropped_tail_pairs,
            "fingerprint": fingerprint(
                {
                    "policy": "source-only-packed-with-eos-v1",
                    "story_ids": [provenance.story_ids[i] for i in selected],
                    "tokenizer": provenance.tokenizer_fingerprint,
                    "dataset": provenance.dataset_fingerprint,
                    "context_length": self.context_length,
                }
            ),
        }

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Source validation window index is out of range.")
        self._ensure_mapping()
        positions = np.arange(index * self.context_length, (index + 1) * self.context_length + 1)
        stories = np.searchsorted(self.source_offsets[1:], positions, side="right")
        original = (
            self.original_starts[stories].astype(np.int64)
            + positions
            - self.source_offsets[stories]
        )
        # Doosre source ki stories hatao, par apne source ke EOS boundaries bachao.
        tensor = torch.from_numpy(np.array(self._tokens[original], dtype=np.int64, copy=True))
        return tensor[:-1].clone(), tensor[1:].clone()


def evaluate_sources(payload, provenance):
    config = TrainingConfig.from_dict(payload["identity"]["configuration"])
    if (
        config.tokenizer_fingerprint != provenance.tokenizer_fingerprint
        or config.dataset_fingerprint != provenance.dataset_fingerprint
        or config.model.context_length != provenance.context_length
    ):
        raise ValueError("Checkpoint and validation provenance are incompatible.")
    device = select_device(config.device).device
    with execution_settings(config, device):
        model = TransformerLanguageModel(config.model).to(device)
        model.load_state_dict(payload["model"], strict=True)
        results = {}
        for source in SOURCES:
            with SourceValidationDataset(provenance, source) as dataset:
                results[source] = {
                    **dataset.view_identity,
                    **evaluate(model, dataset, config.microbatch_size, len(dataset), device),
                }
    return {
        "update": payload["position"]["updates"],
        "sources": results,
        "selection_metric": False,
        "policy": (
            "Separate source-only streams; no cross-source context, same EOS packing and context. "
            "Different windows/tails from the mixed selector; do not average these to reproduce "
            "its loss."
        ),
    }


# Held out metrics
"""Target-exact held-out loss with packed and document-isolated context policies."""

import math
from itertools import islice

import numpy as np
import torch
from torch.nn import functional as F

from transformer.data.loading import SOURCES
from transformer.tokenization.token_preparation import read_token_manifest


def document_windows(length, context_length, stride):
    """Yield (input start, target start, target end) in local story coordinates."""
    if (
        type(length) is not int
        or length < 0
        or type(context_length) is not int
        or context_length < 1
        or type(stride) is not int
        or not 1 <= stride <= context_length
    ):
        raise ValueError("Document windows require length >= 0 and 1 <= stride <= context.")
    for start in range(1, length, stride):
        end = min(start + stride, length)
        # Pichhla context dobara dikhega, par har target sirf ek baar score hoga.
        yield max(0, end - 1 - context_length), start, end


def loss_summary(nll_sum, tokens):
    if tokens <= 0 or not math.isfinite(nll_sum) or nll_sum < 0:
        raise ValueError("Evaluation requires positive target counts and finite nonnegative NLL.")
    mean = nll_sum / tokens
    try:
        perplexity = math.exp(mean)
    except OverflowError:
        perplexity = None
    return {
        "nll_sum": nll_sum,
        "target_tokens": tokens,
        "cross_entropy_nats": mean,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
    }


def _windows(provenance, mode, stride):
    context = provenance.context_length
    if mode == "packed":
        for start in range(0, provenance.window_count * context, context):
            yield start, start + 1, start + context + 1
    elif mode == "document":
        for left, right in zip(provenance.offsets[:-1], provenance.offsets[1:], strict=True):
            left, right = int(left), int(right)
            for context_start, start, end in document_windows(right - left, context, stride):
                yield left + context_start, left + start, left + end
    else:
        raise ValueError("Evaluation mode must be packed or document.")


def score_held_out(model, provenance, *, mode, stride, batch_size, tokenizer):
    if provenance.stream != "test":
        raise ValueError("Held-out scoring requires the prepared test partition.")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("Evaluation batch size must be in [1, 16].")
    if not 1 <= stride <= provenance.context_length:
        raise ValueError("Document stride exceeds model context capacity.")
    if (
        model.config.context_length != provenance.context_length
        or model.config.vocab_size != tokenizer.vocab_size
        or tokenizer.fingerprint != provenance.tokenizer_fingerprint
    ):
        raise ValueError("Evaluation model/tokenizer/provenance are incompatible.")
    manifest = read_token_manifest(
        provenance.directory,
        tokenizer.fingerprint,
        expected_dataset_fingerprint=provenance.dataset_fingerprint,
    )
    info = manifest["streams"]["test"]
    if info["tokens"] < 2 or len(provenance.story_ids) == 0:
        raise ValueError("Held-out test data has no scoreable target tokens.")
    tokens = np.memmap(provenance.directory / info["token_file"], mode="r", dtype=info["dtype"])
    count = len(provenance.story_ids)
    totals = np.zeros(count, dtype=np.float64)
    counts = np.zeros(count, dtype=np.int64)
    unknown = np.zeros(count, dtype=np.int64)
    eos = np.zeros(count, dtype=np.int64)
    window_count = 0
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    windows = iter(_windows(provenance, mode, stride))
    try:
        with torch.inference_mode():
            while batch := list(islice(windows, batch_size)):
                width = max(end - 1 - left for left, _, end in batch)
                x = np.full((len(batch), width), tokenizer.eos_id, dtype=np.int64)
                y = np.full_like(x, tokenizer.eos_id)
                masks = np.zeros_like(x, dtype=bool)
                owners = np.zeros_like(x)
                for row, (left, start, end) in enumerate(batch):
                    length = end - 1 - left
                    x[row, :length] = tokens[left : end - 1]
                    y[row, :length] = tokens[left + 1 : end]
                    # Logit i agla token predict karta hai; context/padding loss mein nahi aayega.
                    masks[row, start - 1 - left : length] = True
                    owners[row, :length] = np.searchsorted(
                        provenance.offsets[1:], np.arange(left + 1, end), side="right"
                    )
                logits = model(torch.as_tensor(x, device=device)).logits
                losses = (
                    F.cross_entropy(
                        logits.reshape(-1, tokenizer.vocab_size),
                        torch.as_tensor(y, device=device).reshape(-1),
                        reduction="none",
                    )
                    .reshape(x.shape)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if not np.isfinite(losses[masks]).all():
                    raise ValueError("Non-finite held-out loss; evaluation stopped.")
                np.add.at(totals, owners[masks], losses[masks])
                np.add.at(counts, owners[masks], 1)
                np.add.at(unknown, owners[masks], (y[masks] == tokenizer.unk_id).astype(int))
                np.add.at(eos, owners[masks], (y[masks] == tokenizer.eos_id).astype(int))
                window_count += len(batch)
    finally:
        model.train(previous_mode)
        tokens._mmap.close()
    sources = {}
    for i, source in enumerate(SOURCES):
        mask = provenance.sources == i
        sources[source] = {
            **loss_summary(float(totals[mask].sum()), int(counts[mask].sum())),
            "documents": int(mask.sum()),
            "unknown_targets": int(unknown[mask].sum()),
            "eos_targets": int(eos[mask].sum()),
        }
    total = int(counts.sum())
    equal_loss = sum(s["cross_entropy_nats"] for s in sources.values()) / len(SOURCES)
    equal_summary = loss_summary(equal_loss, 1)
    return {
        "mode": mode,
        "windows": window_count,
        "combined": {
            **loss_summary(float(totals.sum()), total),
            "source_weights": {s: v["target_tokens"] / total for s, v in sources.items()},
        },
        "equal_source_secondary": {
            "cross_entropy_nats": equal_loss,
            "perplexity": equal_summary["perplexity"],
            "perplexity_overflow": equal_summary["perplexity_overflow"],
            "weights": dict.fromkeys(SOURCES, 0.5),
        },
        "sources": sources,
        "unscored_stored_tokens": int(provenance.offsets[-1]) - total,
        "documents_without_targets": int((counts == 0).sum()),
        "documents": [
            {
                "story_id": identity,
                "source": SOURCES[int(provenance.sources[i])],
                "nll_sum": float(totals[i]),
                "target_tokens": int(counts[i]),
                "unknown_targets": int(unknown[i]),
                "eos_targets": int(eos[i]),
            }
            for i, identity in enumerate(provenance.story_ids)
        ],
    }


# Generation comparison
"""Fixed greedy continuations and synchronized, interleaved decode timing."""

import time

import torch

from transformer.training.engine import synchronize


def greedy_continuation(model, tokenizer, prompt, new_tokens, *, stop_at_eos=True):
    if not isinstance(prompt, str) or not prompt or type(new_tokens) is not int or new_tokens < 1:
        raise ValueError("Generation needs a nonempty prompt and positive token limit.")
    ids = tokenizer.encode(prompt, unknown="error")
    generated = []
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(new_tokens):
                x = torch.tensor([ids[-model.config.context_length :]], device=device)
                token = int(model(x).logits[0, -1].argmax().item())
                generated.append(token)
                ids.append(token)
                if stop_at_eos and token == tokenizer.eos_id:
                    break
    finally:
        model.train(previous_mode)
    return {
        "prompt": prompt,
        "generated_ids": generated,
        "continuation": tokenizer.decode(generated),
        "new_tokens": len(generated),
        "ended_with_eos": generated[-1] == tokenizer.eos_id,
        "stop_reason": "eos" if stop_at_eos and generated[-1] == tokenizer.eos_id else "token_cap",
        "settings": {
            "strategy": "greedy",
            "temperature": 0,
            "top_k": None,
            "top_p": None,
            "sampling_seed": None,
            "max_new_tokens": new_tokens,
            "stop_at_eos": stop_at_eos,
            "context_length": model.config.context_length,
        },
    }


def benchmark_generation(models, tokenizer, prompts, config):
    for model in models.values():
        for prompt in prompts:
            for _ in range(config.benchmark_warmups):
                greedy_continuation(
                    model, tokenizer, prompt, config.benchmark_new_tokens, stop_at_eos=False
                )
        synchronize(next(model.parameters()).device)
    raw = []
    names = tuple(models)
    for repeat in range(config.benchmark_repetitions):
        for prompt_index, prompt in enumerate(prompts):
            order = names if (repeat + prompt_index) % 2 == 0 else names[::-1]
            for name in order:
                model = models[name]
                device = next(model.parameters()).device
                synchronize(device)
                start = time.perf_counter()
                sample = greedy_continuation(
                    model, tokenizer, prompt, config.benchmark_new_tokens, stop_at_eos=False
                )
                synchronize(device)
                elapsed = time.perf_counter() - start
                raw.append(
                    {
                        "member": name,
                        "repetition": repeat,
                        "prompt_index": prompt_index,
                        "order": list(order),
                        "elapsed_seconds": elapsed,
                        "new_tokens": sample["new_tokens"],
                        "tokens_per_second": sample["new_tokens"] / elapsed,
                    }
                )
    summaries = {}
    for name in names:
        rows = [r for r in raw if r["member"] == name]
        summaries[name] = {
            "pooled_tokens_per_second": sum(r["new_tokens"] for r in rows)
            / sum(r["elapsed_seconds"] for r in rows),
            "minimum_trial_tokens_per_second": min(r["tokens_per_second"] for r in rows),
            "maximum_trial_tokens_per_second": max(r["tokens_per_second"] for r in rows),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
            "new_tokens": sum(r["new_tokens"] for r in rows),
            "trials": len(rows),
        }
    return {
        "summaries": summaries,
        "raw": raw,
        "definition": "Batch-one autoregressive end-to-end decoding; model load/warmup excluded, "
        "encoding/transfers/argmax/decoding included; synchronized; no KV cache; "
        "fixed token count with EOS stopping disabled for timing only.",
    }


# Comparison charts
"""Standalone charts from recorded measurements; no inferred or synthetic curves."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def create_charts(result, directory):
    charts = directory / "charts"
    charts.mkdir()
    names = ("model_a", "model_b")
    colors = ("#2368a0", "#c05229")
    labels = [result["runs"][m]["run_id"] for m in names]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, stem):
        fig.legend(
            [Patch(color=c) for c in colors],
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(charts / f"{stem}.png", dpi=160)
        fig.savefig(charts / f"{stem}.svg")
        plt.close(fig)

    for stem, key, rows_key, title in (
        ("training_loss", "training_loss", "updates", "Training loss — all raw updates"),
        (
            "validation_loss",
            "validation_loss",
            "validation_history",
            "Recorded validation selection loss",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for member, color in zip(names, colors, strict=True):
            rows = result["runs"][member][rows_key]
            ax.plot(
                [r["tokens_processed"] for r in rows],
                [r[key] for r in rows],
                color=color,
                lw=1,
                alpha=0.8,
            )
        ax.set(
            title=title,
            xlabel="Training target tokens processed",
            ylabel="Cross-entropy (nats/token)",
        )
        ax.grid(alpha=0.2)
        save(fig, stem)
    sources = ("tinystories", "simplestories")
    for key, stem, ylabel in (
        ("cross_entropy_nats", "held_out_loss", "Cross-entropy (nats/token)"),
        ("perplexity", "held_out_perplexity", "Perplexity (character/special token)"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, (member, color) in enumerate(zip(names, colors, strict=True)):
            vals = [result["scores"][member]["packed"]["sources"][s][key] for s in sources]
            bars = ax.bar(np.arange(2) + (i - 0.5) * 0.34, vals, 0.34, color=color)
            ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(
            xticks=[0, 1],
            xticklabels=["TinyStories", "SimpleStories"],
            ylabel=ylabel,
            title="Primary held-out metric — original packed windows",
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        save(fig, stem)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (member, color) in enumerate(zip(names, colors, strict=True)):
        vals = [
            result["runs"][member][k]
            for k in ("training_core_seconds", "training_session_wall_seconds")
        ]
        bars = ax.bar(np.arange(2) + (i - 0.5) * 0.34, vals, 0.34, color=color)
        ax.bar_label(bars, fmt="%.1f s", padding=3)
    ax.set(
        xticks=[0, 1],
        xticklabels=[
            "Core (batching + transfers + updates)",
            "Session (validation + I/O + audits)",
        ],
        ylabel="Elapsed time (seconds)",
        title="Actual training duration — B has additional source audits",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    save(fig, "training_duration")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (member, color) in enumerate(zip(names, colors, strict=True)):
        summary = result["generation_benchmark"]["summaries"][member]
        bar = ax.bar(i, summary["pooled_tokens_per_second"], 0.5, color=color)
        ax.bar_label(bar, fmt="%.1f", padding=4)
        raw = [
            r["tokens_per_second"]
            for r in result["generation_benchmark"]["raw"]
            if r["member"] == member
        ]
        ax.scatter(np.full(len(raw), i), raw, marker="_", color="black", s=80, zorder=4)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Model A", "Model B"],
        ylabel="Generated tokens/second",
        title=f"Synchronized {result['environment']['device'].upper()} decode throughput "
        "— bars pooled; ticks individual trials",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    save(fig, "generation_throughput")
