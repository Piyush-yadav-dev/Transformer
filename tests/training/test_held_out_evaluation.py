"""Offline synthetic fixtures: mathematical metrics, provenance and read-only evaluation."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_combined_experiment import baseline_with_hash_evidence
from test_combined_experiment import paired_fixture as _paired_fixture
from torch.nn import functional as F

from transformer.configuration.settings import EvaluationConfig
from transformer.data.loading import SourceProvenance, audit_partitions
from transformer.data.loading import TokenWindowDataset
from transformer.evaluation.comparison import inventory, preflight, run_comparison
from transformer.evaluation.metrics import benchmark_generation, greedy_continuation
from transformer.evaluation.metrics import document_windows, loss_summary, score_held_out
from transformer.model import TransformerLanguageModel
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.training.experiments import run_experiment
from transformer.training.engine import evaluate


@pytest.fixture
def paired_fixture(tmp_path):
    return _paired_fixture.__wrapped__(tmp_path)


@pytest.mark.parametrize("length", [0, 1, 2, 4, 5, 6, 9, 13])
def test_document_coverage_exactly_once_with_overlap(length):
    windows = list(document_windows(length, 4, 2))
    targets = []
    for left, start, end in windows:
        assert 0 <= left < start < end <= length
        assert end - left - 1 <= 4
        targets.extend(range(start, end))
    assert targets == list(range(1, length))
    if length >= 6:
        assert windows[0][0] == windows[1][0]  # Context is reused, targets are not.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"document_stride": 0},
        {"batch_size": True},
        {"device": "cuda"},
        {"benchmark_repetitions": 11},
        {"output_directory": "../escape"},
    ],
)
def test_evaluation_configuration_rejects_invalid_inputs(kwargs):
    settings = {
        "comparison_protocol": "pair.json",
        "document_path": "protocol.md",
        "output_directory": "evaluation",
    }
    with pytest.raises(ValueError):
        EvaluationConfig(**(settings | kwargs))


def test_perplexity_uses_exp_of_token_weighted_mean():
    result = loss_summary(3 * math.log(2) + math.log(8), 4)
    assert result["perplexity"] == pytest.approx(math.sqrt(8))
    assert result["perplexity"] != pytest.approx((2 + 8) / 2)
    assert loss_summary(1000, 1)["perplexity_overflow"]
    for nll, count in [(0, 0), (float("nan"), 1), (-1, 1)]:
        with pytest.raises(ValueError):
            loss_summary(nll, count)


def test_equal_source_perplexity_overflow_is_serializable(paired_fixture):
    from transformer.model.transformer import TransformerOutput

    root, protocol = paired_fixture
    views, tokenizer, model = _components(root, protocol)

    def force_extreme_loss(_module, _inputs, output):
        logits = torch.zeros_like(output.logits)
        logits[..., tokenizer.unk_id] = 1000
        return TransformerOutput(logits, None)

    model.register_forward_hook(force_extreme_loss)
    scores = score_held_out(
        model, views["test"], mode="packed", stride=2, batch_size=2, tokenizer=tokenizer
    )
    assert scores["combined"]["cross_entropy_nats"] == 1000
    assert scores["equal_source_secondary"]["perplexity"] is None
    assert scores["equal_source_secondary"]["perplexity_overflow"] is True
    json.dumps(scores, allow_nan=False)


def _components(root, protocol):
    views, _ = audit_partitions(root, protocol.model_a)
    tokenizer = CharacterTokenizer.load(root / protocol.model_a.tokenizer_path)
    model = TransformerLanguageModel(protocol.model_a.model)
    return views, tokenizer, model


def test_metrics_match_independent_reference_and_training_packed_policy(paired_fixture):
    root, protocol = paired_fixture
    views, tokenizer, model = _components(root, protocol)
    p = views["test"]
    packed = score_held_out(model, p, mode="packed", stride=2, batch_size=3, tokenizer=tokenizer)
    with TokenWindowDataset(p.directory, "test", tokenizer.fingerprint, context_length=4) as data:
        reference = evaluate(model, data, 2, len(data), torch.device("cpu"))
    assert packed["combined"]["cross_entropy_nats"] == pytest.approx(
        reference["validation_loss"], abs=2e-7
    )
    assert packed["combined"]["target_tokens"] == reference["validation_tokens"]
    counts = p.target_counts().sum(axis=0)
    assert [
        packed["sources"][s]["target_tokens"] for s in ("tinystories", "simplestories")
    ] == list(counts)
    assert model.training  # Original train mode restored; no gradients built.
    assert all(parameter.grad is None for parameter in model.parameters())
    isolated = score_held_out(
        model, p, mode="document", stride=2, batch_size=3, tokenizer=tokenizer
    )
    serial = score_held_out(model, p, mode="document", stride=2, batch_size=1, tokenizer=tokenizer)
    assert isolated["combined"]["cross_entropy_nats"] == pytest.approx(
        serial["combined"]["cross_entropy_nats"], abs=2e-7
    )
    raw = np.fromfile(p.directory / "test.bin", dtype=np.uint8)
    model.eval()
    expected_sum = 0.0
    with torch.inference_mode():
        for left, right in zip(p.offsets[:-1], p.offsets[1:], strict=True):
            ids = raw[int(left) : int(right)].tolist()
            for target in range(1, len(ids)):
                block_end = min(1 + ((target - 1) // 2 + 1) * 2, len(ids))
                context_start = max(0, block_end - 1 - 4)
                # Independent single-target causal prefix, no padding or batched loss mask.
                x = torch.tensor([ids[context_start:target]])
                logits = model(x).logits[:, -1]
                expected_sum += F.cross_entropy(logits, torch.tensor([ids[target]])).item()
    assert isolated["combined"]["nll_sum"] == pytest.approx(expected_sum, abs=2e-5)
    assert isolated["combined"]["target_tokens"] == len(raw) - len(p.story_ids)
    assert sum(s["eos_targets"] for s in isolated["sources"].values()) == len(p.story_ids)
    wrong = CharacterTokenizer(tuple(sorted((*tokenizer.characters, "z"))))
    with pytest.raises(ValueError, match="incompatible"):
        score_held_out(model, p, mode="packed", stride=2, batch_size=1, tokenizer=wrong)
    with pytest.raises(ValueError, match="test partition"):
        score_held_out(
            model, views["train"], mode="packed", stride=2, batch_size=1, tokenizer=tokenizer
        )


def test_empty_test_view_fails_without_dividing_by_zero(paired_fixture):
    root, protocol = paired_fixture
    views, tokenizer, model = _components(root, protocol)
    empty = replace(views["test"], story_ids=(), offsets=np.array([0]), sources=np.array([]))
    with pytest.raises(ValueError, match="no scoreable"):
        score_held_out(model, empty, mode="document", stride=2, batch_size=1, tokenizer=tokenizer)


class AlwaysEOS(torch.nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(context_length=4)
        self.vocabulary = vocabulary

    def forward(self, x):
        logits = torch.zeros((*x.shape, self.vocabulary))
        logits[..., 1] = 10
        return SimpleNamespace(logits=logits)


def test_qualitative_eos_and_fixed_work_timing_are_distinct():
    tokenizer = CharacterTokenizer(("a", "b"))
    model = AlwaysEOS(tokenizer.vocab_size)
    sample = greedy_continuation(model, tokenizer, "a", 4)
    assert sample["new_tokens"] == 1 and sample["stop_reason"] == "eos"
    assert sample["continuation"] == "<eos>" and model.training
    assert greedy_continuation(model, tokenizer, "a", 4, stop_at_eos=False)["new_tokens"] == 4
    config = EvaluationConfig(
        "pair.json", "protocol.md", "out", benchmark_repetitions=2, benchmark_new_tokens=4
    )
    result = benchmark_generation({"model_a": model, "model_b": model}, tokenizer, ["a"], config)
    assert len(result["raw"]) == 4
    assert result["raw"][0]["member"] == "model_a"
    assert result["raw"][2]["member"] == "model_b"
    assert result["summaries"]["model_a"]["new_tokens"] == 8


def test_complete_comparison_preserves_runs_and_rejects_reuse(paired_fixture):
    pytest.importorskip("matplotlib", reason="Install the evaluation extra for chart integration.")
    root, protocol = paired_fixture
    baseline_with_hash_evidence(root, protocol)
    run_experiment(protocol, root, member="model_b")
    (root / "pair.json").write_text(json.dumps(protocol.to_dict()))
    (root / "evaluation.md").write_text("Synthetic held-out fixture protocol, not research data.")
    config = EvaluationConfig(
        "pair.json",
        "evaluation.md",
        "artifacts/evaluation",
        document_stride=2,
        benchmark_repetitions=1,
        benchmark_new_tokens=2,
    )
    before = inventory(root / "runs")
    result = run_comparison(config, root)
    assert result["status"] == "completed"
    assert before == inventory(root / "runs")
    report = json.loads((root / config.output_directory / "results.json").read_text())
    assert report["test_results_used_for_selection"] is False
    assert len(list((root / config.output_directory / "charts").glob("*.png"))) == 6
    assert report["scores"]["model_a"]["document"]["combined"]["target_tokens"] == 24
    with pytest.raises(FileExistsError):
        run_comparison(config, root)
    history = root / "runs" / protocol.model_a.run_id / "validation_history.json"
    value = json.loads(history.read_text())
    value["history"][0]["validation_loss"] += 0.1
    history.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs from actual logs"):
        preflight(protocol, root)


def test_short_stories_unknown_and_eos_are_scored(tmp_path, monkeypatch):
    from transformer.configuration.settings import TransformerConfig

    tokenizer = CharacterTokenizer(("a", "b"))
    # Synthetic token fixture: unknown character plus two very short stories.
    np.array([2, 0, 1, 3, 1], dtype=np.uint8).tofile(tmp_path / "test.bin")
    monkeypatch.setattr(
        "transformer.evaluation.metrics.read_token_manifest",
        lambda *args, **kwargs: {
            "streams": {
                "test": {
                    "tokens": 5,
                    "token_file": "test.bin",
                    "dtype": "|u1",
                }
            }
        },
    )
    provenance = SourceProvenance(
        tmp_path,
        "test",
        np.array([0, 3, 5]),
        np.array([0, 1]),
        ("first", "second"),
        tokenizer.fingerprint,
        "synthetic",
        4,
    )
    model = TransformerLanguageModel(
        TransformerConfig(
            vocab_size=4,
            num_layers=1,
            num_heads=1,
            embedding_dim=4,
            context_length=4,
        )
    )
    scores = score_held_out(
        model,
        provenance,
        mode="document",
        stride=2,
        batch_size=2,
        tokenizer=tokenizer,
    )
    assert scores["combined"]["target_tokens"] == 3
    assert scores["sources"]["tinystories"]["unknown_targets"] == 1
    assert scores["sources"]["tinystories"]["eos_targets"] == 1
    assert scores["sources"]["simplestories"]["eos_targets"] == 1
    assert scores["sources"]["simplestories"]["target_tokens"] == 1


@pytest.mark.mps
def test_held_out_mps_matches_cpu(paired_fixture):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this machine")
    root, protocol = paired_fixture
    views, tokenizer, model = _components(root, protocol)
    for mode in ("packed", "document"):
        cpu = score_held_out(
            model.cpu(), views["test"], mode=mode, stride=2, batch_size=3, tokenizer=tokenizer
        )
        mps = score_held_out(
            model.to("mps"), views["test"], mode=mode, stride=2, batch_size=3, tokenizer=tokenizer
        )
        assert cpu["combined"]["cross_entropy_nats"] == pytest.approx(
            mps["combined"]["cross_entropy_nats"], abs=2e-5
        )
