"""Offline synthetic tests of frozen controls, milestone sampling, and baseline-only execution."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from transformer.configuration.settings import ComparisonProtocol
from transformer.training.experiments import run_baseline, save_samples, weights_fingerprint
from transformer.training.engine import CheckpointManager
from transformer.training.engine import capture_rng
from transformer.training.engine import run_training


@pytest.fixture
def comparison(training_fixture):
    root, config = training_fixture
    (root / "protocol.md").write_text("Synthetic fixture protocol; no corpus or network.\n")
    config = replace(
        config,
        train_stream="train_tinystories",
        max_updates=None,
        max_training_tokens=64,
        max_validation_batches=10,
    )
    protocol = ComparisonProtocol(
        config,
        replace(config, run_id="synthetic-model-b", train_stream="train"),
        "protocol.md",
        (2, 4),
        ("abcd", "dcba"),
        max_new_tokens=3,
    )
    return root, protocol


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("max_training_tokens", 80),
        ("microbatch_size", 1),
        ("evaluation_interval", 1),
        ("learning_rate", 0.002),
    ],
)
def test_comparison_rejects_unmatched_controls(comparison, field, value):
    _, protocol = comparison
    with pytest.raises(ValueError, match="Unmatched"):
        replace(protocol, model_b=replace(protocol.model_b, **{field: value}))


def test_comparison_roundtrip_and_invalid_milestones(comparison):
    _, protocol = comparison
    assert ComparisonProtocol.from_dict(json.loads(json.dumps(protocol.to_dict()))) == protocol
    for updates in ((2, 3), (4, 2), (True, 4), (2, 5)):
        with pytest.raises(ValueError, match="Sample updates"):
            replace(protocol, sample_updates=updates)


def test_baseline_archives_samples_and_best_without_training_model_b(comparison):
    root, protocol = comparison
    report = run_baseline(protocol, root)
    directory = root / "runs" / protocol.model_a.run_id
    assert report["status"] == "completed"
    assert report["training_tokens"] == 64 and report["optimizer_updates"] == 4
    assert len(report["sessions"]) == 2
    assert report["sessions"][0]["status"] == "stopped"
    assert report["sessions"][1]["resume"]
    assert not (root / "runs" / protocol.model_b.run_id).exists()
    assert not report["test_evaluation_performed"]
    initial = json.loads((directory / "initialization.json").read_text())
    assert initial["updates"] == 0 and initial["optimizer_state_entries"] == 0
    latest = CheckpointManager(directory).load()
    assert weights_fingerprint(latest["model"]) != initial["weights_sha256"]
    validations = json.loads((directory / "validation_history.json").read_text())["history"]
    assert all(row["validation_tokens"] == 20 for row in validations)
    assert report["best_validation"] == min(validations, key=lambda row: row["validation_loss"])
    for update in (2, 4):
        path = directory / f"diagnostics/samples-update-{update:06d}.json"
        samples = json.loads(path.read_text())
        archived = torch.load(directory / samples["checkpoint_file"], weights_only=True)
        assert archived["position"]["updates"] == update
        assert all(
            sample["checkpoint_update"] == update and sample["seed"] is None
            for sample in samples["samples"]
        )
    state = capture_rng(torch.device("cpu"))
    checkpoint_before = copy.deepcopy(CheckpointManager(directory).load())
    (directory / "diagnostics/samples-update-000004.json").unlink()
    save_samples(protocol, root, directory)
    assert torch.equal(state["torch_cpu"], torch.get_rng_state())
    assert checkpoint_before["position"] == CheckpointManager(directory).load()["position"]
    reference = replace(protocol.model_a, run_id="uninterrupted-reference")
    run_training(reference, root)
    reference_weights = CheckpointManager(root / "runs" / reference.run_id).load()["model"]
    for name, value in latest["model"].items():
        assert torch.equal(value, reference_weights[name])
    with pytest.raises(FileExistsError):
        run_baseline(protocol, root)
    resumed = run_baseline(protocol, root, resume=True)
    assert resumed["training_tokens"] == 64
    assert len(resumed["sessions"]) == 2
    (root / "protocol.md").write_text("Changed after training.")
    with pytest.raises(ValueError, match="frozen"):
        run_baseline(protocol, root, resume=True)


def test_baseline_refuses_partial_validation_and_unknown_prompts(comparison):
    root, protocol = comparison
    short = replace(
        protocol,
        model_a=replace(protocol.model_a, max_validation_batches=1),
        model_b=replace(protocol.model_b, max_validation_batches=1),
    )
    with pytest.raises(ValueError, match="every prepared validation"):
        run_baseline(short, root)
    with pytest.raises(ValueError, match="Unknown character"):
        run_baseline(replace(protocol, prompts=("z",)), root)
    assert not (root / "runs" / protocol.model_a.run_id).exists()


def test_interruption_between_sessions_is_recorded_and_recoverable(comparison, monkeypatch):
    from transformer.training import experiments as baseline_experiment

    root, protocol = comparison
    original = baseline_experiment.save_samples

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(baseline_experiment, "save_samples", interrupt)
    result = run_baseline(protocol, root)
    assert result["status"] == "interrupted" and result["updates"] == 2
    directory = root / "runs" / protocol.model_a.run_id
    assert "orchestration_interrupted" in (directory / "experiment_events.jsonl").read_text()
    monkeypatch.setattr(baseline_experiment, "save_samples", original)
    result = run_baseline(protocol, root, resume=True)
    assert result["status"] == "completed" and result["training_tokens"] == 64
    assert len(result["sessions"]) == 2
    archived = directory / "diagnostics/checkpoint-update-000004.pt"
    archived.write_bytes(b"corrupt synthetic fixture")
    with pytest.raises(ValueError, match="different checkpoint"):
        save_samples(protocol, root, directory)
