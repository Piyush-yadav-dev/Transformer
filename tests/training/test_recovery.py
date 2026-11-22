"""Tiny actual updates, interruption, artifact protection, and controlled CPU resume equivalence."""

import json
import signal
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from transformer.model import TransformerLanguageModel
from transformer.training import engine as checkpoint_manager
from transformer.training import engine as training_loop
from transformer.training.engine import CheckpointManager, run_lock
from transformer.training.engine import run_training


def load(root, config, which="latest"):
    return CheckpointManager(root / "runs" / config.run_id).load(which)


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_cpu_resume_exactly_reproduces_uninterrupted_updates(training_fixture):
    root, config = training_fixture
    full = replace(config, run_id="uninterrupted")
    resumed = replace(config, run_id="resumed")
    run_training(full, root)
    partial = run_training(resumed, root, stop_after_updates=2)
    assert partial["status"] == "stopped" and partial["updates"] == 2
    result = run_training(resumed, root, resume=True)
    assert result["status"] == "completed" and result["tokens"] == config.total_tokens
    a, b = load(root, full), load(root, resumed)
    for name in ("model", "optimizer", "scheduler", "sampler", "rng"):
        assert_nested_equal(a[name], b[name])
    for key in ("updates", "tokens", "best_validation", "last_validation"):
        assert_nested_equal(a["position"][key], b["position"][key])

    def updates(conf):
        rows = [
            json.loads(line)
            for line in (root / "runs" / conf.run_id / "metrics.jsonl").read_text().splitlines()
        ]
        return [
            (
                row["optimizer_update"],
                row["tokens_processed"],
                row["training_loss"],
                row["validation_loss"],
                row["learning_rate"],
                row["gradient_norm"],
            )
            for row in rows
            if row["event"] == "update"
        ]

    assert updates(full) == updates(resumed)


def test_exact_token_budget_and_partial_final_cycle(training_fixture):
    root, config = training_fixture
    config = replace(config, max_updates=None, max_training_tokens=28)
    result = run_training(config, root)
    assert result["tokens"] == 28 and result["updates"] == 2
    assert load(root, config)["scheduler"]["completed_updates"] == 2
    rows = [
        json.loads(line)
        for line in (root / "runs" / config.run_id / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["cycle_tokens"] for row in rows if row["event"] == "update"] == [16, 12]


def test_existing_run_and_changed_resume_settings_are_protected(training_fixture):
    root, config = training_fixture
    run_training(config, root, stop_after_updates=1)
    manager = CheckpointManager(root / "runs" / config.run_id)
    index = manager.index_path.read_bytes()
    with pytest.raises(FileExistsError, match="Run already exists"):
        run_training(config, root)
    for changed in (
        replace(config, learning_rate=0.02),
        replace(config, max_updates=5),
        replace(config, seed=3),
    ):
        with pytest.raises(ValueError, match="Incompatible resume"):
            run_training(changed, root, resume=True)
    assert manager.index_path.read_bytes() == index


def test_unknown_checkpoint_state_is_not_discarded(training_fixture):
    root, config = training_fixture
    run_training(config, root, stop_after_updates=1)
    manager = CheckpointManager(root / "runs" / config.run_id)
    state = manager.load()
    del state["rng"]["torch_cpu"]
    manager.commit(state, improved=False)
    with pytest.raises(ValueError, match="RNG state"):
        run_training(config, root, resume=True)


def test_missing_optimizer_history_is_not_silently_reset(training_fixture):
    root, config = training_fixture
    run_training(config, root, stop_after_updates=1)
    manager = CheckpointManager(root / "runs" / config.run_id)
    state = manager.load()
    state["optimizer"]["state"] = {}
    manager.commit(state, improved=False)
    with pytest.raises(ValueError, match="optimizer"):
        run_training(config, root, resume=True)


@pytest.mark.parametrize("change", ["learning_rate", "moment_shape", "step", "extra_group_option"])
def test_optimizer_history_is_validated(training_fixture, change):
    root, config = training_fixture
    run_training(config, root, stop_after_updates=1)
    manager = CheckpointManager(root / "runs" / config.run_id)
    payload = manager.load()
    state = payload["optimizer"]
    item = next(iter(state["state"].values()))
    if change == "learning_rate":
        state["param_groups"][0]["lr"] = 0.3
    elif change == "moment_shape":
        item["exp_avg"] = torch.zeros(1)
    elif change == "step":
        item["step"] = torch.tensor(0.0)
    else:
        state["param_groups"][0]["unsupported_option"] = True
    manager.commit(payload, improved=False)
    with pytest.raises(ValueError, match="[Oo]ptimizer"):
        run_training(config, root, resume=True)


def test_training_float32_is_explicit_and_ambient_dtype_is_restored(training_fixture):
    root, config = training_fixture
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        run_training(config, root, stop_after_updates=1)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(before)
    assert all(value.dtype == torch.float32 for value in load(root, config)["model"].values())


def test_latest_and_best_are_distinct_atomic_references(training_fixture, monkeypatch):
    root, config = training_fixture
    config = replace(config, max_updates=3, evaluation_interval=1)
    losses = iter([3.0, 2.0, 4.0])
    monkeypatch.setattr(
        training_loop,
        "evaluate",
        lambda *args: {"validation_loss": next(losses), "validation_tokens": 20},
    )
    run_training(config, root)
    assert load(root, config)["position"]["updates"] == 3
    assert load(root, config, "best")["position"]["updates"] == 2
    assert load(root, config)["position"]["best_validation"]["loss"] == 2
    assert len(list((root / "runs" / config.run_id / "checkpoints").glob("*.pt"))) == 2


def test_failed_atomic_publication_preserves_previous_checkpoint(training_fixture, monkeypatch):
    root, config = training_fixture
    run_training(config, root, stop_after_updates=1)
    manager = CheckpointManager(root / "runs" / config.run_id)
    before, payload = manager.index_path.read_bytes(), manager.load()
    original = checkpoint_manager.os.replace

    def fail_index(source, destination):
        if destination.name == "checkpoints.json":
            raise OSError("simulated publication failure")
        return original(source, destination)

    monkeypatch.setattr(checkpoint_manager.os, "replace", fail_index)
    with pytest.raises(OSError, match="publication"):
        manager.commit(payload, improved=False)
    assert manager.index_path.read_bytes() == before
    assert manager.load()["position"]["updates"] == 1
    assert not list(manager.directory.glob("*.tmp"))


def test_uncommitted_metrics_are_archived_on_resume(training_fixture):
    root, config = training_fixture
    run_training(config, root, stop_after_updates=1)
    directory = root / "runs" / config.run_id
    with (directory / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted partial line')
    run_training(config, root, resume=True)
    archives = list(directory.glob("metrics.uncommitted-*.jsonl"))
    assert len(archives) == 1 and archives[0].read_bytes() == b'{"uncommitted partial line'
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [row["optimizer_update"] for row in rows if row["event"] == "update"] == [1, 2, 3, 4]
    assert any(row.get("archived_uncommitted_metrics") == archives[0].name for row in rows)


def test_sigint_finishes_cycle_and_resume_is_exact(training_fixture, monkeypatch):
    root, config = training_fixture
    reference = replace(config, run_id="interrupt-reference")
    run_training(reference, root)
    original = TransformerLanguageModel.forward
    raised = False

    def interrupt_once(model, *args, **kwargs):
        nonlocal raised
        result = original(model, *args, **kwargs)
        if model.training and not raised:
            raised = True
            signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(TransformerLanguageModel, "forward", interrupt_once)
    stopped = run_training(config, root)
    assert (
        stopped["status"] == "interrupted" and stopped["updates"] == 1 and stopped["tokens"] == 16
    )
    monkeypatch.setattr(TransformerLanguageModel, "forward", original)
    run_training(config, root, resume=True)
    assert_nested_equal(load(root, config)["model"], load(root, reference)["model"])
    assert_nested_equal(load(root, config)["optimizer"], load(root, reference)["optimizer"])


def test_unexpected_keyboard_exception_keeps_last_safe_checkpoint(training_fixture, monkeypatch):
    root, config = training_fixture
    original = training_loop.optimizer_update

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(training_loop, "optimizer_update", interrupted)
    result = run_training(config, root)
    assert result["status"] == "interrupted" and result["updates"] == 0
    assert load(root, config)["position"]["tokens"] == 0


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_cycle_never_advances_checkpoint(training_fixture, monkeypatch, failure):
    root, config = training_fixture
    original = TransformerLanguageModel.forward

    def invalid(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        if failure == "loss":
            return type(result)(result.logits, result.loss * float("nan"))
        result.loss.register_hook(lambda gradient: gradient * float("nan"))
        return result

    monkeypatch.setattr(TransformerLanguageModel, "forward", invalid)
    with pytest.raises(RuntimeError, match="[Nn]on-finite"):
        run_training(config, root)
    assert load(root, config)["position"]["updates"] == 0


def test_dry_run_and_exclusive_writer(training_fixture):
    root, config = training_fixture
    report = run_training(config, root, dry_run=True)
    assert report["status"] == "dry_run" and report["total_updates"] == 4
    assert not (root / "runs").exists()
    with run_lock(root / "runs" / config.run_id):
        with pytest.raises(RuntimeError, match="Another writer"):
            run_training(config, root)
    assert not (root / "runs" / config.run_id).exists()


def test_cli_tiny_fixture_stop_and_resume(training_fixture):
    root, config = training_fixture
    config_path = root / "training.json"
    config.save(config_path)
    command = [
        sys.executable,
        "-m",
        "transformer",
        "train",
        "--config",
        str(config_path),
        "--project-root",
        str(root),
    ]
    for flags, status in [(["--stop-after-updates", "2"], "stopped"), (["--resume"], "completed")]:
        result = subprocess.run(command + flags, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == status


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Actual MPS training/recovery check pending."
)
def test_mps_checkpoint_records_rng_and_resumes(training_fixture):
    root, config = training_fixture
    config = replace(config, device="mps", max_updates=2, deterministic_algorithms=False)
    run_training(config, root, stop_after_updates=1)
    assert isinstance(load(root, config)["rng"]["torch_mps"], torch.Tensor)
    result = run_training(config, root, resume=True)
    assert result["updates"] == 2
    assert all(torch.isfinite(value).all() for value in load(root, config)["model"].values())
