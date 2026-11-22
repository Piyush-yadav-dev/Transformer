"""Early configuration errors, fixed LR timing, accumulation, and no-gradient validation."""

from dataclasses import replace

import pytest
import torch

from transformer.configuration.settings import TrainingConfig
from transformer.model import TransformerLanguageModel
from transformer.training.engine import WarmupCosineSchedule, make_optimizer, optimizer_update
from transformer.data.loading import WindowSampler
from transformer.training.engine import evaluate


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "../bad"),
        ("token_data_dir", "/absolute"),
        ("tokenizer_path", "../bad"),
        ("device", "cuda"),
        ("seed", -1),
        ("microbatch_size", 0),
        ("gradient_accumulation_steps", True),
        ("max_updates", None),
        ("max_updates", 0),
        ("max_training_tokens", 16),
        ("learning_rate", float("nan")),
        ("learning_rate", 0),
        ("weight_decay", -1),
        ("gradient_clip_norm", 0),
        ("warmup_updates", 4),
        ("minimum_lr_ratio", 1.1),
        ("evaluation_interval", 0),
        ("checkpoint_interval", 0),
        ("max_validation_batches", 0),
        ("train_stream", "test"),
        ("num_workers", 1),
        ("tokenizer_fingerprint", "wrong"),
        ("dataset_fingerprint", "wrong"),
        ("token_data_fingerprint", "wrong"),
        ("deterministic_algorithms", 1),
    ],
)
def test_invalid_configuration_fails_before_run_creation(training_fixture, field, value):
    root, config = training_fixture
    with pytest.raises(ValueError):
        replace(config, **{field: value})
    assert not (root / "runs").exists()


def test_configuration_round_trip_and_token_budget(training_fixture):
    root, config = training_fixture
    target = root / "config.json"
    config.save(target)
    assert TrainingConfig.load(target) == config
    with pytest.raises(FileExistsError):
        config.save(target)
    with pytest.raises(ValueError, match="Unknown"):
        TrainingConfig.from_dict({**config.to_dict(), "unknown": 1})
    with pytest.raises(ValueError, match="multiple"):
        replace(config, max_updates=None, max_training_tokens=18)
    bounded = replace(config, max_updates=None, max_training_tokens=28)
    assert bounded.total_updates == 2 and bounded.total_tokens == 28


def test_scheduler_timing_and_incompatible_resume(training_fixture):
    _, config = training_fixture
    schedule = WarmupCosineSchedule(
        replace(config, max_updates=5, warmup_updates=2, learning_rate=1.0, minimum_lr_ratio=0.1)
    )
    assert [schedule.learning_rate(i) for i in range(1, 6)] == pytest.approx(
        [0.5, 1, 0.775, 0.325, 0.1]
    )
    schedule.completed_updates = 3
    state = schedule.state_dict()
    another = WarmupCosineSchedule(config)
    with pytest.raises(ValueError, match="schedule"):
        another.load_state_dict(state)
    single = WarmupCosineSchedule(replace(config, max_updates=1, warmup_updates=0))
    assert single.learning_rate(1) == config.learning_rate


def test_accumulation_matches_large_batch_including_partial_cycle(
    training_fixture, synthetic_windows, monkeypatch
):
    _, config = training_fixture
    config = replace(
        config, model=replace(config.model, dropout=0), gradient_clip_norm=0.1, warmup_updates=0
    )
    torch.manual_seed(7)
    accumulated = TransformerLanguageModel(config.model).double()
    large = TransformerLanguageModel(config.model).double()
    large.load_state_dict(accumulated.state_dict())
    with torch.no_grad():
        losses = [
            accumulated(x.unsqueeze(0), y.unsqueeze(0)).loss.item()
            for x, y in synthetic_windows[:3]
        ]
    wrong_equal_microbatch_mean = ((losses[0] + losses[1]) / 2 + losses[2]) / 2
    assert abs(sum(losses) / 3 - wrong_equal_microbatch_mean) > 1e-5
    first_optimizer, second_optimizer = (
        make_optimizer(accumulated, config),
        make_optimizer(large, config),
    )
    initial = {
        name: parameter.detach().clone() for name, parameter in accumulated.named_parameters()
    }
    calls = []
    clip = torch.nn.utils.clip_grad_norm_

    def counted_clip(*args, **kwargs):
        calls.append(1)
        return clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    first = optimizer_update(
        accumulated,
        first_optimizer,
        WarmupCosineSchedule(config),
        synthetic_windows,
        WindowSampler(4, 1, False),
        config,
        3,
        torch.device("cpu"),
    )
    large_config = replace(config, microbatch_size=3, gradient_accumulation_steps=1)
    second = optimizer_update(
        large,
        second_optimizer,
        WarmupCosineSchedule(large_config),
        synthetic_windows,
        WindowSampler(4, 1, False),
        large_config,
        3,
        torch.device("cpu"),
    )
    assert len(calls) == 2
    assert first["cycle_tokens"] == 12
    assert first["training_loss"] == pytest.approx(second["training_loss"], abs=1e-12)
    assert first["gradient_norm"] == pytest.approx(second["gradient_norm"], abs=1e-12)
    assert any(
        not torch.equal(initial[name], parameter)
        for name, parameter in accumulated.named_parameters()
    )
    for name, parameter in accumulated.named_parameters():
        torch.testing.assert_close(parameter, large.get_parameter(name), atol=1e-12, rtol=1e-10)
        assert parameter.grad is None
    assert all(state["step"].item() == 1 for state in first_optimizer.state.values())


def test_validation_is_token_weighted_and_restores_mode(training_fixture, synthetic_windows):
    _, config = training_fixture
    torch.manual_seed(13)
    model = TransformerLanguageModel(config.model)
    windows = synthetic_windows[:3]
    observed = []

    def hook(module, args):
        observed.append(
            (module.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )

    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    handle = model.register_forward_pre_hook(hook)
    actual = evaluate(model, windows, 2, 10, torch.device("cpu"))
    handle.remove()
    assert observed == [(False, False, True)] * 2
    assert model.training and actual["validation_tokens"] == 12
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    model.eval()
    with torch.no_grad():
        losses = [model(x.unsqueeze(0), y.unsqueeze(0)).loss.item() for x, y in windows]
        expected = sum(losses) / 3
    wrong_equal_batch_mean = ((losses[0] + losses[1]) / 2 + losses[2]) / 2
    assert abs(expected - wrong_equal_batch_mean) > 1e-5
    assert actual["validation_loss"] == pytest.approx(expected, abs=1e-6)
    evaluate(model, windows, 2, 1, torch.device("cpu"))
    assert not model.training
    model.train()
    handle = model.register_forward_pre_hook(
        lambda *_: (_ for _ in ()).throw(RuntimeError("fixture failure"))
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        evaluate(model, windows, 2, 1, torch.device("cpu"))
    handle.remove()
    assert model.training


def test_sampler_resume_across_epoch_boundaries():
    first = WindowSampler(5, 42, True)
    first.take(8)
    state = first.state_dict()
    restored = WindowSampler(5, 42, True)
    restored.load_state_dict(state)
    assert first.take(17) == restored.take(17)
    state["cursor"] = 10
    with pytest.raises(ValueError, match="position"):
        restored.load_state_dict(state)
