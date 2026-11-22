"""Offline synthetic fixtures for bounded diagnostics and honest benchmark accounting."""

import copy
import json
import signal
from dataclasses import replace

import pytest
import torch

from transformer.benchmarking.worker import diagnostic_continuation, run_overfit
from transformer.benchmarking.hardware_benchmark import (
    estimate_runtime,
    run_benchmark,
    select_candidate,
)
from transformer.benchmarking.worker import MemoryMonitor, memory_limit_reason
from transformer.benchmarking.worker import run_case, summarize_rows
from transformer.configuration.settings import BenchmarkConfig
from transformer.model import TransformerLanguageModel
from transformer.tokenization.character_tokenizer import CharacterTokenizer
from transformer.training.engine import CheckpointManager
from transformer.training.engine import (
    WarmupCosineSchedule,
    batch_from_indices,
    make_optimizer,
    optimizer_update,
    optimizer_update_batches,
)
from transformer.training.engine import execution_settings
from transformer.data.loading import WindowSampler
from transformer.training.engine import run_training


@pytest.mark.parametrize(
    "change",
    [
        {"microbatch_sizes": (2, 1)},
        {"microbatch_sizes": (True,)},
        {"microbatch_sizes": (3,)},
        {"warmup_updates": 0},
        {"measured_updates": 1000},
        {"case_max_seconds": float("nan")},
        {"output_dir": "../escape"},
    ],
)
def test_benchmark_rejects_unbounded_or_ambiguous_settings(training_fixture, change):
    _, config = training_fixture
    with pytest.raises(ValueError):
        BenchmarkConfig(**{"output_dir": "artifacts/synthetic", "training": config, **change})


def test_benchmark_configuration_roundtrip(training_fixture):
    _, config = training_fixture
    settings = BenchmarkConfig("artifacts/synthetic", config)
    assert BenchmarkConfig.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings


def test_cached_batch_math_matches_pipeline(training_fixture, synthetic_windows):
    _, config = training_fixture
    config = replace(config, model=replace(config.model, dropout=0))
    device = torch.device("cpu")
    with execution_settings(config, device):
        first = TransformerLanguageModel(config.model)
        second = copy.deepcopy(first)
        schedule_a, schedule_b = WarmupCosineSchedule(config), WarmupCosineSchedule(config)
        streamed = optimizer_update(
            first,
            make_optimizer(first, config),
            schedule_a,
            synthetic_windows,
            WindowSampler(4, config.seed, False),
            config,
            3,
            device,
        )
        cached = [
            batch_from_indices(synthetic_windows, indices, device) for indices in ([0, 1], [2])
        ]
        resident = optimizer_update_batches(
            second,
            make_optimizer(second, config),
            schedule_b,
            cached,
            config,
            3,
            device,
        )
        for name, parameter in first.state_dict().items():
            assert torch.equal(parameter, second.state_dict()[name])
        assert streamed["training_loss"] == resident["training_loss"]
        assert resident["cycle_tokens"] == 12


def test_incomplete_resident_cycle_refuses_step(training_fixture, synthetic_windows):
    _, config = training_fixture
    device = torch.device("cpu")
    with execution_settings(config, device):
        model = TransformerLanguageModel(config.model)
        before = copy.deepcopy(model.state_dict())
        optimizer, schedule = make_optimizer(model, config), WarmupCosineSchedule(config)
        with pytest.raises(ValueError, match="Incomplete"):
            optimizer_update_batches(model, optimizer, schedule, [], config, 2, device)
        assert schedule.completed_updates == 0
        for key, value in before.items():
            assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize("interrupt", [False, True])
def test_observer_sees_each_complete_boundary_once(training_fixture, interrupt):
    root, config = training_fixture
    events = []

    def observer(event):
        events.append(event.copy())
        if interrupt and event.get("optimizer_update") == 1:
            signal.raise_signal(signal.SIGINT)

    result = run_training(config, root, observer=observer)
    updates = [event for event in events if event["event"] == "update"]
    assert events[0]["event"] == "ready"
    expected = 1 if interrupt else config.total_updates
    assert [event["optimizer_update"] for event in updates] == list(range(1, expected + 1))
    assert (
        CheckpointManager(root / "runs" / config.run_id).load()["position"]["updates"] == expected
    )
    assert result["updates"] == expected
    if not interrupt:
        assert updates[-1]["checkpoint_seconds"] > 0
        assert updates[-1]["validation_seconds"] > 0


def test_diagnostic_sampler_is_bounded_and_restores_mode(training_fixture):
    _, config = training_fixture
    tokenizer = CharacterTokenizer.fit(["abcd"])
    model = TransformerLanguageModel(config.model)
    model.train()
    records = []
    hook = model.register_forward_pre_hook(
        lambda module, inputs: records.append(
            (inputs[0].shape[1], torch.is_grad_enabled(), module.training)
        )
    )
    try:
        result = diagnostic_continuation(model, tokenizer, "abcdabcd", 5)
    finally:
        hook.remove()
    assert 1 <= len(result["generated_ids"]) <= 5
    assert "DIAGNOSTIC ONLY" in result["label"]
    assert all(length <= 4 and not grad and not training for length, grad, training in records)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="Unknown character"):
        diagnostic_continuation(model, tokenizer, "z")


def test_overfit_failure_is_reported_and_artifacts_are_preserved(training_fixture):
    root, config = training_fixture
    tokenizer = CharacterTokenizer.fit(["red blue.\n"])
    tokenizer.save(root / "overfit_tokenizer.json")
    config = replace(
        config,
        model=replace(config.model, vocab_size=tokenizer.vocab_size, dropout=0),
        tokenizer_path="overfit_tokenizer.json",
        tokenizer_fingerprint=tokenizer.fingerprint,
        microbatch_size=1,
        gradient_accumulation_steps=1,
        max_updates=2,
        warmup_updates=0,
    )
    output = root / "synthetic-overfit-check"
    result = run_overfit(config, root, output, success_loss=1e-10)
    assert result["status"] == "failed_to_overfit"
    assert result["updates"] == 2
    saved = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_overfit(config, root, output)
    assert (output / "summary.json").read_bytes() == saved


def test_summary_excludes_warmup_and_weights_by_tokens():
    def row(phase, tokens, seconds):
        return dict(
            phase=phase,
            cycle_tokens=tokens,
            boundary_seconds=seconds,
            update_seconds=seconds / 2,
            validation_seconds=0,
            checkpoint_seconds=0,
        )

    summary = summarize_rows(
        [row("warmup", 100, 50), row("measured", 10, 1), row("measured", 30, 2)]
    )
    assert summary["measured_iterations"] == 2
    assert summary["tokens_per_second"] == 40 / 3


def test_selection_requires_success_and_prefers_smaller_near_best():
    def case(batch, mode, rate=100, status="completed"):
        return dict(
            microbatch_size=batch,
            mode=mode,
            status=status,
            tokens_per_second=rate,
            measured_iterations=8,
        )

    cases = [case(b, "compute") for b in (1, 2, 4)]
    cases += [case(1, "pipeline", 50), case(2, "pipeline", 95), case(4, "pipeline", 100)]
    assert select_candidate(cases, 0.9)["microbatch_size"] == 2
    cases[4]["status"] = "memory_limit"
    assert select_candidate(cases, 0.9)["microbatch_size"] == 4
    assert select_candidate([], 0.9) is None


def test_runtime_projection_charges_proposed_cadence(training_fixture):
    _, config = training_fixture
    config = replace(config, max_updates=10, evaluation_interval=4, checkpoint_interval=3)
    case = dict(
        measured_iterations=4,
        mean_core_update_seconds=1,
        mean_boundary_update_seconds=2,
        validation_event_seconds=[2],
        checkpoint_event_seconds=[2],
        setup_to_ready_seconds=3,
    )
    estimate = estimate_runtime(case, config, [])
    assert estimate["proposed_validation_events"] == 3
    assert estimate["proposed_checkpoint_events"] == 6
    assert estimate["estimated_seconds_at_proposed_cadence"] == 31


def test_memory_names_and_resource_guards(training_fixture):
    _, config = training_fixture
    settings = BenchmarkConfig("artifacts/synthetic", config)
    memory = MemoryMonitor(torch.device("cpu")).snapshot()
    assert memory["mps_driver_allocated_bytes"] is None
    assert memory["process_lifetime_peak_rss_bytes"] > 0
    memory["process_lifetime_peak_rss_bytes"] = settings.max_process_rss_bytes + 1
    assert "RSS" in memory_limit_reason(memory, settings)
    memory["process_lifetime_peak_rss_bytes"] = 1
    memory["maximum_sampled_mps_driver_bytes"] = settings.max_mps_driver_bytes + 1
    assert "driver" in memory_limit_reason(memory, settings)
    memory["maximum_sampled_mps_driver_bytes"] = None
    memory["system_memory_pressure_free_percent"] = 5
    assert "headroom" in memory_limit_reason(memory, settings)


@pytest.mark.parametrize("mode", ["compute", "pipeline"])
def test_bounded_worker_on_offline_fixture(training_fixture, mode):
    root, config = training_fixture
    settings = BenchmarkConfig(
        "artifacts/benchmark",
        config,
        microbatch_sizes=(1,),
        effective_batch_size=2,
        warmup_updates=1,
        measured_updates=4,
        validation_windows=2,
        case_max_seconds=30,
        minimum_system_free_percent=0,
    )
    result = run_case(settings, root, 1, mode)
    assert result["status"] == "completed"
    assert result["measured_iterations"] == 4
    assert result["measured_tokens"] == 4 * 2 * 4
    assert result["tokens_per_second"] > 0
    raw = (root / settings.output_dir / f"{mode}-mb1/iterations.jsonl").read_text()
    assert len(raw.splitlines()) == 5
    with pytest.raises(FileExistsError):
        run_case(settings, root, 1, mode)
    assert (root / settings.output_dir / f"{mode}-mb1/iterations.jsonl").read_text() == raw


def test_compute_worker_checks_expected_token_artifacts(training_fixture):
    root, config = training_fixture
    settings = BenchmarkConfig(
        "artifacts/benchmark", replace(config, token_data_fingerprint="0" * 64)
    )
    with pytest.raises(ValueError, match="token-data fingerprint"):
        run_case(settings, root, 1, "compute")


def test_fresh_process_benchmark_preserves_results_and_never_launches_proposal(training_fixture):
    root, config = training_fixture
    settings = BenchmarkConfig(
        "artifacts/coordinator",
        config,
        microbatch_sizes=(1,),
        effective_batch_size=2,
        warmup_updates=1,
        measured_updates=4,
        sustained_updates=8,
        validation_windows=2,
        case_max_seconds=30,
        sustained_max_seconds=30,
        minimum_system_free_percent=0,
    )
    result = run_benchmark(settings, root)
    assert result["status"] == "completed"
    assert result["selected_microbatch_size"] == 1
    assert result["sustained"]["measured_iterations"] == 8
    assert all(
        case["subprocess_wall_seconds"] >= case["case_wall_seconds"] for case in result["cases"]
    )
    assert (root / settings.output_dir / "proposed_training.json").is_file()
    assert not (root / "runs/phase7-proposed-model-a-unexecuted").exists()
    with pytest.raises(FileExistsError):
        run_benchmark(settings, root)
