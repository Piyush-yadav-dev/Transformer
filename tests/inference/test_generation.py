"""Sampling mathematics, prompt/token policy, rolling positions and actual device behavior."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from transformer.configuration.settings import MAX_PROMPT_CHARACTERS
from transformer.inference import GenerationConfig, GenerationService
from transformer.inference.service import select_next_token


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "other"},
        {"max_new_tokens": -1},
        {"max_new_tokens": 1025},
        {"max_new_tokens": True},
        {"temperature": 1},
        {"seed": 1},
        {"top_k": 1},
        {"unknown_policy": "ignore"},
        {"mode": "sample"},
        {"mode": "sample", "temperature": 0},
        {"mode": "sample", "temperature": -1},
        {"mode": "sample", "temperature": float("inf")},
        {"mode": "sample", "temperature": float("nan")},
        {"mode": "sample", "temperature": True},
        {"mode": "sample", "temperature": 1, "top_k": 0},
        {"mode": "sample", "temperature": 1, "top_k": 1.5},
        {"mode": "sample", "temperature": 1, "seed": -1},
        {"mode": "sample", "temperature": 1, "seed": 2**63},
    ],
)
def test_invalid_sampling_parameters(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_temperature_distribution_and_exact_top_k_ties(monkeypatch):
    generator = torch.Generator().manual_seed(42)
    captured = []

    def draw(probabilities, *args, **kwargs):
        captured.append(probabilities.clone())
        return torch.tensor([0])

    monkeypatch.setattr(torch, "multinomial", draw)
    settings = GenerationConfig(mode="sample", temperature=2, top_k=2)
    token = select_next_token(
        torch.tensor([0.0, 1.3862943611198906, 0.6931471805599453], dtype=torch.float64),
        settings,
        generator,
    )
    assert token == 1
    assert captured[-1].tolist() == pytest.approx([2 / (2 + 2**0.5), 2**0.5 / (2 + 2**0.5)])
    assert select_next_token(torch.zeros(5), settings, generator) == 0
    assert len(captured[-1]) == 2
    assert captured[-1].tolist() == [0.5, 0.5]


def test_sampling_reproducibility_and_top_one_greedy_equivalence():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    settings = GenerationConfig(mode="sample", temperature=1, top_k=3, seed=1)

    def sequence(seed):
        rng = torch.Generator().manual_seed(seed)
        return [select_next_token(logits, settings, rng) for _ in range(40)]

    assert sequence(1) == sequence(1)
    assert sequence(1) != sequence(2)
    assert set(sequence(1)) <= {0, 1, 2}
    assert select_next_token(logits, GenerationConfig()) == select_next_token(
        logits,
        GenerationConfig(mode="sample", temperature=0.5, top_k=1),
        torch.Generator(),
    )
    with pytest.raises(ValueError, match="finite"):
        select_next_token(torch.tensor([float("nan")]), GenerationConfig())
    with pytest.raises(ValueError, match="vocabulary"):
        select_next_token(
            logits, GenerationConfig(mode="sample", temperature=1, top_k=5), torch.Generator()
        )


@pytest.mark.parametrize("prompt", ["", None, "a" * (MAX_PROMPT_CHARACTERS + 1), "\ud800"])
def test_invalid_prompt_fails_before_loading(inference_fixture, prompt):
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="Prompt"):
            service.generate("fixture-baseline-v1", prompt)
        assert service.cache_info()["cached_models"] == 0


def test_unknown_prompt_and_zero_output(inference_fixture):
    with GenerationService(inference_fixture, device="cpu") as service:
        with pytest.raises(ValueError, match="Unknown character"):
            service.generate("fixture-baseline-v1", "a🙂")
        called = []
        service._loaded.model.register_forward_hook(lambda *args: called.append(True))
        result = service.generate(
            "fixture-baseline-v1",
            "a🙂",
            GenerationConfig(max_new_tokens=0, unknown_policy="replace"),
        )
        assert result["continuation"] == "" and result["generated_token_count"] == 0
        assert result["unknown_prompt_tokens"] == 1 and result["stop_reason"] == "max_new_tokens"
        assert not called
        assert result["cache_hit"] and result["model_load_seconds"] == 0


def test_rolling_context_restarts_learned_positions(inference_fixture, monkeypatch):
    monkeypatch.setattr("transformer.inference.service.select_next_token", lambda *a: 2)
    with GenerationService(inference_fixture, device="cpu") as service:
        service.load("fixture-baseline-v1")
        inputs, positions = [], []
        service._loaded.model.register_forward_pre_hook(
            lambda _, args: inputs.append(args[0].tolist()[0])
        )
        service._loaded.model.embeddings.position.register_forward_pre_hook(
            lambda _, args: positions.append(args[0].tolist())
        )
        result = service.generate(
            "fixture-baseline-v1", "abcdab", GenerationConfig(max_new_tokens=3)
        )
        assert inputs == [[4, 5, 2, 3], [5, 2, 3, 2], [2, 3, 2, 2]]
        assert positions == [[0, 1, 2, 3]] * 3
        assert result["initial_prompt_tokens_truncated"] == 2
        inputs.clear()
        positions.clear()
        service.generate("fixture-baseline-v1", "b", GenerationConfig(max_new_tokens=3))
        assert positions == [[0], [0, 1], [0, 1, 2]]


@pytest.mark.parametrize(
    "token,expected,count,reason", [(1, "", 1, "eos"), (0, "<unk>" * 3, 3, "max_new_tokens")]
)
def test_immediate_eos_and_visible_unknown(
    inference_fixture, monkeypatch, token, expected, count, reason
):
    monkeypatch.setattr(
        "transformer.inference.service.select_next_token", lambda *a: token
    )
    with GenerationService(inference_fixture, device="cpu") as service:
        result = service.generate("fixture-baseline-v1", "a", GenerationConfig(max_new_tokens=3))
        assert result["continuation"] == expected
        assert result["generated_token_count"] == count and result["stop_reason"] == reason
        assert result["generated_token_ids"] == [token] * count


def test_repeated_and_concurrent_generation_are_isolated(inference_fixture):
    state = torch.get_rng_state().clone()
    with GenerationService(inference_fixture, device="cpu") as service:
        settings = GenerationConfig(
            mode="sample", temperature=1, top_k=4, seed=1337, max_new_tokens=16
        )
        first = service.generate("fixture-baseline-v1", "abcd", settings)
        second = service.generate("fixture-baseline-v1", "abcd", settings)
        assert first["generated_token_ids"] == second["generated_token_ids"]
        assert all(p.grad is None for p in service._loaded.model.parameters())
        assert not service._loaded.model.training
        greedy = service.generate("fixture-baseline-v1", "abcd")
        assert (
            greedy["generated_token_ids"]
            == service.generate("fixture-baseline-v1", "abcd")["generated_token_ids"]
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(
                pool.map(
                    lambda _: service.generate("fixture-baseline-v1", "abcd", settings), range(2)
                )
            )
        assert all(row["generated_token_ids"] == first["generated_token_ids"] for row in outputs)
        unseeded = service.generate(
            "fixture-baseline-v1",
            "abcd",
            GenerationConfig(mode="sample", temperature=1, max_new_tokens=16),
        )
        effective = unseeded["sampling_settings"]["effective_seed"]
        replay = service.generate(
            "fixture-baseline-v1",
            "abcd",
            GenerationConfig(mode="sample", temperature=1, seed=effective, max_new_tokens=16),
        )
        assert replay["generated_token_ids"] == unseeded["generated_token_ids"]
    assert torch.equal(state, torch.get_rng_state())


def test_maximum_output_limit_and_inference_mode(inference_fixture, monkeypatch):
    monkeypatch.setattr(
        "transformer.inference.service.select_next_token", lambda *args: 2
    )
    with GenerationService(inference_fixture, device="cpu") as service:
        service.load("fixture-baseline-v1")
        states = []

        def observe(_module, _inputs, output):
            states.append(
                (
                    torch.is_grad_enabled(),
                    output.logits.requires_grad,
                    torch.is_inference_mode_enabled(),
                )
            )

        service._loaded.model.register_forward_hook(observe)
        result = service.generate(
            "fixture-baseline-v1", "abcd", GenerationConfig(max_new_tokens=1024)
        )
        assert result["generated_token_count"] == 1024
        assert result["continuation"] == "a" * 1024
        assert states == [(False, False, True)] * 1024
        assert all(p.grad is None for p in service._loaded.model.parameters())


@pytest.mark.mps
def test_real_mps_sampling_and_switching(inference_fixture):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rng = torch.mps.get_rng_state().clone()
    with GenerationService(inference_fixture, device="mps") as service:
        settings = GenerationConfig(mode="sample", temperature=1, seed=42, max_new_tokens=8)
        first = service.generate("fixture-baseline-v1", "abcd", settings)
        assert (
            service.generate("fixture-baseline-v1", "abcd", settings)["generated_token_ids"]
            == first["generated_token_ids"]
        )
        service.generate("fixture-combined-v1", "abcd", settings)
        assert service.cache_info()["cached_models"] == 1
        assert all(p.grad is None for p in service._loaded.model.parameters())
    assert torch.equal(rng, torch.mps.get_rng_state())
