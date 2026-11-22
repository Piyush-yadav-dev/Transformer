"""Causal prediction, aligned loss, modes, shapes, serialization, and actual MPS checks."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from transformer.configuration.settings import TransformerConfig
from transformer.model import TransformerLanguageModel


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("position", [0, 3, 6])
def test_future_changes_cannot_change_prefix_logits(small_config, tied, position):
    model = TransformerLanguageModel(replace(small_config, tie_embeddings=tied)).double().eval()
    assert torch.are_deterministic_algorithms_enabled()
    assert all(not layer.training for layer in model.modules() if isinstance(layer, nn.Dropout))
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [8, 7, 6, 5, 4, 3, 2, 1]])
    changed = tokens.clone()
    changed[:, position + 1 :] = (changed[:, position + 1 :] + 3) % small_config.vocab_size
    with torch.no_grad():
        before = model(tokens).logits
        after = model(changed).logits
        truncated = model(tokens[:, : position + 1]).logits
    torch.testing.assert_close(
        before[:, : position + 1], after[:, : position + 1], rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(before[:, : position + 1], truncated, rtol=1e-10, atol=1e-12)
    assert (before[:, position + 1 :] - after[:, position + 1 :]).abs().max().item() > 1e-6


def test_causality_fixture_detects_deliberately_disabled_mask(small_config):
    model = TransformerLanguageModel(small_config).double().eval()
    for block in model.blocks:
        block.attention.future_mask.zero_()
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 3) % small_config.vocab_size
    with torch.no_grad():
        leaked_difference = (model(tokens).logits[:, :4] - model(changed).logits[:, :4]).abs().max()
    assert leaked_difference.item() > 1e-6


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("length", [1, 3, 8])
def test_shapes_context_boundary_and_batch_independence(small_config, batch, length):
    model = TransformerLanguageModel(small_config).double().eval()
    tokens = torch.randint(small_config.vocab_size, (batch, length))
    with torch.no_grad():
        combined = model(tokens).logits
        separately = torch.cat([model(row.unsqueeze(0)).logits for row in tokens])
    assert combined.shape == (batch, length, small_config.vocab_size)
    assert torch.isfinite(combined).all()
    torch.testing.assert_close(combined, separately, rtol=1e-10, atol=1e-12)


def test_primary_configuration_at_full_context_capacity():
    config = TransformerConfig.load(
        Path(__file__).resolve().parents[2] / "configs/model/primary.json"
    )
    model = TransformerLanguageModel(config).eval()
    tokens = torch.arange(config.context_length).remainder(config.vocab_size).unsqueeze(0)
    with torch.no_grad():
        output = model(tokens)
    assert output.logits.shape == (1, 256, 78)
    assert torch.isfinite(output.logits).all()


def test_loss_uses_every_already_shifted_target_once(small_config):
    model = TransformerLanguageModel(small_config).double().eval()
    stream = torch.tensor([[2, 4, 1, 0, 7, 9], [3, 5, 8, 6, 1, 2]])
    inputs, targets = stream[:, :-1], stream[:, 1:]
    assert not targets.is_contiguous()
    output = model(inputs, targets)
    logits = output.logits
    per_position = logits.logsumexp(dim=-1) - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    direct = F.cross_entropy(logits.transpose(1, 2), targets)
    torch.testing.assert_close(output.loss, per_position.mean(), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(output.loss, direct, rtol=1e-10, atol=1e-12)
    wrong_shift = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets[:, 1:])
    assert (output.loss - wrong_shift).abs().item() > 1e-5
    actual_gradient = torch.autograd.grad(output.loss, logits)[0]
    expected_gradient = (
        logits.softmax(-1) - F.one_hot(targets, small_config.vocab_size)
    ) / targets.numel()
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)


def test_dropout_changes_training_but_evaluation_is_stable(small_config):
    model = TransformerLanguageModel(replace(small_config, dropout=0.5)).double()
    tokens = torch.tensor([[2, 4, 6, 8, 10, 12, 1, 3], [3, 1, 12, 10, 8, 6, 4, 2]])
    model.train()
    with torch.no_grad():
        torch.manual_seed(91)
        first = model(tokens).logits
        torch.manual_seed(92)
        second = model(tokens).logits
        torch.manual_seed(91)
        repeated_seed = model(tokens).logits
    assert not torch.equal(first, second)
    torch.testing.assert_close(first, repeated_seed, rtol=0, atol=0)
    model.eval()
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first_eval, second_eval = model(tokens).logits, model(tokens).logits
    torch.testing.assert_close(first_eval, second_eval, rtol=0, atol=0)
    assert torch.equal(rng_before, torch.random.get_rng_state())


@pytest.mark.parametrize("tied", [True, False])
def test_config_and_weights_reconstruct_equivalent_model(small_config, tied, tmp_path):
    config = replace(small_config, tie_embeddings=tied)
    original = TransformerLanguageModel(config).double().eval()
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [8, 7, 6, 5, 4, 3, 2, 1]])
    with torch.no_grad():
        expected = original(tokens).logits
    config_path, weights_path = tmp_path / "configuration.json", tmp_path / "synthetic-weights.pt"
    config.save(config_path)
    with weights_path.open("xb") as destination:
        torch.save(original.state_dict(), destination)
    restored = TransformerLanguageModel(TransformerConfig.load(config_path)).double().eval()
    with torch.no_grad():
        assert not torch.allclose(expected, restored(tokens).logits)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert not any("future_mask" in name for name in weights)
    restored.load_state_dict(weights, strict=True)
    assert restored.config == original.config
    assert (restored.embeddings.token.weight is restored.vocabulary_projection.weight) is tied
    assert restored.trainable_parameter_count() == original.trainable_parameter_count()
    for block in restored.blocks:
        assert block.attention.future_mask[0, -1] and not block.attention.future_mask[-1, 0]
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_built() or not torch.backends.mps.is_available(),
    reason="MPS unavailable: real architecture/gradient comparisons remain pending.",
)
def test_real_mps_causality_forward_backward_and_cpu_agreement(small_config):
    config = replace(small_config, dropout=0.0)
    cpu = TransformerLanguageModel(config).eval()
    mps = TransformerLanguageModel(config).eval()
    mps.load_state_dict(cpu.state_dict(), strict=True)
    mps = mps.to("mps")
    assert mps.embeddings.token.weight is mps.vocabulary_projection.weight
    assert all(block.attention.future_mask.device.type == "mps" for block in mps.blocks)
    stream = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 6, 9, 12, 2, 5, 8, 11, 1]])
    inputs, targets = stream[:, :-1], stream[:, 1:]
    cpu_output = cpu(inputs, targets)
    mps_output = mps(inputs.to("mps"), targets.to("mps"))
    torch.testing.assert_close(mps_output.logits.cpu(), cpu_output.logits, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(mps_output.loss.cpu(), cpu_output.loss, rtol=3e-4, atol=3e-5)
    cpu_output.loss.backward()
    mps_output.loss.backward()
    for name, parameter in mps.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert cpu.get_parameter(name).grad is not None, name
        torch.testing.assert_close(
            parameter.grad.cpu(), cpu.get_parameter(name).grad, rtol=1e-3, atol=3e-5, msg=name
        )
    changed = inputs.clone()
    changed[:, 4:] = (changed[:, 4:] + 3) % config.vocab_size
    with torch.no_grad():
        after = mps(changed.to("mps")).logits.cpu()
        repeated = mps(inputs.to("mps")).logits.cpu()
    before = mps_output.logits.detach().cpu()
    torch.testing.assert_close(before[:, :4], after[:, :4], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(before, repeated, rtol=3e-4, atol=3e-5)
    assert (before[:, 4:] - after[:, 4:]).abs().max().item() > 1e-4
