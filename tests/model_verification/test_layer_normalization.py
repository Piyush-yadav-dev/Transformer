"""Compare custom normalization and its derivatives with PyTorch's reference primitive."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from transformer.model.transformer import LayerNormalization


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("epsilon", [1e-5, 0.2])
@pytest.mark.parametrize("width", [1, 4])
def test_normalization_values_epsilon_affine_and_gradients(small_config, dtype, epsilon, width):
    config = replace(small_config, embedding_dim=width, num_heads=1, layer_norm_epsilon=epsilon)
    layer = LayerNormalization(config).to(dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-1.4, 2.3, width, dtype=dtype))
        layer.bias.copy_(torch.linspace(0.7, -0.9, width, dtype=dtype))
    values = torch.randn(2, 3, width, dtype=dtype) * 2.7 + 7.3
    values[0, 0] = 3.25
    values[0, 1] = 5.4 + torch.linspace(-1e-4, 1e-4, width, dtype=dtype)
    inputs = values.clone().requires_grad_()
    reference_inputs = values.clone().requires_grad_()
    reference_scale = layer.weight.detach().clone().requires_grad_()
    reference_bias = layer.bias.detach().clone().requires_grad_()
    actual = layer(inputs)
    expected = F.layer_norm(reference_inputs, (width,), reference_scale, reference_bias, epsilon)
    tolerance = (
        {"rtol": 3e-5, "atol": 2e-5}
        if dtype == torch.float32
        else {
            "rtol": 1e-9,
            "atol": 1e-10,
        }
    )
    torch.testing.assert_close(actual, expected, **tolerance)
    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (inputs, layer.weight, layer.bias), probe)
    expected_gradients = torch.autograd.grad(
        expected, (reference_inputs, reference_scale, reference_bias), probe
    )
    gradient_tolerance = {"rtol": 3e-5, "atol": 2e-4} if dtype == torch.float32 else tolerance
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, **gradient_tolerance)
    if width > 1:
        wrong_epsilon = F.layer_norm(
            reference_inputs, (width,), reference_scale, reference_bias, epsilon * 10
        )
        assert (expected - wrong_epsilon).abs().max().item() > 1e-3
