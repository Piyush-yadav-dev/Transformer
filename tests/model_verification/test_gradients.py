"""Bounded double-precision numerical checks and tied-weight gradient accounting."""

from dataclasses import replace

import pytest
import torch

from transformer.model import TransformerLanguageModel
from transformer.model.transformer import CausalSelfAttention
from transformer.model.transformer import FeedForward
from transformer.model.transformer import LayerNormalization


@pytest.mark.parametrize("component", [CausalSelfAttention, LayerNormalization, FeedForward])
def test_small_component_inputs_and_parameters_pass_gradcheck(small_config, component):
    config = replace(small_config, embedding_dim=4, num_heads=2, context_length=3)
    module = component(config).double().eval()
    inputs = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    names, original = zip(*module.named_parameters(), strict=True)
    parameters = tuple(value.detach().clone().requires_grad_() for value in original)

    def evaluate(hidden, *weights):
        return torch.func.functional_call(module, dict(zip(names, weights, strict=True)), (hidden,))

    # Sirf chhote components par finite differences; poore model ka numerical Jacobian nahi.
    assert torch.autograd.gradcheck(
        evaluate, (inputs, *parameters), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=False
    )


def test_tied_gradient_equals_embedding_plus_output_contributions(small_config):
    tied = TransformerLanguageModel(small_config).double().eval()
    untied = TransformerLanguageModel(replace(small_config, tie_embeddings=False)).double().eval()
    untied.load_state_dict(tied.state_dict(), strict=True)
    stream = torch.tensor([[0, 1, 4, 7, 2, 6], [2, 6, 8, 9, 10, 3]])
    tied_result = tied(stream[:, :-1], stream[:, 1:])
    untied_result = untied(stream[:, :-1], stream[:, 1:])
    torch.testing.assert_close(tied_result.logits, untied_result.logits, rtol=1e-10, atol=1e-12)
    tied_result.loss.backward()
    untied_result.loss.backward()
    for model in (tied, untied):
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    expected = untied.embeddings.token.weight.grad + untied.vocabulary_projection.weight.grad
    torch.testing.assert_close(tied.embeddings.token.weight.grad, expected, rtol=1e-9, atol=1e-11)
    for name, parameter in tied.named_parameters():
        if name != "embeddings.token.weight":
            torch.testing.assert_close(
                parameter.grad, untied.get_parameter(name).grad, rtol=1e-9, atol=1e-11
            )
