"""An independent scalar/prefix NumPy reference for the vectorized PyTorch attention."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from transformer.model.transformer import CausalSelfAttention


def prefix_attention_reference(module, hidden, *, scale=True, causal=True, reverse_heads=False):
    """Visit visible token prefixes without tensor head reshaping or a triangular mask."""
    array = hidden.detach().numpy()
    parameters = {name: value.detach().numpy() for name, value in module.named_parameters()}
    batch, length, embedding = array.shape
    width = embedding // module.num_heads
    output = np.empty_like(array)

    def project(name, row, start):
        rows = slice(start, start + width)
        return parameters[f"{name}.weight"][rows] @ row + parameters[f"{name}.bias"][rows]

    for sample in range(batch):
        for position in range(length):
            heads = []
            for start in range(0, embedding, width):
                query = project("query", array[sample, position], start)
                visible = range(position + 1 if causal else length)
                scores = np.array(
                    [
                        np.dot(query, project("key", array[sample, earlier], start))
                        for earlier in visible
                    ]
                )
                scores /= math.sqrt(width) if scale else 1
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                value_sum = sum(
                    probability * project("value", array[sample, earlier], start)
                    for earlier, probability in zip(visible, probabilities, strict=True)
                )
                heads.append(value_sum)
            joined = np.concatenate(heads[::-1] if reverse_heads else heads)
            output[sample, position] = (
                parameters["output_projection.weight"] @ joined
                + parameters["output_projection.bias"]
            )
    return torch.from_numpy(output)


@pytest.mark.parametrize("heads", [1, 2, 3])
@pytest.mark.parametrize("length", [1, 5])
def test_attention_matches_independent_prefix_reference(small_config, heads, length):
    config = replace(small_config, embedding_dim=6, num_heads=heads, context_length=5)
    attention = CausalSelfAttention(config).double().eval()
    hidden = torch.randn(2, length, 6, dtype=torch.float64) * 1.7 + 0.4
    expected = prefix_attention_reference(attention, hidden)
    with torch.no_grad():
        actual = attention(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_reference_fixture_detects_missing_scale_mask_and_reversed_heads(small_config):
    attention = CausalSelfAttention(replace(small_config, embedding_dim=6)).double().eval()
    hidden = torch.randn(2, 5, 6, dtype=torch.float64) * 1.7 + 0.4
    correct = prefix_attention_reference(attention, hidden)
    for change in ({"scale": False}, {"causal": False}, {"reverse_heads": True}):
        incorrect = prefix_attention_reference(attention, hidden, **change)
        assert (correct - incorrect).abs().max().item() > 1e-4, change
