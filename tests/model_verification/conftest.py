"""Deterministic, synthetic-only fixtures scoped to architecture verification."""

import pytest
import torch

from transformer.configuration.settings import TransformerConfig


@pytest.fixture(autouse=True)
def seeded_execution(request):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        if request.node.get_closest_marker("mps") is None:
            torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture
def small_config():
    return TransformerConfig(
        vocab_size=13,
        num_layers=2,
        num_heads=2,
        embedding_dim=8,
        context_length=8,
        feed_forward_expansion=2,
        dropout=0.35,
        initialization_std=0.07,
    )
