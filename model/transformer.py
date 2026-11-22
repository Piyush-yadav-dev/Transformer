# Layer normalization
"""Trainable normalization over the final embedding dimension."""

import torch
from torch import nn

from transformer.configuration.settings import TransformerConfig


class LayerNormalization(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.epsilon = config.layer_norm_epsilon
        self.weight = nn.Parameter(torch.ones(config.embedding_dim))
        self.bias = nn.Parameter(torch.zeros(config.embedding_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Chhoti precision mein mean/variance float32 mein nikalo, taaki rounding kam ho.
        values = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
        mean = values.mean(dim=-1, keepdim=True)
        variance = (values - mean).square().mean(dim=-1, keepdim=True)
        normalized = (values - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized.to(hidden.dtype) * self.weight + self.bias


# Token position embeddings
"""Token lookup plus learned absolute positions, followed by embedding dropout."""

import torch
from torch import nn

from transformer.configuration.settings import TransformerConfig


class TokenPositionEmbeddings(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.dropout(self.token(token_ids) + self.position(positions).unsqueeze(0))


# Causal self attention
"""Explicit projected, scaled, masked multi-head self-attention."""

import math

import torch
from torch import nn

from transformer.configuration.settings import TransformerConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "future_mask",
            torch.ones(config.context_length, config.context_length, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, embedding = hidden.shape

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            # [B,T,E] ko [B,H,T,D] banao; har head apne D features par attention karega.
            return projected.reshape(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.query(hidden))
        keys = split_heads(self.key(hidden))
        values = split_heads(self.value(hidden))
        # sqrt(D) se scale karne par bade dot products softmax ko jaldi saturate nahi karte.
        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Future tokens ko softmax se pehle mask karo, warna next-token answer leak hoga.
        scores = scores.masked_fill(self.future_mask[:sequence, :sequence], float("-inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ values
        # [B,H,T,D] se [B,T,E] lautao; transpose ke baad contiguous layout zaroori hai.
        combined = attended.transpose(1, 2).contiguous().view(batch, sequence, embedding)
        return self.output_dropout(self.output_projection(combined))


# Feed forward
"""Position-wise expansion, GELU, contraction, and residual-branch dropout."""

import torch
from torch import nn

from transformer.configuration.settings import TransformerConfig


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        width = config.embedding_dim * config.feed_forward_expansion
        self.expansion = nn.Linear(config.embedding_dim, width)
        self.activation = nn.GELU(approximate="none")
        self.contraction = nn.Linear(width, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.contraction(self.activation(self.expansion(hidden))))


# Transformer block
"""Pre-normalization attention and feed-forward residual branches."""

import torch
from torch import nn

from transformer.configuration.settings import TransformerConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = LayerNormalization(config)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = LayerNormalization(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Residual addition ke dono tensors [B,T,E] hone chahiye; projections E wapas laati hain.
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


# Language model
"""Autoregressive Transformer logits and optional loss against already-shifted targets."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from transformer.configuration.settings import TransformerConfig


@dataclass(frozen=True)
class TransformerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if not isinstance(config, TransformerConfig):
            raise ValueError("config must be a TransformerConfig object.")
        self.config = config
        self.embeddings = TokenPositionEmbeddings(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = LayerNormalization(config)
        self.vocabulary_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.vocabulary_projection.weight = self.embeddings.token.weight
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        initialized: set[int] = set()
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Tied weight ek hi Parameter hai; final initialization use sirf ek baar mile.
                if id(module.weight) not in initialized:
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initialization_std)
                    initialized.add(id(module.weight))
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNormalization):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def trainable_parameter_count(self) -> int:
        # parameters() shared weight ko ek baar ginta hai; tying double-count nahi hogi.
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_ids(self, value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a tensor shaped [batch, sequence].")
        if value.dtype != torch.int64:
            raise ValueError(
                f"{name} must have dtype torch.int64, as supplied by the token loader."
            )
        if value.device != self.embeddings.token.weight.device:
            raise ValueError(f"{name} and model parameters must be on the same device.")
        if value.numel() == 0:
            raise ValueError(f"{name} must have non-empty batch and sequence dimensions.")
        if bool(((value < 0) | (value >= self.config.vocab_size)).any()):
            raise ValueError(f"{name} must contain IDs in [0, {self.config.vocab_size - 1}].")

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> TransformerOutput:
        self._validate_ids(token_ids, "token_ids")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"Sequence length {token_ids.shape[1]} exceeds "
                f"context_length={self.config.context_length}."
            )
        if targets is not None:
            self._validate_ids(targets, "targets")
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same [batch, sequence] shape as token_ids.")
        hidden = self.embeddings(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.vocabulary_projection(self.final_norm(hidden))
        loss = None
        if targets is not None:
            # Loader targets pehle hi ek token aage hain; yahan dobara shift mat karo.
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return TransformerOutput(logits=logits, loss=loss)
