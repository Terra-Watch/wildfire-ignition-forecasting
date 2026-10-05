"""Attention building blocks: a self-attention block and a query-only cross-attention block. Names of the sub-modules are in full words."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class SelfAttentionBlock(nn.Module):
    """Pre-normalised self-attention followed by a feed-forward layer, both with residual connections. Entries marked as padding are never attended to."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm_attention, self.norm_feed_forward = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv_projection, self.output_projection = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, length, width); padding: (batch, length), True where the entry is empty."""
        batch, length, width = tokens.shape
        normed = self.norm_attention(tokens)
        query, key, value = self.qkv_projection(normed).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        mask = torch.zeros(batch, self.heads, 1, length, device=tokens.device).masked_fill(padding[:, None, None, :], float("-inf")).expand(batch, self.heads, length, length)
        attended = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.dropout(self.output_projection(attended))
        return tokens + self.dropout(self.feed_forward(self.norm_feed_forward(tokens)))


class CrossAttentionBlock(nn.Module):
    """One query token attends over all stream tokens (keys and values come from the tokens only), then a feed-forward layer, both residual.
    The query is rescaled and shifted by a learned function of a conditioning vector (static summary and calendar values); the conditioning starts at zero effect."""

    def __init__(self, width: int, heads: int, dropout: float, condition_size: int):
        super().__init__()
        self.heads = heads
        self.norm_query, self.norm_tokens, self.norm_feed_forward = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.query_projection, self.key_value_projection, self.output_projection = nn.Linear(width, width), nn.Linear(width, 2 * width), nn.Linear(width, width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)
        self.condition_projection = nn.Linear(condition_size, 2 * width)
        nn.init.zeros_(self.condition_projection.weight), nn.init.zeros_(self.condition_projection.bias)

    def forward(self, query_state: torch.Tensor, tokens: torch.Tensor, padding: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """query_state: (batch, 1, width); tokens: (batch, length, width); padding: (batch, length); condition: (batch, condition_size)."""
        batch, length, width = tokens.shape
        scale, shift = self.condition_projection(condition).chunk(2, -1)
        normed = self.norm_query(query_state) * (1 + scale[:, None]) + shift[:, None]
        query = self.query_projection(normed).view(batch, 1, self.heads, width // self.heads).transpose(1, 2)
        key, value = self.key_value_projection(self.norm_tokens(tokens)).view(batch, length, 2, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        mask = torch.zeros(batch, self.heads, 1, length, device=tokens.device).masked_fill(padding[:, None, None, :], float("-inf"))
        attended = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask).transpose(1, 2).reshape(batch, 1, width)
        query_state = query_state + self.dropout(self.output_projection(attended))
        return query_state + self.dropout(self.feed_forward(self.norm_feed_forward(query_state)))
