"""BiLSTM and Transformer encoders for token-level BIO prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMChunker(nn.Module):
    """Predict BIO labels with embeddings and a bidirectional LSTM."""

    def __init__(
        self,
        vocabulary_size: int,
        label_count: int,
        embedding_size: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.3,
        padding_index: int = 0,
    ) -> None:
        """Build the embedding, recurrent encoder, and label projection."""
        super().__init__()
        if min(vocabulary_size, label_count, embedding_size, hidden_size) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_index)
        self.encoder = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, label_count)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return token logits without letting padding alter recurrent states."""
        embeddings = self.embedding(tokens)
        if mask is None:
            encoded, _ = self.encoder(embeddings)
        else:
            lengths = mask.sum(dim=1).to(dtype=torch.long, device="cpu")
            if torch.any(lengths < 1):
                raise ValueError("every sequence must contain at least one valid token")
            packed = pack_padded_sequence(
                embeddings,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_encoded, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(
                packed_encoded,
                batch_first=True,
                total_length=tokens.shape[1],
            )
        return self.classifier(self.dropout(encoded))


class SinusoidalPositionEncoding(nn.Module):
    """Add deterministic sine and cosine position signals."""

    def __init__(self, width: int, maximum_length: int = 512) -> None:
        """Precompute position signals up to the supported sequence length."""
        super().__init__()
        if width < 1 or maximum_length < 1:
            raise ValueError("position width and maximum length must be positive")
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
        )
        encoding = torch.zeros(maximum_length, width)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add position signals matching the input sequence length."""
        if embeddings.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence exceeds configured maximum length")
        return embeddings + self.encoding[:, : embeddings.shape[1]]


class TransformerChunker(nn.Module):
    """Predict BIO labels with a compact Transformer encoder."""

    def __init__(
        self,
        vocabulary_size: int,
        label_count: int,
        embedding_size: int = 128,
        attention_heads: int = 4,
        layer_count: int = 2,
        feedforward_size: int = 256,
        dropout: float = 0.2,
        padding_index: int = 0,
    ) -> None:
        """Build embeddings, positional signals, encoder layers, and classifier."""
        super().__init__()
        if min(vocabulary_size, label_count, embedding_size, attention_heads, layer_count) < 1:
            raise ValueError("model dimensions, heads, and layers must be positive")
        if feedforward_size < 1:
            raise ValueError("feedforward_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if embedding_size % attention_heads:
            raise ValueError("embedding_size must be divisible by attention_heads")
        self.embedding_scale = math.sqrt(embedding_size)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_index)
        self.positions = SinusoidalPositionEncoding(embedding_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=attention_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layer_count,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(embedding_size, label_count)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return token label logits while excluding padding from self-attention."""
        embeddings = self.positions(self.embedding(tokens) * self.embedding_scale)
        padding_mask = None if mask is None else ~mask
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        return self.classifier(encoded)
