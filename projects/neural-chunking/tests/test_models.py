"""Shape and masking tests for both neural chunkers."""

from __future__ import annotations

import pytest
import torch

from neural_chunking.models import BiLSTMChunker, TransformerChunker
from neural_chunking.training import _snapshot_state


@pytest.mark.parametrize("model_type", [BiLSTMChunker, TransformerChunker])
def test_chunker_returns_one_logit_vector_per_token(model_type) -> None:
    """Both architectures should preserve batch and sequence dimensions."""
    model = model_type(vocabulary_size=20, label_count=5, embedding_size=16)
    tokens = torch.tensor([[2, 3, 4], [5, 6, 0]])
    mask = tokens.ne(0)
    assert model(tokens, mask).shape == (2, 3, 5)


def test_transformer_rejects_incompatible_attention_width() -> None:
    """The embedding width must divide evenly across attention heads."""
    with pytest.raises(ValueError, match="divisible"):
        TransformerChunker(20, 5, embedding_size=15, attention_heads=4)


def test_bilstm_valid_logits_do_not_depend_on_batch_padding() -> None:
    """Packing should make valid recurrent outputs invariant to neighbouring lengths."""
    torch.manual_seed(7)
    model = BiLSTMChunker(20, 5, embedding_size=8, hidden_size=8, dropout=0.0)
    model.eval()
    short_tokens = torch.tensor([[2, 3]])
    short_mask = torch.ones_like(short_tokens, dtype=torch.bool)
    padded_tokens = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]])
    padded_mask = padded_tokens.ne(0)
    with torch.inference_mode():
        alone = model(short_tokens, short_mask)[0]
        batched = model(padded_tokens, padded_mask)[0, :2]
    assert torch.allclose(alone, batched, atol=1e-6)


def test_selected_cpu_state_is_an_independent_snapshot() -> None:
    """Later parameter updates must not mutate the state selected by validation."""
    model = BiLSTMChunker(20, 5, embedding_size=8, hidden_size=8)
    selected = _snapshot_state(model)
    first_name, first_parameter = next(iter(model.named_parameters()))
    before = selected[first_name].clone()
    with torch.no_grad():
        first_parameter.add_(1.0)
    assert torch.equal(selected[first_name], before)
    assert not torch.equal(selected[first_name], first_parameter)
