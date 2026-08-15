"""Tests for BIO data handling and exact-span scoring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_chunking.data import (
    Sentence,
    Vocabulary,
    build_vocabularies,
    collate_sentences,
    read_sentences,
    split_by_content_hash,
)
from neural_chunking.metrics import Span, bio_spans, span_metrics


def test_reader_preserves_complete_sentences(tmp_path: Path) -> None:
    """Blank lines should delimit aligned token-label sequences."""
    path = tmp_path / "tiny.txt"
    path.write_text("New B-NP\nYork I-NP\n\nworks B-VP\n\n", encoding="utf-8")
    assert read_sentences(path) == [
        Sentence(("New", "York"), ("B-NP", "I-NP")),
        Sentence(("works",), ("B-VP",)),
    ]


def test_duplicate_tokens_always_share_a_split() -> None:
    """Content hashing should prevent exact duplicate sentences crossing partitions."""
    duplicate_a = Sentence(("Same", "tokens"), ("B-NP", "I-NP"))
    duplicate_b = Sentence(("Same", "tokens"), ("O", "O"))
    rows = [duplicate_a, duplicate_b]
    for index in range(200):
        rows.append(Sentence((f"token-{index}",), ("O",)))
    splits = split_by_content_hash(rows, seed=9)
    duplicate_locations = {
        name for name, values in splits.items() if duplicate_a in values or duplicate_b in values
    }
    assert len(duplicate_locations) == 1


def test_split_requires_a_validation_partition() -> None:
    """A zero validation fraction cannot support model selection."""
    rows = [Sentence((f"token-{index}",), ("O",)) for index in range(200)]
    with pytest.raises(ValueError, match="validation_fraction"):
        split_by_content_hash(rows, validation_fraction=0.0)


def test_vocabulary_rejects_reserved_values_in_observations() -> None:
    """Literal special markers must not be mistaken for padding or unknown tokens."""
    with pytest.raises(ValueError, match="reserved vocabulary"):
        Vocabulary(["word", "<PAD>"], include_unknown=True)


def test_bio_spans_repairs_invalid_continuation() -> None:
    """An unmatched continuation should start a new span instead of disappearing."""
    spans = bio_spans(["I-NP", "I-NP", "O", "B-VP"])
    assert spans == {Span("NP", 0, 2), Span("VP", 3, 4)}


def test_exact_span_metric_requires_matching_boundaries() -> None:
    """Token overlap alone must not count as an exact chunk match."""
    metrics = span_metrics(
        [["B-NP", "I-NP", "O", "B-VP"]],
        [["B-NP", "O", "O", "B-VP"]],
    )
    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.f1 == pytest.approx(0.5)


@pytest.mark.parametrize("tag", ["BAD", "B-", "I-"])
def test_bio_spans_rejects_malformed_tags(tag: str) -> None:
    """Malformed values should raise a clear metric-contract error."""
    with pytest.raises(ValueError, match="invalid BIO tag"):
        bio_spans([tag])


def test_label_padding_is_not_a_prediction_class() -> None:
    """Padding targets should use the loss sentinel rather than a learned label."""
    sentences = [
        Sentence(("one", "two"), ("B-NP", "I-NP")),
        Sentence(("short",), ("O",)),
    ]
    token_vocabulary, label_vocabulary = build_vocabularies(sentences)
    assert token_vocabulary.pad_index == 0
    assert label_vocabulary.pad_index is None
    assert "<PAD>" not in label_vocabulary.value_to_index
    examples = [
        (
            torch.tensor(token_vocabulary.encode(sentence.tokens)),
            torch.tensor(label_vocabulary.encode(sentence.labels)),
        )
        for sentence in sentences
    ]
    _, labels, mask = collate_sentences(examples, token_pad_index=0, label_pad_index=-100)
    assert labels[1, 1].item() == -100
    assert mask.tolist() == [[True, True], [True, False]]
