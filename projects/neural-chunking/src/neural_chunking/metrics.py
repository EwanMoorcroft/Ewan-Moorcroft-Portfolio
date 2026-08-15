"""Token and exact-span metrics for BIO sequence labelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class Span(NamedTuple):
    """A labelled half-open token span."""

    label: str
    start: int
    end: int


@dataclass(frozen=True)
class SpanMetrics:
    """Exact-span precision, recall, and F1 counts."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def bio_spans(tags: list[str] | tuple[str, ...]) -> set[Span]:
    """Convert BIO tags to exact spans, treating invalid continuations as new spans."""
    spans: set[Span] = set()
    active_label: str | None = None
    active_start = 0
    for index, tag in enumerate((*tags, "O")):
        if tag == "O":
            prefix, label = "O", None
        else:
            parts = tag.split("-", maxsplit=1)
            if len(parts) != 2 or parts[0] not in {"B", "I"} or not parts[1]:
                raise ValueError(f"invalid BIO tag {tag!r}")
            prefix, label = parts
        continues = prefix == "I" and label == active_label
        if active_label is not None and not continues:
            spans.add(Span(active_label, active_start, index))
            active_label = None
        if prefix == "B" or (prefix == "I" and not continues):
            if label is None:
                raise ValueError(f"invalid BIO tag {tag!r}")
            active_label = label
            active_start = index
    return spans


def span_metrics(
    expected_sequences: list[list[str]],
    predicted_sequences: list[list[str]],
) -> SpanMetrics:
    """Aggregate exact-span counts across aligned sentence pairs."""
    if len(expected_sequences) != len(predicted_sequences):
        raise ValueError("expected and predicted sentence counts must match")
    true_positive = false_positive = false_negative = 0
    for expected, predicted in zip(expected_sequences, predicted_sequences, strict=True):
        if len(expected) != len(predicted):
            raise ValueError("expected and predicted token counts must match")
        expected_spans = bio_spans(expected)
        predicted_spans = bio_spans(predicted)
        true_positive += len(expected_spans & predicted_spans)
        false_positive += len(predicted_spans - expected_spans)
        false_negative += len(expected_spans - predicted_spans)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SpanMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )
