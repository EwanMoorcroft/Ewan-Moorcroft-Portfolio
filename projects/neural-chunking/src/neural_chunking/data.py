"""BIO sentence loading, duplicate-safe splitting, and tensor batching."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sentence:
    """One token sequence paired with BIO labels."""

    tokens: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject empty or misaligned sentence records."""
        if not self.tokens:
            raise ValueError("a sentence must contain at least one token")
        if len(self.tokens) != len(self.labels):
            raise ValueError("token and label counts must match")


class Vocabulary:
    """Stable string-to-index vocabulary with reserved special values."""

    def __init__(
        self,
        values: Iterable[str],
        include_unknown: bool,
        include_padding: bool = True,
    ) -> None:
        """Build a deterministic vocabulary from observed values."""
        observed = set(values)
        specials = ["<PAD>"] if include_padding else []
        if include_unknown:
            specials.append("<UNK>")
        collisions = observed.intersection(specials)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"observed values collide with reserved vocabulary entries: {names}")
        ordered = specials + sorted(observed)
        self.value_to_index = {value: index for index, value in enumerate(ordered)}
        self.index_to_value = tuple(ordered)
        self.pad_index = self.value_to_index.get("<PAD>")
        self.unknown_index = self.value_to_index.get("<UNK>")

    @classmethod
    def from_index_values(cls, values: Sequence[str]) -> Vocabulary:
        """Restore an exact index order from trusted primitive checkpoint data."""
        if not values or len(values) != len(set(values)):
            raise ValueError("vocabulary values must be non-empty and unique")
        vocabulary = cls.__new__(cls)
        vocabulary.index_to_value = tuple(values)
        vocabulary.value_to_index = {
            value: index for index, value in enumerate(vocabulary.index_to_value)
        }
        vocabulary.pad_index = vocabulary.value_to_index.get("<PAD>")
        vocabulary.unknown_index = vocabulary.value_to_index.get("<UNK>")
        return vocabulary

    def encode(self, values: Sequence[str]) -> list[int]:
        """Encode values, using the unknown index when configured."""
        encoded: list[int] = []
        for value in values:
            if value in self.value_to_index:
                encoded.append(self.value_to_index[value])
            elif self.unknown_index is not None:
                encoded.append(self.unknown_index)
            else:
                raise ValueError(f"unseen label: {value}")
        return encoded

    def decode(self, indices: Sequence[int]) -> list[str]:
        """Decode integer indices into their original strings."""
        return [self.index_to_value[index] for index in indices]

    def __len__(self) -> int:
        """Return the number of indexed values."""
        return len(self.index_to_value)


def _parse_row(raw: str, line_number: int) -> tuple[str, str]:
    """Parse one two-column token-label row with a useful error message."""
    parts = raw.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"line {line_number} must contain a token and BIO label")
    token, label = parts
    valid_prefixed_label = len(label) > 2 and label[:2] in {"B-", "I-"} and bool(label[2:].strip())
    if label != "O" and not valid_prefixed_label:
        raise ValueError(f"line {line_number} has invalid BIO label {label!r}")
    return token, label


def read_sentences(path: Path) -> list[Sentence]:
    """Read blank-line-separated sentences from a two-column UTF-8 file."""
    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            if tokens:
                sentences.append(Sentence(tuple(tokens), tuple(labels)))
                tokens, labels = [], []
            continue
        token, label = _parse_row(raw, line_number)
        tokens.append(token)
        labels.append(label)
    if tokens:
        sentences.append(Sentence(tuple(tokens), tuple(labels)))
    if not sentences:
        raise ValueError("input contains no sentences")
    return sentences


def _content_fraction(sentence: Sentence, seed: int) -> float:
    """Map sentence content and a seed to a stable fraction in [0, 1)."""
    canonical = "\x1f".join(token.lower() for token in sentence.tokens)
    digest = hashlib.sha256(f"{seed}\x1e{canonical}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_by_content_hash(
    sentences: Sequence[Sentence],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 534,
) -> dict[str, list[Sentence]]:
    """Split complete sentences while keeping exact token duplicates together."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no room for a test split")
    splits: dict[str, list[Sentence]] = {"train": [], "validation": [], "test": []}
    boundary = train_fraction + validation_fraction
    for sentence in sentences:
        fraction = _content_fraction(sentence, seed)
        name = (
            "train"
            if fraction < train_fraction
            else "validation"
            if fraction < boundary
            else "test"
        )
        splits[name].append(sentence)
    if any(not rows for rows in splits.values()):
        raise ValueError("each split must contain at least one sentence")
    return splits


def build_vocabularies(sentences: Sequence[Sentence]) -> tuple[Vocabulary, Vocabulary]:
    """Build token and label vocabularies from training sentences only."""
    token_values = (token.lower() for sentence in sentences for token in sentence.tokens)
    label_values = (label for sentence in sentences for label in sentence.labels)
    return Vocabulary(token_values, include_unknown=True), Vocabulary(
        label_values,
        include_unknown=False,
        include_padding=False,
    )


class ChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Encode sentence records as token and label tensors."""

    def __init__(
        self,
        sentences: Sequence[Sentence],
        token_vocabulary: Vocabulary,
        label_vocabulary: Vocabulary,
    ) -> None:
        """Store encoded sentence tensors for repeated training epochs."""
        self.examples = [
            (
                torch.tensor(
                    token_vocabulary.encode([token.lower() for token in sentence.tokens]),
                    dtype=torch.long,
                ),
                torch.tensor(label_vocabulary.encode(sentence.labels), dtype=torch.long),
            )
            for sentence in sentences
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one encoded sentence pair."""
        return self.examples[index]

    def __len__(self) -> int:
        """Return the number of encoded sentences."""
        return len(self.examples)


def collate_sentences(
    examples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    token_pad_index: int = 0,
    label_pad_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sentences and return tokens, labels, and a valid-token mask."""
    token_rows, label_rows = zip(*examples, strict=True)
    tokens = pad_sequence(token_rows, batch_first=True, padding_value=token_pad_index)
    labels = pad_sequence(label_rows, batch_first=True, padding_value=label_pad_index)
    mask = tokens.ne(token_pad_index)
    return tokens, labels, mask
