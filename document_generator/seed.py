"""Deterministic seeding utilities for reproducible randomization."""
import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar


def derive_seed(master_seed, key):
    """Derive a deterministic 32-bit seed from (master_seed, key).

    We hash the pair and take the first 4 bytes. This keeps seeds stable across
    Python versions and platforms.
    """
    data = f"{master_seed}::{key}".encode()
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def make_rng(master_seed, category, entity_id: str | int, field_id):
    """Create a deterministic RNG for one (category, entity_id, field_id)."""
    key = f"{category}:{entity_id}:{field_id}"
    return random.Random(derive_seed(master_seed, key))


T = TypeVar("T")


def weighted_choice(rng: random.Random, items: Sequence[T], weights: Sequence[float]):
    """Pick one item with deterministic weighted sampling."""
    if len(items) != len(weights):
        raise ValueError("items and weights must have same length")

    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to > 0")

    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += float(weight)
        if threshold <= cumulative:
            return item

    return items[-1]
