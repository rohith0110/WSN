"""Isolated, seeded random streams.

Two mechanisms keep runs reproducible and paired across algorithms:

* ``keyed_uniform`` hashes a key (scenario seed, round, packet, sender, ...) into a uniform
  value. The value depends only on the key, so an algorithm that picks a different path never
  shifts the random draws seen by another algorithm.
* ``stream_rng`` gives a NumPy generator for a named purpose (topology, traffic order, agent).
"""

from __future__ import annotations

import hashlib

import numpy as np

_KEY_SEPARATOR = "\x1f"
_UINT64_RANGE = float(2**64)


def keyed_seed(*key: int | str) -> int:
    """Deterministic 64-bit integer derived from the key parts."""
    text = _KEY_SEPARATOR.join(str(part) for part in key)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def keyed_uniform(*key: int | str) -> float:
    """Deterministic value in ``[0, 1)`` derived from the key parts."""
    return keyed_seed(*key) / _UINT64_RANGE


def keyed_symmetric(amplitude: float, *key: int | str) -> float:
    """Deterministic value in ``[-amplitude, +amplitude)``."""
    return (2.0 * keyed_uniform(*key) - 1.0) * amplitude


def stream_rng(*key: int | str) -> np.random.Generator:
    """Independent NumPy generator for one named purpose."""
    return np.random.default_rng(keyed_seed(*key))
