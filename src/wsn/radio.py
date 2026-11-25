"""First-order radio energy model and the abstract, seeded link-reliability model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from wsn.config import RadioConfig
from wsn.models import NodeId
from wsn.random_utils import keyed_symmetric

BITS_PER_BYTE = 8


def bytes_to_bits(byte_count: int) -> int:
    if byte_count <= 0:
        raise ValueError("byte_count must be positive")
    return byte_count * BITS_PER_BYTE


@dataclass(frozen=True, slots=True)
class RadioEnergyModel:
    """``E_tx = k*E_elec + k*eps*d^n`` (free space n=2 below d0, multipath n=4 at/above d0)."""

    e_elec_j_per_bit: float
    epsilon_fs_j_per_bit_m2: float
    epsilon_mp_j_per_bit_m4: float

    @classmethod
    def from_config(cls, config: RadioConfig) -> RadioEnergyModel:
        return cls(
            config.e_elec_j_per_bit,
            config.epsilon_fs_j_per_bit_m2,
            config.epsilon_mp_j_per_bit_m4,
        )

    @property
    def crossover_distance_m(self) -> float:
        """``d0 = sqrt(eps_fs / eps_mp)``."""
        return math.sqrt(self.epsilon_fs_j_per_bit_m2 / self.epsilon_mp_j_per_bit_m4)

    def tx_energy_j(self, bits: int, distance_m: float) -> float:
        if bits <= 0:
            raise ValueError("bits must be positive")
        if distance_m < 0:
            raise ValueError("distance_m must be nonnegative")
        if distance_m < self.crossover_distance_m:
            amplifier = self.epsilon_fs_j_per_bit_m2 * distance_m**2
        else:
            amplifier = self.epsilon_mp_j_per_bit_m4 * distance_m**4
        return bits * self.e_elec_j_per_bit + bits * amplifier

    def rx_energy_j(self, bits: int) -> float:
        if bits <= 0:
            raise ValueError("bits must be positive")
        return bits * self.e_elec_j_per_bit

    def reference_energy_j(self, bits: int) -> float:
        """``E_ref = bits * E_elec``: the electronics-only cost used to normalise energy terms."""
        return self.rx_energy_j(bits)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def edge_key(first: NodeId, second: NodeId) -> tuple[NodeId, NodeId]:
    return (first, second) if first <= second else (second, first)


class LinkQualityModel:
    """Per-link success probability: distance base + fixed offset + offset refreshed per epoch.

    An abstract reliability model, not a propagation simulator. Values depend only on the
    scenario seed, the link and the refresh epoch, so every algorithm sees the same conditions.
    """

    def __init__(
        self,
        config: RadioConfig,
        communication_range_m: float,
        scenario_seed: int,
        edge_lengths_m: dict[tuple[NodeId, NodeId], float],
    ) -> None:
        self._config = config
        self._seed = scenario_seed
        self._static: dict[tuple[NodeId, NodeId], float] = {
            key: self.base_probability(length, config, communication_range_m)
            + keyed_symmetric(config.link_offset_amplitude, scenario_seed, "link-static", *key)
            for key, length in edge_lengths_m.items()
        }
        self._epoch = -1
        self._current: dict[tuple[NodeId, NodeId], float] = {}

    @staticmethod
    def base_probability(distance_m: float, config: RadioConfig, range_m: float) -> float:
        ratio = distance_m / range_m
        raw = (
            config.base_link_quality_intercept
            - config.base_link_quality_distance_coefficient * ratio**2
        )
        return _clip(raw, config.base_link_quality_min, config.base_link_quality_max)

    def epoch_for_round(self, round_number: int) -> int:
        return max(round_number - 1, 0) // self._config.link_refresh_rounds

    def _refresh(self, epoch: int) -> None:
        amplitude = self._config.link_temporal_amplitude
        low = self._config.min_success_probability
        high = self._config.max_success_probability
        self._current = {
            key: _clip(
                static + keyed_symmetric(amplitude, self._seed, "link-temporal", epoch, *key),
                low,
                high,
            )
            for key, static in self._static.items()
        }
        self._epoch = epoch

    def success_probability(self, first: NodeId, second: NodeId, round_number: int) -> float:
        epoch = self.epoch_for_round(round_number)
        if epoch != self._epoch:
            self._refresh(epoch)
        key = edge_key(first, second)
        if key not in self._current:
            raise KeyError(f"No potential link between {first} and {second}")
        return self._current[key]
