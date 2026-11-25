from __future__ import annotations

import math

import pytest

from wsn.config import RadioConfig
from wsn.radio import LinkQualityModel, RadioEnergyModel, bytes_to_bits

RADIO = RadioEnergyModel.from_config(RadioConfig())
BITS = bytes_to_bits(512)


def test_bytes_convert_to_bits() -> None:
    assert BITS == 4096


def test_crossover_distance_matches_formula() -> None:
    assert RADIO.crossover_distance_m == pytest.approx(math.sqrt(1e-11 / 1.3e-15))
    assert RADIO.crossover_distance_m == pytest.approx(87.7, abs=0.1)


def test_tx_energy_is_positive_and_increases_with_distance() -> None:
    energies = [RADIO.tx_energy_j(BITS, d) for d in (0.0, 5.0, 20.0, 60.0, 90.0, 150.0)]
    assert all(e > 0 for e in energies)
    assert energies == sorted(energies)
    assert len(set(energies)) == len(energies)


def test_tx_energy_uses_free_space_below_crossover() -> None:
    d = RADIO.crossover_distance_m - 1.0
    expected = BITS * 5e-8 + BITS * 1e-11 * d**2
    assert RADIO.tx_energy_j(BITS, d) == pytest.approx(expected)


def test_tx_energy_uses_multipath_at_and_above_crossover() -> None:
    d0 = RADIO.crossover_distance_m
    for d in (d0, d0 + 10.0):
        expected = BITS * 5e-8 + BITS * 1.3e-15 * d**4
        assert RADIO.tx_energy_j(BITS, d) == pytest.approx(expected)


def test_rx_energy_is_electronics_only() -> None:
    assert RADIO.rx_energy_j(BITS) == pytest.approx(BITS * 5e-8)
    assert RADIO.reference_energy_j(BITS) == RADIO.rx_energy_j(BITS)


@pytest.mark.parametrize("bits", [0, -8])
def test_invalid_bits_rejected(bits: int) -> None:
    with pytest.raises(ValueError, match="bits"):
        RADIO.tx_energy_j(bits, 10.0)


def test_negative_distance_rejected() -> None:
    with pytest.raises(ValueError, match="distance"):
        RADIO.tx_energy_j(BITS, -1.0)


def _link_model(seed: int = 3) -> LinkQualityModel:
    return LinkQualityModel(RadioConfig(), 30.0, seed, {(0, 1): 10.0, (1, 2): 29.0})


def test_base_probability_falls_with_distance_and_is_clipped() -> None:
    config = RadioConfig()
    near = LinkQualityModel.base_probability(1.0, config, 30.0)
    far = LinkQualityModel.base_probability(30.0, config, 30.0)
    assert near > far
    assert far == pytest.approx(0.78)
    assert LinkQualityModel.base_probability(500.0, config, 30.0) == config.base_link_quality_min


def test_success_probability_is_bounded_and_deterministic() -> None:
    first, second = _link_model(), _link_model()
    for round_number in (1, 25, 26, 300):
        p = first.success_probability(0, 1, round_number)
        assert 0.5 <= p <= 0.99
        assert p == second.success_probability(1, 0, round_number)


def test_success_probability_refreshes_per_epoch() -> None:
    model = _link_model()
    assert model.success_probability(1, 2, 1) == model.success_probability(1, 2, 25)
    epochs = {model.success_probability(1, 2, r) for r in range(1, 25 * 8, 25)}
    assert len(epochs) > 1


def test_unknown_link_raises() -> None:
    with pytest.raises(KeyError):
        _link_model().success_probability(0, 2, 1)
