"""Offered-load generation: a fixed, seeded schedule of sensing events per round."""

from __future__ import annotations

from wsn.config import TrafficConfig
from wsn.models import OfferedPacket
from wsn.random_utils import keyed_uniform, stream_rng


class TrafficGenerator:
    """Events depend only on (scenario seed, round, sensor), never on the routing algorithm.

    ``uniform``: every original sensor offers ``offered_packets_per_node_per_round`` packets.
    ``bursty``: in addition, each sensor independently bursts with ``burst_probability`` per
    round, multiplying its packets by ``burst_multiplier`` for that round.
    """

    def __init__(self, config: TrafficConfig, node_count: int, scenario_seed: int) -> None:
        self._config = config
        self._node_count = node_count
        self._seed = scenario_seed
        self._slots_per_source = config.offered_packets_per_node_per_round * config.burst_multiplier

    def packets_from(self, source: int, round_number: int) -> int:
        count = self._config.offered_packets_per_node_per_round
        bursting = (
            self._config.profile == "bursty"
            and keyed_uniform(self._seed, "burst", round_number, source)
            < self._config.burst_probability
        )
        return count * self._config.burst_multiplier if bursting else count

    def packet_id(self, round_number: int, source: int, index: int) -> int:
        return (round_number * self._node_count + source) * self._slots_per_source + index

    def offered_count(self, round_number: int) -> int:
        return sum(self.packets_from(source, round_number) for source in range(self._node_count))

    def packets_for_round(self, round_number: int) -> list[OfferedPacket]:
        """All events of one round, in a deterministic scenario-seeded processing order."""
        packets = [
            OfferedPacket(self.packet_id(round_number, source, index), source, round_number)
            for source in range(self._node_count)
            for index in range(self.packets_from(source, round_number))
        ]
        order = stream_rng(self._seed, "processing-order", round_number).permutation(len(packets))
        return [packets[int(position)] for position in order]
