"""Discrete-round packet engine: energy charging, stochastic links, drops and metrics."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from wsn.config import config_fingerprint
from wsn.metrics import LifetimeEvent, RoundMetrics, RunSummary
from wsn.models import (
    HopOutcome,
    NodeId,
    OfferedPacket,
    PacketContext,
    PacketOutcome,
    PacketTrace,
)
from wsn.network import NetworkState, NetworkView, Scenario
from wsn.random_utils import keyed_uniform
from wsn.routing.base import FeedbackRouter, Router
from wsn.topology import reachable_from_sink

ProgressCallback = Callable[[int, int], None]
"""Called with (completed_round, max_rounds) after every simulated round."""

LINK_ATTEMPT_INDEX = 0
"""v1 makes exactly one attempt per hop (no link-level retries)."""


class RoutingContractError(RuntimeError):
    """A router returned a hop that is out of range or dead: a bug in the router."""


@dataclass(frozen=True, slots=True)
class EnergySnapshot:
    round_number: int
    energy_j: tuple[float, ...]


@dataclass(slots=True)
class SimulationResult:
    scenario: Scenario
    algorithm: str
    round_metrics: list[RoundMetrics]
    summary: RunSummary
    traces: list[PacketTrace]
    energy_snapshots: list[EnergySnapshot]


@dataclass(slots=True)
class _RoundTally:
    offered: int = 0
    delivered: int = 0
    from_alive_sources: int = 0
    hops_attempted: int = 0
    delivered_hops: int = 0
    drops: Counter[PacketOutcome] = field(default_factory=Counter)


@dataclass(slots=True)
class _PacketRun:
    """Mutable bookkeeping for one packet while it is being forwarded."""

    packet: OfferedPacket
    current: NodeId
    path: list[NodeId]
    visited: set[NodeId]
    hops: int = 0

    def context(self) -> PacketContext:
        return PacketContext(
            self.packet.packet_id, self.packet.source, self.packet.round_number, self.hops
        )


class Simulation:
    """Runs one router on one scenario. Create a new instance per run."""

    def __init__(
        self,
        scenario: Scenario,
        router: Router,
        *,
        max_rounds: int | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        self.scenario = scenario
        self.router = router
        self.config = scenario.config
        self.max_rounds = max_rounds if max_rounds is not None else self.config.scenario.max_rounds
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.model_fingerprint = model_fingerprint
        self.state = NetworkState(scenario)
        self.view = NetworkView(scenario, self.state)
        self._topology = scenario.topology
        self._sink = self._topology.sink_id
        self._bits = scenario.packet_bits
        self._capacity = self.config.traffic.tx_capacity_per_round
        self._max_hops = self.config.scenario.max_hops
        self._decision_time_s = 0.0
        self._decisions = 0
        self._traces: list[PacketTrace] = []

    # ----------------------------------------------------------------- energy

    def _drain(self, node: NodeId, amount_j: float) -> bool:
        """Charge ``amount_j`` if affordable; otherwise exhaust the node. Returns affordability."""
        available = self.state.energy_j[node]
        affordable = amount_j <= available
        charged = amount_j if affordable else available
        self.state.energy_j[node] = available - charged if affordable else 0.0
        self.state.consumed_j[node] += charged
        if self.state.energy_j[node] <= 0.0:
            self.state.energy_j[node] = 0.0
            if self.state.death_round[node] is None and available > 0.0:
                self.state.death_round[node] = self.state.round_number
        return affordable

    def _died_now(self, node: NodeId, was_alive: bool) -> int:
        return 1 if was_alive and self.state.energy_j[node] <= 0.0 else 0

    # ----------------------------------------------------------------- packets

    def _finish(self, run: _PacketRun, outcome: PacketOutcome) -> PacketOutcome:
        if isinstance(self.router, FeedbackRouter):
            self.router.on_packet_end(run.context(), outcome)
        packet = run.packet
        if packet.packet_id % self.config.output.trace_sample_every == 0:
            self._traces.append(
                PacketTrace(
                    packet.packet_id,
                    packet.round_number,
                    packet.source,
                    tuple(run.path),
                    outcome,
                    run.hops,
                )
            )
        return outcome

    def _report_hop(self, run: _PacketRun, outcome: HopOutcome) -> None:
        if isinstance(self.router, FeedbackRouter):
            self.router.on_hop_result(self.view, run.context(), outcome)

    def _choose(self, run: _PacketRun) -> NodeId | None:
        started = time.perf_counter()
        choice = self.router.select_next_hop(
            run.current, self._sink, run.visited, self.view, run.context()
        )
        self._decision_time_s += time.perf_counter() - started
        self._decisions += 1
        return choice

    def _validate_choice(self, sender: NodeId, receiver: NodeId) -> None:
        if receiver not in self._topology.neighbors[sender]:
            raise RoutingContractError(
                f"{self.router.name} chose {receiver}, which is out of range of {sender}"
            )
        if not self.view.is_alive(receiver):
            raise RoutingContractError(f"{self.router.name} chose dead node {receiver}")

    def _forward(self, packet: OfferedPacket) -> tuple[PacketOutcome, int]:
        """Forward one packet hop by hop. Returns (outcome, hops attempted)."""
        run = _PacketRun(packet, packet.source, [packet.source], {packet.source})
        if not self.view.is_alive(packet.source):
            return self._finish(run, PacketOutcome.SOURCE_DEAD), 0
        while True:
            outcome = self._step(run)
            if outcome is not None:
                return outcome, run.hops

    def _step(self, run: _PacketRun) -> PacketOutcome | None:
        """One forwarding decision and attempt; returns the outcome once the packet ends."""
        sender = run.current
        if not self.view.is_alive(sender):
            return self._finish(run, PacketOutcome.ENERGY_EXHAUSTED)
        if run.hops >= self._max_hops:
            return self._finish(run, PacketOutcome.MAX_HOPS)
        if self.state.tx_attempts[sender] >= self._capacity:
            return self._finish(run, PacketOutcome.CONGESTION)
        receiver = self._choose(run)
        if receiver is None:
            return self._finish(run, PacketOutcome.NO_ROUTE)
        if receiver in run.visited:
            return self._finish(run, PacketOutcome.LOOP_PREVENTED)
        self._validate_choice(sender, receiver)
        return self._attempt(run, sender, receiver)

    def _attempt(self, run: _PacketRun, sender: NodeId, receiver: NodeId) -> PacketOutcome | None:
        tx_j = self.view.tx_energy_j(sender, receiver)
        energy_before = self.state.energy_j[sender]
        if not self._drain(sender, tx_j):
            # Rejected before transmission: the sender is exhausted, nothing is sent.
            self._report_hop(
                run,
                HopOutcome(
                    sender,
                    receiver,
                    False,
                    False,
                    False,
                    PacketOutcome.ENERGY_EXHAUSTED,
                    energy_before,
                    1,
                ),
            )
            return self._finish(run, PacketOutcome.ENERGY_EXHAUSTED)
        self.state.tx_attempts[sender] += 1
        run.hops += 1
        charged = tx_j
        exhausted = self._died_now(sender, energy_before > 0.0)
        if receiver != self._sink:
            receiver_before = self.state.energy_j[receiver]
            rx_j = self.view.rx_energy_j(receiver)
            receiver_ok = self._drain(receiver, rx_j)
            charged += rx_j if receiver_ok else receiver_before
            exhausted += self._died_now(receiver, receiver_before > 0.0)
            if not receiver_ok:
                self._report_hop(
                    run,
                    HopOutcome(
                        sender,
                        receiver,
                        True,
                        False,
                        False,
                        PacketOutcome.ENERGY_EXHAUSTED,
                        charged,
                        exhausted,
                    ),
                )
                return self._finish(run, PacketOutcome.ENERGY_EXHAUSTED)
        success = keyed_uniform(
            self.scenario.seed,
            "hop",
            run.packet.round_number,
            run.packet.packet_id,
            sender,
            receiver,
            LINK_ATTEMPT_INDEX,
        ) < self.view.link_success_probability(sender, receiver)
        if not success:
            self._report_hop(
                run,
                HopOutcome(
                    sender,
                    receiver,
                    True,
                    False,
                    False,
                    PacketOutcome.LINK_FAILURE,
                    charged,
                    exhausted,
                ),
            )
            return self._finish(run, PacketOutcome.LINK_FAILURE)
        run.path.append(receiver)
        delivered = receiver == self._sink
        terminal = PacketOutcome.DELIVERED if delivered else None
        self._report_hop(
            run,
            HopOutcome(sender, receiver, True, True, delivered, terminal, charged, exhausted),
        )
        if delivered:
            return self._finish(run, PacketOutcome.DELIVERED)
        run.visited.add(receiver)
        run.current = receiver
        return None

    # ----------------------------------------------------------------- rounds

    def _run_round(self, round_number: int) -> _RoundTally:
        self.state.round_number = round_number
        self.state.tx_attempts = [0] * self._topology.node_count
        tally = _RoundTally()
        for packet in self.scenario.traffic.packets_for_round(round_number):
            tally.offered += 1
            if self.view.is_alive(packet.source):
                tally.from_alive_sources += 1
            outcome, hops = self._forward(packet)
            tally.hops_attempted += hops
            if outcome is PacketOutcome.DELIVERED:
                tally.delivered += 1
                tally.delivered_hops += hops
            else:
                tally.drops[outcome] += 1
        return tally

    def run(self, progress: ProgressCallback | None = None) -> SimulationResult:
        node_count = self._topology.node_count
        initial_total = self.state.initial_energy_j * node_count
        snapshot_every = self.config.output.energy_snapshot_every
        snapshots = [EnergySnapshot(0, tuple(self.state.energy_j))]
        reachable = reachable_from_sink(self._topology, self.state.alive_flags())
        ever_reachable = set(reachable)
        first_disconnection: int | None = None
        totals = _RoundTally()
        metrics: list[RoundMetrics] = []
        stop_reason = "max_rounds"
        round_number = 0

        while round_number < self.max_rounds:
            round_number += 1
            tally = self._run_round(round_number)
            _accumulate(totals, tally)
            alive = self.state.alive_flags()
            reachable = reachable_from_sink(self._topology, alive)
            lost = {node for node in ever_reachable if alive[node] and node not in reachable}
            if lost and first_disconnection is None:
                first_disconnection = round_number
            ever_reachable |= reachable
            metrics.append(self._round_metrics(round_number, tally, totals, alive, reachable))
            if round_number % snapshot_every == 0:
                snapshots.append(EnergySnapshot(round_number, tuple(self.state.energy_j)))
            if progress is not None:
                progress(round_number, self.max_rounds)
            if not reachable:
                stop_reason = "no_sensor_can_reach_sink"
                break

        if snapshots[-1].round_number != round_number:
            snapshots.append(EnergySnapshot(round_number, tuple(self.state.energy_j)))
        simulated_offered = totals.offered
        metrics.extend(self._network_down_rounds(round_number, totals, reachable))
        drop_counts = {outcome.value: count for outcome, count in totals.drops.items()}
        unsimulated = totals.offered - simulated_offered

        summary = RunSummary(
            algorithm=self.router.name,
            seed=self.scenario.seed,
            node_count=node_count,
            config_fingerprint=config_fingerprint(self.config),
            model_fingerprint=self.model_fingerprint,
            rounds_executed=round_number,
            max_rounds=self.max_rounds,
            stop_reason=stop_reason,
            offered_packets=totals.offered,
            offered_packets_simulated=totals.offered - unsimulated,
            delivered_packets=totals.delivered,
            packets_from_alive_sources=totals.from_alive_sources,
            total_hops_attempted=totals.hops_attempted,
            delivered_hops_total=totals.delivered_hops,
            fnd=self._death_event(1),
            hnd=self._death_event(math.ceil(node_count / 2)),
            lnd=self._death_event(node_count),
            first_disconnection=LifetimeEvent(first_disconnection),
            final_alive_nodes=self.state.alive_count(),
            final_reachable_fraction=len(reachable) / node_count,
            initial_energy_total_j=initial_total,
            remaining_energy_total_j=math.fsum(self.state.energy_j),
            consumed_energy_total_j=math.fsum(self.state.consumed_j),
            drop_counts=drop_counts,
            policy_decisions=self._decisions,
            policy_compute_time_s=self._decision_time_s,
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return SimulationResult(
            self.scenario, self.router.name, metrics, summary, self._traces, snapshots
        )

    def _network_down_rounds(
        self, last_simulated: int, totals: _RoundTally, reachable: set[NodeId]
    ) -> list[RoundMetrics]:
        """Rounds after an early stop: traffic is still offered but nothing can be delivered.

        The network state is frozen, so these rounds are accounted for without simulation.
        Keeping them makes offered-load PDR comparable between runs that stop at different times.
        """
        alive = self.state.alive_flags()
        rows: list[RoundMetrics] = []
        for round_number in range(last_simulated + 1, self.max_rounds + 1):
            offered = self.scenario.traffic.offered_count(round_number)
            tally = _RoundTally(offered=offered, from_alive_sources=0)
            tally.drops[PacketOutcome.NETWORK_DOWN] = offered
            for source in range(self._topology.node_count):
                if alive[source]:
                    tally.from_alive_sources += self.scenario.traffic.packets_from(
                        source, round_number
                    )
            _accumulate(totals, tally)
            rows.append(
                self._round_metrics(round_number, tally, totals, alive, reachable, simulated=False)
            )
        return rows

    def _death_event(self, deaths_needed: int) -> LifetimeEvent:
        """First round by which at least ``deaths_needed`` sensors had died (None if never)."""
        rounds = sorted(r for r in self.state.death_round if r is not None)
        return LifetimeEvent(rounds[deaths_needed - 1] if len(rounds) >= deaths_needed else None)

    def _round_metrics(
        self,
        round_number: int,
        tally: _RoundTally,
        totals: _RoundTally,
        alive: list[bool],
        reachable: set[NodeId],
        *,
        simulated: bool = True,
    ) -> RoundMetrics:
        node_count = self._topology.node_count
        remaining = math.fsum(self.state.energy_j)
        alive_count = sum(alive)
        return RoundMetrics(
            round=round_number,
            simulated=simulated,
            algorithm=self.router.name,
            seed=self.scenario.seed,
            alive_nodes=alive_count,
            dead_nodes=node_count - alive_count,
            mean_remaining_energy_j=remaining / node_count,
            total_remaining_energy_j=remaining,
            cumulative_energy_used_j=math.fsum(self.state.consumed_j),
            offered_packets_round=tally.offered,
            delivered_packets_round=tally.delivered,
            offered_packets_cumulative=totals.offered,
            delivered_packets_cumulative=totals.delivered,
            cumulative_offered_pdr=100.0 * totals.delivered / totals.offered,
            round_offered_pdr=100.0 * tally.delivered / tally.offered,
            reachable_alive_fraction=len(reachable) / node_count,
            link_failure_drops=tally.drops[PacketOutcome.LINK_FAILURE],
            congestion_drops=tally.drops[PacketOutcome.CONGESTION],
            energy_drops=tally.drops[PacketOutcome.ENERGY_EXHAUSTED],
            no_route_drops=tally.drops[PacketOutcome.NO_ROUTE],
            source_dead_drops=tally.drops[PacketOutcome.SOURCE_DEAD],
            loop_prevented_drops=tally.drops[PacketOutcome.LOOP_PREVENTED],
            max_hops_drops=tally.drops[PacketOutcome.MAX_HOPS],
            network_down_drops=tally.drops[PacketOutcome.NETWORK_DOWN],
            hops_attempted_round=tally.hops_attempted,
        )


def _accumulate(totals: _RoundTally, tally: _RoundTally) -> None:
    totals.offered += tally.offered
    totals.delivered += tally.delivered
    totals.from_alive_sources += tally.from_alive_sources
    totals.hops_attempted += tally.hops_attempted
    totals.delivered_hops += tally.delivered_hops
    totals.drops.update(tally.drops)


def run_simulation(
    scenario: Scenario,
    router: Router,
    *,
    max_rounds: int | None = None,
    model_fingerprint: str | None = None,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    simulation = Simulation(
        scenario, router, max_rounds=max_rounds, model_fingerprint=model_fingerprint
    )
    return simulation.run(progress)
