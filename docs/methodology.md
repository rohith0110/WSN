# Methodology

All values below are the defaults in `configs/default.yaml`. They are simulation assumptions
chosen to make battery depletion observable in a student-scale experiment, not physical
constants, and the model is not an IEEE 802.15.4 PHY/MAC implementation.

## 1. Simulation model

### Topology

- `N = 50` sensors placed uniformly at random (seeded) in a 100 m × 100 m field; sink at
  (50, 50) with unlimited energy.
- A potential link exists between two devices when their distance `d <= 30 m` (inclusive).
- Placements where some sensor cannot reach the sink are regenerated (logged), up to 100 tries;
  if none is connected the run fails with an explicit error. Parameters are never changed silently.
- Alive density of node `v` = `min(alive_degree(v) / 10, 1)`, counting alive neighbors in range
  (the sink counts as an alive neighbor; the node itself never does).

### Radio energy (first-order model)

`k = packet_bytes × 8 = 4096 bits`, `E_elec = 50 nJ/bit`, `ε_fs = 10 pJ/bit/m²`,
`ε_mp = 0.0013 pJ/bit/m⁴`, `d0 = sqrt(ε_fs / ε_mp) ≈ 87.7 m`.

```text
E_tx(k, d) = k·E_elec + k·ε_fs·d²   if d < d0
E_tx(k, d) = k·E_elec + k·ε_mp·d⁴   if d >= d0
E_rx(k)    = k·E_elec
```

- The sender pays `E_tx` for every attempted hop, successful or not.
- A non-sink receiver pays `E_rx` for every attempted hop, even when decoding fails
  (simplifying assumption: its receiver electronics were active for the attempt).
- The sink's battery is not modelled; it pays nothing.
- A node that cannot afford its required energy is drained to 0 J (it dies) and the packet is
  dropped as `energy_exhausted`; energy never goes negative. Sender exhaustion happens *before*
  transmission (receiver not charged); receiver exhaustion happens after the sender transmitted.
- **Omitted:** idle listening, sensing, processing and data aggregation energy; energy recovery.
- Ledger check per run: `initial = remaining + consumed` (error is reported in `summary.json`).

### Link reliability (abstract, not a propagation model)

```text
p_base(d) = clip(0.98 − 0.20·(d / 30)², 0.60, 0.99)
p(u, v, t) = clip(p_base(d) + offset_static(u, v) + offset_epoch(u, v, ⌊(t−1)/25⌋), 0.50, 0.99)
```

`offset_static ∈ [−0.08, 0.08]` is fixed per link; `offset_epoch ∈ [−0.05, 0.05]` is redrawn
every 25 rounds. Each attempt succeeds when `U < p`, where `U` is a keyed pseudo-random value
derived from (scenario seed, round, packet id, sender, receiver, attempt index). Because the
value depends only on that key, one algorithm choosing a different route never shifts the random
draws of another. Routers can see `p(u, v, t)`, never `U`.

### Traffic, capacity and packet lifecycle

- Every original sensor offers 1 packet per round (`uniform`), independent of the algorithm.
  `bursty`: with probability 0.10 per sensor per round, that sensor offers 3× packets.
- Offered events from dead or disconnected sensors still count as offered load.
- Packets in a round are processed one at a time in a seeded shuffled order; each is forwarded
  hop by hop until delivered or dropped.
- Each sensor may make at most 12 transmission attempts per round (own + relayed). A sender
  that has used its capacity drops the packet as `congestion`. Observable utilization is
  `used / capacity` (a service-capacity proxy, not a queue).
- Safety: the packet's visited set (loops → `loop_prevented`) and `max_hops = 2N`.
- Outcomes (mutually exclusive): `delivered`, `source_dead`, `no_route`, `link_failure`,
  `energy_exhausted`, `congestion`, `loop_prevented`, `max_hops`, `network_down`.
- One attempt per hop (no retries). Delay is measured in hops, never in milliseconds.
- **Early stop:** when no alive sensor can reach the sink, nothing can be delivered any more.
  The run stops simulating, and each remaining round up to the horizon is still accounted for
  (its offered packets are dropped as `network_down`; rows have `simulated = false`). This keeps
  offered-load PDR comparable between runs that fail at different times
  (see [decisions.md §4](decisions.md)).

## 2. Routing algorithms

All routers receive the same read-only `NetworkView` (positions, alive status, residual energy,
current link estimates, current-round utilization, neighborhoods) and return an alive, in-range,
unvisited neighbor or `None`. The engine charges energy, draws link outcomes and records drops.

### A. Shortest path

Breadth-first search from the sink over alive nodes not yet visited by the packet; choose the
eligible neighbor with the smallest hop distance (ties: lowest id). Recomputed every decision.

### B. Energy-aware heuristic

Dijkstra from the sink with directed edge costs, recomputed every decision:

```text
cost(u → v) = 1.0
            + 1.0 · (E_tx(u, v) + E_rx(v)) / E_ref        E_ref = k·E_elec (J)
            + 1.2 · (1 − p(u, v))
            + 1.5 · (1 − energy_fraction(v))
            + 0.5 · utilization(v)
            + 0.2 · (1 − density(v))
```

For the sink the `E_rx`, depletion, utilization and density terms are 0. All terms are
nonnegative and the base is positive, so costs are strictly positive.

### C. Tabular Q-Learning over ranked candidate slots

**Candidates.** For each eligible neighbor `v` of the current node `u`:

```text
score(v) = +0.30 · energy_fraction(v)
           +0.25 · p(u, v)
           +0.35 · clip((dist(u, sink) − dist(v, sink)) / 30, −1, 1)
           −0.20 · utilization(v)
           +0.05 · density(v)
           −0.10 · E_tx(u, v) / E_ref
```

The sink uses energy fraction 1, utilization 0 and density 1. Candidates are sorted by score
(ties: lowest id) and the top `K = 5` are kept; a directly reachable sink always gets a slot
(it replaces the 5th if needed). **Actions are slot indices 0..K−1**, so the same table works on
unseen topologies. Missing slots are masked.

**State** (each dimension in three bins: low / medium / high, boundaries in config):

| Dimension | Boundaries |
|---|---|
| current node energy fraction | 0.33, 0.66 |
| current node distance to sink / field diagonal | 0.15, 0.35 |
| current node utilization | 0.34, 0.67 |
| current node alive density | 0.4, 0.8 |
| best candidate link reliability (sentinel 3 if none) | 0.75, 0.88 |
| best candidate energy fraction (sentinel 3 if none) | 0.33, 0.66 |

This is an approximate local observation (at most 3⁶ = 729 states, plus sentinels), not a fully
observable Markov state; per-neighbor detail is carried by the ranking.

**Reward per attempted hop** (base terms once per attempt, then the terminal term):

```text
r = −0.05
    −0.15 · charged_energy(tx + rx) / E_ref
    −0.30 · utilization(v)
    −0.40 · (1 − energy_fraction(v))
    +0.50 · clip(progress / 30, −1, 1)
    −2.0  · nodes newly exhausted by this attempt
    +10   if delivered to the sink
    −10   if the packet failed on this attempt or before its next attempt
```

**Update.** `Q(s,a) ← Q(s,a) + α [r + γ · max_{a' valid} Q(s',a') − Q(s,a)]` with α = 0.15,
γ = 0.95. Terminal transitions bootstrap 0. `s'` is the state at the next node, computed when
the packet continues; the max runs over valid slots only. If the packet ends before its next
attempt (no candidates, congestion, exhaustion or hop limit), the transition is closed as a
failure. Unseen states read as 0.

**Exploration.** ε-greedy over valid slots, ε decaying linearly per episode from 1.0 to 0.05;
greedy ties are broken by the agent's seeded RNG during training. **Evaluation** uses ε = 0, no
updates, and a deterministic tie-break (lowest slot, i.e. the best-ranked candidate).

### D. Cluster-head routing (LEACH baseline and adaptive rotation)

Every `rotation_rounds = 20` rounds the router elects cluster heads from observable state, and
each alive non-head sensor joins the nearest alive head in radio range. A packet first goes
from its source to the source's head (one hop). Heads, gateways and sensors with no head in
range then follow a backbone route: Dijkstra from the sink where entering a head or the sink
costs 1 and entering any other sensor costs `1 + gateway_penalty` (0.5), so routes prefer the
head backbone.

- **LEACH** (baseline): a sensor that has not led in the current cycle of `round(1/p)` epochs
  becomes a head when a keyed random draw falls below `T = p / (1 − p · (e mod 1/p))`, with
  `p = 0.10`. Each sensor leads at most once per cycle.
- **Adaptive** (`cluster_adaptive`): heads are chosen greedily by
  `0.5·energy + 0.2·density + 0.2·(1 − d_sink/diagonal) − 0.1·relay_load`, skipping sensors
  below 10% battery, until every alive sensor is a head or a head's neighbor (a dominating
  set, so every member reaches its head in one hop). `relay_load` is how often the router chose
  that node as next hop in the previous epoch, normalised by the busiest node.

**Data aggregation is not modelled.** A head forwards each member reading individually, so the
offered load, packet accounting and energy costs are identical to the other strategies and
the comparison stays like-for-like. Clustering in published work usually gains its benefit
from aggregation; without it, the extra member-to-head hop is pure cost (see results).

### E. Deep Q-network (DQN)

Same ranked candidate slots, reward, discount, exploration schedule and training seeds as the
tabular agent; only the value function differs. The observation is
`[energy, distance to sink / diagonal, utilization, density]` of the current node, followed by
`K × [energy, link reliability, clipped progress, utilization, density, E_tx/E_ref − 1]` for
the ranked candidates (zero-padded) and a `K`-slot validity mask (39 inputs for `K = 5`).

A fully connected network (39 → 64 → 64 → 5, ReLU, 7,045 weights) outputs one Q-value per
slot. Training uses experience replay (20,000 transitions, batches of 64, one gradient step
every 4 transitions after 1,000 warm-up transitions), a target network copied every 500
gradient steps, Huber loss (δ = 1) and Adam (learning rate 0.001). Targets bootstrap only from
valid slots of the next state (`max` over the mask) and use 0 for terminal transitions. The
network is written in NumPy rather than PyTorch (see [decisions.md](decisions.md)); its
backpropagation is checked against finite differences in the tests. Evaluation is greedy and
frozen, exactly as for the tabular agent.

## 3. Training and evaluation protocol

- Training seeds `1000..1039` (one fresh topology per episode, 40 episodes, ≤ 400 rounds each).
  Evaluation seeds `2000..2009`. Disjointness is enforced by config validation, the trainer and
  the model compatibility check.
- Coefficients and hyperparameters were not tuned on evaluation seeds. The defaults are the
  starting values proposed in the specification; no tuning loop was run.
- For every evaluation seed, all algorithms run on the **same** `Scenario` object: same
  placement, initial energy, link offsets, link evolution, offered events and keyed draws.
- The model artifacts (`models/q_table.json`, `models/dqn.json`) store the Q-table or network
  weights, ranking weights, reward weights, discount, training seeds, config fingerprint and
  format version. Both learned policies train on the same seeds with the same reward and
  exploration schedule.

## 4. Metrics and statistics

| Metric | Definition |
|---|---|
| Alive nodes | Sensors with energy > 0 |
| Mean residual energy (J) | Mean over all original sensors, dead ones count 0 |
| Energy used (J) | Initial total − remaining total |
| Offered packets | Every scheduled event over the full horizon (incl. dead/disconnected sources and network-down rounds) |
| Offered-load PDR (%) | 100 · delivered / offered |
| Operational PDR (%) | 100 · delivered / offered during simulated rounds (supplementary) |
| Active-source PDR (%) | 100 · delivered / offered by sources alive at offer time (supplementary) |
| Mean delivered hops | Mean hops of delivered packets only (N/A if none) |
| FND / HND / LND | First round with ≥ 1 / ≥ ⌈N/2⌉ / N dead sensors; `null` + `censored` if not reached |
| First disconnection | First round an alive sensor that could previously reach the sink no longer can |
| Throughput | Delivered packets / simulated rounds |
| Drops by cause | Counts per outcome |
| Policy compute time | Wall-clock time inside `select_next_hop` (not part of the energy model) |

Statistics in `aggregate.csv`: sample mean, sample standard deviation and a percentile bootstrap
95% CI of the mean (5,000 resamples, fixed seed), with `n_runs`. For censored lifetime metrics the
mean is reported only when every run observed the event; otherwise `observed_fraction` is
reported and the mean is withheld (averaging only observed runs would be biased).
`paired_differences.csv` gives per-seed differences (A − B) for the same scenario with a bootstrap
CI over seeds.

## Output schemas

`round_metrics.csv` (one row per round; comparison files add `variant`):
`round, simulated, algorithm, seed, alive_nodes, dead_nodes, mean_remaining_energy_j,
total_remaining_energy_j, cumulative_energy_used_j, offered_packets_round,
delivered_packets_round, offered_packets_cumulative, delivered_packets_cumulative,
cumulative_offered_pdr, round_offered_pdr, reachable_alive_fraction, link_failure_drops,
congestion_drops, energy_drops, no_route_drops, source_dead_drops, loop_prevented_drops,
max_hops_drops, network_down_drops, hops_attempted_round`. Drop columns are per round.

`summary.json`: `algorithm, seed, node_count, config_fingerprint, model_fingerprint,
rounds_executed, max_rounds, stop_reason, packets{offered, offered_simulated_rounds, delivered,
offered_by_alive_sources, drops{…}}, metrics{offered_load_pdr_pct, active_source_pdr_pct,
operational_pdr_pct, mean_delivered_hops, mean_attempted_hops_per_offered,
throughput_packets_per_round, final_alive_nodes, final_reachable_fraction},
lifetime_events{fnd, hnd, lnd, first_disconnection: {round, censored}},
energy{initial_total_j, remaining_total_j, consumed_total_j, ledger_error_j}, pdr_definitions,
units, policy{decisions, compute_time_s}, timestamp_utc`.

`per_run.csv`: one row per (variant, seed, algorithm) with the headline metrics, lifetime rounds
(empty when censored) and `drops_<cause>` columns. `aggregate.csv`: `variant, algorithm, metric,
n_runs, n_observed, observed_fraction, mean, std, ci95_low, ci95_high, note`.
`sampled_packet_traces.jsonl`: packets whose id is a multiple of `output.trace_sample_every`,
with their real path and outcome. `node_energy_snapshots.csv`: `round, node_id, energy_j` every
`output.energy_snapshot_every` rounds plus round 0 and the last simulated round.

## 5. Results (measured)

Everything in this section comes from runs executed on 2026-11-25 with the code and configs in
this repository: 10 held-out evaluation fields (seeds 2000–2009) for every scenario and variant,
paired across all six strategies. Regenerate them with the commands in each heading. Full
tables (std, CIs, every metric) are in each folder's `summary.md`, `aggregate.csv` and
`paired_differences.csv`. Values below are means with the 95% bootstrap CI in brackets.

### 5.1 Training

`python -m wsn.cli train --algorithm q_learning` (44 s) and
`python -m wsn.cli train --algorithm dqn` (187 s), 40 episodes each on training seeds
1000–1039. Mean reward per attempt rose from about −1.1 in the first episodes (fully random
exploration) to +0.4 for Q-Learning and +1.1 for the DQN on the last episode; the best late
training episode reached 31% (Q-Learning) and 50% (DQN) offered-load PDR. Episodes use
different topologies, so the curves are noisy (`models/training_curve.png`,
`models/dqn_training_curve.png`). These are training numbers, not evaluation results.

### 5.2 Standard scenario: 50 nodes, 10 fields, horizon 2,000 rounds

`python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json --scenario standard --output results/comparisons/standard`

| Metric | Shortest path | Energy-aware | LEACH | Adaptive clusters | Q-Learning | DQN |
|---|---|---|---|---|---|---|
| Delivered packets (of 100,000) | 6,168 [5,390, 7,099] | **6,336** [5,568, 7,248] | 5,167 | 4,718 | 3,553 | 6,320 [5,564, 7,198] |
| First node death, FND (round) | 47.6 [43.6, 52.2] | **102.7** [86.0, 118.4] | 55.2 | 58.9 | 63.4 | 90.5 [81.7, 100.4] |
| First disconnection (round) | **175.0** | 167.3 | 156.5 | 149.5 | 117.4 | 174.4 |
| Energy used (J, of 10 J) | 5.85 | **5.53** | 7.42 | 7.64 | 9.47 | 8.18 |
| Mean hops per delivered packet | 2.06 | **1.99** | 2.50 | 2.85 | 3.40 | 2.18 |
| Delivery rate while connected (%) | 39.7 | **53.3** | 34.7 | 46.2 | 27.4 | 46.1 |
| Half the sensors dead (HND) | 1/10 fields | 1/10 | 3/10 | 1/10 | 10/10 (round 128.5) | 7/10 |
| Decision time per run (s) | **0.41** | 5.88 | 0.89 | 0.83 | 0.92 | 1.21 |

Paired differences on the same fields (mean [95% CI]):

| Contrast | Delivered packets | FND (rounds) | Energy used (J) |
|---|---|---|---|
| Energy-aware − shortest path | +168 [+126, +213] | +55.1 [+40.4, +68.4] | −0.32 [−0.48, −0.15] |
| DQN − energy-aware | −16 [−67, +37] | −12.2 [−23.2, −0.9] | +2.65 [+1.80, +3.50] |
| DQN − Q-Learning | +2,767 [+2,209, +3,275] | +27.1 [+18.1, +35.3] | −1.29 [−2.35, −0.30] |
| Q-Learning − energy-aware | −2,783 [−3,327, −2,208] | −39.3 [−55.3, −22.4] | +3.94 [+3.47, +4.33] |
| Adaptive clusters − LEACH | −449 [−574, −335] | +3.7 [−7.6, +14.5] | +0.22 [−0.16, +0.62] |

### 5.3 Harsh conditions (bursty traffic, weak links, sink on the edge)

`python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json --scenario stress --output results/comparisons/stress`

| Variant | Metric | Shortest path | Energy-aware | LEACH | Adaptive | Q-Learning | DQN |
|---|---|---|---|---|---|---|---|
| Bursty traffic | Delivered | 6,099 | **6,334** | 5,082 | 4,624 | 3,500 | 6,315 |
| | FND | 40.6 | **86.2** | 47.5 | 47.7 | 57.2 | 76.2 |
| Weak radio links | Delivered | 4,682 | **4,797** | 3,796 | 3,286 | 3,212 | 4,778 |
| | FND | 55.1 | **113.5** | 68.6 | 73.4 | 98.3 | 108.2 |
| Sink at the edge | Delivered | 4,567 | **4,771** | 3,700 | 3,392 | 2,432 | 4,707 |
| | FND | 35.7 | **84.3** | 42.9 | 42.3 | 49.6 | 72.3 |

The energy-aware heuristic delays FND relative to shortest path in every variant (paired CIs
exclude 0: +45.6, +58.4 and +48.6 rounds). The DQN's delivery is statistically
indistinguishable from the heuristic's with bursty traffic and weak links (CIs include 0) and
slightly lower with the sink on the edge (−64 [−111, −18]).

### 5.4 Network size: 25 / 50 / 100 sensors

`python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json --scenario scalability --output results/comparisons/scalability`

The 25-sensor variant uses a 35 m range so the field can connect; the others use 30 m. The
learned policies were trained only on 50-sensor fields and are evaluated unchanged on 25 and
100 (ranked slots make that possible).

| Sensors | Metric | Shortest path | Energy-aware | LEACH | Adaptive | Q-Learning | DQN |
|---|---|---|---|---|---|---|---|
| 25 | Delivered | 4,538 | **4,651** | 3,791 | 3,139 | 1,847 | 4,616 |
| | FND | 83.8 | **134.1** | 80.6 | 84.1 | 71.7 | 120.8 |
| 50 | Delivered | 6,168 | **6,336** | 5,167 | 4,718 | 3,553 | 6,320 |
| | FND | 47.6 | **102.7** | 55.2 | 58.9 | 63.4 | 90.5 |
| 100 | Delivered | 12,460 | **13,600** | 9,219 | 9,192 | 7,516 | 13,480 |
| | FND | 32.5 | **120.4** | 32.4 | 33.6 | 56.3 | 102.3 |
| 100 | Decision time per run (s) | **2.3** | 44.1 | 5.7 | 5.5 | 4.2 | 5.4 |

The heuristic's advantage over shortest path grows with density (+1,140 [+1,012, +1,279]
delivered and +87.9 rounds FND at 100 sensors) because a denser ring around the sink gives it
more relays to spread load over. Its decision cost also grows fastest: a full weighted
Dijkstra per hop costs about 19× shortest path at 100 sensors, while the DQN's cost stays
near the cheap strategies.

### 5.5 Interpretation

- **The energy-aware heuristic is the best strategy in all 6 distinct test conditions** (the
  50-sensor size variant repeats the standard field): first on
  delivered packets, first sensor death and energy used in every variant.
- **The DQN nearly matches it.** Its delivery is statistically tied with the heuristic on the
  standard field, with bursty traffic and with weak links (paired CIs include 0) and within 2%
  in the other conditions. It keeps the first sensor alive far longer than shortest path
  (90.5 vs 47.6 rounds) and transfers to network sizes it never trained on. It uses more
  energy than the heuristic (+2.65 J [+1.80, +3.50] of 10 J) and 0.19 more hops per delivered
  packet.
- **The tabular Q-Learning policy is the weakest.** Its 6-number local state cannot represent
  the per-neighbor detail that the DQN reads from the candidate features, so it detours
  (3.4 hops per delivered packet vs 2.0) and drains energy. The same training data with a
  richer function approximator (the DQN) fixes most of that: +2,767 delivered packets on the
  same fields. An earlier Q-table, trained identically except for its exploration random
  stream, delivered 4,813 packets on these same 10 fields instead of 3,553: the tabular
  policy is highly sensitive to its training trajectory.
- **Clustering without aggregation does not pay.** Both cluster routers add a member-to-head
  hop that moves readings sideways before they head to the sink, so they use 27–31% more
  energy than shortest path and deliver fewer packets. The adaptive rotation keeps sensors alive no
  longer than LEACH (FND difference +3.7 [−7.6, +14.5]) and delivers fewer packets. Gains
  reported for clustering in the literature come from data aggregation at the heads, which
  this comparison deliberately leaves out to keep packet accounting equal.
- **The energy hole decides the lifetime.** In every strategy the sensors within one hop of
  the sink die first and cut off the rest of the field while most sensors still hold charge:
  HND was reached in at most 1 of 10 fields for the two best strategies. Routing can only
  spread the relay load across that ring; sink mobility or aggregation would change the picture.

### 5.6 ns-3 validation (optional extension C)

The two ns-3 programs in `ns3/` were built against ns-3.49 (MinGW-w64 g++ 15.2 on Windows,
modules core, network, mobility, propagation, spectrum, antenna, lr-wpan, energy). Steps and
commands are in the README.

**Link level.** `wsn-link-validation` sends 5,000 IEEE 802.15.4 frames (100-byte payload,
2.4 GHz O-QPSK error model, 0 dBm, −106.58 dBm sensitivity, no ACK or retry) per distance from 1
to 80 m, under log-distance path loss with Rayleigh fading (Nakagami m = 1). A 512-byte reading
needs 6 such frames, so its success is `PSR_frame⁶`. Compared with the simulator's
distance-based probability over its 30 m range (`results/ns3/summary.md`,
`results/ns3/link_validation.png`):

| Path loss exponent | Mean absolute gap | Mean gap (ns-3 − model) | ns-3 at 30 m | Model at 30 m |
|---|---|---|---|---|
| 2.7 | 0.077 | +0.077 | 0.966 | 0.780 |
| 3.0 | **0.056** | +0.056 | 0.872 | 0.780 |
| 3.5 | 0.074 | −0.051 | 0.498 | 0.780 |
| 4.0 | 0.295 | −0.284 | 0.020 | 0.780 |

The abstract model sits closest to an outdoor channel with exponent 3 (within 6 percentage
points on average) and between the exponent-3 and 3.5 curves at the edge of range. It falls
off more gently with distance than any physical curve: real 802.15.4 links stay near-perfect
up to a point and then collapse quickly.

**Field level.** `wsn-scenario-replay` replays field 2000 (exported with
`python -m wsn.cli ns3-export`) for 50 rounds of shortest-path routing at exponent 3:

| Source | Delivery rate | Mean hops |
|---|---|---|
| This simulator (batteries and capacity unlimited, so only links fail) | 77.0% | 1.83 |
| ns-3, one reading on air at a time | 98.0% | 1.90 |
| ns-3, all readings of a round competing (CSMA/CA) | 84.2% | 1.86 |

Each ns-3 reading is a single 100-byte frame, while the simulator models 512-byte readings,
so ns-3 is optimistic here; the link study above shows a 6-frame reading loses noticeably more
per hop. Contention alone costs ns-3 about 14 points of delivery, an effect the simulator does
not model at all: its congestion model is a per-node capacity cap, not a shared channel.

## 6. Limitations

- Abstract link model: a distance-based success probability with random offsets; no
  propagation, interference, collisions or MAC (the ns-3 study quantifies two of these gaps).
- Simplified congestion: a per-round transmission cap; no queues, buffering or retransmission.
- Homogeneous initial energy, static topology, no mobility, discrete rounds (no real time).
- Idle, sensing and processing energy are omitted; reception is charged even on failed attempts.
- Learned policies use local observations; neither can see global structure such as the
  energy hole. Training compute is not charged to sensor batteries.
- Routers have global, instantaneous knowledge of alive status and energy for path computation
  (both baselines and the backbone of the cluster routers); no control-message overhead.
- Cluster routing does not aggregate data, so it cannot show the benefit clustering usually
  gets from aggregation.
- The ns-3 field replay uses one 100-byte frame per reading and static routes from the
  initial topology; it validates link and contention behaviour, not battery depletion.

## 7. Extensions

Data aggregation at cluster heads (with explicit accounting of what an aggregated packet
represents), sink mobility, retransmissions, an ns-3 replay that fragments 512-byte readings
and charges an ns-3 energy model, reward and hyperparameter tuning on validation seeds, and a
policy-gradient agent over the same candidate features.
