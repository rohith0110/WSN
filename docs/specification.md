# AI-Driven Energy-Aware Routing and Lifetime Optimization in Wireless Sensor Networks

> **Status:** Implementation specification for an autonomous coding agent<br>
> **Project type:** Final-year B.Tech / Computer Science research-and-simulation project<br>
> **Primary deliverable:** A working, reproducible Python application that simulates a wireless sensor network (WSN), implements three routing strategies (shortest path, energy-aware heuristic, tabular Q-Learning), evaluates their behavior fairly, and visualizes actual results in a Streamlit dashboard.

---

## 0. Instructions to the coding agent (read this first)

You are implementing this project **end to end**, not merely generating scaffolding, pseudocode, or a demonstration UI.

1. Read this entire specification before writing code. Treat the **core requirements** and **acceptance criteria** as binding.
2. Work incrementally: simulation physics → routing baselines → Q-Learning → experiments → dashboard → tests/docs.
3. Build a **real simulation**. No mock results, hard-coded metrics, fabricated graphs, fake training, or buttons that do nothing.
4. Use the technology and design choices below unless there is a concrete technical reason to depart. If you change one, document the reason in `docs/decisions.md` and update tests.
5. Execute commands, run tests, fix failures, and verify the application where your environment permits. If something cannot be executed, explicitly state what was not verified.
6. Keep core simulation/routing logic independent of Streamlit. The dashboard should call the same tested simulation APIs as the CLI.
7. Preserve deterministic behavior from seeds; do not use unseeded global randomness.
8. Do not claim that AI outperforms baselines unless experiments support it. Negative, neutral, or mixed results are valid scientific outcomes.
9. Build the minimum viable system fully **before** pursuing optional deep RL, NS-3 integration, or cluster-head selection.
10. At completion, provide: implemented features, reproducible commands, tests run/results, limitations, and files generated.

### Definition of done

A reviewer can clone the repository, install dependencies, run one simulation, train a Q-Learning agent, compare all three algorithms on held-out seeds, open an interactive dashboard, inspect saved outputs, and run passing automated tests **without editing source files**.

---

## 1. Problem and objectives

A WSN consists of energy-constrained wireless sensor nodes that generate readings and forward packets toward a base station (the **sink**). Multi-hop forwarding can conserve transmission energy, but repeatedly choosing the shortest route may overuse certain relay nodes. Dead relays can fragment the network even if many other nodes still have energy.

Build a simulator and adaptive routing system that chooses **next-hop neighbors** based on:

1. **Residual energy:** available battery at candidate relay nodes.
2. **Node density:** how many usable neighbors exist around a node.
3. **Link quality:** estimated probability that a packet succeeds over a wireless link.
4. **Traffic/load:** how busy a node is during the current simulation round.
5. **Distance:** physical link length and progress toward the sink.

The objective is to extend meaningful network operation while keeping a reasonable packet delivery ratio (PDR) and energy cost. These objectives can conflict; document the trade-off instead of optimizing only battery survival.

### Research questions

- Compared with hop-count shortest-path routing, does adaptive Q-Learning delay first-node death, half-node death, or network disconnection?
- Does Q-Learning outperform a **simple energy-aware heuristic**, not just a naive baseline?
- Under what node densities, traffic profiles, and link conditions does each algorithm work best?
- What delivery, delay, and computational overhead trade-offs accompany lifetime gains?

### In scope (required)

- Static 2D WSN topology with configurable node count, area, sink, and communication range.
- Per-node battery levels, a first-order radio energy model, link-quality-based packet outcomes, variable traffic load, and node deaths.
- Real packet generation, hop-by-hop forwarding, success/failure tracking, bounded paths, and routing around dead nodes.
- Three working routing algorithms.
- Trainable, persistable **tabular Q-Learning** with explicitly defined state/action/reward and held-out evaluation.
- Repeatable multi-seed experiments, metrics, CSV/JSON artifacts, plots, and a Streamlit dashboard.
- CLI, configuration, documentation, and automated tests.

### Out of scope for version 1

Physical hardware, real radio/PHY emulation, mobility, security/cryptography, a cloud backend, Kubernetes, DQN/PPO, mandatory clustering, and NS-3/OMNeT++/Cooja integration. These may be added after all core acceptance criteria pass.

---

## 2. Stack and environment

**Language:** Python 3.11+ (prefer the locally available supported version; ensure compatibility with Python 3.13 if used).<br>
**Required:** NumPy, NetworkX, Pandas, Matplotlib, Streamlit, PyYAML, pytest.<br>
**Recommended:** Plotly for an interactive topology/chart view, Ruff for linting.<br>
**Optional:** Gymnasium (if a standard `reset`/`step` RL wrapper is useful), PyTorch (only for optional DQN).<br>
**No paid API, physical sensor, GPU, or external dataset is needed.**

Use a `pyproject.toml` with an editable install and a `dev` extra; keep dependency ranges compatible rather than assuming all latest packages work together. Prefer standard-library `argparse` for CLI to avoid another dependency. Development must work on Windows and Linux; use `pathlib` for paths.

Target commands:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to the OS
python -m pip install -e ".[dev]"
python -m pytest -q
python -m wsn.cli simulate --config configs/default.yaml --algorithm shortest_path --seed 2000 --output results/smoke
python -m wsn.cli train --config configs/default.yaml --episodes 40 --output models/q_table.json
python -m wsn.cli compare --config configs/default.yaml --model models/q_table.json --seeds 2000,2001,2002 --output results/comparison
streamlit run dashboard/app.py
```

The CLI must implement useful `--help` output, validation errors, and exit codes. A `quickstart` or smoke-test mode with much shorter runs is desirable. Never silently start a lengthy training job when the user only opens the dashboard.

---

## 3. Reference project structure

The agent may make small improvements to names, but preserve equivalent separation of concerns.

```text
ai-wsn-routing/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── smoke.yaml
│   └── experiments.yaml
├── src/wsn/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py              # Node, Packet, Link, Scenario, immutable events
│   ├── topology.py            # Graph generation and connectivity
│   ├── radio.py               # Energy equations and link quality
│   ├── simulator.py           # Round engine and packet delivery
│   ├── traffic.py             # Offered-load generation, deterministic schedules
│   ├── random_utils.py       # Isolated seeded streams and keyed link draws
│   ├── routing/
│   │   ├── base.py            # Router interface
│   │   ├── shortest_path.py
│   │   ├── energy_aware.py
│   │   └── q_learning.py
│   ├── rl/
│   │   ├── state.py           # Discretization and ranked candidate actions
│   │   ├── reward.py
│   │   ├── trainer.py
│   │   └── serialization.py
│   ├── metrics.py
│   ├── experiments.py
│   ├── reporting.py
│   └── visualization.py
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_radio.py
│   ├── test_topology.py
│   ├── test_simulator.py
│   ├── test_routing.py
│   ├── test_q_learning.py
│   ├── test_metrics.py
│   └── test_reproducibility.py
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── decisions.md
├── models/                   # Generated, not committed by default
└── results/                  # Generated, not committed by default
```

Use typed dataclasses and/or lightweight validation for models. Avoid a monolithic `main.py`. Design clear interfaces so each routing strategy interacts with the **same** environment.

---

## 4. Simulation model: exact rules

### 4.1 Topology

- `N` sensor nodes have unique integer IDs `0..N-1` and fixed `(x, y)` positions in metres.
- The sink has a separate ID, e.g. `"sink"`, an `(x, y)` location, and unlimited energy in v1.
- An undirected **potential link** exists between a pair of devices when Euclidean distance `d <= communication_range_m`. Actual forwarding requires the receiver to be alive. Treat the potential topology as symmetric but each packet attempt as stochastic.
- Node density is the number of **alive neighbors** within radio range; normalize as `min(degree / density_reference, 1)` with `density_reference=10` by default. Never count the node itself.
- Default topology generator: seeded uniform random node positions in a rectangular area. Optionally reject/discard an initial placement not connected to the sink, up to a configurable retry limit. Log topology regeneration. If no connected network can be generated, fail clearly instead of secretly changing parameters.
- Save positions and link parameters once per scenario. All algorithms evaluated on the same seed must use the **same initial topology**.

### 4.2 Starting values (editable, not universal scientific constants)

Use these as the **default configuration**, with a lighter `smoke.yaml` for tests:

| Parameter | Default | Notes |
|---|---:|---|
| Sensor nodes | 50 | Experiment with 25, 50, 100 |
| Area width/height | 100 m / 100 m | Static 2D field |
| Sink position | (50 m, 50 m) | Center; allow edge alternative |
| Radio communication range | 30 m | Must ensure initial connectivity or report failure |
| Initial sensor energy | 0.2 J | Chosen to make depletion observable; tunable |
| Packet payload size | 512 bytes | Energy calculations use **bits = bytes × 8** |
| Regular offered traffic | 1 packet per original sensor per round | Fixed exogenous workload |
| Per-node transmission capacity | 12 hop attempts/round | A simple traffic/congestion abstraction |
| Maximum simulation length | 2,000 rounds | Stop sooner if no sensors remain operational |
| Maximum hops per packet | `2 * N` | Also reject route loops explicitly |
| Link-level retry count | 0 | One attempt per hop in v1 |
| Topology retries | 100 | Configurable |
| Link quality refresh period | 25 rounds | Time-varying wireless conditions |

**Important:** These values are simulation assumptions for a reproducible student project. Do not present the model as a real IEEE 802.15.4 PHY/MAC implementation. Results must state the assumptions.

### 4.3 First-order radio energy model

Use a conventional simplified WSN radio model with correct SI units.

Let `k` be packet bits, `d` be link distance in metres, and:

- `E_elec = 50e-9 J/bit`
- `epsilon_fs = 10e-12 J/bit/m^2`
- `epsilon_mp = 0.0013e-12 J/bit/m^4`
- `d0 = sqrt(epsilon_fs / epsilon_mp)` (approximately 87.7 m)

Transmit energy:

```text
E_tx(k, d) = k * E_elec + k * epsilon_fs * d^2,  if d < d0
E_tx(k, d) = k * E_elec + k * epsilon_mp * d^4,  if d >= d0
```

Receive energy:

```text
E_rx(k) = k * E_elec
```

Rules:

- Deduct TX energy from the sender **for every attempted hop**, successful or failed.
- Deduct RX energy from a participating non-sink receiver **for every attempted hop**, even if packet decoding fails; this models receiver electronics being active for the reception attempt. State this simplifying assumption in the report.
- Do not deduct sink battery energy. Track sink receptions separately.
- A node that cannot afford its required energy becomes exhausted (`energy_j = 0`) and the packet is dropped with an explicit cause. Never allow negative energy.
- No energy recovery. Idle/sensing energy and aggregation are excluded from v1 and must be described as omissions.
- Recompute alive graph/connectivity whenever a node dies. Inactive devices are not valid forwarding targets.
- Maintain an energy ledger: initial energy = remaining energy + energy consumed, within numerical tolerance.

**Calibration tests:** For fixed bits, TX energy increases with distance; RX does not depend on distance; crossing `d0` uses the correct branch; all energies remain nonnegative.

### 4.4 Link quality model

The existence of an edge does **not** guarantee successful packet transmission.

For every potential edge, use a repeatable base reliability based on normalized distance. Example:

```text
p_base(d) = clip(0.98 - 0.20 * (d / communication_range_m)^2,
                 0.60, 0.99)
```

Assign an additional seeded per-link offset in `[-0.08, +0.08]`. Refresh a small seeded time-varying offset every `link_refresh_rounds`; clamp the final `p_success` to `[0.50, 0.99]`.

On each attempted hop, draw a Bernoulli success/failure using a deterministic **keyed** pseudo-random value (scenario seed + round + offered packet ID + sender + receiver + attempt index). This avoids one algorithm shifting another algorithm's random-number sequence just because it selects a different path. This is an **abstract reliability model**, not a radio propagation simulator.

The router can **observe current estimated link reliability**, but cannot observe future random successes before choosing an action.

### 4.5 Traffic, congestion, and packet lifecycle

Implement a discrete-round simulation:

1. A seeded scenario determines offered packet events for **each original sensor**, independent of routing algorithm. Default = one offered packet/sensor/round. Support an optional `bursty` traffic profile.
2. If an offered source is dead or disconnected, the event remains counted in **offered load** but cannot be delivered. This avoids improving PDR merely by letting dead sensors stop generating data.
3. Shuffle source-processing order deterministically within each round using a scenario-derived seed.
4. Each live source attempts to forward its packet hop by hop until it reaches the sink, fails, or hits a safety limit.
5. Every non-sink transmitting node may attempt at most `tx_capacity_per_round` hops each round (including its own traffic and relaying). Reset usage counters at the start of each round.
6. Define the **observable load** of a node as `used_tx_attempts_this_round / tx_capacity_per_round` (clamped to `[0, 1]`). This is a service-capacity proxy, **not** a simulated queue. Do not label it “queue length.”
7. If a sender has exhausted its per-round capacity, drop the packet with reason `congestion`. No unmodelled buffer or retry is allowed.
8. At each hop, reject dead neighbors, neighbors not in radio range, previously visited nodes, and nodes violating explicit safety constraints. Enforce both a visited-node set and `max_hops`.
9. Record outcome reason: `delivered`, `source_dead`, `no_route`, `link_failure`, `energy_exhausted`, `congestion`, `loop_prevented`, or `max_hops` (and other documented precise reasons if needed).
10. Delay in v1 is **hop count** and/or defined abstract units per hop, not real milliseconds. Never label it latency in ms unless you implement a validated time model.

For graph construction, exclude dead nodes from traversable paths. When node death occurs during a packet, that packet fails appropriately, and later packets see the updated network. Prevent double charging for the same attempted hop.

---

## 5. Router interface and algorithms

### 5.1 Required router contract

All routers must expose an equivalent method such as:

```python
select_next_hop(
    current_node_id,
    sink_id,
    visited_node_ids,
    network_view,
    packet_context,
) -> node_id | None
```

`network_view` exposes **only current/observable** positions, remaining energies, alive statuses, current link estimates, current-round utilization, and neighborhood information. Keep decisions independent of hidden future events or random success draws. `None` means no route/eligible next hop.

The engine, not the router, handles energy charging, stochastic transmission, drop causes, packet delivery, metrics, and node deaths. The routing policy must not mutate network state directly.

### 5.2 Algorithm A: shortest-path baseline

Compute a shortest path from current node to sink on the **currently alive** graph, using hop count as edge cost (NetworkX BFS/shortest path). Choose the first next hop. Handle unreachable sink gracefully. Recompute when topology changes or when visited-node constraints invalidate a previously computed route. Never forward to an already visited node.

### 5.3 Algorithm B: energy-aware heuristic baseline

Use weighted shortest-path routing over the currently alive graph. A positive edge weight may include:

```text
cost(u, v) = 1.0
           + 1.0 * (estimated_E_tx(u,v) + estimated_E_rx(v)) / E_ref
           + 1.2 * (1 - p_success(u,v))
           + 1.5 * (1 - residual_energy_fraction(v))
           + 0.5 * utilization(v)
           + 0.2 * (1 - normalized_alive_degree(v))
```

- Set `E_ref = packet_bits * E_elec` and document its units.
- For sink `v`, energy depletion, congestion, and receiver energy penalties should be zero.
- All edge weights must remain nonnegative.
- This is a **tunable heuristic**; record exact coefficients in config and results.
- Recompute weights as energy and traffic change (do not cache stale weighted paths).

The heuristic should be useful even if Q-Learning performs poorly. It is an important fair baseline.

### 5.4 Algorithm C: tabular Q-Learning (required AI)

**Do not attempt a Q-table keyed by raw continuous sensor states and arbitrary node IDs across random topologies**; this creates sparse, non-transferable tables. Use a **shared, topology-transferable ranked-candidate action formulation**:

**Candidate construction:**

1. List eligible currently alive, in-range, unvisited neighbors.
2. For each neighbor, calculate a deterministic `candidate_score` using: residual energy fraction (positive), current link quality (positive), progress toward sink (positive, may be negative for a necessary detour), current utilization (negative), normalized alive-neighbor density (small positive), and normalized TX cost (negative).
3. Sort by score descending, breaking ties by stable node ID. Retain up to `K=5` candidate slots. Ensure the sink, when directly reachable and eligible, is included even if it would otherwise fall outside the top K.
4. Actions are **slot indices** `0..K-1`, not permanent physical node IDs. The same action means “choose the candidate in this rank position,” enabling use on unseen topologies. Apply an action mask for missing slots.
5. Do not require monotonic progress toward the sink, since detours can be necessary. Loop prevention and hop limits provide safety.

Example ranking score (configurable; keep a single implementation used in training and evaluation):

```text
score(v) = +0.30 * neighbor_energy_fraction
           +0.25 * link_success_probability
           +0.35 * clip(progress_to_sink / radio_range, -1, 1)
           -0.20 * neighbor_utilization
           +0.05 * neighbor_normalized_density
           -0.10 * (estimated_E_tx / E_ref)
```

For the unlimited-energy sink, define its energy fraction as `1`, utilization as `0`, and density term as `1` rather than dividing by undefined quantities.

**Observation/state:** Discretize into low/medium/high bins (explicit configurable boundaries) to create a bounded table. Required state dimensions:

- Current node residual energy fraction.
- Current node distance to sink, normalized by field diagonal.
- Current node utilization.
- Current node alive-neighbor density.
- Best eligible neighbor link quality (or sentinel if none).
- Best eligible neighbor energy fraction (or sentinel if none).

`state = (energy_bin, sink_distance_bin, load_bin, degree_bin, best_link_bin, best_neighbor_energy_bin)`.

State is recomputed at each hop **after** candidate eligibility is determined. Document that this is an approximate local observation and that candidate ranking supplies per-neighbor features the coarse state cannot represent. Avoid claiming this is a globally optimal or fully observable Markov model.

**Action selection:** Epsilon-greedy among **currently valid ranked slots only**. Random tie-breaking must use the agent's seeded RNG. During evaluation, `epsilon=0` and **no learning or updates**. Define a deterministic tie-break rule for evaluation.

**Q-update:** After a hop outcome or terminal drop:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a' valid} Q(s',a') - Q(s,a)]
```

If the packet is terminal (delivered or dropped), use bootstrap value `0`. Do not take `max()` over invalid action slots. The next state `s'` corresponds to the next-hop node only if transmission succeeds and the packet continues.

Suggested defaults: `alpha=0.15`, `gamma=0.95`, `epsilon_start=1.0`, `epsilon_end=0.05`, `K=5`, linear epsilon decay across training interactions. Initialize unseen Q-values at zero.

**Reward per attempt (starting proposal, tune using training data only):**

```text
r = -0.05                                 # hop/attempt overhead
    -0.15 * (charged_tx_plus_rx / E_ref)  # energy cost
    -0.30 * neighbor_utilization          # congestion pressure
    -0.40 * (1 - neighbor_energy_frac)    # avoid battery depletion
    +0.50 * clip(progress / radio_range, -1, 1)
    +10.0 if delivered_to_sink
    -10.0 if packet_terminated_in_failure
    -2.0 for each node newly exhausted on this attempt
```

Make the implementation unambiguous about reward accumulation: apply the base terms **once per attempted hop**, then any terminal bonus/penalty. Treat sink energy fraction as `1`. An invalid/no-candidate action produces a terminal failure with penalty, not an invented next state.

Keep a training history: episode, packets offered/delivered, mean reward, epsilon, alive nodes, and Q-table size. Save Q-table, state-bin definitions, candidate-score weights, reward weights, config fingerprint, training seed list, and training version in a portable JSON model artifact. Reject incompatible model/config formats with a useful message.

**Important training/evaluation distinction:** Train a **single shared ranked-slot policy** on multiple training topologies/seeds; do not train or re-tune it on the held-out evaluation topologies. This supports a meaningful out-of-sample comparison. During evaluation, every algorithm starts with the same fresh initial energy and receives the same offered traffic per scenario.

---

## 6. Experiment design and statistical evaluation

### 6.1 Train/evaluation isolation

- Define disjoint deterministic seed sets, for example training seeds `1000..1039` and evaluation seeds `2000..2009`.
- Each Q-Learning training episode resets the network (initial energy, capacity counters, connectivity, random events); seeds/topologies vary across episodes.
- A training episode is a bounded run using the same simulator as evaluation. Start with `40` episodes of up to `400` rounds, configurable.
- During evaluation, load the saved model, set epsilon to zero, and disable Q updates. Never silently train on evaluation data.
- Tune coefficients/rewards and hyperparameters with training/validation seeds, **not** the final held-out seeds.
- Within each evaluation scenario, reuse initial placement, initial energy, link offsets, link-quality evolution, offered event schedule, and keyed stochastic draws across algorithms.
- Preserve identical simulation rules for all three policies. Track policy decision overhead separately.

### 6.2 Scenarios

The default `compare` command should support:

- **Smoke:** 1 seed, 25 nodes, 50–100 rounds; quick correctness/demo.
- **Standard:** 50 nodes, 10 evaluation seeds, 2,000 maximum rounds.
- **Scalability:** 25 / 50 / 100 nodes, paired seeds and otherwise matching parameters.
- **Stress:** bursty traffic and/or degraded link quality, plus different sink positions.

Allow `compare` to configure a subset to manage runtime. Clearly distinguish smoke results from scientifically meaningful multi-seed results.

### 6.3 Mandatory metrics

Record each round and each evaluation run:

| Metric | Definition |
|---|---|
| Alive nodes | Count of sensor nodes with energy > 0 |
| Mean residual energy (J) | Average remaining battery across all original sensors (dead nodes contribute 0) |
| Energy used (J) | Initial total sensor energy minus remaining total sensor energy |
| Offered packets | All external sensor traffic events, including events from now-dead sensors |
| Delivered packets | Offered packets successfully received at sink |
| Offered-load PDR (%) | `100 * delivered / offered`; label explicitly |
| Active-source PDR (%) | `100 * delivered / packets_from_alive_sources` when denominator > 0; supplementary |
| Packet hop count | Actual successful/attempted hops per offered packet as clearly documented |
| Mean delivered-packet hop count | Mean hops for **delivered** packets only; `N/A` if none |
| FND | First round any sensor reaches 0 J |
| HND | First round at least half of sensors have died |
| LND | First round all original sensors have died |
| Alive-to-sink reachability (%) | Fraction of original sensor IDs currently alive and graph-connected to sink |
| Time to first disconnection | First round a previously sink-reachable alive sensor loses graph connectivity (optional but recommended) |
| Throughput | Delivered packets per simulation round |
| Congestion/link/energy drops | Counts broken down by explicit drop cause |
| Policy compute time | Wall time spent making routing decisions (report separately from simulated energy) |

If an event (FND/HND/LND) has not occurred when the run stops, save **`null` and a `censored=true` indicator**; never pretend it happened at the final round. Report the fraction of runs where each event was observed. Keep round-based units clear: there is no justified conversion to seconds in v1.

### 6.4 Fair comparison and results

- Compare **paired scenarios**: same evaluation seed and network settings for each algorithm.
- Report per-seed results, mean, standard deviation, and 95% confidence intervals (bootstrap over paired seeds is acceptable; label method and sample size).
- Where useful, report paired differences (`Q-Learning - heuristic` and `Q-Learning - shortest path`) rather than comparing unpaired means.
- For censored lifetime metrics, show event observation rate and avoid naive average of observed-only events as though uncensored.
- Generate at least: alive nodes vs round, residual energy vs round, offered-load PDR vs round (and cumulative PDR), delivered packets vs round, drop reason chart, and per-algorithm metric comparison. Distinguish individual-seed traces from multi-seed aggregates.
- Never generate an example plot from fabricated measurements in an output folder that resembles experimental results. If placeholders are needed in UI development, mark them **DEMO / SYNTHETIC** and remove from final experiment reports.
- Do not guarantee Q-Learning will win. Discuss possible reasons if it does not.

---

## 7. Dashboard: functional requirements

Build `dashboard/app.py` with a clean engineering/research dashboard and **real data**.

**Layout / functionality:**

1. **Overview:** title, experiment description, selected algorithm/seed, current alive count, energy remaining, offered-load PDR, delivered packets, FND/HND/LND status.
2. **Network topology:** interactive or clear static 2D node plot using actual saved coordinates. Colors: healthy / low-energy / dead; distinct sink marker; optional edges and currently selected packet path. Ensure any displayed path comes from a real recorded trace.
3. **Simulation controls:** algorithm selection; number of nodes; seed; radio range; initial energy; traffic mode; run length. Validate input and warn if requested topology cannot be connected.
4. **Run:** launch a small simulation with a button and visible progress; save resulting artifacts. Do not silently rerun expensive simulations on every Streamlit rerender.
5. **Comparison:** select an existing comparison run and show side-by-side real metrics, time-series charts, and honest discussion of trade-offs. Show not-reached lifetime events as `Not reached` rather than `0` or the run limit.
6. **RL model:** display whether a model is loaded, its training configuration and policy status (`trained/frozen`), and training history if available. Do not imply a model exists until one is actually trained.
7. **Export:** download relevant metrics CSV and summary JSON from the current real run. Provide meaningful error/empty states when there are no results.

Use Streamlit Session State for user selections/current runs as needed. Keep the app usable on a laptop screen. Use readable visual hierarchy; no fake futuristic filler, unsupported claims, or unexplained jargon. For a huge run, encourage CLI-based experiment generation and dashboard loading rather than blocking the UI.

---

## 8. CLI and artifact contracts

Commands are required to be implemented through `python -m wsn.cli`.

| Command | Purpose |
|---|---|
| `simulate` | Run one specified algorithm/seed and export trace + metrics |
| `train` | Train Q-Learning and export a versioned model and learning curves |
| `compare` | Evaluate all three algorithms on paired held-out seeds and export aggregate results |
| `validate-config` | Parse config, check units/constraints, print resolved settings (recommended) |

Example results layout:

```text
results/
├── runs/<run_id>/
│   ├── resolved_config.yaml
│   ├── scenario.json
│   ├── nodes_initial.csv
│   ├── round_metrics.csv
│   ├── summary.json
│   ├── sampled_packet_traces.jsonl
│   └── topology.png
└── comparisons/<comparison_id>/
    ├── experiment_config.yaml
    ├── per_run.csv
    ├── aggregate.csv
    ├── summary.md
    └── plots/
        ├── alive_nodes.png
        ├── residual_energy.png
        ├── packet_delivery.png
        └── drop_reasons.png
models/
├── q_table.json
└── training_history.csv
```

Minimum `round_metrics.csv` columns: `round, algorithm, seed, alive_nodes, dead_nodes, mean_remaining_energy_j, total_remaining_energy_j, cumulative_energy_used_j, offered_packets_round, delivered_packets_round, offered_packets_cumulative, delivered_packets_cumulative, cumulative_offered_pdr, reachable_alive_fraction, link_failure_drops, congestion_drops, energy_drops, no_route_drops`.

`summary.json` must include algorithm, seed, config/version fingerprint, rounds actually executed, packet counts, PDR definitions, metrics, censored lifetime events, units, and timestamp. Use stable schemas and explicitly document them in `docs/methodology.md`. Packet traces may be sampled to limit file size, but aggregate accounting must be complete.

**Config rules:** Use YAML, validate positive sizes and ranges (`0 < link_probability <= 1`, `0 < gamma < 1`, `0 < alpha <= 1`, etc.), record all defaults, and fail clearly on unknown algorithm names or missing model paths. Do not change physics across algorithms.

---

## 9. Suggested configuration schema

Create a real `configs/default.yaml` matching the implemented validation model. Suggested fields:

```yaml
scenario:
  node_count: 50
  area_width_m: 100.0
  area_height_m: 100.0
  sink_x_m: 50.0
  sink_y_m: 50.0
  communication_range_m: 30.0
  initial_energy_j: 0.2
  packet_bytes: 512
  max_rounds: 2000
  max_hops_multiplier: 2
  topology_require_connected: true
  topology_max_tries: 100
  seed: 2000

radio:
  e_elec_j_per_bit: 5.0e-8
  epsilon_fs_j_per_bit_m2: 1.0e-11
  epsilon_mp_j_per_bit_m4: 1.3e-15
  base_link_quality_intercept: 0.98
  base_link_quality_distance_coefficient: 0.20
  link_offset_amplitude: 0.08
  link_temporal_amplitude: 0.05
  link_refresh_rounds: 25
  min_success_probability: 0.50
  max_success_probability: 0.99

traffic:
  profile: uniform
  offered_packets_per_node_per_round: 1
  tx_capacity_per_round: 12
  link_retries: 0

routing:
  rl_candidate_limit: 5
  density_reference: 10
  # Put heuristic and RL candidate ranking weights here.

q_learning:
  alpha: 0.15
  gamma: 0.95
  epsilon_start: 1.0
  epsilon_end: 0.05
  train_episodes: 40
  train_max_rounds: 400
  training_seed_start: 1000
  # Put state-bin boundaries and reward coefficients here.

evaluation:
  seeds: [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
  algorithms: [shortest_path, energy_aware, q_learning]
```

Keep implemented parameters centralized; no unexplained magic numbers scattered through code. `configs/smoke.yaml` should finish quickly with deterministic settings and enough rounds to exercise packet delivery and some packet failures.

---

## 10. Implementation order and milestones

**Milestone 1 — Scaffold and correctness foundations**

- Package imports, configuration and CLI help work.
- Sensor/sink/link/packet models and topology generator implemented.
- Deterministic seeded scenario and initial connected topology saved.
- Tests validate graph symmetry, distance/range, units, and config.

**Milestone 2 — Physics and working simulation**

- First-order radio model, probabilistic links, round processing and fixed offered traffic.
- Real packet delivery/drop reasons, node death, route constraints, logging and metrics.
- A minimal router is sufficient to prove end-to-end packet delivery.
- Tests cover energy accounting, dead nodes, congestion, cycles, and determinism.

**Milestone 3 — Two honest baselines**

- Functional shortest-path and weighted energy-aware routers.
- Same scenario and packet engine for both.
- Export plots and CSV from actual simulation data.

**Milestone 4 — Q-Learning**

- Candidate-ranked action formulation, discretization, valid action masking, epsilon-greedy, TD updates, training loop.
- Checkpoint/JSON model load-save, training history, truly frozen evaluation policy.
- Tests confirm Q updates, terminal bootstrap, invalid action masking, and model roundtrip.

**Milestone 5 — Comparison / science**

- Paired seeds, disjoint train/evaluation data, censoring, statistical summaries, CLI comparison and written output.
- Stress and scalability configurations; no performance claims without data.

**Milestone 6 — Dashboard, polish and handoff**

- Dashboard loads the same real artifacts and offers useful controls.
- Run end-to-end smoke workflow, test suite, formatting/lint checks, and readme command validation.
- Provide limitations, next steps, and architecture/methodology docs.

**Priority:** A working, measured version of milestones 1–5 is more valuable than a polished dashboard built on nonfunctional simulation.

---

## 11. Required automated tests / acceptance criteria

The coding agent must implement tests that prove at least the following:

### Physics and simulation

- `E_tx(k, d)` is positive and increases with distance; `E_rx(k)` is correct and independent of distance.
- Radio model uses correct piecewise threshold at `d0` and correct bits/bytes conversion.
- Neither a node's energy nor a network's total energy ever becomes negative.
- `initial_total_energy ≈ remaining_total_energy + consumed_total_energy` within a sensible float tolerance.
- A dead sensor cannot transmit, receive, or serve as a relay; the sink behaves according to its unlimited-energy assumption.
- No routing attempt targets a node beyond communication range.
- A failed attempted hop still consumes TX and RX energy as specified, unless the attempt is rejected **before** transmission for insufficient energy/capacity.
- Packet outcomes are mutually exclusive; `delivered + dropped = offered` at the end of each completed round.
- Packet paths terminate: visited-node checks and hop limit prevent infinite loops.
- If the network has no path, packets fail with a recorded reason instead of crashing.
- Congestion behavior changes with configured capacity and is not a decorative field.

### Algorithms and RL

- The shortest-path algorithm selects a valid hop on a known tiny topology.
- The heuristic responds sensibly to low-energy or unreliable alternative routes.
- RL candidates are sorted deterministically and have valid masks.
- Q update matches a hand-calculated numeric example.
- Terminal Q updates have no bootstrap; evaluation never changes any Q-value.
- Saving and reloading a model reproduces its deterministic greedy actions on a fixed observation.
- Candidate ranking demonstrably responds to **all five** promised factors (energy, density, quality, traffic, distance).

### Experiments and product

- Same scenario/seed creates identical initial positions, offered traffic, and link draws for all algorithms.
- Re-running the same seeded algorithm/config/model yields identical scientific outputs (exclude runtime measurements and timestamps).
- Evaluation seed set is disjoint from training seeds; no Q-table update during comparison.
- Censored FND/HND/LND remain null, never replaced by max rounds.
- CSV/JSON schema parses; dashboard graphs reflect stored metrics, not hard-coded sample numbers.
- `pytest` passes and all documented CLI commands are runnable.
- A fresh user can follow README setup and complete the smoke workflow.

**Useful edge tests:** 1-node network near sink, permanently disconnected node with connectivity requirement disabled, no available neighbor, exactly-at-range link, just-below-zero energy budget, one congested relay, equal-scored candidates, empty model, incompatible model metadata, zero delivered packets, and topology-generation failure.

---

## 12. Reporting and scientific integrity

In `docs/methodology.md`, explicitly document:

1. Simulation assumptions, units, radio model, link model, traffic model, and exclusions (no real MAC/PHY, no idle/sensing energy).
2. Mathematical equations for radio energy and Q update, state discretization, action ranking, reward function.
3. Baseline definitions, topology and traffic generation, random-seed pairing, training/validation/evaluation separation.
4. Metric definitions, event censoring, statistical method and number of seeds.
5. Experimental observations **as obtained from actual executions**; avoid unsupported claims or cherry-picked results.
6. Limitations: simplistic wireless success model, simplified congestion, homogeneous fixed energy, static topology, discrete-round time, tabular RL approximations, training/compute cost omitted from sensor battery model.
7. Suggested extension: clustering and rotating cluster-head selection, richer traffic, mobility, alternative link models, DQN, and NS-3 validation.

**Recommended final report plots:** topology snapshots at early/middle/late rounds; survival curves; cumulative offered-load delivery; energy trajectories; delivery vs lifetime trade-off; drop breakdown; training reward curve; and paired-metric comparisons with uncertainty.

---

## 13. Optional follow-on features (do not block core delivery)

### A. Cluster-head routing extension

Add LEACH-inspired cluster formation and periodically rotate a cluster head based on energy, neighbor coverage, distance to sink, and traffic. Ordinary nodes report to their CH; CHs forward/aggregate toward the sink. **If aggregation is added, change packet accounting and energy costs explicitly:** an aggregated packet does not represent the same workload as one raw sensor reading. Compare against a correctly implemented clustering baseline before claiming novelty.

### B. Deep reinforcement learning

Only after the tabular agent works: replace ranked-slot Q-values with a compact DQN that accepts numerical observations plus padded candidate features and action masks. Evaluate on the same held-out scenarios, report training overhead, and verify there is a real reason for added complexity.

### C. NS-3 validation

Reproduce selected scenarios in a network simulator with detailed radio/channel behavior. Treat this as a separate validation study, not a dependency of the Python application.

---

## 14. External technical references

These are starting references for API conventions and the simplified radio model. Use them to guide implementation, not to copy code blindly.

- **NetworkX shortest paths:** https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
- **NetworkX `shortest_path` API:** https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html
- **Gymnasium custom environments (optional):** https://gymnasium.farama.org/main/tutorials/environment_creation/
- **Streamlit Session State:** https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- **Simplified first-order WSN radio-model example:** https://www.mdpi.com/1424-8220/19/13/3017
- **Energy-aware routing / clustering research context:** https://www.cs.purdue.edu/homes/fahmy/papers/tmc04.pdf

This README is the **engineering specification**, not an experimental result. Validate every finding with the implementation and record departures from assumptions.

---

## 15. Final implementation-agent handoff prompt

> Implement this repository from the specification in `README.md`. Build the actual physics-backed WSN simulator, the shortest-path and energy-aware baselines, the ranked-candidate Q-Learning policy with real training and frozen evaluation, the experiment runner, real output artifacts, and the Streamlit dashboard. Follow the milestone order. Do not stop at scaffolding, do not fabricate plots/metrics, and do not claim improvement unless measured. Run tests and the documented smoke workflow; fix issues you can verify. If a feature is genuinely blocked, leave a precise documented limitation, not a misleading stub. Finish by summarizing files changed, commands used, measured results (if any), tests passed, and remaining limitations.
