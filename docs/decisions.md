# Design decisions and departures from the specification

Each entry records what was decided, why, and where it shows up in code or tests.

## 1. Sink identifier is the integer `N`, written as `"sink"` in outputs

Sensors are `0..N-1`; the sink is stored as the last position with internal id `N`. Keeping
every id an integer makes the routing, BFS/Dijkstra and Q-Learning code simpler and fully typed.
All written artifacts (`nodes_initial.csv`, `scenario.json`, packet traces) use the label
`"sink"`, as the specification suggests. (`topology.py`, `reporting.py`)

## 2. Hand-written BFS and Dijkstra instead of NetworkX calls per decision

The specification suggests NetworkX shortest paths. Routers recompute routes on **every**
decision (so deaths, energy and load are never stale) while excluding the packet's visited nodes.
Building a NetworkX subgraph view per decision was the dominant cost, so the routers run a BFS
(shortest path) or Dijkstra (energy-aware) directly over the topology's adjacency lists.
NetworkX is still used to build and check the topology graph, and
`test_shortest_path_hop_count_matches_networkx` checks the hand-written BFS against
`networkx.single_source_shortest_path_length`.

## 3. Energy-aware costs are computed from the sink outward

Edge cost `cost(u -> v)` depends on the receiver `v`. A single Dijkstra from the sink over
reversed edges gives every node's cost-to-sink; the router then picks the neighbor minimising
`cost(current -> v) + cost_to_sink(v)`. This is exact for the configured cost and avoids one
Dijkstra per candidate. The receiver part of the cost is computed once per node per decision.

## 4. Early stop with network-down accounting

A run stops when **no alive sensor can reach the sink** (no packet can be delivered any more and
baseline routers would spend no energy). Stopping early would otherwise let a run that dies
sooner report a higher offered-load PDR, because its denominator would be smaller. To keep runs
comparable, every remaining round up to `max_rounds` is still accounted for: its scheduled
packets are added to offered load and recorded with drop reason `network_down`, and
`round_metrics.csv` contains those rounds with `simulated = false`. Lifetime events that did not
happen before the stop stay `null` / censored. (`Simulation._network_down_rounds`)

Consequence: with the default 2,000-round horizon, offered-load PDR is small because most of
the horizon is network-down time. `operational_pdr_pct` (delivered / offered during simulated
rounds) and `delivered_packets` are reported alongside it.

## 5. Epsilon decays per episode, not per interaction

The specification suggests linear decay across training interactions. The number of interactions
per episode is not known in advance (episodes stop when the network disconnects), so epsilon
decays linearly per episode from `epsilon_start` (first episode) to `epsilon_end` (last).
(`rl/trainer.py::epsilon_for_episode`)

## 6. Reward and ranking weights are stored as magnitudes

Config values for candidate-ranking and reward weights are nonnegative magnitudes; the sign of
each term is fixed in code and documented in the formula docstrings and `configs/default.yaml`.
This lets config validation reject negative values instead of silently flipping a term.

## 7. When the receiver cannot afford reception

If the receiver cannot pay its receive energy, the sender has already transmitted (it is
charged TX), the receiver's remaining energy is drained to zero (it dies) and the packet is
dropped as `energy_exhausted`. If the **sender** cannot afford TX, nothing is transmitted, the
sender's remaining energy is drained (it dies), the receiver is not charged, and the packet is
dropped as `energy_exhausted`. Drained remainders count as consumed energy so the ledger holds.

## 8. A relay that dies on reception cannot forward

If a node's energy reaches exactly zero while receiving a packet, the packet was decoded but the
node is dead before its next decision, so the packet is dropped as `energy_exhausted`.

## 9. Q-Learning failure penalty for packets that fail before the next attempt

A transition that succeeded (packet moved to the next node) normally bootstraps from the next
node's state. If the packet then fails *before* another action is taken (no candidate at the next
node, the next node is congested or dead, or the hop limit is reached), the pending transition is
closed as terminal with the failure penalty. (`QLearningRouter.on_packet_end`)

## 10. Router feedback hooks live on a separate base class

`Router` only defines `select_next_hop`. `FeedbackRouter` adds `on_hop_result` and
`on_packet_end`; the engine calls them only for routers that learn. Baselines carry no empty
hook methods.

## 11. Bursty traffic model

`bursty`: each sensor independently bursts with probability `burst_probability` per round,
multiplying that round's packets by `burst_multiplier`. Bursts are drawn from the keyed random
stream, so they are identical for every algorithm on the same seed.

## 12. Link retries fixed at zero

`traffic.link_retries` exists in the config schema but validation requires `0`; retries are not
modelled in v1 (one attempt per hop, as the specification states).

## 13. Dashboard reads saved artifacts only

The dashboard runs simulations through the same `run_simulation` API as the CLI, saves the
artifacts, and then renders from the saved files (`wsn.artifacts`, `wsn.dashboard.data`).
Every chart and every sentence that names a better strategy is computed from stored numbers
(`wsn.dashboard.insights`); tests check that a different result produces a different sentence.

## 14. Dashboard charts and replay are custom SVG, mounted by script

Streamlit's chart widgets cannot draw the replay (two synchronised sensor fields with moving
packets) or give the line charts per-series toggles and a shared hover readout, so the
dashboard renders its own SVG through `st.html` with JavaScript enabled. Streamlit sanitises
that HTML with a profile that strips `<svg>` and drops any element whose text looks like
markup. `wsn.dashboard.components.mount_svgs` therefore replaces each SVG with an empty slot
and adds a script that inserts the SVG after sanitisation; all embedded JSON escapes `<` as
`<`. The SVG is built from escaped text only; no user input reaches it unescaped. The
sanitiser itself is left on.

## 15. One acting/learning loop for both learned strategies

`routing/slot_policy.py` holds candidate ranking, epsilon-greedy over valid slots, reward,
deferred bootstrapping and frozen evaluation. Tabular Q-Learning and the DQN only implement
`observe`, `q_values` and `learn`, so both are guaranteed to make decisions and receive rewards
identically.

## 16. DQN in NumPy instead of PyTorch

The specification suggests PyTorch for the optional DQN. The network is small (39 → 64 → 64 → 5),
so a NumPy implementation (He initialisation, ReLU, Huber loss, Adam, replay, target network)
keeps the install light, deterministic from one seed and fully type-checked. Its gradients are
verified against finite differences in `tests/test_dqn.py`.

## 17. Cluster routing without data aggregation

The specification allows aggregation only with explicit changes to packet accounting. To keep
every strategy's offered load, delivered-packet meaning and energy costs identical, cluster
heads forward each member reading individually. LEACH is implemented as the clustering baseline
the specification asks for, and the adaptive rotation is compared against it.

## 18. ns-3 built natively on Windows with MinGW

WSL and Docker were unavailable (hardware virtualisation disabled), so ns-3.49 was built with
CMake, Ninja and MinGW-w64 g++ 15.2, enabling only the modules the validation needs. The two
validation programs live in `ns3/` and are copied into ns-3's `scratch/` folder to build. The
Python package never depends on ns-3: `ns3-validate` analyses the CSV files the programs write.

## 19. Not implemented

A Gymnasium wrapper was not added: training drives the simulator directly through the router
interface, so a `reset`/`step` wrapper would duplicate it without a consumer.
