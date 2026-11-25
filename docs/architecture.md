# Architecture

```text
configs/*.yaml ──> config.py (typed, validated dataclasses)
                      │
                      ▼
            network.build_scenario(config, seed)
            ├─ topology.py      seeded placement, potential links, reachability
            ├─ radio.py         first-order energy model + keyed link reliability
            └─ traffic.py       fixed offered-event schedule per round
                      │  (one Scenario is shared by every algorithm on that seed)
                      ▼
            simulator.Simulation(scenario, router)
            ├─ NetworkState     energies, deaths, per-round TX usage (engine-owned)
            ├─ NetworkView      read-only observations handed to routers
            └─ per round: offered packets → hop-by-hop forwarding → metrics
                      │
          ┌───────────┼──────────────────────────┐
          ▼           ▼                          ▼
 routing/shortest_path, energy_aware, clustering   routing/q_learning, dqn (+ slot_policy, rl/*)
                      │
                      ▼
 metrics.py (RoundMetrics, RunSummary, bootstrap CI) ─> reporting.py (CSV/JSON/PNG/MD)
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
  cli.py         rl/trainer.py   experiments.py (paired comparison, aggregates)
                      │
                      ▼
  rl/serialization.py (portable JSON model)

  artifacts.py (typed readers) ─> wsn/dashboard (Streamlit pages) <─ dashboard/app.py launcher

  ns3/*.cc (C++, built inside ns-3) ─> CSV ─> ns3_validation.py ─> results/ns3
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | YAML → frozen dataclasses; rejects unknown keys and out-of-range values; config fingerprint |
| `models.py` | Shared value types: packet outcomes, packet context, hop feedback, traces |
| `random_utils.py` | Keyed uniform draws (hash of seed + round + packet + link) and named NumPy streams |
| `topology.py` | Seeded uniform placement, retry until connected (logged), BFS reachability |
| `radio.py` | `E_tx`, `E_rx`, `d0`; per-link reliability = distance base + fixed offset + epoch offset |
| `traffic.py` | Offered packets per round, deterministic processing order, bursty profile |
| `network.py` | `Scenario`, engine-owned `NetworkState`, router-facing read-only `NetworkView` |
| `simulator.py` | Round loop, capacity, energy charging, keyed Bernoulli link draws, drop causes, early stop |
| `routing/base.py` | `Router.select_next_hop` contract; `FeedbackRouter` for learning routers |
| `routing/shortest_path.py` | Hop-count BFS from the sink over alive, unvisited nodes |
| `routing/energy_aware.py` | Dijkstra from the sink with the weighted energy/reliability/load cost |
| `routing/slot_policy.py` | Shared acting/learning loop for ranked-slot policies: ranking, epsilon-greedy, reward, bootstrapping, frozen evaluation |
| `routing/q_learning.py` | Tabular Q-table and TD update on top of the slot policy |
| `routing/dqn.py` | DQN router: feature observation, network Q-values, replay-based learning |
| `routing/clustering.py` | LEACH and adaptive cluster-head election, member-to-head hop, head backbone routing |
| `rl/dqn.py` | NumPy MLP, Huber loss, Adam, replay buffer, target network, masked bootstrap |
| `rl/state.py` | Candidate scoring and ranking (K slots), discretised 6-dimensional state |
| `rl/reward.py` | Per-attempt reward with terminal bonus/penalty |
| `rl/trainer.py` | One shared policy trained across training seeds; history records |
| `rl/serialization.py` | Versioned JSON models (Q-table and DQN); compatibility and seed-leak checks |
| `experiments.py` | Experiment plans (scenario variants), paired runs, aggregates, paired differences |
| `reporting.py` | Writes run/comparison/training artifacts and plots |
| `visualization.py` | Matplotlib figures from recorded data |
| `artifacts.py` | Typed loaders for saved runs and comparisons |
| `ns3_validation.py` | ns-3 CSV parsing, field export, link and field comparisons, report |
| `dashboard/` | Streamlit app: `app.py` shell and navigation, `views/` pages, `charts.py` and `replay.py` SVG builders, `insights.py` findings computed from stored numbers, `theme.py` tokens and stylesheet |
| `cli.py` | `validate-config`, `simulate`, `train`, `compare` |

## Invariants enforced by the engine

- Routers never mutate state; they receive a `NetworkView` and return a node id or `None`.
- The engine rejects (raises `RoutingContractError`) a chosen hop that is out of range or dead,
  and drops a packet as `loop_prevented` if the router picks a visited node.
- Every offered packet ends with exactly one outcome; `delivered + Σ drops = offered` per round.
- Energy never goes negative; `initial = remaining + consumed` (ledger) holds.
- Every random draw that affects outcomes is keyed by scenario seed and event identity, so all
  algorithms on one seed see the same topology, traffic, link conditions and per-attempt draws.
