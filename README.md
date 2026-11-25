# AI-Driven Energy-Aware Routing and Lifetime Optimization in Wireless Sensor Networks

A reproducible Python simulator of a wireless sensor network (WSN) that compares six routing
strategies on the same physics, the same random fields and the same traffic:

| Strategy | Kind | Idea |
|---|---|---|
| Shortest path | Baseline | Fewest hops to the sink over the currently alive graph |
| Energy-aware | Heuristic | Weighted shortest path: transmit/receive energy, link reliability, relay battery, relay load, density |
| LEACH clusters | Clustering baseline | Sensors take random turns as cluster heads; members report through their head |
| Adaptive clusters | Clustering | Heads chosen by battery, coverage, distance to sink and recent load |
| Q-Learning | Learned (tabular) | Picks one of the top-K ranked neighbors from a table of discretised local states |
| Deep Q-network | Learned (neural) | Same choice, from a small neural network over per-neighbor features |

Both learned strategies train on their own fields and are evaluated frozen on fields they never
saw. An optional ns-3 study checks the simulator's radio assumptions against a detailed
IEEE 802.15.4 model. Results are explored in a Streamlit dashboard.

**Headline result** (10 held-out fields, 50 sensors): the energy-aware heuristic keeps the first
sensor alive 2.2× longer than shortest path and delivers the most packets in all 6 distinct test
conditions (the 50-sensor size test repeats the standard field). The deep Q-network comes within 2%
of it on delivery (statistically tied in 3
conditions); tabular Q-Learning and clustering without data aggregation trail the baselines.
Details and caveats: [docs/methodology.md](docs/methodology.md#5-results-measured).

- Specification this project implements: [docs/specification.md](docs/specification.md)
- Code structure: [docs/architecture.md](docs/architecture.md)
- Model, equations, metrics and measured results: [docs/methodology.md](docs/methodology.md)
- Departures from the specification and why: [docs/decisions.md](docs/decisions.md)

> A simulation with simplified assumptions (first-order radio energy, abstract link
> reliability, discrete rounds). It is not an IEEE 802.15.4 PHY/MAC implementation; the ns-3
> study measures how far its link model is from one. Time is in rounds, delay in hops, energy in
> joules.

## Setup

Install Python 3.11 or newer (tested on 3.14, Windows) and Git. Run all commands
from the project root. The Python simulator and dashboard need no GPU, API keys,
external services or ns-3 installation. Dependencies are declared in `pyproject.toml`;
their version ranges are bounded, but there is no dependency lockfile.

Clone the repository:

```bash
git clone https://github.com/rohith0110/WSN.git
cd WSN
```

On Windows PowerShell, use the virtual environment's Python directly, so activation
does not depend on the PowerShell execution policy:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

For the remaining examples, activate the environment with
`.\.venv\Scripts\Activate.ps1`, or replace `python` with
`.\.venv\Scripts\python.exe`. On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop
the server with Ctrl+C. A fresh clone has no saved results or trained models:
**Run your own** works immediately with a baseline strategy; generate the data
below to populate the comparison and learning pages.

## Quick start

```bash
# 1. One simulation (about 5 seconds)
python -m wsn.cli simulate --config configs/default.yaml --algorithm energy_aware --seed 2000 --output results/smoke

# 2. Train both learned strategies on training fields 1000-1039 (about 1 + 3 minutes)
python -m wsn.cli train --algorithm q_learning
python -m wsn.cli train --algorithm dqn

# 3. Compare all six strategies on 10 held-out fields (about 2 minutes)
python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json --scenario standard --output results/comparisons/standard

# 4. Open the dashboard
python -m streamlit run dashboard/app.py
```

The full experiment set behind the documented results adds:

```bash
python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json --scenario stress --output results/comparisons/stress
python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json --scenario scalability --output results/comparisons/scalability
```

A fast end-to-end check uses `configs/smoke.yaml` (25 sensors, 100 rounds):

```bash
python -m wsn.cli train --config configs/smoke.yaml --algorithm q_learning --output models/smoke/q_table.json
python -m wsn.cli train --config configs/smoke.yaml --algorithm dqn --output models/smoke/dqn.json
python -m wsn.cli compare --config configs/smoke.yaml --model models/smoke/q_table.json --dqn-model models/smoke/dqn.json --scenario smoke --output results/comparisons/smoke
```

## Dashboard

`streamlit run dashboard/app.py` opens Routing Lab:

- **Results**: the answer first. Which strategy keeps a field alive longest, the key figures
  behind it, sensors alive round by round, rankings, and the winner under every test condition.
- **Replay**: pick two strategies and a field, press play, and watch both drain the same
  sensors side by side, with recorded packets moving along their real paths.
- **Compare**: one test condition, any measure, every strategy, with paired differences and
  their uncertainty, curves over time, why packets were lost, and CSV downloads.
- **Run your own**: choose the number of sensors, battery, radio range, traffic and strategy;
  the run is saved and replayed on the spot.
- **Learned strategies**: how the two learned policies were trained.
- **Reality check**: the simulator's link model against ns-3 measurements.

Every number, chart and sentence comes from saved runs in `results/` and `models/`; nothing is
computed for show. Pages explain what to run when data is missing.

## CLI reference

Every command has `--help`. Exit codes: `0` success, `1` runtime failure (for example no
connected topology could be generated), `2` invalid input (bad config, unknown algorithm,
missing or incompatible model).

| Command | Purpose |
|---|---|
| `validate-config --config FILE` | Parse, validate and print the resolved configuration |
| `simulate --algorithm NAME [--seed N] [--model FILE] [--dqn-model FILE] [--rounds N] [--output DIR]` | One run; writes trace, metrics and plots |
| `train [--algorithm q_learning\|dqn] [--episodes N] [--output FILE]` | Train a learned policy; writes the model, its training history and curve next to it |
| `compare [--model FILE] [--dqn-model FILE] [--seeds LIST] [--algorithms LIST] [--scenario NAME] [--output DIR]` | Paired evaluation on held-out fields |
| `replot --comparison DIR` | Redraw a comparison's PNG plots from its saved CSVs |
| `ns3-export [--seed N] [--output FILE]` | Export a field (positions and next hops) for the ns-3 replay |
| `ns3-validate --csv FILE [--replay CSV ...] [--output DIR]` | Compare ns-3 measurements with the simulator |

Algorithm names: `shortest_path`, `energy_aware`, `leach`, `cluster_adaptive`, `q_learning`,
`dqn`. Learned models are checked before use: candidate count and ranking weights must match
the config, and evaluation fields must not overlap their training fields.

Named experiment scenarios (`--scenario`, defined in `configs/experiments.yaml`), all with 10
evaluation fields unless `--seeds` narrows them:

| Scenario | What varies |
|---|---|
| `smoke` | 1 field, 25 sensors, 100 rounds (correctness check only) |
| `standard` | 50 sensors, up to 2,000 rounds |
| `scalability` | 25 / 50 / 100 sensors |
| `stress` | bursty traffic, weak radio links, sink on the field edge |

## ns-3 validation (optional)

The simulator never needs ns-3. To reproduce the validation in
[docs/methodology.md](docs/methodology.md#56-ns-3-validation-optional-extension-c):

1. Download ns-3.49 from nsnam.org and build it with the LR-WPAN module, for example
   `cmake -S ns-3.49 -B build -G Ninja -DNS3_ENABLED_MODULES="lr-wpan;energy;propagation;mobility;spectrum;network;core"`
   then `cmake --build build` (Linux, macOS, or Windows with MinGW-w64).
2. Copy `ns3/wsn-link-validation.cc` and `ns3/wsn-scenario-replay.cc` into ns-3's `scratch/`
   folder, re-run the CMake configure step and build again.
3. Run them and analyse the output:

```bash
ns3.49-wsn-link-validation --output=results/ns3/ns3_psr.csv --exponents=2.7,3.0,3.5,4.0 --packets=5000 --step=1 --maxDistance=80
python -m wsn.cli ns3-export --seed 2000 --output results/ns3/field_2000.txt
ns3.49-wsn-scenario-replay --scenario=results/ns3/field_2000.txt --mode=isolated --output=results/ns3/replay_isolated.csv
ns3.49-wsn-scenario-replay --scenario=results/ns3/field_2000.txt --mode=contention --output=results/ns3/replay_contention.csv
python -m wsn.cli ns3-validate --csv results/ns3/ns3_psr.csv --replay results/ns3/replay_isolated.csv results/ns3/replay_contention.csv --output results/ns3
```

On Windows, put ns-3's `build/lib` folder and the MinGW `bin` folder on `PATH` so the
programs find their DLLs.

## Outputs

```text
results/<run folder>/            # from `simulate` (or the dashboard)
  resolved_config.yaml  scenario.json  nodes_initial.csv  round_metrics.csv
  node_energy_snapshots.csv  summary.json  sampled_packet_traces.jsonl  topology.png
results/comparisons/<name>/      # from `compare`
  experiment_config.yaml  per_run.csv  aggregate.csv  paired_differences.csv
  round_metrics.csv  summary.md  plots/*.png
  runs/<variant>_seed<N>_<algorithm>/   full run data, so every run can be replayed
results/ns3/                     # from the ns-3 study
  ns3_psr.csv  replay_*.csv  field_2000.txt  link_validation.png  summary.md
models/
  q_table.json  training_history.csv  training_curve.png
  dqn.json  dqn_training_history.csv  dqn_training_curve.png
```

`results/` and `models/` are generated and ignored by Git; rerun the commands to recreate them.
Schemas are documented in [docs/methodology.md](docs/methodology.md#output-schemas).

## What is stored on GitHub

The repository includes the Python source, dashboard and its theme, configuration
files, tests, ns-3 C++ source, project metadata and documentation. The numerical
results described above are documented findings from local experiments; a clone
does not include those experiment files.

The following stay on the machine and are excluded by `.gitignore`:

| Local files | Reason / how to recreate |
|---|---|
| `results/` | Simulation traces, CSV tables, plots and ns-3 measurements; rerun simulations and comparisons |
| `models/` | Learned policies, training histories and curves; rerun both training commands |
| `.venv/`, `*.egg-info/` | Installed Python environment and package metadata; repeat setup |
| `__pycache__/`, tool caches, `build/`, `dist/` | Generated files; recreated by the corresponding tools |
| Local agent state directories | Development-only state, unrelated to running the project |
| `.env` files and `.streamlit/secrets.toml` | Local credentials, if added later; the project needs none |

No precompiled ns-3 binaries are included. Build the optional programs separately
using the instructions above. There is no hosted dashboard deployment; run it locally.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named wsn` | From the project root, run `python -m pip install -e ".[dev]"` using the same Python that launches the app |
| `streamlit` command is missing | Use `python -m streamlit run dashboard/app.py` in the installed environment |
| Results, learning or validation pages are empty | Generate comparisons, train models or run the optional ns-3 study respectively; generated files are excluded from Git |
| Model is incompatible with a configuration | Train a new model with that configuration and pass its path explicitly; smoke models use `configs/smoke.yaml` |
| Training and evaluation seeds overlap | Use separate fields, such as training seeds 1000–1039 and evaluation seeds 2000–2009 |
| No connected topology can be generated | Increase communication range or revise field size / sensor count; validate the YAML with `python -m wsn.cli validate-config --config FILE` |
| Port 8501 is occupied | Add `--server.port 8502` to the Streamlit command |

## Quality checks

```bash
python -m ruff format --check src tests dashboard
python -m ruff check src tests dashboard
python -m mypy            # strict mode: src, tests and dashboard
python -m pytest -q
```

## Project layout

```text
configs/      default.yaml, smoke.yaml, experiments.yaml
src/wsn/      simulator, routers, learning, experiments, reporting (see docs/architecture.md)
src/wsn/dashboard/   Streamlit app: pages, charts, replay, theme
dashboard/    app.py launcher for `streamlit run`
ns3/          ns-3 validation programs (C++)
tests/        pytest suite
docs/         specification, architecture, methodology and results, decisions
```
