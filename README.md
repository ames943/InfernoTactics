# InfernoTactics

InfernoTactics is a noncommercial wildfire-response research project. Its authoritative simulator is now `infernotactics/src/world`: a georeferenced wildland-fire world with explicit resource travel, work, return, and refill, plus policies that act through the same feasible candidate-action contract.

This replaces the course-project cellular automaton and zone-dispatch environment. It does **not** reproduce the Palisades Fire exactly, simulate building combustion, or provide an operational dispatch system. Synthetic examples are illustrative; a successful run is not historical validation or evidence of real-world response effectiveness.

## Start here

Run these commands from the nested project directory:

```sh
cd infernotactics
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Offline, explicitly synthetic scenario; no real-data downloads required.
python -m src.world.cli synthetic --out data/world-smoke --seed 7

# The same World drives command-line rollouts and the local viewer.
python -m src.world.cli run --scenario data/world-smoke --seed 7 --policy none --out logs/world-none
streamlit run src/viz/world_app.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Existing command outputs are protected; use `--overwrite` only when you intend to replace the selected destination. Full CLI episodes use the scenario's horizon, so use the viewer's Step control for interactive exploration rather than expecting an immediate result from a six-hour simulation.

For the complete workflow, see [the project README](infernotactics/README.md) and [the tutorial](infernotactics/TUTORIAL.md).

## One world, one contract

- **Geography:** the real-data builder uses an outward-snapped **30 m EPSG:5070** grid, not the historical README's 200 m grid. The Palisades core study bounds are longitude `[-118.605, -118.440]`, latitude `[34.030, 34.150]`, with a 3 km simulation buffer and an explicit core mask. The offline synthetic fixture is a smaller illustrative landscape, not a measured Palisades reconstruction.
- **Environmental inputs:** explicitly sourced terrain, supported fuel models, canopy, asset exposure, hourly weather, and moisture. Missing or incompatible real inputs are errors; they are not silently replaced by synthetic fuels or frozen weather. The builder checks source hashes, units, coordinate systems, and validity windows.
- **Spread:** Pyretechnics-backed wildland spread and moisture coupling. Buildings are exposure assets, not combustible building-fire cells. Physical approximations and input uncertainty remain limitations.
- **Operations:** engines, handcrews, and helicopters have explicit individual state and constrained actions. Road/off-road travel, access, supply, work duration, return, and refill matter; dispatch does not instantly erase fire. Operational parameters require source labels, including when they are assumptions.
- **Time:** a canonical 60-second coupling interval joins fire, resource events, and completed treatments. Outputs use elapsed seconds and timestamps, not ambiguous course-project ticks.
- **Observation:** policies receive the current simulated state (`oracle_current`), current weather and unit records, feasible candidates and masks, and provenance. This is perfect current simulated fire observation, not a sensor or forecasting model; future realized weather/fire are not policy observations.
- **Policy:** no response, heuristic, and learned masked-candidate policies use the same World action interface. Training is experimental and does not bypass environmental or historical validity gates.

## Train and compare experimental policies

```sh
cd infernotactics
python -m src.world.cli train --scenario data/world-smoke --steps 512 --seed 7 --out models/world-smoke.pt
python -m src.world.cli run --scenario data/world-smoke --seed 8 --policy checkpoint --checkpoint models/world-smoke.pt --out logs/world-learned
python -m src.world.cli evaluate --scenario data/world-smoke --checkpoint models/world-smoke.pt --seeds 10000,10001 --out logs/world-evaluation
```

These commands are a workflow example, not a claim that a short training run produces a useful policy. Use paired held-out comparisons against both no response and heuristic response. Keep training and evaluation seeds/scenarios separate, report uncertainty and per-episode outcomes, and do not interpret synthetic policy scores as incident validation.

## Build a sourced historical scenario

```sh
cd infernotactics
python -m src.world.cli build --inputs /path/to/INPUTS.json --out data/palisades-world
```

The explicit manifest format is documented in [`src/world/prepare.py`](infernotactics/src/world/prepare.py). A hindcast needs supported fuel/canopy, elevation, asset exposure, valid hourly weather and moisture, road coverage, and sourced resource parameters. Existing `src/data_pipeline` fetchers remain acquisition utilities; simply running the old GIS download sequence does not produce a complete valid World scenario. A station-weather CSV alone is not the gridded weather contract. Moisture priors must be labeled with uncertainty and do not make a historical scenario ready.

Run a built scenario with `--longitude` and `--latitude` together to select an ignition. Coordinates must lie on supported, valid, burnable cells. For example, the synthetic fixture includes longitude `-118.5225`, latitude `34.09`; do not assume an arbitrary address is inside every scenario.

A raw unsuppressed simulation compared with a late, suppressed final incident perimeter is not validation of spread physics or response effectiveness. Numerical, historical, suppression, and policy evidence are distinct gates.

## Local viewer and security

The local Streamlit viewer owns one persistent World per session. Explicit controls create the world, ignite it, dispatch units, step, and run to the horizon; rendering does not advance the simulation. It shows projected fuel/fire, resource routes, quality labels, and timestamped events. [`Simulation.html`](Simulation.html) now points to this authoritative viewer rather than running a separate sector simulation.

The previous HTML exposed a map-service credential. Its owner must revoke or rotate that credential; removing it from the current file does not revoke copies or repository history. There is no live operational control endpoint.

## Historical course artifacts

Stored checkpoints, logs, figures, validation reports, and source datasets remain historical artifacts. In particular, `InfernoModel`, `RelativeInfernoModel`, zone-head/v8 checkpoints, and their old training scores describe the retired simulator; they are **unsupported by the World checkpoint loader**. There is no compatibility shim or automatic conversion. Train a World-format policy instead.

The obsolete `src/env`, `src/train`, `src/models`, and `src/validation` implementations, their simulator-specific tests, experimental trainer/diagnostic entrypoints, and the v8 Colab training notebook have been retired. The supported simulator, training, evaluation, and viewer entrypoints are the World CLI and Streamlit commands above. Preserved historical artifacts must not be presented as a second runnable authoritative simulator or current validation evidence.

## Layout

```text
InfernoTactics/
  README.md                    # This entrypoint
  Simulation.html              # Instructions for the authoritative local viewer
  infernotactics/
    README.md                  # Current implementation and interfaces
    TUTORIAL.md                # Current reproducible workflow
    requirements.txt
    src/world/                 # Scenario, fire, operations, policies, CLI
    src/viz/world_app.py        # Local World viewer
    src/data_pipeline/         # Environmental acquisition utilities
    tests/world/               # Current contract and engine tests
    data/                      # Source inputs and prepared scenarios
    models/                    # World policies and retained historical checkpoints
    logs/                      # Run output and retained historical evidence
```
