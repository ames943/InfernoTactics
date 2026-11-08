# InfernoTactics World tutorial

This workflow uses one authoritative `src.world.World` for fire spread, resource operations, policy training, evaluation, and visualization. It replaces the course-project `FireSim` / `InfernoEnv` training workflow. All examples are research experiments, not operational fire-management advice.

## 1. Install

From the repository's `infernotactics` directory:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell activation is `.venv\Scripts\Activate.ps1`. Use the pinned Pyretechnics dependency in `requirements.txt`; do not substitute the removed cellular automaton if installation fails. Weather acquisition also uses the ecCodes/cfgrib stack; platform-specific native dependencies may be needed.

## 2. Create an offline scenario

```sh
python -m src.world.cli synthetic --out data/world-smoke --seed 7
```

This creates an explicitly labeled synthetic fixture centered near longitude `-118.5225`, latitude `34.09` within the Palisades study region. Its terrain, fuels, weather, and resource parameters are illustrative inputs, not observations of the actual incident. The default fixture is 96 by 96 cells at **30 m** resolution in **EPSG:5070** and has a six-hour horizon.

The World uses an explicit scenario contract and provenance rather than the retired eight-channel `grid_static.npy` format. Keep the generated scenario files together. An existing destination is an error unless you explicitly supply `--overwrite`; that flag replaces the selected output, so choose paths carefully.

## 3. Inspect and operate the same World locally

```sh
streamlit run src/viz/world_app.py
```

In the viewer:

1. Choose the offline synthetic source or load a prepared scenario directory, then explicitly create the world.
2. Enter ignition longitude and latitude. The synthetic fixture includes `-118.5225`, `34.09`. Submit ignition explicitly; invalid, out-of-domain, nonburnable, or unsupported cells are rejected.
3. Select an available unit, operation, and feasible target. Dispatch through the explicit form control.
4. Use **Step** to advance one coupling interval or **Run to horizon** to continue the episode.
5. Inspect the projected fuel/fire map, unit positions/routes, data-quality labels, and timestamped event table.

One World persists in the Streamlit session. Widget reruns and rendering do not mutate it. A coupling interval is 60 simulated seconds, except the final partial interval. Fire and operations advance together; a dispatched resource must travel and work before its treatment is available. Explicit return/refill and availability constraints matter. No action instantly rewrites burning cells into safe cells.

The browser's `Simulation.html` is an instruction/link page, not another simulator or a polling client. This local research viewer has no live emergency-dispatch connection.

## 4. Write a reproducible rollout

```sh
python -m src.world.cli run --scenario data/world-smoke --longitude -118.5225 --latitude 34.09 --seed 7 --policy none --out logs/world-none
python -m src.world.cli run --scenario data/world-smoke --longitude -118.5225 --latitude 34.09 --seed 7 --policy heuristic --out logs/world-heuristic
```

Supply longitude and latitude together, or omit both to use the scenario's default ignition. The available policies are `none`, `heuristic`, and `checkpoint`; `checkpoint` additionally requires `--checkpoint`. Every policy acts through the same World action contract and feasibility rules.

Rollouts write `events.jsonl`, `metrics.json`, and timestamped NPZ snapshots to the selected output directory. Read status and provenance as well as outcome metrics: reaching a domain boundary is not containment, and a synthetic run is not a validated hindcast. Invalid inputs should exit nonzero rather than producing a successful-run report. A full six-hour episode may take substantial wall-clock time; use Step in the viewer when exploring interactively.

## 5. Train an experimental response policy

```sh
python -m src.world.cli train --scenario data/world-smoke --steps 512 --seed 7 --out models/world-smoke.pt
python -m src.world.cli run --scenario data/world-smoke --seed 8 --policy checkpoint --checkpoint models/world-smoke.pt --out logs/world-learned
```

The learned policy scores explicit unit/operation/target candidates with a feasibility mask; wait remains available. PPO and rule-based response share the same feasible candidate generator, not separate action semantics. Recorded training configuration and scenario provenance are part of reproducibility. The small command above demonstrates the interface, not convergence or useful firefighting performance.

The baseline observation mode is **`oracle_current`**: perfect current simulated fire state. It is not a real sensor system. Policy observations contain current grid/weather/unit records, candidate actions and masks, elapsed time, and provenance—not future realized fire or weather and not mutable engine objects. Resource costs, burned area, and asset exposure are modeled research outcomes, not direct monetary or operational guarantees.

Only the new World checkpoint format is supported. Old `InfernoModel`, `RelativeInfernoModel`, fixed-zone, and v8 checkpoints are historical artifacts and must be rejected rather than converted or silently loaded.

## 6. Compare held-out policies

```sh
python -m src.world.cli evaluate --scenario data/world-smoke --checkpoint models/world-smoke.pt --seeds 10000,10001 --out logs/world-evaluation
```

Keep evaluation seeds and scenario draws disjoint from training. Compare learned response with **both** no response and heuristic response on paired scenarios. Report per-episode metrics and uncertainty, not just one best rollout. Use more held-out episodes than this short interface example when drawing research conclusions.

Policy evaluation does not establish historical spread validity or suppression-effect validity. A policy can exploit inaccurate assumptions in a simulator; better synthetic rewards are not evidence that dispatch would help on a real incident. Read the current [project README](README.md) for implemented policy and evaluation details.

## 7. Build a sourced historical scenario

```sh
python -m src.world.cli build --inputs /path/to/INPUTS.json --out data/palisades-world
```

Do not run the removed `src.env.grid_builder`. The World builder reads an explicit local-input manifest, whose full format is documented in [`src/world/prepare.py`](src/world/prepare.py). It expects:

- A schema version, name, mode, UTC start, duration, study bounds, buffer, and source registry.
- Local source paths with SHA-256 hashes, provider, version, license, retrieval time, source CRS, units, and validity windows. Data applicability is distinct from download time.
- Elevation, supported fuel models, canopy cover/height/base-height/bulk-density, and asset exposure; source-unit scale/offset conversions are explicit.
- Contiguous hourly weather records, horizontal vector basis, and all required moisture fields. Future realized values remain environmental inputs, not policy observations.
- Explicit roads, closures where supplied, resources, operational parameters, and source labels for assumptions. Station names alone do not define unit capabilities.

Real hindcasts use the Palisades bounds `[-118.605, 34.030, -118.440, 34.150]` in west/south/east/north order, a 3 km buffer, and an outward-snapped **30 m EPSG:5070** grid with a separate core mask. Categorical fuels must remain categories; wind vectors must respect CRS/vector conventions. Missing source coverage or unsupported fuels are errors, not invitations to invent a replacement layer.

The `src/data_pipeline` acquisition utilities remain available for source data. For example, the existing roads, buildings, elevation, population, station-weather, and perimeter fetchers download supporting evidence; they do not by themselves form a complete valid World scenario. LANDFIRE and HRRR acquisition support the new sourced workflow. Follow their documented inputs and the builder contract rather than assuming raw downloads are already aligned.

A KSMO station CSV cannot stand in for the required spatial weather fields. Moisture priors must be explicitly labeled with uncertainty, and prior-mode data do not confer historical-readiness status. Missing real fuels/weather/coverage must remain visible. Do not relabel the offline synthetic fixture as measured Palisades data.

## 8. Interpret outcomes honestly

Separate these questions:

1. **Numerical correctness:** do units, terrain, weather interpolation, fire integration, routing, and treatment timing satisfy their contracts?
2. **Historical spread evidence:** do simulations agree with appropriate timed observations under documented input uncertainty?
3. **Suppression evidence:** are modeled travel, delivery, treatment duration, and effectiveness supported independently?
4. **Policy performance:** does response improve paired held-out outcomes relative to no response and a feasible heuristic?

An unsuppressed simulation compared with a late final perimeter shaped by a large real firefighting response does not validate spread physics or policy benefit. Asset exposure is not building combustion, structure-loss prediction, or a casualty estimate. Ground walking speed, environmental downscaling, treatment persistence, and resource parameter choices include approximations; provenance and sensitivity are essential.

## 9. Historical artifacts and retired commands

Existing data, figures, logs, checkpoint files, and stored reports have been retained. They describe the retired course-project system unless explicitly generated by the new World workflow. Retention does not imply loader compatibility or current validation.

The obsolete `src/env`, `src/train`, `src/models`, and `src/validation` implementations, their simulator-only tests, experimental course trainers/diagnostics, and the v8 Colab training notebook are retired. There are no legacy import shims, old zone-action adapters, or second authoritative simulator. Use `python -m src.world.cli` and `streamlit run src/viz/world_app.py` instead.

## Troubleshooting

| Symptom | Action |
|---|---|
| Output already exists | Choose a new destination or explicitly use `--overwrite` for that command. |
| Ignition is rejected | Check both coordinate arguments, scenario extent, validity, burnability, and supported fuels. |
| A unit cannot dispatch | Inspect status, supply, operation, access, weather/closure constraints, and current feasible candidates. |
| Historical scenario build fails | Fix the specific source hash, units, CRS, validity window, coverage, or missing-field error; do not switch silently to synthetic data. |
| Old checkpoint is rejected | Train a World-format policy. Historical checkpoints have no compatibility loader. |
| Full run appears slow | CLI rollouts run to the scenario horizon; explore one 60-second step at a time in the viewer. |
| Legacy `src.env` / `src.train` command fails | Those entrypoints are intentionally retired; use the World commands above. |
