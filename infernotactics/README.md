# InfernoTactics research world

The authoritative simulator is `src.world.World`: projected landscape inputs, timestamped weather and fuel moisture, Pyretechnics fire physics, and explicit per-unit resource operations. This is noncommercial research, not operational dispatch, coupled atmospheric simulation, or building combustion.

## Runtime and reproducible offline scenario

Python 3.11+, a virtual environment, and `pip install -r requirements.txt` are required. The fire backend is pinned to `pyretechnics==2026.10.20`; its noncommercial license applies.

```sh
.venv/bin/python -m src.world.cli synthetic --out data/world-smoke --seed 7
```

The synthetic fixture is explicitly illustrative: 96×96 cells at 30 m, EPSG:5070, flat grass fuel, six hours of hourly weather, a directed road loop, and engine/handcrew/helicopter resources. It is never a substitute for Palisades environmental data.

Scenario schema 1 stores scalar metadata/resources/provenance in `scenario.json`, arrays in non-pickle NPZ files, and directed roads in JSON. SHA-256 hashes detect altered artifacts; paths cannot escape the scenario directory. Existing outputs require explicit CLI `--overwrite`. Cells are square and north-up; projected units are metres, model time seconds, weather validity UTC, moisture kg/kg dry fuel, and asset value building-footprint square metres—not monetary loss. Valid nonburnable land is distinct from nodata.

The world couples operations and physics at fixed 60-second boundaries. Viewer sampling cannot advance time or change random draws. Completed treatments activate at the next boundary, with a recorded delay of at most 60 seconds. Historical burned area is never erased. Domain-edge escape censors area results. No-new-growth duration is not proof of permanent containment.

## Evidence and limitations

Numerical, operations, historical, suppression, and policy readiness are separate. Synthetic checks establish software behavior only. Historical readiness requires held-out, time-resolved incident observations, pre-event fuels, complete weather/moisture coverage, and an independently justified protocol. Suppression effectiveness remains experimental without independent measurements. The research observation mode is `oracle_current`, not a sensor model; policies receive no future weather analysis.

Environmental sources: [LANDFIRE LF2023](https://landfire.gov/data/lf2023), USGS 3DEP, NOAA HRRR and [NFDRS4](https://github.com/firelab/NFDRS4/tree/84223b95fd395a62ea344257f80a9949cee71e6e). HRRR resampling is not terrain wind downscaling. Missing live/foliar moisture and wind uncertainty must be explicit rather than silently fabricated.

Old checkpoint artifacts and recorded perimeter scores are historical experiment artifacts, not validation of this world or compatible policy checkpoints.

## Moisture preprocessing

`src.world.moisture.run_nfdrs4` runs the public NFDRS4 CLI pinned to commit `84223b95fd395a62ea344257f80a9949cee71e6e`. Supply a complete preceding calendar year of hourly SI weather, native meteorological cell and vegetation configuration IDs, station ID/latitude/elevation, and the complete upstream initialization text. Station ID belongs in the run configuration, not the initialization schema. The adapter writes explicit local-offset FW21 timestamps from `timeZoneOffset` and normalizes output to UTC; this avoids the pinned upstream UTC day-rollover conversion defect without discarding duplicate records.

Required build attestation records source commit, binary SHA-256, and build-patch SHA-256 (null for a pristine build). On Darwin the pinned source requires a portability-only correction: `lib/utctime/src/utctime.cpp` declaration of `get_utc_timestamp` must return `Time64_T`, matching its header, rather than `time_t`. This does not change moisture equations. Build bundled `extern/config4cpp` with make, configure CMake with its include/library paths, and build `NFDRS4_cli`.

Input/cache keys include hourly weather, complete station configuration, foliar input, and executable/build hashes. Dead/live output percentages are converted to kg/kg. Foliar moisture is separately provenance-tagged and time-covered; herbaceous moisture is not foliar moisture. Missing history or observations require explicit nondegenerate intervals for **all** moisture fields in an uncertainty manifest. Seeded uniform prior sampling is labeled `assumed_prior`, never a historical-readiness certificate.

## Explicit environmental inputs

```sh
.venv/bin/python -m src.data_pipeline.fetch_landfire --cache data/lf-cache --out data/lf-crop --bbox -118.64 34.00 -118.40 34.18 --valid-from 2023-01-01T00:00:00Z --valid-until 2025-01-01T00:00:00Z
.venv/bin/python -m src.data_pipeline.fetch_hrrr --start 2025-01-07T00:00:00Z --end 2025-01-07T01:00:00Z --cache data/hrrr-cache --out data/hrrr-native.npz
.venv/bin/python -m src.world.cli build --inputs INPUTS.json --out data/palisades-world
```

LANDFIRE resolves LF2023 FBFM40/CC/CH/CBH/CBD from the official paginated catalog, verifies provider checksums when published, and records local SHA-256 regardless. Full CONUS packages can be large; local exact-product files are supported. The example validity dates are **not** a certification: replace them with provider-supported intervals covering the intended hindcast; never use post-event fuels.

HRRR uses published GRIB indexes and byte ranges, retaining native 3 km geometry, UTC valid times and vector orientation. Wind vectors are rotated to earth east/north before interpolation and meteorological wind-from calculation. Analysis `PRATE` is instantaneous precipitation rate, **not** hourly accumulation for NFDRS; separately source accumulation history rather than integrating an invented record. Source resolution, terrain-wind limits, and moisture uncertainty survive into provenance. Source files and explicit manifests are required; missing products or coverage are acquisition errors, never synthetic replacements.

The local INPUTS schema is documented in `src/world/prepare.py`: each hashed source includes provider/version/acquisition/CRS/units/validity; layer references specify explicit conversion and band/member. The Palisades target is the densified projected envelope plus a 3 km buffer at 30 m, with a separate `core_mask`. Road acquisition coverage extends 20 km beyond the fire domain; external depots remain external, never edge-snapped. Conservative interpolation marks output invalid if any contributing input is nodata. Hindcast moisture priors require all six explicit intervals, and wind speed/direction intervals remain an uncertainty limitation. Generated prior fields are labeled `assumed_prior`; preparation alone never certifies historical accuracy.

## Resource operations and authoritative viewer

Unit actions select an explicit unit and concrete projected point. Directed road edge geometry and speed caps determine engine travel; last-mile access is bounded. Handcrew access uses a DEM eight-neighbor graph and an approximate grade-adjusted walking relation, excluding nodata, water and configured excessive slopes. Helicopters use wind-projected cruise speed with a positive minimum and explicit corridor/weather constraints. None of these illustrative constraints authorizes real operations.

Mobilization, travel, work, return and FIFO base refill all consume elapsed time. Only return interrupts busy work; return starts at the actual current route position. Water consumes litres and produces an earned, configured-radius dose; line work earns metres along a segment rather than clearing a disk. Weather/closures can block travel or suspend work without teleporting units or spending paused supplies. Shared base IDs share one refill service slot.

```sh
.venv/bin/python -m src.world.cli run --scenario data/world-smoke --seed 7 --policy none --out logs/world-smoke-none
.venv/bin/python -m src.world.cli run --scenario data/world-smoke --seed 7 --policy heuristic --out logs/world-smoke-response
.venv/bin/python -m streamlit run src/viz/world_app.py
```

Runs write audited `events.jsonl`, physical `metrics.json`, and timestamped non-pickle NPZ snapshots. Invalid inputs exit nonzero without a successful-run metrics report. The viewer owns one World per session; only explicit ignition/step/run controls mutate it. Its projected map, unit routes and timestamped event table show the same state as the command line. The former static sector demo is replaced by local viewer instructions; its previously embedded map-service credential must be rotated by the owner.

## Experimental policy training and held-out comparison

```sh
.venv/bin/python -m src.world.cli train --scenario data/world-smoke --steps 512 --seed 7 --out models/world-smoke.pt
.venv/bin/python -m src.world.cli evaluate --scenario data/world-smoke --checkpoint models/world-smoke.pt --seeds 10000,10001 --out logs/world-smoke-eval
```

The masked actor scores the same explicit feasible candidates as the earliest-arrival heuristic; wait is always available. Features use configured scenario bounds/capacities, current observation and feasible ETA—not future weather or hidden fire/suppression parameters. The actor and pooled critic each use two 64-unit tanh layers. PPO records Adam learning rate 3e-4, gamma .99, GAE lambda .95, clip .2, value coefficient .5, entropy coefficient .01, gradient norm .5, 256 transitions/update, four epochs and minibatch 64. Reward is incremental normalized burned area, newly exposed footprint area and normalized busy-unit time; it is not an economic damage estimate.

Checkpoints record schema/backend, feature ordering, scenario/input hashes, training seeds and hyperparameters. Loading uses tensor-only deserialization and rejects old course weights or mismatched inputs. Training seeds 0–999 are disjoint from the default evaluation panel 10000–10031. Paired evaluation runs no-response, earliest-feasible heuristic and checkpoint on identical seeds. Output contains `evaluation.json` and per-seed/policy rollout artifacts. Physical outcomes remain separate from reward; deltas are policy-minus-baseline, and smaller synthetic area is not independently validated suppression benefit. Burned-area aggregates exclude censored pairs. Improving training reward does not establish policy quality or readiness.
