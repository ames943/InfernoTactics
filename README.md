# InfernoTactics

InfernoTactics is a geospatial wildfire research platform with two coupled systems:

1. a stochastic cellular fire simulator over a real Los Angeles study area; and
2. a reinforcement-learning dispatch policy that assigns water teams, trench crews,
   rescue vehicles, and helicopters to evolving fire-relative targets.

The current implementation uses a 316 x 595 grid at 30 m resolution. Terrain,
buildings, roads, population, weather, fire stations, and LANDFIRE fuel models are
converted into one validated world bundle. A checkpoint records the exact world
fingerprint it was trained on, so incompatible maps and policies fail clearly.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the detailed system model,
data contracts, simulator mechanics, policy design, known limitations, and the
implemented package structure.

## Setup

The reproducible environment is named `ai-ml`:

```powershell
conda env create -f environment.yml
conda activate ai-ml
```

If the environment already exists:

```powershell
conda activate ai-ml
python -m pip install -e ".\infernotactics[dev]"
```

## Build the world

Run from `infernotactics/` after activating `ai-ml`:

```powershell
python -m infernotactics.data.fetch_elevation
python -m infernotactics.data.fetch_fuels
python -m infernotactics.data.fetch_population
python -m infernotactics.data.fetch_buildings
python -m infernotactics.data.fetch_roads
python -m infernotactics.data.fetch_weather
python -m infernotactics.data.build_manifest
python -m infernotactics.world.builder
```

`fetch_perimeter` is optional and is used for validation rather than training.
The grid builder requires the LANDFIRE fuel raster by default. Set
`INFERNO_ALLOW_HEURISTIC_FUEL=1` only to reproduce the historical heuristic grid.

## Train and test

```powershell
cd infernotactics
python -m pytest
python -m infernotactics.training.trainer
```

Useful training settings include `INFERNO_RUN_TAG`, `INFERNO_N_EPISODES`,
`INFERNO_LEARNING_RATE`, `INFERNO_EVAL_EVERY`, `INFERNO_CHECKPOINT_EVERY`,
and `INFERNO_MAX_DISPATCH_SLOTS`. The default run tag is
`relative_v12_landfire`, intentionally separate from historical checkpoints.

Versioned checkpoints contain model, optimizer, return-normalizer, random-number,
episode, and world-identity state. Re-running the same run resumes the next episode
and appends to its logs.

## Simulation API

The supported local API entry point is:

```powershell
uvicorn infernotactics.api.server:app --reload
```

The bundled historical checkpoint defaults to the preserved historical world.
For a new checkpoint, configure the matching `INFERNO_CHECKPOINT`,
`INFERNO_GRID_STATIC`, and `INFERNO_GRID_META` paths together.

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

Existing clients use the `default` simulation session. New clients can create an
isolated session with `POST /api/sessions`, pass its `session_id` to reset/step/state
requests, and delete it when finished. Session count is capped by
`INFERNO_MAX_SESSIONS` (default `8`).
