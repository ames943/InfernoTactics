# InfernoTactics tutorial

Commands in this guide are run from `A:\AI\InfernoTactics\infernotactics` in PowerShell.

## 1. Environment

```powershell
cd A:\AI\InfernoTactics
conda env create -f environment.yml
conda activate ai-ml
cd infernotactics
```

To refresh an existing environment, run `python -m pip install -e ".[dev]"`.

## 2. Build or refresh the geographic world

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

The runtime pair is `data/grid_static.npy` plus `data/grid_meta.json`. The default
builder requires LANDFIRE FBFM40. `source_manifest.json` records provider, source
URL, file size, and hash for every input. Large source/generated binaries are
ignored by Git; never copy a grid without its matching metadata.

## 3. Verify the installation

```powershell
python -m pytest
python -c "from infernotactics.operations import InfernoEnv; e=InfernoEnv(seed=42); o=e.reset(seed=42); print(o['grid'].shape, e.n_zones, e.world.fingerprint)"
```

The current observation shape is `(9, 316, 595)`: eight static channels plus fire
state. There are 32 macro-zones.

## 4. Train

```powershell
$env:INFERNO_RUN_TAG="my_landfire_run"
$env:INFERNO_N_EPISODES="2000"
$env:INFERNO_CHECKPOINT_EVERY="10"
$env:INFERNO_EVAL_EVERY="50"
python -m infernotactics.training.trainer
```

Artifacts are isolated by run tag:

- `models/checkpoints_<run-tag>/`: complete versioned checkpoints;
- `logs/runs/<run-tag>/`: configuration, episode/evaluation CSVs, optional traces;
- `logs/tensorboard/<run-tag>/`: TensorBoard event data.

Re-running the same command resumes after the completed episode and appends to its
logs. A checkpoint with a different world fingerprint is rejected.

| Setting | Purpose | Default |
| --- | --- | --- |
| `INFERNO_LEARNING_RATE` | Adam learning rate | `0.0001` |
| `INFERNO_AUX_COEFF` | Semantic-target auxiliary loss | `0.05` |
| `INFERNO_MAX_DISPATCH_SLOTS` | Maximum choices in one tick | `10` |
| `INFERNO_TRACE_EVERY` | Tick trace interval; `0` disables | `0` |
| `INFERNO_FORCE_CPU` | Disable DirectML/CUDA when `1` | `0` |
| `INFERNO_PROGRESS_ENABLED` | Rich live progress when `1` | `1` |

Older `INFERNO_V8_*` names remain temporary aliases for historical commands.

## 5. Evaluate and inspect

```powershell
python -m infernotactics.training.evaluation `
  --checkpoint "models/checkpoints_my_landfire_run/latest.pt" `
  --episodes 5 `
  --random-points 30
python -m infernotactics.policy.heuristic
tensorboard --logdir logs/tensorboard
python -m infernotactics.training.plotting --run-tag my_landfire_run
```

Use held-out ignition locations and multiple seeds. A single rollout is useful for
debugging, not for comparing policies.

## 6. Run the interactive simulation

From the repository root:

```powershell
uvicorn infernotactics.api.server:app --reload --host 127.0.0.1 --port 8000
```

The bundled historical checkpoint uses the preserved legacy grid. For a newly
trained policy, set matching `INFERNO_CHECKPOINT`, `INFERNO_GRID_STATIC`, and
`INFERNO_GRID_META` paths together before launch.

## 7. Reproducibility rules

- Treat the grid, metadata, source manifest, config, and checkpoint as one identity.
- Do not compare an old policy on a rebuilt world without labeling the change.
- Keep stochastic seeds and held-out ignitions with reported metrics.
- Training success does not validate physics; perimeter calibration is separate.
