# InfernoTactics — Best Model (self-contained)

Everything needed to load the trained model and run it on real fire
scenarios, without needing the rest of the InfernoTactics repo.

## Quick start

```bash
cd best_model
pip install -r requirements.txt
python run_example.py
```

That resets the environment on the real Palisades Fire ignition point
(Skull Rock trailhead) and runs one deterministic rollout, printing the
resource/zone dispatched each tick and the final outcome (reward, buildings
destroyed, contained or not). Verified working end-to-end from a clean copy
of this folder before this was pushed.

## What's in here

```
inferno_best_model.pt   trained weights (PyTorch state_dict)
run_example.py          minimal working example (loads model, runs a rollout)
requirements.txt        the packages needed to run it
src/
  models/                InfernoModel + its CNN/MLP/actor-critic/classification-head sub-modules
  env/                   InfernoEnv (the simulation itself) + fire_sim (fire-spread physics)
  data_pipeline/          config.py (paths) + real_depots.json (real LAFD station data)
data/
  grid_static.npy         precomputed 9-layer grid (elevation, roads, buildings, population, etc.)
  grid_meta.json          grid metadata (resolution, CRS, bounds)
  roads.graphml           real LA road network, for resource routing
  palisades_weather_jan2025.csv   real Jan 7-8 2025 weather driving the fire
```

This is a full copy of the model + environment code and the exact data
files it depends on — the model's weights are meaningless without the
environment producing observations in the same format, so both are
included together.

## Using it in your own code

```python
import sys
sys.path.insert(0, "best_model/src")   # or wherever this folder lives

import torch
from env.inferno_env import InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES
from models.inferno_model import InfernoModel

env = InfernoEnv()
obs = env.reset(ignition_point=TRAINING_IGNITION_POINT, scenario="single", seed=0)

model = InfernoModel(n_grid_channels=obs["grid"].shape[0])
model.load_state_dict(torch.load("best_model/inferno_best_model.pt", map_location="cpu"))
model.eval()

grid_t, scalars_t = InfernoModel.obs_to_tensors(obs)
action_logits, value, classification_logits = model(grid_t, scalars_t)
resource_idx = action_logits["resource_type"].argmax(dim=-1).item()
zone_idx = action_logits["zone"].argmax(dim=-1).item()
resource_type = RESOURCE_TYPES[resource_idx]

obs, reward, done, info = env.step((resource_type, zone_idx))
```

You can also run on the other real ignition points already wired into the
environment:

```python
from env.inferno_env import VALIDATION_IGNITION_POINTS, MULTI_IGNITION_TRAINING_SCENARIO
```

## What this checkpoint can and can't do — read before trusting the output

This is episode 1550 from the zone-head-repair training run
(`src/train/train_zonehead_fix.py` in the full repo, run tag
`zonehead_fix1_2k`) — the best combined-performing checkpoint found across
that run, evaluated deterministically over 5 episodes/scenario:

| Scenario | Reward | Buildings destroyed | Containment |
|---|---|---|---|
| `single_training` (Skull Rock, the real Palisades ignition point) | +44.2 | 0 | 100% |
| `stone_canyon` (a Santa Ana-corridor multi-ignition point) | -43,161 | 150 | 66.7% |

**It has NOT been shown to generalize to other/held-out ignition points**
(e.g. `mandeville_canyon`, `getty_view_park` in the full repo) — it
performs far worse than even a simple rule-based heuristic there. Treat
this as a policy specialized to the scenarios above, not a general-purpose
wildfire strategist, if you're presenting or publishing results from it.

Environment version: **v5** (8 real LAFD stations, frozen 2026-07-28) — see
`src/env/inferno_env.py` for the resource model and reward function.
