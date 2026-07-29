# InfernoTactics — Best Model (self-contained release)

A trained reinforcement-learning agent that dispatches real LAFD-style fire
resources (water tenders, brush-patrol/trench crews, rescue vehicles,
helicopters) against a physically simulated wildfire on a real 30m-resolution
grid of the LA Westside (Topanga → Brentwood/Bel-Air → Westwood/UCLA),
grounded in the real Palisades Fire (Jan 7–31, 2025). This folder is a
complete, drop-in copy of the model, the environment it was trained against,
and the exact real-world data both depend on — nothing else from the parent
repo is required.

Everything below is accurate as of this checkpoint. Read the **Honest
performance & limitations** section before quoting numbers from it anywhere.

## Quick start

```bash
cd best_model
pip install -r requirements.txt
python run_example.py
```

This resets the environment on the real Palisades Fire ignition point
(the Skull Rock trailhead, the fire's actual documented origin), loads the
model, and runs one deterministic rollout — printing the resource/zone
dispatched each tick and the final outcome (reward, buildings destroyed,
contained or not). This was run from a clean copy of this exact folder
before it was pushed, so it is verified working, not just assumed to.

## How the model works

Four ML concepts are fused into one ~247K-parameter model
(`src/models/inferno_model.py`):

1. **CNN branch** (`cnn_branch.py`) — reads the 9-channel spatial grid
   (elevation, slope, building density/height, road mask, fuel density,
   water mask, population density, and the live fire state) at full 30m
   resolution. It produces two things from the same convolutional trunk:
   a full-resolution per-cell feature map (for the classification head)
   and a pooled 128-dim vector (for fusion with the MLP branch). It also
   exposes a *per-zone* pooled feature (`zone_pooled`) — each of the 32
   macro-zones' own local features, kept separate rather than averaged
   away — which turned out to matter a lot (see below).
2. **MLP branch** (`mlp_branch.py`) — reads the global scalars: wind speed
   and direction, humidity, how many units of each resource type are
   currently available, and elapsed time.
3. **Classification head** (`classification_head.py`) — a 1×1 conv over
   the CNN's per-cell features, predicting a 4-class fire state
   (Safe / Fuel / Threat / Blaze) for every single grid cell.
4. **Actor-critic** (`actor_critic.py`) — a factored policy: a
   `resource_type` head (4-way) and a `zone` head (32-way), sampled
   independently, plus a scalar critic. The action space is deliberately
   *not* one flat 128-way distribution — resource type and target zone are
   different questions with different failure modes.

One architectural detail worth knowing if you dig into the code: the zone
head does **not** read the same shared fused trunk the resource-type head
and critic use. It reads the raw MLP output plus each zone's own
`zone_pooled` features directly. This was a deliberate fix, not the
original design — see "Why the zone head is special" below.

## The environment it was trained against

`src/env/inferno_env.py` (`InfernoEnv`) + `src/env/fire_sim.py`
(`FireSim`) implement a deterministic, non-ML cellular-automaton fire
spread model, wrapped in a Gym-style RL environment:

- **Fire spread**: per-cell ignition probability = base rate × flammability
  × directional slope factor × directional wind factor × road resistance
  × humidity suppression, combined across all 8 neighbors, plus
  wind-scaled **ember spotting** (burning cells can ignite fuel 4–12 cells
  downwind, bypassing roads — this is how real fires jump containment
  lines, and it's in the simulation).
- **Resources**: real LAFD station addresses and apparatus — 8 stations,
  15 total units (`src/data_pipeline/real_depots.json`). Ground units
  (water_team, trench_crew, rescue_vehicle) route via real road-network
  Dijkstra shortest paths (`data/roads.graphml`); helicopters route via
  straight-line air distance and can reach a zone (zone 18) that is
  unreachable by any ground unit via the real road graph.
- **Weather**: real hourly Jan 7–8, 2025 KSMO (Santa Monica Airport) ASOS
  data — the actual Santa Ana event that drove the real fire, including
  humidity crashing to 0.67% and sustained 25.3 mph wind
  (`data/palisades_weather_jan2025.csv`).
- **Reward**: +50 for extinguishing fire, up to −200 per building
  destroyed (population-density-weighted, since not all buildings sit in
  equally populated areas), −10 for a wasted dispatch, and a travel-time
  penalty.
- **Validation**: the bare, unsuppressed fire-spread simulation was checked
  numerically against the real historical fire perimeter (LA County WFIGS
  data) — IoU 0.636, recall 90.7% against where the real fire actually
  burned, robust across 4 random seeds and 2 independent real timestamps.
  This is a real, if imperfect, physical simulation, not just a
  plausible-looking toy.

## Why the zone head is special (worth knowing before you trust its zone choices)

A direct diagnostic found that in earlier versions of this model, the zone
head was effectively **ignoring fire location entirely** — its output
correlated at r=-0.05 with where the fire actually was, even though the
exact same CNN features, pooled per-zone instead of globally averaged,
correlated at r=0.97. Global average pooling was diluting a handful of
burning cells across ~188,000 total grid cells to the point of erasure
before the zone head ever saw them.

The fix (baked into this checkpoint): the zone head now scores each zone
from its own `zone_pooled` features through a small per-zone MLP (shared
weights across zones), trained with an auxiliary supervised loss pulling
its output toward the real per-zone fire distribution every tick, and
cold-started so it can't fall back on a constant bias. After 2000 episodes
of training with this fix, zone-logit-to-real-fire correlation rose to a
sustained 0.31–0.39 — the zone head genuinely reads its input now, and this
checkpoint is the first in the project to show scenario-specific dispatch
behavior (different scenarios genuinely produce different chosen zones,
not one fixed global favorite).

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

Other real ignition points already wired into the environment:

```python
from env.inferno_env import VALIDATION_IGNITION_POINTS, MULTI_IGNITION_TRAINING_SCENARIO
```

`VALIDATION_IGNITION_POINTS` (`mandeville_canyon`, `getty_view_park`) are
real, held-out chaparral/WUI-adjacent locations never trained on.
`MULTI_IGNITION_TRAINING_SCENARIO` (Topanga ridge / Sullivan Canyon /
Stone Canyon) are additional real locations along the same Santa Ana
corridor. **Both of these are exactly the scenarios the limitations
section below is about — read that before drawing conclusions from running
on them.**

To read the per-cell fire-state classification instead of just acting:

```python
predicted_class = classification_logits.argmax(dim=1)  # (B, H, W), 0-3
# 0=Safe, 1=Fuel, 2=Threat, 3=Blaze  (see src/models/classification_head.py)
```

## Honest performance & limitations — read before quoting this anywhere

This is episode 1550 from the zone-head-repair training run
(`train_zonehead_fix.py`, run tag `zonehead_fix1_2k`) — the best
combined-performing checkpoint found across that entire run, evaluated
deterministically over 5 episodes/scenario:

| Scenario | Reward | Buildings destroyed | Containment |
|---|---|---|---|
| `single_training` (Skull Rock, the real Palisades ignition point) | +44.2 | 0 | **100%** |
| `stone_canyon` (a Santa Ana-corridor multi-ignition point) | −43,161 | 150 | 66.7% |

For context, a simple rule-based heuristic policy (nearest-fire dispatch,
same real routing) gets **−29,411** reward and 80% containment on the exact
same `single_training` scenario — this model beats that heuristic by
roughly +29,400 reward there. That comparison is real and reproducible.

**What it does NOT do:** it has not been shown to generalize to ignition
points it wasn't trained on. On both held-out validation points
(`mandeville_canyon`, `getty_view_park`), this family of models performs
*far worse than the simple heuristic above* — this is diagnosed as the
policy specializing to its training scenario(s) rather than learning a
transferable "how to fight any fire" strategy. If you run this checkpoint
on an arbitrary new ignition point, treat the result as an experiment, not
a guarantee, and expect it may perform badly.

**A structural ceiling, independent of training quality:** a capacity
analysis found the real fire in some scenarios (e.g. `stone_canyon`)
exceeds the entire 8-unit resource fleet by tick 2–4 of the episode, while
the environment's one-dispatch-per-tick action space needs 8 ticks just to
commit that whole fleet, and each helicopter is then locked for ~15 ticks
(mostly reload time) before it can be redeployed. Some scenarios may not be
containable by *any* policy under this resource model — a low score there
isn't necessarily a policy failure.

**Model-quality caveats, for full transparency:**
- `fuel_density`, one of the 9 grid layers, is a placeholder heuristic, not
  real LANDFIRE fuel-model data.
- 1 of the 32 macro-zones is unreachable by any ground resource via the
  real road graph — only a helicopter can reach it. This is intentional
  (a real asymmetry of air vs. ground response), not a bug.
- The classification head's own auxiliary loss is a course-project
  requirement (fusing a "Classification" concept into the model) — it
  is not load-bearing for the dispatch policy's decisions.

Environment version: **v5** (8 real LAFD stations, 15 units, frozen as of
2026-07-28 — anything trained before this environment version is not
numerically comparable to this checkpoint).

## File manifest

```
inferno_best_model.pt         trained weights (PyTorch state_dict, ~1MB, ~247K params)
run_example.py                minimal, verified-working example
requirements.txt              the packages needed to run it
src/
  models/
    inferno_model.py           ties CNN + MLP + classification + actor-critic together
    cnn_branch.py               spatial grid -> per-cell features + pooled vector + per-zone features
    mlp_branch.py                global scalars -> feature vector
    classification_head.py      per-cell 4-class fire-state prediction
    actor_critic.py              factored (resource_type, zone) policy + critic
  env/
    inferno_env.py               the Gym-style RL environment (resources, reward, action space)
    fire_sim.py                  deterministic cellular-automaton fire-spread physics
  data_pipeline/
    config.py                    real-world bounding box + local data paths
    real_depots.json             real LAFD station addresses, coordinates, apparatus rosters
data/
  grid_static.npy                precomputed 9-layer grid (elevation, slope, buildings, roads,
                                  fuel, water, population), 595x316 cells @ 30m resolution
  grid_meta.json                 grid metadata (resolution, CRS, bounds)
  roads.graphml                  real LA road network (OSMnx/OpenStreetMap), for resource routing
  palisades_weather_jan2025.csv  real Jan 7-8 2025 hourly weather (NOAA ASOS, station KSMO)
```

## Troubleshooting

- **`osmnx`/`geopandas` install issues**: these pull in compiled geospatial
  dependencies (GDAL, GEOS, PROJ). If `pip install` fails, try a conda/mamba
  environment instead of plain pip — geospatial packages are usually far
  less painful there.
- **Shape mismatch on `load_state_dict`**: make sure you're constructing
  `InfernoModel` with `n_grid_channels` taken from an actual
  `env.reset()["grid"].shape[0]` call (currently 9), not a hardcoded
  guess — and don't pass a custom `adaptive_pool_size` or `n_value_heads`,
  this checkpoint was trained with the defaults.
- **Slow first run**: `InfernoEnv()` builds a routing tree over the full
  road graph once per station on construction (roughly one second total);
  this is a one-time cost per process, not per episode.
