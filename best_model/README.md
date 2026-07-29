# InfernoTactics — Best Model (self-contained release, v8: fire-relative actions)

A reinforcement-learning agent that dispatches real LAFD-style fire
resources (water tenders, brush-patrol/trench crews, rescue vehicles,
helicopters) against a physically simulated wildfire on a real 30m-resolution
grid of the LA Westside (Topanga → Brentwood/Bel-Air → Westwood/UCLA),
grounded in the real Palisades Fire (Jan 7–31, 2025). This folder is a
complete, drop-in copy of the model, the environment it was trained against,
and the exact real-world data both depend on — nothing else from the parent
repo is required.

Everything below is accurate as of this checkpoint. Read the **Honest
performance & limitations** section before quoting numbers from it anywhere
— this checkpoint is an early-stage result, not a finished, fully-trained
policy (see that section for exactly what that means).

## Quick start

```bash
cd best_model
pip install -r requirements.txt
python run_example.py
```

This resets the environment on the real Palisades Fire ignition point
(the Skull Rock trailhead, the fire's actual documented origin), loads the
model, and runs one deterministic rollout — printing the resource/semantic
target dispatched each tick and the final outcome (reward, buildings
destroyed, contained or not). This was run from a clean copy of this exact
folder before it was pushed, so it is verified working, not just assumed to.

## What changed: fire-relative actions (v8)

Every earlier version of this project (through v7) used an **absolute**
action space: the policy picked one of 32 fixed macro-zone ids directly.
That meant "dispatch to zone 18" had to be memorized per ignition point —
it could not transfer to a fire starting anywhere else, and every
architecture through v7 failed hard on held-out ignition points as a result
(see the project's other write-ups for the full investigation: task-conflict
fixes, credit-assignment fixes, a GAE regression bisect — all real, all
insufficient, because the action space itself was the bottleneck).

This checkpoint instead uses **fire-relative** actions
(`src/train/relative_actions.py`). Every tick, the current fire state is
resolved into a small set of semantic candidates *wherever they currently
are*, not fixed geographic locations:

- `active_fire` — the zone with the most currently-burning cells
- `downwind_fire_front` — unburned fuel scored by wind alignment + distance
  to active fire (where the fire is *about to* spread)
- `adjacent_fuel` — unburned fuel directly bordering active fire (a firebreak
  candidate)
- `threatened_population` — the zone with the highest population density
  under active threat
- `nearest_reachable_fire` — the active-fire zone with the shortest real
  travel time for the resource type being considered (routing-aware, so it
  differs by resource type)
- `noop` — always valid, dispatch nothing this tick

The model (`src/models/relative_model.py`, `RelativeInfernoModel`) never
outputs an absolute zone. It picks a `resource_type` (4-way) and then scores
these ≤6 candidates for that resource, using each candidate's own resolved
features (local fire/fuel/population/travel-time stats) rather than a fixed
zone embedding. "Dispatch the helicopter to the fire" means the same thing
regardless of where the fire started.

## How the model works

1. **CNN branch** (`cnn_branch.py`) — same as prior versions: reads the
   9-channel spatial grid at full 30m resolution, producing a per-cell
   feature map (for classification) and both a globally-pooled and a
   *per-zone*-pooled feature vector.
2. **MLP branch** (`mlp_branch.py`) — global scalars: wind, humidity,
   resource availability, elapsed time.
3. **Classification head** (`classification_head.py`) — per-cell 4-class
   fire-state prediction (course-project requirement; not load-bearing for
   dispatch decisions).
4. **Target head** (new in v8) — scores each of the ≤6 resolved semantic
   candidates using that candidate's own local features plus a learned
   resource-type embedding, rather than emitting a fixed 32-way zone
   distribution.

## The environment it was trained against

`src/env/inferno_env.py` (`InfernoEnv`) + `src/env/fire_sim.py`
(`FireSim`) implement a deterministic, non-ML cellular-automaton fire
spread model, wrapped in a Gym-style RL environment — unchanged from prior
versions:

- **Fire spread**: per-cell ignition probability = base rate × flammability
  × directional slope factor × directional wind factor × road resistance
  × humidity suppression, combined across all 8 neighbors, plus
  wind-scaled **ember spotting** (burning cells can ignite fuel 4–12 cells
  downwind, bypassing roads).
- **Resources**: real LAFD station addresses and apparatus — 8 stations,
  15 total units (`src/data_pipeline/real_depots.json`). Ground units route
  via real road-network Dijkstra shortest paths (`data/roads.graphml`);
  helicopters route via straight-line air distance.
- **Weather**: real hourly Jan 7–8, 2025 KSMO (Santa Monica Airport) ASOS
  data (`data/palisades_weather_jan2025.csv`).
- **Reward**: +50 for extinguishing fire, up to −200 per building destroyed
  (population-density-weighted), −10 for a wasted dispatch, and a
  travel-time penalty.
- **Validation**: the bare, unsuppressed fire-spread simulation was checked
  numerically against the real historical fire perimeter (LA County WFIGS
  data) — IoU 0.636, recall 90.7%.

Environment version: **v5** (8 real LAFD stations, 15 units, frozen as of
2026-07-28).

## Using it in your own code

```python
import sys
sys.path.insert(0, "best_model/src")

import torch
from env.inferno_env import InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES, SCALAR_KEYS
from models.relative_model import RelativeInfernoModel
from train.relative_actions import TARGET_TYPES, decode_action, resolve_relative_targets

env = InfernoEnv()
obs = env.reset(ignition_point=TRAINING_IGNITION_POINT, scenario="single", seed=0)

model = RelativeInfernoModel(
    n_grid_channels=obs["grid"].shape[0], n_scalars=len(SCALAR_KEYS),
    n_resources=len(RESOURCE_TYPES), n_zones=env.n_zones,
)
model.load_state_dict(torch.load("best_model/inferno_best_model.pt", map_location="cpu", weights_only=True))
model.eval()

grid_t, scalars_t = RelativeInfernoModel.obs_to_tensors(obs)
target_zones, target_features = resolve_relative_targets(env, obs)
logits, value, classification_logits = model(
    grid_t, scalars_t, torch.from_numpy(target_zones).unsqueeze(0), torch.from_numpy(target_features).unsqueeze(0)
)
resource_idx = logits["resource_type"].argmax(dim=-1).item()
target_idx = logits["target"][0, resource_idx].argmax(dim=-1).item()
action = decode_action(resource_idx, target_idx, target_zones)  # (resource_type, zone_id) or None

obs, reward, done, info = env.step(action)
```

Other real ignition points already wired into the environment:

```python
from env.inferno_env import VALIDATION_IGNITION_POINTS, MULTI_IGNITION_TRAINING_SCENARIO
```

`VALIDATION_IGNITION_POINTS` (`mandeville_canyon`, `getty_view_park`) are
real, held-out chaparral/WUI-adjacent locations never trained on.
`MULTI_IGNITION_TRAINING_SCENARIO` (Topanga ridge / Sullivan Canyon /
Stone Canyon) are additional real locations along the same Santa Ana
corridor.

## Honest performance & limitations — read before quoting this anywhere

**This checkpoint is from episode 10 of a 2000-episode training run — a
smoke test, not a finished/converged policy.** It is included here because,
even at this very early stage, it demonstrates the structural benefit of the
fire-relative action space, which is the actual point of v8. It is not yet
a "the model is done" result.

Deterministic eval, 2–3 episodes/scenario:

| Scenario | Reward | Buildings destroyed | Containment |
|---|---|---|---|
| `single_training` (Skull Rock) | +27.6 | 0 | 100% |
| `mandeville_canyon` (held out) | +39.7 | 0 | 100% |
| `getty_view_park` (held out) | +31.0 | 0 | 100% |
| `stone_canyon` (multi-ignition, hard) | −182.7 (high variance: +38 to −637 across 3 eps) | 0.7 avg | 100%* |

*Every prior architecture (v1–v7) scored strongly negative on both held-out
validation points (roughly −17k and −81k reward). Getting positive reward
and full containment there after just 10 episodes of training is a real,
structural result: the semantic action space makes "dispatch to wherever
the fire actually is" available from episode 1, instead of requiring
hundreds of episodes to memorize a fixed zone id per location.

**But an unbiased 20-point random-ignition sweep tells a more honest
story:** across 20 randomly sampled unseen WUI ignition points (2 episodes
each), containment was 100%, but **aggregate reward was −244.6, not
positive** — 7 of 20 points suffered real building loss (1–4 buildings each).
Traced tick-by-tick, this is not a bug: the policy dispatches a helicopter
to the active fire immediately and correctly every time, but this
checkpoint has only learned that one reflex — it has not yet learned to
also send a faster-arriving ground unit as backup when a fire starts near
buildings and the helicopter's travel time (which can be several minutes)
would arrive too late to prevent structure loss. The two curated validation
points above happened to be easy draws (small fires, no nearby buildings in
the fire's path); the random sweep shows that isn't representative.

**A structural ceiling, independent of training quality:** a capacity
analysis found the real fire in some scenarios (e.g. `stone_canyon`)
exceeds the entire 8-unit resource fleet by tick 2–4 of the episode, while
the environment's one-dispatch-per-tick action space needs 8 ticks just to
commit that whole fleet, and each helicopter is then locked for ~15 ticks
(mostly reload time) before it can be redeployed. Some scenarios may not be
containable by *any* policy under this resource model.

**What this checkpoint does NOT show:** whether the generalization
advantage holds up after real training (hundreds–thousands of episodes),
whether the model learns to mix resource types rather than defaulting to
helicopter-only, and how it performs on the full multi-ignition training
scenarios once trained. Treat every number above as an early, promising
signal — not a finished result.

**A training-loop bug was found and fixed before this checkpoint's later
episodes:** the gradient update recomputed the resource-type action
distribution without re-masking it by which resource types were actually
available that tick (the rollout that generated the action *did* mask it).
This caused a rollout/update distribution mismatch on the majority of ticks
in any scenario with resource congestion (confirmed: 135 of ~171 steps in a
`stone_canyon` rollout) — the same class of bug that caused catastrophic
policy collapse in earlier (pre-v8) architectures. It's fixed in the code
shipped here.

**Model-quality caveats, unchanged from prior versions:**
- `fuel_density`, one of the 9 grid layers, is a placeholder heuristic, not
  real LANDFIRE fuel-model data.
- 1 of the 32 macro-zones is unreachable by any ground resource via the
  real road graph — only a helicopter can reach it. This is intentional
  (a real asymmetry of air vs. ground response), not a bug.

## File manifest

```
inferno_best_model.pt          trained weights (PyTorch state_dict, episode 10 of a 2000-ep run)
run_example.py                 minimal, verified-working example
requirements.txt               the packages needed to run it
src/
  models/
    relative_model.py           RelativeInfernoModel: CNN + MLP + classification + fire-relative actor-critic
    cnn_branch.py                spatial grid -> per-cell features + pooled vector + per-zone features
    mlp_branch.py                global scalars -> feature vector
    classification_head.py      per-cell 4-class fire-state prediction
  train/
    relative_actions.py          resolves fire-relative semantic candidates from the current observation
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
  environment instead of plain pip.
- **Shape mismatch on `load_state_dict`**: construct `RelativeInfernoModel`
  with `n_grid_channels`/`n_scalars` taken from a real `env.reset()` call
  and `SCALAR_KEYS`, not hardcoded guesses.
- **Slow first run**: `InfernoEnv()` builds a routing tree over the full
  road graph once per station on construction (roughly one second total);
  this is a one-time cost per process, not per episode.
