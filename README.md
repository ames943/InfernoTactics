# InfernoTactics

A reinforcement-learning agent that learns to fight a real wildfire.

It's built on a real 3D-mappable grid of the LA Westside (Topanga → Brentwood
/ Bel-Air → Westwood/UCLA) and grounded in the real Palisades Fire
(Jan 7–31, 2025 — ~23,400 acres burned, ~6,800 structures destroyed, 12
deaths). Real terrain, buildings, roads, population density, and the actual
Jan 7–8, 2025 Santa Ana weather that drove the fire all feed a CNN+MLP
actor-critic that decides, tick by tick, where to send real LAFD-style fire
resources (water tenders, brush-patrol/trench crews, rescue vehicles,
helicopters) across a physically simulated, spreading fire.

This is a UCLA ML/neural-network course final project. The course requirement
was to genuinely fuse four ML concepts — not bolt them together, actually
have each one do real work — and that requirement shaped the whole
architecture:

| Concept | What it actually does here |
|---|---|
| **CNN** | Reads the spatial grid (terrain, roads, buildings, fire) |
| **MLP** | Reads the global scalars (wind, humidity, resource availability, time) |
| **Classification** | An auxiliary head labeling every single grid cell Safe / Fuel / Threat / Blaze |
| **RL (Actor-Critic)** | Drives the actual resource-dispatch decisions; the critic's TD-error is the same signal covered in the course's dopamine/reward-prediction-error lecture |

If you just want a trained model to run against fires immediately — no
training, no data pipeline, nothing else from this repo required — see
[`best_model/`](best_model/), a fully self-contained release folder with its
own README.

If you want to reproduce training, extend the research, or understand *why*
the project looks the way it does, keep reading.

---

## Table of contents

1. [Architecture, in depth](#architecture-in-depth)
2. [The simulation itself](#the-simulation-itself)
3. [Real-world data grounding](#real-world-data-grounding)
4. [Results — RL vs. a rule-based heuristic](#results--rl-vs-a-rule-based-heuristic)
5. [The research journey — training this thing was not simple](#the-research-journey--training-this-thing-was-not-simple)
6. [Known limitations, stated honestly](#known-limitations-stated-honestly)
7. [Repo structure](#repo-structure)
8. [Setting up from a fresh clone](#setting-up-from-a-fresh-clone)

---

## Architecture, in depth

`src/models/inferno_model.py` ties four sub-modules into one ~247K-parameter
model:

**CNN branch** (`cnn_branch.py`) reads the 9-channel grid observation
(elevation, slope, building density, building height, road mask, fuel
density, water mask, population density, and the live fire state) at full
30m-per-cell resolution. Stride-1, same-padding convolutions keep a
full-resolution per-cell feature map alive for the classification head,
while a separate pooled path downsamples to a fixed 128-dim vector for
fusion with the MLP branch. The CNN also exposes each of the environment's
32 macro-zones' own pooled features separately (`zone_pooled`) rather than
only a single global average — this turned out to be load-bearing (see
below).

**MLP branch** (`mlp_branch.py`) reads the global scalars: wind speed and
direction, humidity, how many units of each resource type are currently
available, and elapsed time.

**Classification head** (`classification_head.py`) is a 1×1 convolution
over the CNN's per-cell features, predicting a 4-class fire state (Safe /
Fuel / Threat / Blaze — the fire simulation's 5th raw state, Burned Out,
collapses into Safe, since a burned-out cell can't re-ignite and behaves
identically to Safe for every tactical purpose) for every grid cell
independently. Ground truth is free — it's literally the fire simulation's
own state array — so this head trains against real signal, not a proxy.

**Actor-critic** (`actor_critic.py`) is a *factored* policy: a
`resource_type` head (4-way: water_team / trench_crew / rescue_vehicle /
helicopter) and a `zone` head (32-way, one of the grid's macro-zones),
sampled independently rather than as one flat 128-way distribution, plus a
scalar critic. Factoring the action space this way was deliberate —
resource type and target zone are genuinely different questions with
different failure modes, and collapsing them into one distribution would
make debugging (and interpreting) which part of a bad decision was wrong
much harder.

One detail worth knowing if you read the code: the zone head does **not**
read the same shared fused trunk the resource-type head and critic use — it
reads the raw MLP output plus each zone's own `zone_pooled` features
directly. That split exists because of a real bug found mid-project (see
the research journey section): a direct diagnostic showed the zone head was
correlating at r=-0.05 with actual fire location, i.e. essentially ignoring
it, while the same underlying features pooled per-zone instead of globally
correlated at r=0.97. Global average pooling was diluting a handful of
burning cells across ~188,000 total grid cells to the point of erasure
before the zone head ever saw them. Feeding it the per-zone features
directly, plus an auxiliary supervised loss and a cold-started head, raised
that correlation to a sustained 0.31–0.39 and produced the first genuinely
scenario-specific dispatch behavior in the project.

## The simulation itself

`src/env/fire_sim.py` (`FireSim`) is a deterministic cellular-automaton fire
model — no ML involved, this is the physics the RL agent has to work
against. States: Safe / Fuel / Threat / Blaze / Burned Out. Per-cell
ignition probability combines a base rate, cell flammability, a directional
slope factor, a directional wind factor, road resistance (roads
meaningfully suppress spread — a real road corridor cuts ignition
probability 4–5x), and humidity suppression, across all 8 neighbors. On top
of adjacency spread, burning cells have a wind-scaled chance of **ember
spotting** — igniting a fuel cell 4–12 cells downwind, bypassing roads
entirely — which is how real fires jump containment lines; in a synthetic
test this happened in 26/30 trials at 45 mph vs. 0/30 at 5 mph.

`src/env/inferno_env.py` (`InfernoEnv`) wraps that physics in a Gym-style
RL environment. Every tick (2 simulated minutes), the agent picks one
`(resource_type, target_zone)` action or no-op. Resources are grounded in
real LAFD data (`src/data_pipeline/real_depots.json`): 8 real stations, 15
total units. Ground units (water_team, trench_crew, rescue_vehicle) route
via real road-network Dijkstra shortest paths (`data/roads.graphml`);
helicopters route by straight-line air distance and can reach a zone that
no ground unit can via the real road graph — a genuine, intentional
asymmetry of real air response, not an oversight.

Reward: +50 for extinguishing fire, up to −200 per building destroyed
(scaled by real population density, so losing a building in a dense area
costs more than in a sparse one), −10 for a wasted dispatch (arrives late,
or the target is already resolved), and a travel-time penalty.

Weather isn't a fixed assumption — real hourly Jan 7–8, 2025 KSMO (Santa
Monica Airport) ASOS data drives wind and humidity throughout an episode,
including humidity crashing to 0.67% and sustained 25.3 mph wind, the
textbook Santa Ana signature of the actual event.

## Real-world data grounding

| Data | Source | Notes |
|---|---|---|
| Roads | OSMnx (OpenStreetMap) | 3,760 nodes, 9,523 edges |
| Buildings | LA GeoHub LARIAC4 | 46,585 footprints with real heights |
| Elevation | USGS 3DEP 10m DEM | 1783×948px, 0–686m |
| Weather | NOAA ASOS station KSMO | real Jan 7–8 2025 hourly Santa Ana event |
| Population | US Census/ACS block-group data | normalized 0–1 grid layer |
| Resource depots | Real LAFD Fire Station Directory | 8 stations, apparatus grounded where determinable |
| Fire perimeter (validation) | WFIGS / LA GeoHub Palisades Perimeter | numerically compared to sim output, see below |

The bare, unsuppressed fire-spread simulation (no resources dispatched) was
checked numerically against the real historical fire perimeter:
**IoU 0.636, recall 90.7%, precision 68.1%** against a Jan 21, 2025
snapshot — a genuinely good result for an untuned cellular automaton, robust
across 4 random seeds and confirmed against a second, independent, earlier
real timestamp (Jan 11). `fire_sim.py`'s parameters were never tuned against
this validation data. Error analysis found ~52% of the area the simulation
under-predicts sits on road cells with reduced fuel density — a real,
explained consequence of existing modeling choices, not an unexplained gap.

## Results — RL vs. a rule-based heuristic

`src/train/heuristic_policy.py` implements a simple rule-based baseline
(nearest-fire dispatch, same real routing) for comparison, using the exact
same environment interface as the trained policy.

| Scenario | RL reward | RL destroyed | RL containment | Heuristic reward |
|---|---|---|---|---|
| `single_training` (the real Palisades ignition point) | **+44.2** | 0 | **100%** | −29,411 |
| `stone_canyon` (a Santa Ana-corridor multi-ignition point) | −43,161 | 150 | 66.7% | −300 |

The headline result: on its training scenario, the RL agent beats the
heuristic by roughly **+29,400 reward**, going from 80% to 100% containment
and 0 buildings destroyed. That result is real, reproducible, and not
cherry-picked — see the limitations section for exactly where it stops
holding.

## The research journey

This section exists so nobody has to rediscover what's already been ruled
out. The short version: getting from "a model that can forward-pass" to
"a model that reliably beats a simple heuristic" took multiple full rewrites
and several genuinely negative results, each with real mechanistic evidence
behind it rather than guesswork.

**Generalizing across ignition points (v1–v3, task-conflict hypothesis).**
The first fully-trained model solved its one training scenario but badly
underperformed the heuristic on two held-out validation points — classic
overfitting to a single scenario. Three targeted architectural fixes were
tried to address a suspected reward-scale/critic-sharing conflict between
scenarios: per-scenario return normalization, a scenario-identity input,
and separate value heads per scenario. **All three came back flat** on
every scenario except the one already-solved anchor — ruling out
reward-scale conflict as the primary cause.

**Entropy collapse (v4–v6, credit-assignment hypothesis).** Introducing
GAE and multi-epoch PPO updates (the standard modern recipe) caused the
policy to collapse to a deterministic, wrong action within ~100–200
episodes, every time, across many hyperparameter variants (PPO epoch
count, gradient clip magnitude, learning rate, advantage clipping,
multi-trajectory batching) — seven independently negative results in a
row. A bisect test proved the pre-rewrite training loop never collapsed on
the same environment, localizing the regression to the GAE-based rewrite
itself, not to RL being fundamentally unstable here. A feature-by-feature
toggle diff then isolated the specific culprit to GAE. A critic-quality
diagnostic explained the actual mechanism: the critic tracks value
reliably during long, still-burning episodes, but is *reliably
anti-correlated* with returns during short, already-won episodes — and
that regime grows more common as the policy improves, so a policy getting
better feeds itself an increasingly wrong learning signal. This closed the
causal chain from "the collapse is a mystery" to "the collapse is a
specific, diffable interaction between GAE and this environment's episode
structure," which is a very different, much more useful thing to know.

**The zone head was ignoring fire location (representation hypothesis).**
Described in the architecture section above — found via direct probing of
logit variance and correlation with real per-zone fire state, fixed with a
per-zone scorer plus an auxiliary supervised loss.

**A structural capacity ceiling.** A diagnostic on the resource dispatch
arithmetic found that some scenarios' fire exceeds the entire 8-unit
resource fleet within the first 2–4 ticks, while the environment's
one-dispatch-per-tick action space needs 8 ticks just to commit that whole
fleet, and helicopters are then locked ~15 ticks (mostly reload) before
redeploying. Some scenarios may simply not be containable by *any* policy
under this resource model — a real, environment-level ceiling independent
of training quality, not a hidden training failure.

The best checkpoint produced by this whole line of work is what's packaged
in [`best_model/`](best_model/).

## Known limitations

- **Generalization to new ignition points is unsolved.** The model performs
  far worse than a simple heuristic on ignition points it wasn't trained
  near. This has been investigated extensively (see above) and is the
  single biggest open item in the project.
- **A structural resource-capacity ceiling exists** independent of policy
  quality — some scenarios may be uncontainable by design under the current
  8-unit resource model and one-dispatch-per-tick action space.
- **`fuel_density`** (one of the 9 grid layers) is a placeholder heuristic,
  not real LANDFIRE fuel-model data.
- **1 of 32 macro-zones is unreachable by ground resources** via the real
  road graph — only a helicopter can reach it. This is an intentional,
  realistic asymmetry, not a bug.
- **The classification head's auxiliary loss is a course requirement**,
  fusing a required "Classification" concept into the model — it is not
  load-bearing for the dispatch policy's own decisions.
- **The 3D visualization is intentionally deferred** — it was scoped as the
  last step, after training produced real results worth showing, not before.

## Repo structure

```
infernotactics/
├── data/                    real-world data + generated grid/sim outputs (gitignored except grid_meta.json)
├── src/
│   ├── data_pipeline/       scripts pulling real data (OSM, USGS, LA GeoHub, NOAA), config.py, real_depots.json
│   ├── env/                 InfernoEnv (RL environment) + FireSim (fire-spread physics)
│   ├── models/              CNN, MLP, classification head, actor-critic, InfernoModel
│   ├── train/                training loops (the full experiment history — each script is a
│   │                         self-contained, documented run) + heuristic_policy.py + eval.py
│   ├── validation/           numerical fire-spread-vs-real-perimeter validation (WFIGS)
│   └── viz/                  pydeck/Streamlit visualization (not yet started, deliberately deferred)
├── models/                   checkpoints from every training run (each run gets its own subfolder)
├── logs/                     per-run training/eval/diagnostic logs (csv/json/console output)
└── notebooks/

best_model/                  self-contained release: trained weights + everything needed to run
                              them, independent of the rest of this repo (see its own README)

Simulation                   in-progress CesiumJS 3D visualization prototype (separate from the
                              Python RL project above; not yet wired to model output)
```

## Setting up from a fresh clone

This section covers what's needed to train a new model from scratch. It
does not repeat the research history above — see `logs/` and the
diagnostic scripts in `src/train/` for the full detail behind any specific
finding.

### Why data setup is a separate step

`data/*.tif`, `*.geojson`, `*.graphml`, `*.npy`, and `*.png` are gitignored
(see `.gitignore`) — they're either large binaries or straightforward to
regenerate from public sources, so they aren't committed. `models/` (model
checkpoints) **is** committed, so warm-starting from a prior run works
immediately after cloning with no extra setup.

### 1. Environment

Requires Python 3.14 (this is what the project was developed and pickled
against; earlier versions may still work but are untested).

```bash
cd infernotactics
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch real-world data

Each script pulls from a live public source (OSM, USGS 3DEP, WorldPop, LA
GeoHub, NOAA ASOS) — see the docstring at the top of each file for the exact
source and any sourcing caveats. Run from `infernotactics/`:

```bash
python -m src.data_pipeline.fetch_elevation    # -> data/elevation.tif
python -m src.data_pipeline.fetch_population   # -> data/population.tif
python -m src.data_pipeline.fetch_buildings    # -> data/buildings.geojson
python -m src.data_pipeline.fetch_roads        # -> data/roads.graphml
python -m src.data_pipeline.fetch_weather      # -> data/palisades_weather_jan2025.csv (already committed, but safe to re-fetch)
```

`fetch_perimeter.py` (real fire perimeter, used only by validation, not
training) is optional at this stage:

```bash
python -m src.data_pipeline.fetch_perimeter    # -> data/palisades_perimeter_*.geojson
```

### 3. Build the simulation grid

Rasterizes everything from step 2 into the aligned static layer stack the
env and CNN branch consume:

```bash
python -m src.env.grid_builder    # -> data/grid_static.npy, data/grid_meta.json
```

### 4. Train

`src/train/` contains the full experiment history (each script is a
self-contained, one-off run — see individual docstrings). The current/final
line is `train_zonehead_randign.py` (randomized-ignition training,
warm-started from the best prior checkpoint, already present in
`models/checkpoints_zonehead_zonehead_fix1_2k/` from the clone):

```bash
INFERNO_RUN_TAG=<your_run_name> INFERNO_N_EPISODES=2000 python -m src.train.train_zonehead_randign
```

`INFERNO_RUN_TAG` is required (keeps each run's checkpoints/logs in their
own `models/checkpoints_zonehead_<tag>/` dir instead of colliding). Optional
env vars: `INFERNO_N_EPISODES` (default 2000), `INFERNO_CHECKPOINT_EVERY`
(default 25), `INFERNO_EVAL_EVERY` (default 50), `INFERNO_WARM_START_CKPT`
(default: the zonehead_fix1_2k checkpoint referenced above).

The run is crash-safe/resumable: re-running the same command with the same
`INFERNO_RUN_TAG` picks up from `resume_state.pt` in that run's checkpoint
dir if it exists.

To train a model from scratch (no warm start), use
`src/train/train_actor_critic.py` instead — the earliest, no-warm-start
entry point in this line.
