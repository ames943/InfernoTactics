# InfernoTactics coding-tool handoff

Start here, then read `APPROVED_PLAN.md` in this directory. This handoff records observed repository state, architectural intent, verification evidence, and remaining work. It is not a declaration that the approved implementation is complete.

## 1. User intent and scope

The user approved replacing the course-project wildfire simulator with a versioned georeferenced world, maintained fire physics, reproducible environmental inputs, explicit resource operations, and response-policy training/evaluation. Execute the approved plan step by step, verifying a stage before moving on. The latest request was to write this handoff for another coding tool, not to continue implementing features during handoff creation.

Scientific scope: one California study area (Palisades), noncommercial research, CPU-first local execution, wildland fire plus asset exposure. Not structure combustion, operational dispatch, or exact reproduction of an incident. Keep numerical, operations, historical, suppression, and policy evidence separate. No claim of real-world accuracy from synthetic tests or improving RL reward.

## 2. Repository orientation

- Git repository root: `InfernoTactics/` (this directory's parent).
- Python/application root: `InfernoTactics/infernotactics/`. Run Python commands there, normally `.venv/bin/python -m src.world.cli ...`.
- Parent `README.md`: current architecture and high-level workflow.
- Nested `README.md` and `TUTORIAL.md`: current workflow and input details; inspect these before designing another convention.
- `Simulation.html`: now a static pointer to the authoritative local Streamlit viewer, not a simulation.
- `context.txt`, `experiments/`, notebooks, old checkpoint/log files: historical course-project material; not authoritative contracts or scientific ground truth.
- There was no configured language server when queried earlier. Recheck if your tool config supplies one.

Preserve unexpected working-tree changes. The checkout at handoff is substantially further along than the visible early implementation transcript: do not overwrite current files with earlier snippets. Use current code as evidence and inspect version-control state before changing anything. No git reset/cleanup was performed for this handoff.

## 3. Current implementation map (observed files)

Under `infernotactics/src/world/`:

| File | Responsibility |
|---|---|
| `contracts.py` | Scenario, Action, Event, Treatment; structural/geospatial/resource/road validation |
| `scenario.py` | Version-1 JSON + NPZ + roads persistence, SHA-256 verification, safe internal paths, no pickle |
| `synthetic.py` | Deterministic labeled 96×96 / 30 m / six-hour flat grass fixture |
| `fire.py` | Pinned Pyretechnics adapter, ignition, unit/axis conversion, spotting continuation, treatment modifiers, domain exit |
| `prepare.py` | Explicit provenance-tagged local source import and scenario construction |
| `moisture.py` | NFDRS4 integration / explicit uncertainty-prior pathway; inspect exact current API |
| `routing.py` | Directed road and last-mile/access routing |
| `resources.py` | Per-unit travel, work, return/refill, constraints/events |
| `world.py` | Authoritative canonical 60-second fire/operations coupling and owned observations |
| `actions.py` | Shared feasible explicit candidate actions (the initial `candidates.py` is no longer present) |
| `learning.py` | Masked candidate PPO, checkpoint metadata/loading |
| `rollout.py` | Common episode runner and output artifacts |
| `evaluation.py` | Paired none/heuristic/checkpoint evaluation on held-out seeds |
| `cli.py` | Current commands: synthetic, build, run, train, evaluate |

Other relevant files:

- `src/data_pipeline/fetch_landfire.py`, `fetch_hrrr.py`: new acquisition implementations.
- Existing fetchers for DEM/roads/buildings/population/weather/perimeter remain; acquisition is not authoritative preparation or validation.
- `src/viz/world_app.py`: authoritative Streamlit UI.
- `tests/world/`: test_contract, test_fire, test_prepare, test_acquisition, test_moisture, test_routing, test_resources, test_world, test_learning, test_evaluation.
- `src/env`, `src/train`, `src/models`, `src/validation` currently list as empty directories. Old active course runtime appears cut over; inspect historical experiment/notebook references rather than restoring aliases.

**Known missing approved deliverable:** `src/world/validation.py` is absent from the observed world listing, and `cli.py` has no `score` subcommand. Existing `evaluation.py` is paired policy evaluation, not the approved timestamped observed-perimeter scoring/uncertainty/readiness implementation. Finish stage 7 rather than treating the presence of an evaluation module as completion.

## 4. Evidence and progress: do not conflate these

### Directly observed earlier in this session

- Existing `.venv` actually uses CPython 3.11.18 (system `python3` was 3.9.6).
- Pyretechnics distribution version: `2026.10.20`.
- Runtime import check printed torch 2.15.0, Streamlit 1.66.1, SciPy 1.12.0.
- `python -m src.world.cli synthetic --out data/world-smoke --seed 7` succeeded, including scenario save/load validation.
- `pytest tests/world/test_contract.py -q`: **36 passed in 0.22s** at that point. Later checkout changes mean this is historical evidence, not a fresh full-suite result.
- Repeating synthetic without overwrite returned exit 2 and a specific existing-output error.
- A recent recovered `pytest tests/world/test_fire.py -q` call returned only `..........` in available tool output: **no complete summary or exit status was observed**. Do not call it a completed passing suite. `proc://` later showed no running jobs.

### Present artifacts, inspected during handoff

- `data/world-smoke/`: scenario.json, landscape.npz, weather.npz, roads.json.
- `logs/world-smoke-none/` and `logs/world-smoke-response/`: metrics.json, events.jsonl, snapshots through `snapshot-021600.npz`.
- `models/world-smoke.pt` exists (~49.8 KB).
- These establish file presence, not how they were produced, successful verification, policy quality, or scientific validity. Inspect metadata/content before relying on them.
- No `logs/world-smoke-eval/` was visible in the inspected logs listing.
- No current browser proof, completed 512-step training output, paired checkpoint evaluation output, or complete final suite result was observed by this handoff author.

Earlier todo state said only stage 1 complete and stage 2 active. It is stale relative to current files. Reconcile by evidence, not by blindly implementing already-present stages or marking them verified because files exist. Two physics subagents hit provider 429 cooldowns after writing files; their completion summaries were unavailable. Their code/tests need ordinary review and verification, not assumption of completion.

## 5. Critical invariants

- Scenario affine order `(a,b,c,d,e,f)`; square north-up projected metre cells; source validity masks are explicit; no clipping invalid ignition into the domain.
- Weather shape `(T,H,W)` or `(T,1,1)`; full UTC band coverage under `provenance.weather_intervals`; no implicit infinite last band. Fuel moisture is kg/kg oven-dry fuel, not relative humidity.
- Landscape includes real supported fuel model codes, canopy fields, DEM/slope/aspect, valid mask, and asset footprint area m². Valid nonburnable/water differs from nodata.
- Serializer refuses existing output paths. CLI owns explicit `--overwrite`. NPZ uses `allow_pickle=False`; artifacts have hashes; paths must remain inside scenario directory.
- Authoritative `World.step(actions)` advances 60 seconds (or final partial interval); observation/render cadence must never integrate physics or consume random draws.
- Preserve burned/arrival history. Quiet growth is a duration, not permanent containment. External boundary escape is censored, not contained.
- Fire backend uses a south-to-north row axis, minutes, 10 m wind km/h, spread m/min; public arrays are north-up SI. Conversion belongs only at the adapter boundary.
- Pyretechnics reconstructs spotting RNG each solver invocation. Use SeedSequence([scenario_seed,coupling_index]) per canonical call, carry pending spot_ignitions explicitly, fresh dictionaries, adaptive `max_cells_per_timestep=0.4`. Reproducibility is defined at coupling resolution, not arbitrary partition invariance.
- Treatments earned during a resource interval affect the next fire interval; maximum coupling delay 60 seconds. Never instant extinguishing or erasing burns. Water is experimental spread attenuation; completed line is earned metre-by-metre, not instant disk clearing.
- Unit identity explicit; ground travel respects directed geometry and access; return starts at current position; work/supply/refill consume time; FIFO shared-base service; reject invalid batches atomically before mutation.
- Same feasibility rules for actions, heuristic, learned policy and actual dispatch. Wait always exists. Observations own their arrays; no future realized weather or hidden mutable engine objects.
- PPO exact feature order/backend/schema/scenario/input hashes and training seeds must be checkpoint-compatible; old course weights are not pretrained new-world weights.
- Train seeds 0–999, default eval 10000–10031. Paired physical outcomes are separate from reward. Censored burn area must not become uncensored success statistics.

## 6. Immediate continuation sequence

1. Read `APPROVED_PLAN.md`, current parent/nested docs, and current implementation/tests relevant to each unresolved criterion. Identify intervening user edits before changing files.
2. Reconcile current code against stages 2–6 with targeted actual smoke runs; do not rewrite them from this historical handoff.
3. Complete stage 7: `validation.py`, `score` CLI, exact-time/CRS/provenance perimeter scoring, seeded uncertainty ensembles, honest readiness report with evidence IDs/reasons. Keep observed suppression contamination and source resolution limitations explicit.
4. Verify current train/checkpoint load/evaluate path; inspect existing checkpoint metadata before deciding whether regeneration is needed. Smoke training demonstrates the pathway, not policy quality.
5. Browser-open the actual Streamlit UI and exercise ignition, a real dispatch, stepping, changed routes/unit/fire state, and audit events. Server startup alone is not UI proof.
6. Run the integrated suite once after changes; update existing docs for actual behavior; remove temporary smoke scripts (keep requested/established run artifacts as appropriate). Deliver exact exercised evidence and remaining external scientific prerequisites.

Useful approved commands (run from nested project root; paths may already exist):

```sh
.venv/bin/python -m src.world.cli --help
.venv/bin/python -m src.world.cli synthetic --out data/world-smoke --seed 7
.venv/bin/python -m src.world.cli run --scenario data/world-smoke --seed 7 --policy none --out logs/world-smoke-none
.venv/bin/python -m src.world.cli run --scenario data/world-smoke --seed 7 --policy heuristic --out logs/world-smoke-response
.venv/bin/python -m src.world.cli train --scenario data/world-smoke --steps 512 --seed 7 --out models/world-smoke.pt
.venv/bin/python -m src.world.cli evaluate --scenario data/world-smoke --checkpoint models/world-smoke.pt --seeds 10000,10001 --out logs/world-smoke-eval
.venv/bin/python -m pytest tests/world
.venv/bin/python -m streamlit run src/viz/world_app.py --server.headless true
```

Use fresh diagnostic paths or intentional `--overwrite`; do not blindly delete artifacts. The existing venv is usable; do not recreate or reinstall it without checking a missing dependency. Requirements include cfgrib/ecCodes; actual installation of those was not verified in the earlier session.

## 7. Scientific boundaries and source research

The approved real-data builder targets EPSG:5070 / 30 m, Palisades bbox west=-118.605 south=34.030 east=-118.440 north=34.150, with a 3 km fire-domain buffer and distinct study mask. Roads need a 20 km acquisition buffer to retain external depots. HRRR resampling is not terrain downscaling. Post-event fuel maps cannot be historical inputs.

Additional files in this handoff preserve acquisition research. They are historical findings and can be stale: inspect actual upstream metadata and current implementation. In particular, `world-input-research.md` initially noted missing canopy products on catalog page 0; `world-landfire-links.md` resolves them on page 1. Do not interpret the older note as an actual unavailable product.

NFDRS4 source is pinned to commit `84223b95fd395a62ea344257f80a9949cee71e6e`. Require a full preceding year of hourly native meteorological history and documented station/config metadata, cache by input/config/binary hash, convert moisture percent to fractions, and require separate foliar provenance. Missing history/live observations require explicit interval priors and block historical readiness. Do not silently choose a best estimate.

Historical validation cannot be manufactured by code: independent held-out time-resolved incidents, environmental coverage, incident-level splits, and a pre-registered domain-reviewed protocol remain external prerequisites. Palisades alone cannot meet the approved ≥3 independent-incident certification gate. Suppression stays experimental without independent calibration.

## 8. Security and communication

The former `Simulation.html` contained a public embedded Cesium/map-service token and polling endpoint. Current static replacement removes that live integration. **Owner must rotate/revoke the previously exposed credential**, including considering repository history; do not reproduce the token in docs or tests. Removal does not revoke it.

Engineering preferences from the task: correctness and six-month maintainability, boring explicit design, no compatibility shims for the obsolete runtime, no hidden fake environmental defaults, evidence-first reporting. Never report an unobserved test result as passing. Ask only when an actual user tradeoff cannot be resolved from repository/data; do not punt reachable implementation work.
