# 3D simulation viewer

Ties the trained policy to a real-terrain 3D view of the fire. The model and the
environment run in Python; the browser only replays what they produced.

```
InfernoEnv + RelativeInfernoModel
        |
        |  export_trajectory.py     deterministic rollout -> trajectories.json
        v
render_basemap.py                   grid_static.npy -> terrain texture
        |
        |  build_player.py          template + assets -> ONE html file
        v
../../../Simulation3D.html          open it directly, no server, no network
```

## Regenerating

```bash
cd infernotactics
.venv/bin/python src/viz/export_trajectory.py            # rollouts  (~30 s)
.venv/bin/python src/viz/build_player.py                 # single-file viewer
open ../Simulation3D.html
```

`export_trajectory.py` options worth knowing:

| flag | default | notes |
|---|---|---|
| `--checkpoint` | `best_model/inferno_best_model.pt` | any relative-action checkpoint |
| `--scenarios` | `anchor,mandeville,getty,multi` | `mandeville`/`getty` are the held-out points |
| `--traffic-mode` | `legacy` | `legacy` matches the dynamics the v8 weights were trained under; `synthetic` runs the v9 traffic/delay model those weights never saw |
| `--post-ticks` | `14` | dispatch-free aftermath frames so the fleet can be seen standing down; excluded from the scored outcome |

## What is real and what is rendering

Real, straight out of the simulation:

- terrain from the USGS 3DEP elevation layer of `grid_static.npy`
- basemap from the real OSM road mask, LARIAC building footprints, Census
  population and the coastline water mask
- fire state per tick, per cell, from `fire_sim.py`
- every dispatch, arrival, suppression, trench, rescue and structure loss,
  with the exact cell the physics touched
- the real Jan 7-8 2025 KSMO weather series, reward stream and the semantic
  action the policy chose each tick
- ground vehicles follow real Dijkstra routes over the OSM graph; helicopters
  fly straight lines from Station 114 at Van Nuys, which is genuinely north of
  the grid, so they enter from off-map

Rendering choices, all surfaced in the viewer's provenance panel:

- **vertical exaggeration 3.4x** - 686 m of relief across 18 km is nearly flat
  at true scale, and the canyon structure the fire model depends on disappears
- **vehicle markers are symbolic in size** - a 17 m helicopter is 0.57 of one
  30 m cell, i.e. sub-pixel; marker scale also adapts to camera distance
- **formation spread** - the policy dispatches whole groups of one type to one
  zone, so their reconstructed positions coincide; they are fanned out laterally
  so five aircraft read as five. Timing, routing and effects are untouched
- **return legs** - the environment models the post-effect busy/reload period as
  a duration, not a path. The viewer animates the unit flying home over that
  timer, because a unit that teleported back would read as a bug

## Checkpoint compatibility

v8 checkpoints were trained before the v9 traffic scalars existed, so they carry
an 8-column MLP input weight while the current env emits 11 scalars. The three
v9 scalars are appended **last** in `SCALAR_KEYS`, so `export_trajectory.py`
widens the first layer and zero-fills the new columns - function-identical to the
original v8 network on the eight scalars it was actually trained on. The adapter
is reported on stdout and shown in the viewer.

## Debug / figure capture

Query parameters on `Simulation3D.html`:

| param | effect |
|---|---|
| `?ep=0..3` | episode index |
| `&tick=N` | seek to a frame |
| `&still=1` | render a settled frame and stop, instead of animating (also enables `preserveDrawingBuffer` so screenshot tools capture it) |
| `&cam=overview\|fire\|air\|free` | camera preset |
| `&spin=0` | disable the slow overview orbit |
| `&debug=1` | overlay GL state, draw counts, camera and elevation values |

Note when screenshotting with headless Chrome: `--screenshot` under software GL
drops the WebGL layer behind the `backdrop-filter` HUD panels and yields a black
scene. Capture through the DevTools Protocol (`Page.captureScreenshot`) instead.

## Files

| file | role |
|---|---|
| `export_trajectory.py` | runs model + env, writes `trajectories.json` |
| `render_basemap.py` | static layers -> terrain texture |
| `build_player.py` | inlines three.js, texture, elevation and rollouts into one html |
| `player_template.html` | the viewer (shaders, particles, unit models, camera, HUD) |
| `vendor/three.module.min.js` | vendored r169 build, inlined at build time |
| `trajectories.json` | generated; safe to delete and rebuild |
