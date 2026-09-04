# InfernoTactics Python project

This directory contains the data pipeline, simulator, containment environment,
policy, training code, tests, checkpoints, and experiment output.

Use the repository-level [README](../README.md) for setup and commands, and
[architecture guide](../docs/ARCHITECTURE.md) for the complete design and data flow.

From this directory, with the `ai-ml` Conda environment active:

```powershell
python -m pytest
python -m infernotactics.data.build_manifest
python -m infernotactics.world.builder
python -m infernotactics.training.trainer
```

Large source rasters and generated grids are ignored by Git and rebuilt locally.
The JSON metadata and provenance manifest remain versioned so the world contract is
reviewable. Existing bundled checkpoints were trained on the preserved legacy grid;
new LANDFIRE-backed training uses the default `grid_static.npy`/`grid_meta.json`.
