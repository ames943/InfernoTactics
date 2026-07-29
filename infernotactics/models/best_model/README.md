# InfernoTactics — Best Model Checkpoint

`inferno_best_model.pt` is a plain PyTorch `state_dict()` for `InfernoModel`
(see `src/models/inferno_model.py`), taken from episode 1550 of the
zone-head-repair training run (`src/train/train_zonehead_fix.py`,
run tag `zonehead_fix1_2k`). It is the single best-performing checkpoint
found across that run.

Environment version: **v5** (the 8-real-LAFD-station resource model, frozen
as of 2026-07-28 — see `src/env/inferno_env.py`). This checkpoint is only
valid against that environment; it is not comparable to earlier v1-v4
checkpoints.

## Loading it

```python
import torch
from src.models.inferno_model import InfernoModel
from src.env.inferno_env import InfernoEnv

env = InfernoEnv()
n_grid_channels = env.reset(seed=0)["grid"].shape[0]

model = InfernoModel(n_grid_channels=n_grid_channels)
model.load_state_dict(torch.load("inferno_best_model.pt", map_location="cpu"))
model.eval()
```

Run a rollout with `InfernoModel.obs_to_tensors(obs)` to convert a raw
`env.reset()`/`env.step()` observation into the batched tensors `forward()`
expects, then take `action_logits["resource_type"].argmax()` /
`action_logits["zone"].argmax()` for a deterministic policy.

## What this checkpoint can and can't do

Evaluated deterministically (`logs/eval_log_zonehead_zonehead_fix1_2k.csv`,
episode 1550), 5 episodes/scenario:

| Scenario | Reward | Buildings destroyed | Containment |
|---|---|---|---|
| `single_training` (Skull Rock, real Palisades ignition point) | +44.2 | 0 | 100% |
| `stone_canyon` (Santa Ana-corridor multi-ignition point) | -43,161 | 150 | 66.7% |

This is the best *combined* result across the whole 2000-episode run —
`single_training` is solved outright, and `stone_canyon` hits the best
containment rate seen anywhere in that run (vs. a 0%/-121k floor at most
other checkpoints).

**Known limitation, honestly**: this checkpoint has NOT been shown to
generalize to the held-out validation ignition points
(`mandeville_canyon`, `getty_view_park`) — it performs far worse than even
a simple rule-based heuristic there. It is trained/validated on the
scenarios listed above only. See the project's known-limitations notes for
the full generalization-gap writeup if you need the caveats for a
presentation or paper.
