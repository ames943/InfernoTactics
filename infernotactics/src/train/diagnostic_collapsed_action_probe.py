"""
Diagnostic ONLY -- no training, no gradient updates, no writes to any shared
checkpoint/log path. Loads already-saved checkpoints from the v4/v5/v6
credit-assignment-collapse investigation and asks: which action did each
COLLAPSED policy actually lock onto?

Run tag "diag_collapseprobe_v1" -- writes only to
logs/diag_collapseprobe_v1_results.json. Does not touch
models/checkpoints_bisect/ (the live bisect job) or any training script.

For each checkpoint that still exists on disk, runs a forward pass (no_grad,
model.eval()) on a handful of states from scenario='single' (Skull Rock,
real weather, deterministic seed=0, noop-only rollout so the states are
"what the policy would have seen," not confounded by this script's own
dispatch choices) and reports both heads' softmax distributions, argmax, and
max probability -- i.e. how deterministic the collapse actually is and
whether the locked action is state-invariant or state-dependent.

All four checkpoints below were warm-started through build_model() in
train_actor_critic_multi_v4.py / _v6.py, which appends one anchor-identity
flag to the scalar input (N_SCALARS = len(SCALAR_KEYS) + 1 = 9) -- confirmed
via each checkpoint's mlp.net.0.weight shape ([32, 9]) before writing this
script. is_anchor=True is used here since scenario='single' (Skull Rock) IS
the anchor point (see build_scalars()'s docstring in either training script).

Checkpoint provenance (see logs/console_run_v4*.log, logs/console_run_v5*.log
timestamps -- checked before writing this script, not assumed):
  - v4_baseline (PPO_EPOCHS=4) and epochs1 (PPO_EPOCHS=1, 40-ep smoke test)
    both wrote to the now-EMPTY models/checkpoints_multi_v4/ -- unrecoverable,
    skipped.
  - The EPOCHS=1 500-episode REAL run (console_run_v5.log) and the tight-
    gradient-clip run (console_run_v5_tightclip.log) both wrote to the SAME
    unsuffixed models/checkpoints_multi_v5/ (no INFERNO_RUN_TAG); tightclip
    ran later (16:48 vs 16:34) and overwrote the real run's files. Only
    tightclip's final (collapsed, entropy~1e-4, interrupted ep29/50) state is
    recoverable from that directory -- reported as "tight_grad_clip" below;
    the plain EPOCHS=1 real run's own checkpoint is gone, not reported.
  - low_lr (INFERNO_LEARNING_RATE=3e-5) -> models/checkpoints_multi_v5_lrtest/
  - adv_clip (INFERNO_ADV_CLIP=1) -> models/checkpoints_multi_v5_advcliptest/
  - v6 multi-trajectory -> models/checkpoints_multi_v6/

    python -m src.train.diagnostic_collapsed_action_probe
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import RESOURCE_TYPES, SCALAR_KEYS, TRAINING_IGNITION_POINT, InfernoEnv, flatten_scalars  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402

RUN_TAG = "diag_collapseprobe_v1"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ANCHOR_FLAG_DIM = 1
N_SCALARS = len(SCALAR_KEYS) + ANCHOR_FLAG_DIM

CHECKPOINTS = {
    "v4_baseline (PPO_EPOCHS=4)": os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v4", "latest.pt"),
    "epochs1 (PPO_EPOCHS=1, smoke test)": os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v4", "latest.pt"),
    "tight_grad_clip": os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v5", "latest.pt"),
    "low_lr (3e-5)": os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v5_lrtest", "latest.pt"),
    "adv_clip": os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v5_advcliptest", "latest.pt"),
    "v6_multi_trajectory": os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v6", "latest.pt"),
}
# v4_baseline and epochs1 both point at the same (now-empty) directory -- both
# get reported as unavailable below rather than silently deduplicated, so the
# gap is visible in the output instead of just missing from the table.

PROBE_TICKS = (0, 5, 10, 20, 40)  # ticks at which a state snapshot is taken (noop rollout)


def build_scalars(obs_scalars, is_anchor):
    return np.concatenate([flatten_scalars(obs_scalars), [1.0 if is_anchor else 0.0]]).astype(np.float32)


def collect_probe_states(env):
    """noop-only rollout from TRAINING_IGNITION_POINT, real weather,
    deterministic seed=0 -- states are what a deployed policy would actually
    see (fire evolving under real physics), untouched by any dispatch this
    script itself would make, so all checkpoints are probed on the identical
    state sequence."""
    obs = env.reset(seed=0, ignition_point=TRAINING_IGNITION_POINT, use_real_weather=True)
    states = []
    if 0 in PROBE_TICKS:
        states.append((0, obs))
    t = 0
    done = False
    while not done and t < max(PROBE_TICKS):
        obs, _reward, done, _info = env.step(None)
        t += 1
        if t in PROBE_TICKS:
            states.append((t, obs))
    return states


def probe_checkpoint(path, model, states, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()

    per_state = []
    with torch.no_grad():
        for tick, obs in states:
            grid, _ = InfernoModel.obs_to_tensors(obs, device=device)
            scalars = torch.from_numpy(build_scalars(obs["scalars"], is_anchor=True)).unsqueeze(0)
            if device is not None:
                scalars = scalars.to(device)
            action_logits, _value, _cls = model(grid, scalars)

            resource_probs = torch.softmax(action_logits["resource_type"][0], dim=0)
            zone_probs = torch.softmax(action_logits["zone"][0], dim=0)
            r_idx = int(torch.argmax(resource_probs))
            z_idx = int(torch.argmax(zone_probs))

            per_state.append({
                "tick": tick,
                "resource_type": RESOURCE_TYPES[r_idx],
                "resource_max_prob": float(resource_probs[r_idx]),
                "resource_probs": {rt: float(resource_probs[i]) for i, rt in enumerate(RESOURCE_TYPES)},
                "zone": z_idx,
                "zone_max_prob": float(zone_probs[z_idx]),
            })

    locked_resources = {s["resource_type"] for s in per_state}
    locked_zones = {s["zone"] for s in per_state}
    return {
        "per_state": per_state,
        "resource_state_invariant": len(locked_resources) == 1,
        "zone_state_invariant": len(locked_zones) == 1,
        "locked_resource_type": next(iter(locked_resources)) if len(locked_resources) == 1 else sorted(locked_resources),
        "locked_zone": next(iter(locked_zones)) if len(locked_zones) == 1 else sorted(locked_zones),
        "avg_resource_max_prob": sum(s["resource_max_prob"] for s in per_state) / len(per_state),
        "avg_zone_max_prob": sum(s["zone_max_prob"] for s in per_state) / len(per_state),
    }


def main():
    device = torch.device("cpu")  # small forward passes, no need for MPS; avoids any device contention with the live bisect job
    print(f"[{RUN_TAG}] Building InfernoEnv (read-only diagnostic; scenario='single', Skull Rock, real weather)...")
    env = InfernoEnv(seed=0)
    states = collect_probe_states(env)
    print(f"[{RUN_TAG}] Collected {len(states)} probe states at ticks {[t for t, _ in states]}")

    obs0 = states[0][1]
    model = InfernoModel(n_grid_channels=obs0["grid"].shape[0], n_scalars=N_SCALARS).to(device)

    results = {}
    seen_paths = set()
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            results[name] = {"available": False, "reason": f"{path} does not exist"}
            print(f"[{RUN_TAG}] SKIP {name}: not found at {path}")
            continue
        if path in seen_paths:
            results[name] = {"available": False,
                              "reason": f"{path} was already overwritten by another run in this same dir; "
                                        f"this checkpoint's own state is not separately recoverable"}
            print(f"[{RUN_TAG}] SKIP {name}: {path} shared/overwritten, not separately recoverable")
            continue
        seen_paths.add(path)

        print(f"[{RUN_TAG}] Probing {name} <- {path}")
        result = probe_checkpoint(path, model, states, device)
        result["available"] = True
        result["checkpoint_path"] = path
        results[name] = result

    print(f"\n{'run':38s} {'locked resource_type':>22s} {'locked zone':>12s} {'max prob (res/zone)':>20s}")
    print("-" * 96)
    for name, r in results.items():
        if not r.get("available"):
            print(f"{name:38s} {'UNAVAILABLE':>22s} {'-':>12s} {r['reason']}")
            continue
        res_label = r["locked_resource_type"] if r["resource_state_invariant"] else f"VARIES {r['locked_resource_type']}"
        zone_label = str(r["locked_zone"]) if r["zone_state_invariant"] else f"VARIES {r['locked_zone']}"
        prob_label = f"{r['avg_resource_max_prob']:.3f} / {r['avg_zone_max_prob']:.3f}"
        print(f"{name:38s} {res_label:>22s} {zone_label:>12s} {prob_label:>20s}")

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"{RUN_TAG}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")
    return results


if __name__ == "__main__":
    main()
