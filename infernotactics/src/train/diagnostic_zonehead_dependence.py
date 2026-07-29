"""
Diagnostic ONLY -- no training, no gradient updates, no writes to any shared
checkpoint/log path. Run tag "diag_zonehead_dependence_v1", writes only to
logs/diag_zonehead_dependence_v1_results.json. Does not touch
models/checkpoints_zonehead_zonehead_v1/ (read-only) or any training script.

Question: does the new ZoneHead's output (actor_critic.py, fed each zone's
own average-pooled per-cell features -- see cnn_branch.CNNBranch's
zone_pooled) actually DEPEND on the observation, or does it -- like the old
plain-Linear zone head -- still emit a near-constant vector regardless of
what's actually burning where?

Loads models/checkpoints_zonehead_zonehead_v1/episode_0300.pt (the run
locked to zone 30) and forward-passes 20 genuinely different observations:
4 scenarios (single_training, stone_canyon, topanga_ridge,
mandeville_canyon) x 5 tick snapshots each (0/15/40/75/120, via a plain
noop rollout from reset -- fire spread is deterministic-given-seed and
doesn't depend on dispatch actions, so noop is sufficient to get genuinely
different fire extents without needing any policy).

For each of the 20 observations, records:
  - the full 32-dim zone logit vector
  - the 4-dim resource_type logit vector (control -- same trunk, old plain
    Linear head, expected to already depend on input if the model saw any
    real training signal at all)
  - per-zone active (Threat+Blaze) cell counts (ground truth "where's the
    fire really at" to correlate zone logits against)
  - the 128-dim CNN pooled vector, 128-dim MLP output, and 256-dim fused
    vector (pooled_cnn concat mlp_features) that feeds actor_critic --
    traces the "is zone_logit input-dependent" question one step further
    back: if the fused vector itself barely varies across genuinely
    different fire states, no zone head design could fix that downstream.

    python -m src.train.diagnostic_zonehead_dependence
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.fire_sim import BLAZE, THREAT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    MULTI_IGNITION_TRAINING_SCENARIO,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
)
from models.inferno_model import InfernoModel  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints_zonehead_zonehead_v1", "episode_0300.pt")

SCENARIOS = {
    "single_training": TRAINING_IGNITION_POINT,
    "stone_canyon": MULTI_IGNITION_TRAINING_SCENARIO[2],
    "topanga_ridge": MULTI_IGNITION_TRAINING_SCENARIO[0],
    "mandeville_canyon": VALIDATION_IGNITION_POINTS["mandeville_canyon"],
}
CHECKPOINT_TICKS = [0, 15, 40, 75, 120]  # early/mid/late spread within a 150-tick episode
SEED = 4242


def _per_zone_active_counts(sim, zones):
    counts = []
    for z in zones:
        r0, r1 = z["row_range"]
        c0, c1 = z["col_range"]
        region = sim.state[r0:r1, c0:c1]
        counts.append(int(np.isin(region, (THREAT, BLAZE)).sum()))
    return counts


def collect_observations(env):
    """Returns a list of dicts: {label, obs} for all 20 (scenario, tick) pairs."""
    records = []
    for name, point in SCENARIOS.items():
        obs = env.reset(ignition_point=point, scenario="single", seed=SEED, use_real_weather=True)
        tick = 0
        done = False
        for target_tick in CHECKPOINT_TICKS:
            while tick < target_tick:
                obs, _reward, done, _info = env.step(None)  # noop -- fire spreads regardless of dispatch
                tick += 1
                if done:
                    break
            active_counts = _per_zone_active_counts(env.sim, env.zones)
            records.append({
                "label": f"{name}@t{target_tick}",
                "scenario": name,
                "tick": target_tick,
                "obs": obs,
                "active_counts": active_counts,
                "total_active": sum(active_counts),
            })
            if done:
                break
    return records


def main():
    env = InfernoEnv(seed=0)
    env.reset(seed=0)
    n_grid_channels = env.reset(seed=0)["grid"].shape[0]

    model = InfernoModel(n_grid_channels=n_grid_channels)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    records = collect_observations(env)
    print(f"Collected {len(records)} observations: "
          f"{[r['label'] for r in records]}")
    print(f"total_active fire-cell counts per observation: {[r['total_active'] for r in records]}")

    zone_logits_all = []
    resource_logits_all = []
    pooled_cnn_all = []
    mlp_out_all = []
    fused_all = []
    zone_pooled_all = []
    active_counts_all = []

    with torch.no_grad():
        for rec in records:
            grid, scalars = InfernoModel.obs_to_tensors(rec["obs"])
            per_cell_features, pooled_cnn, zone_pooled = model.cnn(grid)
            mlp_features = model.mlp(scalars)
            fused = torch.cat([pooled_cnn, mlp_features], dim=1)
            action_logits, value = model.actor_critic(fused, zone_pooled, mlp_features)

            zone_logits_all.append(action_logits["zone"][0].numpy())
            resource_logits_all.append(action_logits["resource_type"][0].numpy())
            pooled_cnn_all.append(pooled_cnn[0].numpy())
            mlp_out_all.append(mlp_features[0].numpy())
            fused_all.append(fused[0].numpy())
            zone_pooled_all.append(zone_pooled[0].numpy())  # (n_zones, per_cell_channels)
            active_counts_all.append(rec["active_counts"])

    zone_logits_all = np.stack(zone_logits_all)        # (20, 32)
    resource_logits_all = np.stack(resource_logits_all)  # (20, 4)
    pooled_cnn_all = np.stack(pooled_cnn_all)          # (20, 128)
    mlp_out_all = np.stack(mlp_out_all)                # (20, 128)
    fused_all = np.stack(fused_all)                    # (20, 256)
    zone_pooled_all = np.stack(zone_pooled_all)        # (20, 32, 32) -- (obs, zone, channel)
    active_counts_all = np.array(active_counts_all, dtype=np.float64)  # (20, 32)

    # --- 3. per-zone-logit variance across the 20 observations ---
    zone_logit_var = zone_logits_all.var(axis=0)  # (32,)
    print(f"\n=== zone logit variance across 20 obs (per zone, 32 values) ===")
    print(np.array2string(zone_logit_var, precision=6, suppress_small=False))
    print(f"mean={zone_logit_var.mean():.6f}  min={zone_logit_var.min():.6f}  max={zone_logit_var.max():.6f}  "
          f"n_below_1e-4={(zone_logit_var < 1e-4).sum()}/32")

    # --- 4. resource_type logit variance across the 20 observations (control) ---
    resource_logit_var = resource_logits_all.var(axis=0)  # (4,)
    print(f"\n=== resource_type logit variance across 20 obs (4 values, CONTROL) ===")
    print(np.array2string(resource_logit_var, precision=6, suppress_small=False))
    print(f"mean={resource_logit_var.mean():.6f}  min={resource_logit_var.min():.6f}  max={resource_logit_var.max():.6f}")

    # --- 5. correlation between zone logits and real per-zone active-fire counts ---
    # (a) flattened over all 20*32=640 (logit, count) pairs
    flat_corr = np.corrcoef(zone_logits_all.ravel(), active_counts_all.ravel())[0, 1]
    # (b) per-observation correlation across the 32 zones (does the head prefer
    #     zones with more fire WITHIN a given state), averaged over the 20 obs
    per_obs_corrs = []
    for i in range(zone_logits_all.shape[0]):
        zl, ac = zone_logits_all[i], active_counts_all[i]
        if ac.std() > 0 and zl.std() > 0:
            per_obs_corrs.append(float(np.corrcoef(zl, ac)[0, 1]))
        else:
            per_obs_corrs.append(float("nan"))
    print(f"\n=== correlation: zone logits vs real per-zone active-fire counts ===")
    print(f"flattened (all 640 obs-zone pairs): r={flat_corr:.4f}")
    print(f"per-observation (32 zones each), mean over 20 obs: "
          f"r_mean={np.nanmean(per_obs_corrs):.4f}  values={[round(c, 3) if c == c else None for c in per_obs_corrs]}")

    # --- 6. trace back: variance of CNN pooled vector / MLP output / fused vector ---
    def _summarize_component_variance(arr, name):
        var = arr.var(axis=0)
        print(f"\n=== {name} variance across 20 obs ({arr.shape[1]} components) ===")
        print(f"mean={var.mean():.6e}  median={np.median(var):.6e}  min={var.min():.6e}  max={var.max():.6e}  "
              f"n_below_1e-6={(var < 1e-6).sum()}/{len(var)}")
        return var

    pooled_var = _summarize_component_variance(pooled_cnn_all, "CNN pooled vector (128-dim)")
    mlp_var = _summarize_component_variance(mlp_out_all, "MLP output (128-dim)")
    fused_var = _summarize_component_variance(fused_all, "fused vector feeding actor_critic (256-dim)")

    # Bonus, not explicitly requested but needed to interpret the above: the
    # ZoneHead's OTHER input -- zone_pooled, each zone's own per-cell-feature
    # average (the thing specifically added so the zone head could see real
    # per-zone spatial content instead of only the globally-pooled vector,
    # which the trace-back above just showed is ~constant). Variance here
    # computed two ways: across observations for each (zone, channel) pair
    # flattened, AND per-zone (does zone 14's feature vector vary across obs
    # more/less than zone 30's, etc -- 32 per-zone summaries).
    zone_pooled_var_flat = zone_pooled_all.reshape(20, -1).var(axis=0)  # (32*32,)
    zone_pooled_var_per_zone = zone_pooled_all.var(axis=0).mean(axis=1)  # (32,) -- mean over channels, per zone
    print(f"\n=== zone_pooled (ZoneHead's per-zone spatial input) variance across 20 obs ===")
    print(f"flattened over all 32 zones x 32 channels: mean={zone_pooled_var_flat.mean():.6e}  "
          f"min={zone_pooled_var_flat.min():.6e}  max={zone_pooled_var_flat.max():.6e}  "
          f"n_below_1e-6={(zone_pooled_var_flat < 1e-6).sum()}/{len(zone_pooled_var_flat)}")
    print(f"per-zone (mean variance across that zone's 32 channels, 32 zones):")
    print(np.array2string(zone_pooled_var_per_zone, precision=6, suppress_small=False))
    # correlation between zone_pooled's per-zone signal strength and real fire:
    # does the zone whose OWN pooled features vary most across obs correspond
    # to zones with more fire variance?
    active_var_per_zone = active_counts_all.var(axis=0)
    zp_vs_fire_corr = np.corrcoef(zone_pooled_var_per_zone, active_var_per_zone)[0, 1]
    print(f"correlation(zone_pooled per-zone variance, real active-fire-count per-zone variance) "
          f"across the 32 zones: r={zp_vs_fire_corr:.4f}")

    results = {
        "checkpoint": CHECKPOINT_PATH,
        "labels": [r["label"] for r in records],
        "total_active_per_obs": [r["total_active"] for r in records],
        "zone_logit_variance": zone_logit_var.tolist(),
        "resource_logit_variance": resource_logit_var.tolist(),
        "zone_logit_vs_fire_corr_flattened": float(flat_corr),
        "zone_logit_vs_fire_corr_per_obs_mean": float(np.nanmean(per_obs_corrs)),
        "zone_logit_vs_fire_corr_per_obs": per_obs_corrs,
        "pooled_cnn_variance_summary": {"mean": float(pooled_var.mean()), "min": float(pooled_var.min()),
                                         "max": float(pooled_var.max())},
        "mlp_out_variance_summary": {"mean": float(mlp_var.mean()), "min": float(mlp_var.min()),
                                      "max": float(mlp_var.max())},
        "fused_variance_summary": {"mean": float(fused_var.mean()), "min": float(fused_var.min()),
                                    "max": float(fused_var.max())},
        "zone_pooled_variance_flat_summary": {"mean": float(zone_pooled_var_flat.mean()),
                                               "min": float(zone_pooled_var_flat.min()),
                                               "max": float(zone_pooled_var_flat.max())},
        "zone_pooled_variance_per_zone": zone_pooled_var_per_zone.tolist(),
        "zone_pooled_vs_fire_variance_corr": float(zp_vs_fire_corr),
        "zone_logits_all": zone_logits_all.tolist(),
        "resource_logits_all": resource_logits_all.tolist(),
        "active_counts_all": active_counts_all.tolist(),
    }
    out_path = os.path.join(LOG_DIR, "diag_zonehead_dependence_v1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
