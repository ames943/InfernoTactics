"""
Diagnostic ONLY -- no change to any training mechanic. Question: does the
value head actually learn to predict returns in this environment at all? If
it never does (explained variance near/below 0), then ANY advantage
estimator built on it -- GAE, TD, whatever -- is effectively noise dressed up
as signal, which would explain both why the OLD loop (full Monte-Carlo
returns, doesn't need a good critic to work) stays healthy on this
environment and why every GAE variant tested so far (E+D+A, and the
fresh-per-minibatch-recompute fix attempt) collapsed regardless of the
specific staleness/normalization details.

Reuses train_actor_critic.py's OWN rollout/return/RunningMeanStd/device code
via direct import (not reimplemented) so this is guaranteed to exercise the
exact same mechanism the bisect already proved healthy. The only new code is
update_policy_with_diagnostics(): a verbatim copy of that file's
update_policy() body, with two extra lines per tick capturing
(value_scalar.item(), g_t) for the correlation/explained-variance analysis --
every loss/backward/clip/step call is identical, so this cannot itself change
training dynamics.

Warm-started from models/checkpoints_bisect/episode_0260.pt (the bisect's own
healthy, solved-on-v5 reference) and run for 200 MORE episodes of the old
loop's exact mechanism, matching how every toggle-diff experiment in this
investigation has warm-started. Adam optimizer state is NOT warm (never
serialized by the original run); this only affects transient step sizes, not
the diagnostic in question (value predictions vs actual returns).

Per episode, across all ticks in that episode:
  - corr_V_return: Pearson correlation between V(s_t) and normalized return G_t
  - explained_variance: 1 - Var(G_t - V(s_t)) / Var(G_t) -- near 1.0 = critic
    predicts well, near 0 = no better than predicting the mean, negative =
    worse than predicting the mean
  - V_mean/V_std vs return_mean/return_std for direct scale comparison

    python -m src.train.diagnostic_value_head_quality
"""

import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import TRAINING_IGNITION_POINT, InfernoEnv  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.train_actor_critic import (  # noqa: E402
    GAMMA,
    VALUE_LOSS_COEFF,
    CLASSIFICATION_LOSS_COEFF,
    ENTROPY_COEFF,
    GRAD_CLIP_NORM,
    RunningMeanStd,
    _is_mps_unimplemented_error,
    collect_rollout,
    compute_returns,
    get_device,
)
from torch.distributions import Categorical  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RUN_TAG = "diag_valuehead_v1"
N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 200))
BASE_SEED = 2000
LEARNING_RATE = 3e-4
STATUS_EVERY = 5

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"{RUN_TAG}.csv")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints_bisect", "episode_0260.pt")

TRAIN_LOG_FIELDS = [
    "episode", "device", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "corr_V_return", "explained_variance", "var_return", "var_residual",
    "V_mean", "V_std", "return_mean", "return_std", "wall_time_s",
]


def _mean_std_var(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var), var


def _pearson_corr(a, b):
    n = len(a)
    mean_a, _, var_a = _mean_std_var(a)
    mean_b, _, var_b = _mean_std_var(b)
    if var_a < 1e-12 or var_b < 1e-12:
        return None  # one series is ~constant -- correlation undefined, not "0"
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / n
    return cov / math.sqrt(var_a * var_b)


def update_policy_with_diagnostics(model, optimizer, steps, device, return_normalizer):
    """Verbatim copy of train_actor_critic.update_policy()'s body -- same
    loss terms, same .backward() calls, same two-group gradient clipping,
    same single optimizer.step() -- with two extra lines per tick that stash
    (value_scalar.item(), g_t) for this diagnostic. No training-mechanic
    change: removing those two stash lines reproduces the original function
    exactly."""
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = return_normalizer.normalize(raw_returns)
    return_normalizer.update(raw_returns)

    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    optimizer.zero_grad()
    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0
    value_predictions = []  # <-- diagnostic only
    return_targets = []     # <-- diagnostic only

    for step, g_t in zip(steps, returns):
        grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
        scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
        action_logits, value, classification_logits = model(grid_t, scalars_t)

        resource_dist = Categorical(logits=action_logits["resource_type"][0])
        zone_dist = Categorical(logits=action_logits["zone"][0])
        log_prob = (
            resource_dist.log_prob(torch.tensor(step["resource_idx"], device=device))
            + zone_dist.log_prob(torch.tensor(step["zone_idx"], device=device))
        )
        entropy = resource_dist.entropy() + zone_dist.entropy()

        value_scalar = value.squeeze()
        g_t_tensor = torch.tensor(g_t, dtype=value_scalar.dtype, device=device)
        advantage = (g_t_tensor - value_scalar).detach()

        policy_loss = -log_prob * advantage
        value_loss = (value_scalar - g_t_tensor) ** 2
        classification_loss = F.cross_entropy(
            classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
        )

        tick_loss = (
            policy_loss + VALUE_LOSS_COEFF * value_loss
            + CLASSIFICATION_LOSS_COEFF * classification_loss - ENTROPY_COEFF * entropy
        )
        tick_loss.backward()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        classification_loss_sum += classification_loss.item()
        entropy_sum += entropy.item()
        value_predictions.append(float(value_scalar.item()))  # <-- diagnostic only
        return_targets.append(float(g_t))                      # <-- diagnostic only

    torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM)
    torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    n = len(steps)
    return {
        "policy_loss": policy_loss_sum / n, "value_loss": value_loss_sum / n,
        "classification_loss": classification_loss_sum / n, "entropy": entropy_sum / n,
        "value_predictions": value_predictions, "return_targets": return_targets,
    }


def build_model(n_grid_channels, device):
    model = InfernoModel(n_grid_channels=n_grid_channels).to(device)
    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    model.load_state_dict(torch.load(WARM_START_CKPT, map_location=device, weights_only=False))
    print(f"[{RUN_TAG}] Warm-started from {WARM_START_CKPT} (plain load, old-loop architecture).")
    return model


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    device = get_device()
    print(f"[{RUN_TAG}] {N_EPISODES} episodes, device={device}  "
          f"(old loop's exact rollout/update mechanism, value-head instrumentation only)")

    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)
    model = build_model(probe_obs["grid"].shape[0], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()

    with open(TRAIN_LOG_PATH, "w", newline="") as train_file:
        train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
        train_writer.writeheader()

        for ep in range(1, N_EPISODES + 1):
            ep_t0 = time.perf_counter()
            seed = BASE_SEED + ep

            try:
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, TRAINING_IGNITION_POINT, device, seed
                )
                result = update_policy_with_diagnostics(model, optimizer, steps, device, return_normalizer)
            except Exception as e:
                if not _is_mps_unimplemented_error(e):
                    raise
                print(f"[{RUN_TAG}] MPS op not implemented -- falling back to CPU for the rest of this run.")
                device = torch.device("cpu")
                model = model.to(device)
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, TRAINING_IGNITION_POINT, device, seed
                )
                result = update_policy_with_diagnostics(model, optimizer, steps, device, return_normalizer)

            V = result["value_predictions"]
            G = result["return_targets"]
            V_mean, V_std, _ = _mean_std_var(V)
            return_mean, return_std, var_return = _mean_std_var(G)
            residual = [g - v for g, v in zip(G, V)]
            _, _, var_residual = _mean_std_var(residual)
            explained_variance = (1 - var_residual / var_return) if var_return > 1e-12 else None
            corr = _pearson_corr(V, G)

            wall_s = time.perf_counter() - ep_t0
            train_writer.writerow({
                "episode": ep, "device": str(device), "n_ticks": len(steps),
                "reward": total_reward, "buildings_destroyed": buildings_destroyed,
                "contained": contained, "policy_loss": result["policy_loss"],
                "value_loss": result["value_loss"], "classification_loss": result["classification_loss"],
                "entropy": result["entropy"], "corr_V_return": corr,
                "explained_variance": explained_variance, "var_return": var_return,
                "var_residual": var_residual, "V_mean": V_mean, "V_std": V_std,
                "return_mean": return_mean, "return_std": return_std, "wall_time_s": wall_s,
            })
            train_file.flush()

            if ep % STATUS_EVERY == 0:
                corr_str = f"{corr:+.3f}" if corr is not None else "n/a"
                ev_str = f"{explained_variance:+.3f}" if explained_variance is not None else "n/a"
                print(f"[status @ ep {ep:4d}/{N_EPISODES}] reward={total_reward:10.1f}  "
                      f"value_loss={result['value_loss']:8.3f}  corr(V,G)={corr_str}  "
                      f"explained_var={ev_str}  V={V_mean:+.3f}+/-{V_std:.3f}  "
                      f"G={return_mean:+.3f}+/-{return_std:.3f}  device={device}", flush=True)

    print(f"\n[{RUN_TAG}] COMPLETE: {N_EPISODES} episodes. Full log: {TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()
