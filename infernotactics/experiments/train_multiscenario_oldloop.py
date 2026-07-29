"""
Multi-scenario generalization attempt using the OLD training loop -- the ONE
mechanism this whole investigation has proven healthy (the bisect: 300 clean
episodes on the frozen v5 environment, entropy never collapsed, solved
single_training at +29,453.3 vs heuristic). Deliberately does NOT use GAE,
minibatching, or per-episode advantage z-scoring -- those three (individually
and in combination) are the established collapse mechanism from the
toggle-diff investigation. This reuses train_actor_critic.py's OWN
collect_rollout/compute_returns/RunningMeanStd/update_policy via direct
import (unmodified) so the update mechanics are guaranteed identical to what
the bisect already validated.

Unlike v1 (which trained on ONLY the 3 MULTI_IGNITION_TRAINING_SCENARIO
points, never single_training) and v2/v3 (which added single_training plus
task-conflict fixes -- scenario one-hot, per-scenario return normalizers,
per-scenario value heads -- all on the OLD 4-depot environment), this run
trains on all 4 points (single_training as anchor + the 3 multi-ignition
points) via uniform random per-episode sampling, on the frozen v5 8-station
environment, with NONE of v2/v3's task-conflict fixes -- testing whether the
plain old loop, with no fixes at all, can generalize once the environment's
resource model is realistic (the effectiveness sweep found helicopter
dominates ~5x over every other single type on single_training; if the
harder scenarios share that helicopter-favoring optimum, generalization may
be more tractable than v1's environment ever allowed).

Warm-started from models/checkpoints_bisect/episode_0260.pt. mandeville_canyon
and getty_view_park are evaluated every EVAL_EVERY episodes but NEVER trained
on (held-out generalization check).

Built-in stop points (NOT training-mechanic changes -- pure run-orchestration):
  - Entropy collapse: if BOTH resource_entropy and zone_entropy stay below 5%
    of their theoretical max for COLLAPSE_STREAK_NEEDED consecutive episodes,
    stop immediately and report. Threshold is a SUSTAINED-streak check, not an
    instant one -- the old loop is known to dip low and recover (e.g. down to
    0.0012-0.0023 for single episodes in the original 2000-episode run), so an
    instant single-episode trigger would be the same false-positive mistake
    v4's early entropy-floor attempt made.
  - Hard pause at episode 800 (only the first time it's reached -- a resumed
    invocation starting past 800 skips this): prints a held-out-scenario trend
    report and stops unconditionally, regardless of the requested
    INFERNO_N_EPISODES, so a human can decide whether continuing past 800 is
    warranted before more wall-clock time is spent.

Resumable: saves models/checkpoints_multiscenario_<tag>/resume_state.pt every
CHECKPOINT_EVERY episodes (model + optimizer + return-normalizer stats +
completed_episodes). Re-running with the same INFERNO_RUN_TAG picks up where
it left off instead of restarting from the warm-start checkpoint.

    INFERNO_RUN_TAG=v7 INFERNO_N_EPISODES=1500 python -m src.train.train_multiscenario_oldloop
"""

import csv
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    MULTI_IGNITION_TRAINING_SCENARIO,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
)
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402
from train.train_actor_critic import (  # noqa: E402
    GRAD_CLIP_NORM,  # noqa: F401 (imported for parity/documentation; used inside update_policy itself)
    LEARNING_RATE,
    RESOURCE_ENTROPY_MAX,
    ZONE_ENTROPY_MAX,
    RunningMeanStd,
    _is_mps_unimplemented_error,
    collect_rollout,
    compute_returns,  # noqa: F401 (used internally by update_policy; re-exported for clarity)
    get_device,
    update_policy,
)

RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
if not RUN_TAG:
    raise SystemExit("INFERNO_RUN_TAG is required (distinct checkpoint dir per run).")

N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 1500))
BASE_SEED = 2000
STATUS_EVERY = 20
CHECKPOINT_EVERY = 20
EVAL_EVERY = 20
EVAL_EPISODES = 3
REPORT_EPISODES = {100, 300, 600, 900, 1200, 1500}
PAUSE_AT_EPISODE = 800
COLLAPSE_ENTROPY_FRAC = 0.05
COLLAPSE_STREAK_NEEDED = 15

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_multiscenario_{RUN_TAG}")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"train_log_multiscenario_{RUN_TAG}.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, f"eval_log_multiscenario_{RUN_TAG}.csv")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints_bisect", "episode_0260.pt")

TRAINING_SCENARIOS = [
    ("single_training", TRAINING_IGNITION_POINT),
    ("topanga_ridge", MULTI_IGNITION_TRAINING_SCENARIO[0]),
    ("sullivan_canyon", MULTI_IGNITION_TRAINING_SCENARIO[1]),
    ("stone_canyon", MULTI_IGNITION_TRAINING_SCENARIO[2]),
]
EVAL_SCENARIOS = TRAINING_SCENARIOS + list(VALIDATION_IGNITION_POINTS.items())
VALIDATION_NAMES = set(VALIDATION_IGNITION_POINTS.keys())

# v5 (frozen 8-station env) heuristic baseline -- see train_actor_critic_multi_v4.py's
# own HEURISTIC_BASELINE (re-baselined via heuristic_policy.py on this same environment).
HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -19082.0, "avg_buildings_destroyed": 65.2, "containment_rate": 0.80},
    "topanga_ridge": {"avg_reward": -3446.3, "avg_buildings_destroyed": 12.8, "containment_rate": 0.80},
    "sullivan_canyon": {"avg_reward": -7821.5, "avg_buildings_destroyed": 31.6, "containment_rate": 0.80},
    "stone_canyon": {"avg_reward": -45379.9, "avg_buildings_destroyed": 167.0, "containment_rate": 0.60},
    "mandeville_canyon": {"avg_reward": 49.0, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
    "getty_view_park": {"avg_reward": 73.6, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
}

TRAIN_LOG_FIELDS = [
    "episode", "device", "scenario", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "resource_entropy", "zone_entropy", "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario", "avg_reward", "avg_buildings_destroyed", "avg_buildings_saved",
    "containment_rate", "action_lock", "action_lock_fraction",
]


def build_model_and_state(device):
    n_grid_channels = InfernoEnv(seed=BASE_SEED).reset(seed=BASE_SEED)["grid"].shape[0]
    model = InfernoModel(n_grid_channels=n_grid_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()

    if os.path.exists(RESUME_STATE_PATH):
        state = torch.load(RESUME_STATE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return_normalizer.mean = state["return_normalizer_mean"]
        return_normalizer.var = state["return_normalizer_var"]
        return_normalizer.count = state["return_normalizer_count"]
        start_ep = state["completed_episodes"] + 1
        print(f"[multiscenario:{RUN_TAG}] RESUMED from {RESUME_STATE_PATH} at episode {start_ep}.")
        return model, optimizer, return_normalizer, start_ep

    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    model.load_state_dict(torch.load(WARM_START_CKPT, map_location=device, weights_only=False))
    print(f"[multiscenario:{RUN_TAG}] Warm-started from {WARM_START_CKPT} (fresh optimizer/normalizer).")
    return model, optimizer, return_normalizer, 1


def save_resume_state(model, optimizer, return_normalizer, completed_episodes):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "return_normalizer_mean": return_normalizer.mean, "return_normalizer_var": return_normalizer.var,
        "return_normalizer_count": return_normalizer.count, "completed_episodes": completed_episodes,
    }, RESUME_STATE_PATH)


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    results = {}
    for name, point in EVAL_SCENARIOS:
        result = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                              use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                              track_actions=True)
        results[name] = result
        mc = result["most_common_action"]
        eval_writer.writerow({
            "episode": episode_num, "scenario": name, "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
            "action_lock": str(mc["action"]), "action_lock_fraction": mc["fraction_of_ticks"],
        })
        baseline = HEURISTIC_BASELINE.get(name)
        delta_str = ""
        if baseline is not None:
            delta = result["avg_reward"] - baseline["avg_reward"]
            delta_str = f"  vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})"
        held_out_tag = " [HELD-OUT]" if name in VALIDATION_NAMES else ""
        print(f"    [eval @ ep {episode_num}] {name}{held_out_tag}: avg_reward={result['avg_reward']:.1f}  "
              f"destroyed={result['avg_buildings_destroyed']:.1f}  containment={result['containment_rate']:.0%}  "
              f"action_lock={mc['action']} ({mc['fraction_of_ticks']:.1%}){delta_str}", flush=True)
    eval_file.flush()
    return results


def main():
    device = get_device()
    print(f"[multiscenario:{RUN_TAG}] target={N_EPISODES} episodes, device={device}  "
          f"training scenarios={[n for n, _ in TRAINING_SCENARIOS]}  "
          f"held-out={sorted(VALIDATION_NAMES)}")

    env = InfernoEnv(seed=BASE_SEED)
    env.reset(seed=BASE_SEED)
    model, optimizer, return_normalizer, start_ep = build_model_and_state(device)
    scenario_rng = random.Random(BASE_SEED + 999 + start_ep)

    write_mode = "a" if start_ep > 1 and os.path.exists(TRAIN_LOG_PATH) else "w"
    train_file = open(TRAIN_LOG_PATH, write_mode, newline="")
    eval_file = open(EVAL_LOG_PATH, write_mode, newline="")
    train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
    eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
    if write_mode == "w":
        train_writer.writeheader()
        eval_writer.writeheader()

    collapse_streak = 0
    recent_rewards = []
    stopped_reason = None

    try:
        for ep in range(start_ep, N_EPISODES + 1):
            ep_t0 = time.perf_counter()
            seed = BASE_SEED + ep
            scenario_name, ignition_point = scenario_rng.choice(TRAINING_SCENARIOS)

            try:
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy(
                    model, optimizer, steps, device, return_normalizer
                )
            except Exception as e:
                if not _is_mps_unimplemented_error(e):
                    raise
                print(f"[multiscenario] MPS op not implemented -- falling back to CPU.")
                device = torch.device("cpu")
                model = model.to(device)
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy(
                    model, optimizer, steps, device, return_normalizer
                )

            wall_s = time.perf_counter() - ep_t0
            recent_rewards.append(total_reward)
            if len(recent_rewards) > STATUS_EVERY:
                recent_rewards.pop(0)

            train_writer.writerow({
                "episode": ep, "device": str(device), "scenario": scenario_name, "n_ticks": len(steps),
                "reward": total_reward, "buildings_destroyed": buildings_destroyed, "contained": contained,
                "policy_loss": policy_loss, "value_loss": value_loss,
                "classification_loss": classification_loss, "entropy": entropy,
                "resource_entropy": resource_entropy, "zone_entropy": zone_entropy,
                "value_grad_norm": value_grad_norm, "other_grad_norm": other_grad_norm, "wall_time_s": wall_s,
            })
            train_file.flush()

            if (resource_entropy < COLLAPSE_ENTROPY_FRAC * RESOURCE_ENTROPY_MAX
                    and zone_entropy < COLLAPSE_ENTROPY_FRAC * ZONE_ENTROPY_MAX):
                collapse_streak += 1
            else:
                collapse_streak = 0

            if ep % STATUS_EVERY == 0:
                print(f"[status @ ep {ep:5d}/{N_EPISODES}] scenario={scenario_name:16s} "
                      f"avg_reward(last{len(recent_rewards)})={sum(recent_rewards)/len(recent_rewards):10.1f}  "
                      f"resource_entropy={resource_entropy:.3f}/{RESOURCE_ENTROPY_MAX:.3f}  "
                      f"zone_entropy={zone_entropy:.3f}/{ZONE_ENTROPY_MAX:.3f}  "
                      f"collapse_streak={collapse_streak}  device={device}", flush=True)

            if ep % CHECKPOINT_EVERY == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt"))
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))
                save_resume_state(model, optimizer, return_normalizer, ep)

            if ep % EVAL_EVERY == 0 and ep not in REPORT_EPISODES and ep != PAUSE_AT_EPISODE:
                run_eval_suite(model, env, ep, eval_writer, eval_file, device)

            if collapse_streak >= COLLAPSE_STREAK_NEEDED:
                stopped_reason = (f"ENTROPY COLLAPSE: both heads stayed below {COLLAPSE_ENTROPY_FRAC:.0%} of max "
                                   f"for {collapse_streak} consecutive episodes (through ep {ep}).")
                print(f"\n[multiscenario:{RUN_TAG}] STOPPING -- {stopped_reason}", flush=True)
                break

            if ep in REPORT_EPISODES or ep == PAUSE_AT_EPISODE:
                print(f"\n[multiscenario:{RUN_TAG}] ===== MILESTONE @ ep {ep} =====", flush=True)
                run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                print(f"[multiscenario:{RUN_TAG}] ===== end milestone =====\n", flush=True)

            if ep == PAUSE_AT_EPISODE:
                stopped_reason = f"SCHEDULED PAUSE at episode {PAUSE_AT_EPISODE} for held-out-trend review."
                print(f"\n[multiscenario:{RUN_TAG}] STOPPING -- {stopped_reason}", flush=True)
                break

        else:
            stopped_reason = f"Reached target N_EPISODES={N_EPISODES}."

    finally:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))
        save_resume_state(model, optimizer, return_normalizer,
                           ep if 'ep' in dir() else start_ep - 1)
        train_file.close()
        eval_file.close()

    print(f"\n[multiscenario:{RUN_TAG}] STOPPED at episode {ep}. Reason: {stopped_reason}")


if __name__ == "__main__":
    main()
