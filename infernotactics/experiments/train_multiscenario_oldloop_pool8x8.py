"""
Zone-head spatial-resolution experiment -- isolated copy of
train_multiscenario_oldloop.py, changed in exactly one way: the CNN's pooled
branch uses adaptive_pool_size=(8,8) instead of the default (4,4) (see
cnn_branch.py's CNNBranch -- the constructor parameter added specifically so
this experiment does NOT touch the module-level ADAPTIVE_POOL_SIZE default
that the concurrent v7 run (INFERNO_RUN_TAG=v7, PID-tracked separately) and
its models/checkpoints_multiscenario_v7/resume_state.pt depend on).

Hypothesis under test: v7 showed single_training locking onto (helicopter,
zone 18) and stone_canyon locking onto (helicopter, zone 19) -- adjacent
80-cell zones -- in a way that's perfectly anti-correlated across 9
consecutive checkpoints (the policy holds one or the other, never both).
zone 18/19 are ~20x37 cells each; the OLD (4,4) pooling's downsampled
79x148 feature map means each of the 16 pooled bins averages roughly a
20x37-cell region -- plausibly collapsing both zones into literally the
same pooled feature, so the actor's zone head structurally cannot tell them
apart no matter how much training happens. (8,8) pooling quadruples the
bins (64 total), roughly halving each bin's footprint in each spatial
dimension -- if that's the real mechanism, zone 18 vs 19 should become
separable and the anti-correlation should break.

Everything else is identical to train_multiscenario_oldloop.py: same OLD
training loop (train_actor_critic.py's collect_rollout/update_policy,
unmodified import -- no GAE, no minibatching, no advantage z-scoring), same
4 training scenarios (single_training + topanga_ridge/sullivan_canyon/
stone_canyon), same 2 held-out validation scenarios, same BASE_SEED=2000,
same per-episode scenario sampling via the same seeded random.Random.

Warm start: partially loaded from the SAME models/checkpoints_bisect/
episode_0260.pt v7 warm-starts from. Verified beforehand (see
verify_pool8x8.py) that changing adaptive_pool_size from (4,4) to (8,8)
changes exactly one parameter tensor's shape -- cnn.pool_fc.0.weight,
(128,1024) -> (128,4096) -- because only the flattened pooled-bin count
feeds that Linear's input dim; its bias ((128,), tied only to the output
dim) and every other layer (mlp, actor_critic, classifier) are shape-
identical, since POOLED_DIM=128 itself doesn't change. So the warm start
copies every matching-shape tensor unchanged and leaves ONLY
cnn.pool_fc.0.weight randomly initialized -- everything the old checkpoint
learned (per-cell CNN trunk, MLP, actor/critic heads, classifier) transfers;
only the newly-higher-resolution pooled-vector's fusion-in weights start
fresh.

Extra logging beyond the base script: per-scenario softmax max-probability
of the argmax (resource_type, zone) pair at tick 0 of each eval scenario's
deterministic rollout (see _confidence_probe) -- the base script's
action_lock/action_lock_fraction (from eval_policy(track_actions=True))
already reports the argmax action and how much of a whole eval rollout it
holds; this adds the actual softmax confidence at that action, matching the
diagnostic_collapsed_action_probe.py convention used earlier in this
investigation. Written as new EVAL_LOG_FIELDS columns, no change to eval.py.

Refuses to run under INFERNO_RUN_TAG=v7 (or empty) -- must be a distinct
tag, distinct checkpoint dir, distinct log files, per explicit instruction
not to touch the concurrent 1500-episode v7 run.

    INFERNO_RUN_TAG=pool8x8 INFERNO_N_EPISODES=300 python -m src.train.train_multiscenario_oldloop_pool8x8
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
    RESOURCE_TYPES,
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

ADAPTIVE_POOL_SIZE = (8, 8)  # this experiment's one intentional change (default is (4,4))

RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
if not RUN_TAG:
    raise SystemExit("INFERNO_RUN_TAG is required (distinct checkpoint dir per run).")
if RUN_TAG == "v7":
    raise SystemExit("Refusing to run under RUN_TAG='v7' -- that is the concurrent 1500-episode "
                      "run this experiment must not touch. Pick a distinct tag (e.g. 'pool8x8').")

N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 300))
BASE_SEED = 2000
STATUS_EVERY = 20
CHECKPOINT_EVERY = 20
EVAL_EVERY = 20
EVAL_EPISODES = 3
REPORT_EPISODES = {100, 200, 300}
PAUSE_AT_EPISODE = 100_000  # effectively disabled -- this run is fixed at 300 episodes, no scheduled pause
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

# Same v5 (frozen 8-station env) heuristic baseline train_multiscenario_oldloop.py uses --
# these are environment/heuristic-policy facts, independent of the model architecture
# change under test here, so reusing them unchanged is correct, not a shortcut.
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
    "t0_resource_argmax", "t0_resource_max_prob", "t0_zone_argmax", "t0_zone_max_prob",
]


def _confidence_probe(model, env, ignition_point, device):
    """One deterministic forward pass at tick 0 of `ignition_point`'s reset
    observation -- reports the argmax (resource_type, zone) and each head's
    softmax max-probability there, matching the
    diagnostic_collapsed_action_probe.py convention (argmax + max_prob per
    head). This is a SINGLE state's snapshot, complementary to
    action_lock/action_lock_fraction (which already summarizes the argmax
    action's frequency across an ENTIRE deterministic eval rollout) -- not a
    replacement for it. No gradients, no env mutation beyond this one reset."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        obs = env.reset(ignition_point=ignition_point, seed=BASE_SEED, use_real_weather=True)
        grid, scalars = InfernoModel.obs_to_tensors(obs, device=device)
        action_logits, _value, _cls = model(grid, scalars)
        resource_probs = torch.softmax(action_logits["resource_type"][0], dim=0)
        zone_probs = torch.softmax(action_logits["zone"][0], dim=0)
        r_idx = int(torch.argmax(resource_probs))
        z_idx = int(torch.argmax(zone_probs))
        result = {
            "t0_resource_argmax": RESOURCE_TYPES[r_idx],
            "t0_resource_max_prob": float(resource_probs[r_idx]),
            "t0_zone_argmax": z_idx,
            "t0_zone_max_prob": float(zone_probs[z_idx]),
        }
    if was_training:
        model.train()
    return result


def build_model_and_state(device):
    n_grid_channels = InfernoEnv(seed=BASE_SEED).reset(seed=BASE_SEED)["grid"].shape[0]
    model = InfernoModel(n_grid_channels=n_grid_channels, adaptive_pool_size=ADAPTIVE_POOL_SIZE).to(device)
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
        print(f"[pool8x8:{RUN_TAG}] RESUMED from {RESUME_STATE_PATH} at episode {start_ep}.")
        return model, optimizer, return_normalizer, start_ep

    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    old_sd = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    new_sd = model.state_dict()
    skipped = [k for k in old_sd if old_sd[k].shape != new_sd[k].shape]
    loaded = {k: v for k, v in old_sd.items() if k not in skipped}
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    print(f"[pool8x8:{RUN_TAG}] Warm-started from {WARM_START_CKPT} (partial load -- "
          f"architecture-changed keys reinitialized, not loaded): {skipped}")
    assert set(missing) == set(skipped), f"unexpected missing keys beyond the known shape-mismatch: {missing}"
    assert not unexpected, f"unexpected keys in checkpoint not in model: {unexpected}"
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
        probe = _confidence_probe(model, env, point, device)
        results[name] = result
        mc = result["most_common_action"]
        eval_writer.writerow({
            "episode": episode_num, "scenario": name, "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
            "action_lock": str(mc["action"]), "action_lock_fraction": mc["fraction_of_ticks"],
            **probe,
        })
        baseline = HEURISTIC_BASELINE.get(name)
        delta_str = ""
        if baseline is not None:
            delta = result["avg_reward"] - baseline["avg_reward"]
            delta_str = f"  vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})"
        held_out_tag = " [HELD-OUT]" if name in VALIDATION_NAMES else ""
        print(f"    [eval @ ep {episode_num}] {name}{held_out_tag}: avg_reward={result['avg_reward']:.1f}  "
              f"destroyed={result['avg_buildings_destroyed']:.1f}  containment={result['containment_rate']:.0%}  "
              f"action_lock={mc['action']} ({mc['fraction_of_ticks']:.1%})  "
              f"t0=({probe['t0_resource_argmax']},{probe['t0_zone_argmax']}) "
              f"p={probe['t0_resource_max_prob']:.3f}/{probe['t0_zone_max_prob']:.3f}{delta_str}", flush=True)
    eval_file.flush()
    return results


def main():
    device = get_device()
    print(f"[pool8x8:{RUN_TAG}] target={N_EPISODES} episodes, device={device}  "
          f"adaptive_pool_size={ADAPTIVE_POOL_SIZE}  "
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
                print(f"[pool8x8] MPS op not implemented -- falling back to CPU.")
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
                print(f"\n[pool8x8:{RUN_TAG}] STOPPING -- {stopped_reason}", flush=True)
                break

            if ep in REPORT_EPISODES or ep == PAUSE_AT_EPISODE:
                print(f"\n[pool8x8:{RUN_TAG}] ===== MILESTONE @ ep {ep} =====", flush=True)
                run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                print(f"[pool8x8:{RUN_TAG}] ===== end milestone =====\n", flush=True)

            if ep == PAUSE_AT_EPISODE:
                stopped_reason = f"SCHEDULED PAUSE at episode {PAUSE_AT_EPISODE} for held-out-trend review."
                print(f"\n[pool8x8:{RUN_TAG}] STOPPING -- {stopped_reason}", flush=True)
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

    print(f"\n[pool8x8:{RUN_TAG}] STOPPED at episode {ep}. Reason: {stopped_reason}")


if __name__ == "__main__":
    main()
