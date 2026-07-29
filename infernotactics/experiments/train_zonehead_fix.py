"""
Three fixes applied together, testing whether the zone head can be made to
actually use the excellent per-zone fire signal a diagnostic already found
sitting unused in its own input (diagnostic_zonehead_dependence.py, checkpoint
models/checkpoints_zonehead_zonehead_v1/episode_0300.pt): zone_pooled
correlates r=0.97 with real per-zone Threat+Blaze counts, but the zone
head's OUTPUT varied 26x less than the untouched resource_type head's and
correlated only r=-0.05 to -0.11 with real fire -- the head had the
information and wasn't using it.

Fix 1 (models/actor_critic.py, models/inferno_model.py): ZoneHead now reads
the RAW MLP branch output instead of the shared post-trunk hidden state h =
trunk(fused) that resource_type_head/value_head use. The per-zone weight-
sharing (same small MLP applied to every (global_features, zone_features)
pair) already existed in the prior version -- what's new is removing the
shared-trunk bottleneck, which mixed the (separately confirmed near-
constant) globally-pooled CNN vector together with the MLP output through
ONE Linear layer optimized jointly for all three heads' losses. Param count:
see verify step below.

Fix 2 (this file, update_policy_with_zone_aux): auxiliary cross-entropy loss
training the zone logits toward the REAL per-zone active-fire distribution
(Threat+Blaze counts per zone, normalized; uniform if no fire anywhere yet),
computed from each stored tick's own grid channel -- no extra env access
needed at update time. Weighted by AUX_ZONE_LOSS_COEFF (default 0.1 -- see
module-level comment for why) relative to the existing policy/value/
classification/entropy terms. Gives the zone head dense per-tick supervised
gradient, the same role the classification head already plays for the
per-cell CNN trunk, instead of relying only on the episode-end policy
gradient signal that (per the whole v4-v6 investigation) is known to be
high-variance and slow for a 150-tick sparse-reward episode.

Fix 3: NOT explicitly coded here as separate logic -- warm-starting from
models/checkpoints_bisect/episode_0260.pt (the pre-zonehead checkpoint, same
as train_zonehead_shaping.py used) naturally cold-starts zone_head, since
that checkpoint has no actor_critic.zone_head.net.* keys at all (it predates
ZoneHead entirely -- its own zone_head.weight/zone_head.bias are the OLD
plain-Linear architecture's keys, a different shape/name, so they load as
"unexpected" and get dropped, never overwriting the new zone_head). Verified
before writing this script's warm-start loader (see console output).
Everything else (cnn, mlp, classifier, actor_critic.trunk/resource_type_head/
value_head) loads unchanged.

Reuses train_actor_critic.py's OWN compute_returns/RunningMeanStd/get_device
via direct import (unmodified) -- old-loop mechanics only, no GAE/
minibatching/advantage-z-scoring. collect_rollout and update_policy are new
variants (need to additionally capture/consume per-tick zone-fire targets).

Direct test tracked at every eval (per the pre-agreed protocol):
  (a) do single_training and stone_canyon's deterministic-eval argmax zones
      DIFFER at this checkpoint?
  (b) correlation between zone logits and real per-zone fire counts, over a
      fixed set of 10 diverse (scenario, tick) probe observations -- started
      at r=-0.05 in the diagnostic; the leading indicator is whether it
      climbs toward/above ~0.3 BEFORE (a) necessarily changes.

    INFERNO_RUN_TAG=zonehead_fix1 INFERNO_N_EPISODES=400 python -m src.train.train_zonehead_fix
"""

import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.fire_sim import BLAZE, THREAT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    MULTI_IGNITION_TRAINING_SCENARIO,
    RESOURCE_TYPES,
    TRAINING_IGNITION_POINT,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402
from train.train_actor_critic import (  # noqa: E402
    CLASSIFICATION_LOSS_COEFF,
    ENTROPY_COEFF,
    GAMMA,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    RESOURCE_ENTROPY_MAX,
    VALUE_LOSS_COEFF,
    ZONE_ENTROPY_MAX,
    RunningMeanStd,
    _is_mps_unimplemented_error,
    compute_returns,
    get_device,
)

RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
if not RUN_TAG:
    raise SystemExit("INFERNO_RUN_TAG is required (distinct checkpoint dir per run).")
if RUN_TAG in ("v7", "zonehead_v1", "pool8x8"):
    raise SystemExit(f"Refusing to run under RUN_TAG={RUN_TAG!r} -- that collides with an "
                      f"earlier experiment's paths. Pick a distinct tag.")

N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 400))
BASE_SEED = 2000
# 0.1: between ENTROPY_COEFF (0.01, a light regularizer) and
# CLASSIFICATION_LOSS_COEFF (0.3, the existing dense-per-tick-supervision
# precedent this aux loss is modeled on) -- chosen as a starting point
# meant to give real gradient signal without letting the aux term dominate
# the policy loss; verified sane (not wildly out of scale vs policy_loss)
# in the 25-episode smoke test before the real run, not just asserted here.
AUX_ZONE_LOSS_COEFF = float(os.environ.get("INFERNO_AUX_ZONE_COEFF", 0.1))
STATUS_EVERY = 20
CHECKPOINT_EVERY = int(os.environ.get("INFERNO_CHECKPOINT_EVERY", 25))
EVAL_EVERY = CHECKPOINT_EVERY  # kept equal so the eval table is written after every checkpoint, not just at the end
EVAL_EPISODES = 3

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_zonehead_{RUN_TAG}")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"train_log_zonehead_{RUN_TAG}.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, f"eval_log_zonehead_{RUN_TAG}.csv")
PROBE_LOG_PATH = os.path.join(LOG_DIR, f"probe_log_zonehead_{RUN_TAG}.jsonl")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints_bisect", "episode_0260.pt")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")

TRAINING_SCENARIOS = [
    ("single_training", TRAINING_IGNITION_POINT),
    ("stone_canyon", MULTI_IGNITION_TRAINING_SCENARIO[2]),
]
HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -19082.0},
    "stone_canyon": {"avg_reward": -45379.9},
}
KNOWN_GOOD = {"single_training": ("helicopter", 18)}  # the pre-existing, known-good solution

# Fixed probe set for direct-test metric (b): 2 scenarios x 5 ticks each,
# same construction as diagnostic_zonehead_dependence.py's 20-observation set
# but narrowed to this run's 2 training scenarios (10 total) so it's cheap to
# recompute at every eval checkpoint.
PROBE_TICKS = [0, 15, 40, 75, 120]

TRAIN_LOG_FIELDS = [
    "episode", "device", "scenario", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "aux_zone_loss", "entropy",
    "resource_entropy", "zone_entropy", "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario", "avg_reward", "avg_buildings_destroyed", "avg_buildings_saved",
    "containment_rate", "action_lock", "action_lock_fraction",
]
PROBE_LOG_FIELDS = [
    "episode", "zone_logit_corr_flat", "zone_logit_corr_per_obs_mean",
    "single_training_argmax_zone", "stone_canyon_argmax_zone", "zones_differ",
]


def _zone_active_fire_counts(fire_channel, zones):
    counts = []
    for z in zones:
        r0, r1 = z["row_range"]
        c0, c1 = z["col_range"]
        sub = fire_channel[r0:r1, c0:c1]
        counts.append(int(np.count_nonzero((sub == THREAT) | (sub == BLAZE))))
    return counts


def build_model_and_state(n_grid_channels, device):
    """Resumable: if RESUME_STATE_PATH exists (a prior attempt under this
    SAME RUN_TAG got at least CHECKPOINT_EVERY episodes in before being
    killed), loads model+optimizer+return_normalizer state from there and
    continues from completed_episodes+1 -- does NOT re-apply the Fix-3
    cold-start warm-start logic (that only ever happens once, at true
    episode 1). Otherwise warm-starts fresh from WARM_START_CKPT exactly as
    before (zone_head cold, everything else loaded unchanged) and starts at
    episode 1."""
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
        print(f"[zonehead_fix:{RUN_TAG}] RESUMED from {RESUME_STATE_PATH} at episode {start_ep}.")
        return model, optimizer, return_normalizer, start_ep

    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    old_sd = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(old_sd, strict=False)
    expected_missing = {"actor_critic.zone_head.net.0.weight", "actor_critic.zone_head.net.0.bias",
                        "actor_critic.zone_head.net.2.weight", "actor_critic.zone_head.net.2.bias"}
    expected_unexpected = {"actor_critic.zone_head.weight", "actor_critic.zone_head.bias"}
    assert set(missing) == expected_missing, f"unexpected missing-key set: {missing}"
    assert set(unexpected) == expected_unexpected, f"unexpected extra-key set: {unexpected}"
    print(f"[zonehead_fix:{RUN_TAG}] Warm-started from {WARM_START_CKPT} -- "
          f"zone_head cold (Fix 3), everything else loaded unchanged.")
    return model, optimizer, return_normalizer, 1


def save_resume_state(model, optimizer, return_normalizer, completed_episodes):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "return_normalizer_mean": return_normalizer.mean, "return_normalizer_var": return_normalizer.var,
        "return_normalizer_count": return_normalizer.count, "completed_episodes": completed_episodes,
    }, RESUME_STATE_PATH)


def collect_rollout(env, model, ignition_point, device, seed):
    """Same as train_actor_critic.collect_rollout, plus each tick's
    zone_fire_counts (real per-zone Threat+Blaze counts, Fix 2's aux
    target) computed straight from that tick's own grid channel -- no
    extra env access needed later in update_policy_with_zone_aux."""
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps = []
    total_reward = 0.0
    buildings_destroyed = 0
    done = False
    info = None

    with torch.no_grad():
        while not done:
            grid_t, scalars_t = InfernoModel.obs_to_tensors(obs, device=device)
            action_logits, _value, _classification_logits = model(grid_t, scalars_t)
            resource_idx = int(Categorical(logits=action_logits["resource_type"][0]).sample())
            zone_idx = int(Categorical(logits=action_logits["zone"][0]).sample())

            fire_state_target = fire_state_to_class(torch.from_numpy(obs["grid"][-1]).long())
            zone_fire_counts = _zone_active_fire_counts(obs["grid"][-1], env.zones)

            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, reward, done, info = env.step(action)

            steps.append({
                "grid": obs["grid"],
                "scalars": flatten_scalars(obs["scalars"]),
                "resource_idx": resource_idx,
                "zone_idx": zone_idx,
                "fire_state_target": fire_state_target,
                "zone_fire_counts": zone_fire_counts,
                "reward": reward,
            })
            total_reward += reward
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs

    return steps, total_reward, buildings_destroyed, info["contained"]


def update_policy_with_zone_aux(model, optimizer, steps, device, return_normalizer):
    """Same mechanics as train_actor_critic.update_policy (full-episode MC
    returns, running normalizer, per-tick backward(), two clip groups, one
    optimizer.step()), plus Fix 2's auxiliary zone cross-entropy term. See
    module docstring. Returns the same tuple as update_policy plus
    mean aux_zone_loss."""
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = return_normalizer.normalize(raw_returns)
    return_normalizer.update(raw_returns)

    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    optimizer.zero_grad()
    policy_loss_sum = value_loss_sum = classification_loss_sum = aux_zone_loss_sum = entropy_sum = 0.0
    resource_entropy_sum = zone_entropy_sum = 0.0

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
        resource_entropy = resource_dist.entropy()
        zone_entropy = zone_dist.entropy()
        entropy = resource_entropy + zone_entropy

        value_scalar = value.squeeze()
        g_t_tensor = torch.tensor(g_t, dtype=value_scalar.dtype, device=device)
        advantage = (g_t_tensor - value_scalar).detach()

        policy_loss = -log_prob * advantage
        value_loss = (value_scalar - g_t_tensor) ** 2
        classification_loss = F.cross_entropy(
            classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
        )

        # Fix 2: soft-label cross-entropy toward the REAL per-zone active-fire
        # distribution -- uniform if no active fire anywhere yet (tick 0 of
        # every episode), normalized counts otherwise. Manual (not
        # F.cross_entropy's soft-label path, for portability across torch
        # versions) -log_softmax dotted with the target distribution.
        zone_counts = torch.tensor(step["zone_fire_counts"], dtype=torch.float32, device=device)
        total = zone_counts.sum()
        zone_target = zone_counts / total if total > 0 else torch.full_like(zone_counts, 1.0 / len(zone_counts))
        aux_zone_loss = -(zone_target * F.log_softmax(action_logits["zone"][0], dim=0)).sum()

        tick_loss = (
            policy_loss
            + VALUE_LOSS_COEFF * value_loss
            + CLASSIFICATION_LOSS_COEFF * classification_loss
            + AUX_ZONE_LOSS_COEFF * aux_zone_loss
            - ENTROPY_COEFF * entropy
        )
        tick_loss.backward()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        classification_loss_sum += classification_loss.item()
        aux_zone_loss_sum += aux_zone_loss.item()
        entropy_sum += entropy.item()
        resource_entropy_sum += resource_entropy.item()
        zone_entropy_sum += zone_entropy.item()

    value_grad_norm = torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM)
    other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    n = len(steps)
    return (
        policy_loss_sum / n, value_loss_sum / n, classification_loss_sum / n, aux_zone_loss_sum / n, entropy_sum / n,
        resource_entropy_sum / n, zone_entropy_sum / n,
        float(value_grad_norm), float(other_grad_norm),
    )


def save_checkpoint(model, episode):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"episode_{episode:04d}.pt"))
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    results = {}
    for name, point in TRAINING_SCENARIOS:
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
        print(f"    [eval @ ep {episode_num}] {name}: avg_reward={result['avg_reward']:.1f}  "
              f"destroyed={result['avg_buildings_destroyed']:.1f}  containment={result['containment_rate']:.0%}  "
              f"action_lock={mc['action']} ({mc['fraction_of_ticks']:.1%}){delta_str}", flush=True)
    eval_file.flush()
    return results


def run_direct_test_probe(model, env, episode_num, device, probe_writer):
    """The pre-agreed direct test, computed fresh at every eval checkpoint:
    (a) do single_training/stone_canyon's deterministic-eval argmax zones
        (from run_eval_suite's action_lock, already computed) differ?
    (b) correlation between zone logits and real per-zone active-fire
        counts, over a fixed 10-observation probe set (2 scenarios x 5
        ticks, noop rollout -- same construction as
        diagnostic_zonehead_dependence.py). Started at r=-0.05 in that
        diagnostic; this is the leading indicator to watch."""
    was_training = model.training
    model.eval()
    zone_logits_all, active_counts_all = [], []
    with torch.no_grad():
        for name, point in TRAINING_SCENARIOS:
            obs = env.reset(ignition_point=point, scenario="single", seed=BASE_SEED, use_real_weather=True)
            tick = 0
            done = False
            for target_tick in PROBE_TICKS:
                while tick < target_tick:
                    obs, _r, done, _info = env.step(None)
                    tick += 1
                    if done:
                        break
                grid_t, scalars_t = InfernoModel.obs_to_tensors(obs, device=device)
                action_logits, _value, _cls = model(grid_t, scalars_t)
                zone_logits_all.append(action_logits["zone"][0].cpu().numpy())
                active_counts_all.append(_zone_active_fire_counts(obs["grid"][-1], env.zones))
                if done:
                    break
    if was_training:
        model.train()

    zone_logits_all = np.stack(zone_logits_all)
    active_counts_all = np.array(active_counts_all, dtype=np.float64)
    flat_corr = float(np.corrcoef(zone_logits_all.ravel(), active_counts_all.ravel())[0, 1])
    per_obs_corrs = []
    for i in range(zone_logits_all.shape[0]):
        zl, ac = zone_logits_all[i], active_counts_all[i]
        if ac.std() > 0 and zl.std() > 0:
            per_obs_corrs.append(float(np.corrcoef(zl, ac)[0, 1]))
    per_obs_mean = float(np.mean(per_obs_corrs)) if per_obs_corrs else float("nan")

    # (a) argmax zone at tick 0 of each scenario's own reset (cheap, direct
    # single-state readout -- action_lock in the eval CSV is the whole-
    # rollout version and is also printed by run_eval_suite for cross-check).
    argmax_zones = {}
    with torch.no_grad():
        for name, point in TRAINING_SCENARIOS:
            obs = env.reset(ignition_point=point, scenario="single", seed=BASE_SEED, use_real_weather=True)
            grid_t, scalars_t = InfernoModel.obs_to_tensors(obs, device=device)
            action_logits, _value, _cls = model(grid_t, scalars_t)
            argmax_zones[name] = int(torch.argmax(action_logits["zone"][0]))

    zones_differ = argmax_zones["single_training"] != argmax_zones["stone_canyon"]
    row = {
        "episode": episode_num, "zone_logit_corr_flat": flat_corr, "zone_logit_corr_per_obs_mean": per_obs_mean,
        "single_training_argmax_zone": argmax_zones["single_training"],
        "stone_canyon_argmax_zone": argmax_zones["stone_canyon"], "zones_differ": zones_differ,
    }
    probe_writer.writerow(row)
    print(f"    [DIRECT TEST @ ep {episode_num}] (a) argmax zones: single_training={argmax_zones['single_training']} "
          f"stone_canyon={argmax_zones['stone_canyon']}  DIFFER={zones_differ}  |  "
          f"(b) zone_logit-vs-real-fire corr: flat={flat_corr:.4f}  per_obs_mean={per_obs_mean:.4f}", flush=True)
    return row


def main():
    device = get_device()
    print(f"[zonehead_fix:{RUN_TAG}] target={N_EPISODES} episodes, device={device}  "
          f"AUX_ZONE_LOSS_COEFF={AUX_ZONE_LOSS_COEFF}  "
          f"training scenarios={[n for n, _ in TRAINING_SCENARIOS]}")

    env = InfernoEnv(seed=BASE_SEED)
    env.reset(seed=BASE_SEED)
    n_grid_channels = env.reset(seed=BASE_SEED)["grid"].shape[0]
    model, optimizer, return_normalizer, start_ep = build_model_and_state(n_grid_channels, device)
    scenario_rng = random.Random(BASE_SEED + 999 + start_ep)

    probe_csv_path = PROBE_LOG_PATH.replace(".jsonl", ".csv")
    resuming = start_ep > 1 and os.path.exists(TRAIN_LOG_PATH)
    write_mode = "a" if resuming else "w"
    train_file = open(TRAIN_LOG_PATH, write_mode, newline="")
    eval_file = open(EVAL_LOG_PATH, write_mode, newline="")
    probe_file = open(probe_csv_path, write_mode, newline="")
    train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
    eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
    probe_writer = csv.DictWriter(probe_file, fieldnames=PROBE_LOG_FIELDS)
    if write_mode == "w":
        train_writer.writeheader()
        eval_writer.writeheader()
        probe_writer.writeheader()

    recent_rewards = []
    stopped_reason = None
    ep = start_ep - 1

    try:
        for ep in range(start_ep, N_EPISODES + 1):
            ep_t0 = time.perf_counter()
            seed = BASE_SEED + ep
            scenario_name, ignition_point = scenario_rng.choice(TRAINING_SCENARIOS)

            try:
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, aux_zone_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy_with_zone_aux(
                    model, optimizer, steps, device, return_normalizer
                )
            except Exception as e:
                if not _is_mps_unimplemented_error(e):
                    raise
                print(f"[zonehead_fix] MPS op not implemented -- falling back to CPU.")
                device = torch.device("cpu")
                model = model.to(device)
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, aux_zone_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy_with_zone_aux(
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
                "classification_loss": classification_loss, "aux_zone_loss": aux_zone_loss, "entropy": entropy,
                "resource_entropy": resource_entropy, "zone_entropy": zone_entropy,
                "value_grad_norm": value_grad_norm, "other_grad_norm": other_grad_norm, "wall_time_s": wall_s,
            })
            train_file.flush()

            if ep % STATUS_EVERY == 0:
                print(f"[status @ ep {ep:5d}/{N_EPISODES}] scenario={scenario_name:16s} "
                      f"avg_reward(last{len(recent_rewards)})={sum(recent_rewards)/len(recent_rewards):10.1f}  "
                      f"aux_zone_loss={aux_zone_loss:.4f}  "
                      f"resource_entropy={resource_entropy:.3f}/{RESOURCE_ENTROPY_MAX:.3f}  "
                      f"zone_entropy={zone_entropy:.3f}/{ZONE_ENTROPY_MAX:.3f}  device={device}", flush=True)

            if ep % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, ep)
                save_resume_state(model, optimizer, return_normalizer, ep)

            if ep % EVAL_EVERY == 0:
                print(f"\n[zonehead_fix:{RUN_TAG}] ===== eval @ ep {ep} =====", flush=True)
                run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                run_direct_test_probe(model, env, ep, device, probe_writer)
                probe_file.flush()
                print(f"[zonehead_fix:{RUN_TAG}] ===== end eval =====\n", flush=True)

        stopped_reason = f"Reached target N_EPISODES={N_EPISODES}."

    finally:
        save_checkpoint(model, ep)
        save_resume_state(model, optimizer, return_normalizer, ep)
        train_file.close()
        eval_file.close()
        probe_file.close()

    print(f"\n[zonehead_fix:{RUN_TAG}] STOPPED at episode {ep}. Reason: {stopped_reason}")


if __name__ == "__main__":
    main()
