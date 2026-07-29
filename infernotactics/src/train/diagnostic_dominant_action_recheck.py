"""
Read-only diagnostic ONLY -- no training, no gradient updates, no checkpoint
writes. Run tag "diag_dominant_action_recheck_v1", writes only to
logs/diag_dominant_action_recheck_v1_results.json. Loads (never modifies)
models/checkpoints_zonehead_zonehead_fix1_2k/episode_*.pt.

Redoes the zonehead_fix1_2k run's "direct test" properly. The version
computed live during training used tick-0 argmax as its (a) metric -- a weak
test, since at tick 0 every scenario is just a ~3-cell ignition disk inside
a ~188,000-cell grid, so identical argmax across scenarios is the EXPECTED
outcome regardless of whether the policy has learned to differentiate
anything (the same issue diagnostic_zonehead_dependence.py's global-CNN-
pooled-vector variance check ran into). The eval log's WHOLE-ROLLOUT
dominant action (eval_policy(..., track_actions=True)'s most_common_action,
aggregated over an entire ~150-tick deterministic episode as fire actually
spreads) is the metric that showed real differentiation from ~ep500 onward
(single_training -> (helicopter, 18), stone_canyon -> (water_team, 21)) --
this script re-derives that number directly from the saved checkpoints,
scenario by scenario, rather than trusting the training run's own live log.

For every checkpoint from episode 500 to 2000 (step 25, 61 checkpoints):
  - single_training, stone_canyon: avg_reward, containment_rate,
    avg_buildings_destroyed, dominant (resource_type, zone) + its fraction
    of ticks, via eval_policy(deterministic=True, seed=BASE_SEED,
    n_episodes=3, track_actions=True) -- identical protocol to the training
    run's own run_eval_suite(), so numbers should reproduce exactly (this is
    itself a useful check -- if they don't reproduce, something is wrong).
  - The actual direct test: is single_training solved (>=+40 reward, 100%
    containment) AND does stone_canyon's dominant zone differ from
    single_training's dominant zone at this SAME checkpoint?

Then, among the 5 checkpoints with the best combined reward (single_training
+ stone_canyon avg_reward, summed), evaluates the 2 held-out validation
scenarios (mandeville_canyon, getty_view_park) the same way, against their
heuristic baselines (+30.5, +46.1).

    python -m src.train.diagnostic_dominant_action_recheck
"""

import json
import os
import sys

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
from train.train_actor_critic import get_device  # noqa: E402

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints_zonehead_zonehead_fix1_2k")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUT_PATH = os.path.join(LOG_DIR, "diag_dominant_action_recheck_v1_results.json")

BASE_SEED = 2000
EVAL_EPISODES = 3
CHECKPOINTS = list(range(500, 2001, 25))  # 500, 525, ..., 2000 -- 61 checkpoints

TRAINING_SCENARIOS = [
    ("single_training", TRAINING_IGNITION_POINT),
    ("stone_canyon", MULTI_IGNITION_TRAINING_SCENARIO[2]),
]
HELDOUT_SCENARIOS = list(VALIDATION_IGNITION_POINTS.items())
HEURISTIC_BASELINE = {
    "single_training": -19082.0,
    "stone_canyon": -45379.9,
    "mandeville_canyon": 30.5,
    "getty_view_park": 46.1,
}
SOLVED_REWARD_THRESHOLD = 40.0  # single_training's known-good solution is +42.4
SOLVED_CONTAINMENT_THRESHOLD = 1.0


def eval_checkpoint(model, env, device, scenarios):
    results = {}
    for name, point in scenarios:
        r = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                         use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                         track_actions=True)
        mc = r["most_common_action"]
        results[name] = {
            "avg_reward": r["avg_reward"],
            "containment_rate": r["containment_rate"],
            "avg_buildings_destroyed": r["avg_buildings_destroyed"],
            "dominant_action": mc["action"],
            "dominant_action_fraction": mc["fraction_of_ticks"],
        }
    return results


def main():
    device = get_device()
    env = InfernoEnv(seed=BASE_SEED)
    env.reset(seed=BASE_SEED)
    n_grid_channels = env.reset(seed=BASE_SEED)["grid"].shape[0]

    per_checkpoint = {}
    for ep in CHECKPOINTS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt")
        model = InfernoModel(n_grid_channels=n_grid_channels).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        model.eval()

        r = eval_checkpoint(model, env, device, TRAINING_SCENARIOS)
        st, sc = r["single_training"], r["stone_canyon"]

        st_solved = st["avg_reward"] >= SOLVED_REWARD_THRESHOLD and st["containment_rate"] >= SOLVED_CONTAINMENT_THRESHOLD
        zones_differ = st["dominant_action"][1] != sc["dominant_action"][1]
        direct_test_pass = st_solved and zones_differ

        per_checkpoint[ep] = {
            "single_training": st, "stone_canyon": sc,
            "single_training_solved": st_solved,
            "dominant_zones_differ": zones_differ,
            "direct_test_pass": direct_test_pass,
            "combined_reward": st["avg_reward"] + sc["avg_reward"],
        }
        print(f"ep {ep:5d}: single_training reward={st['avg_reward']:9.1f} contain={st['containment_rate']:.0%} "
              f"dom={st['dominant_action']}({st['dominant_action_fraction']:.1%})  |  "
              f"stone_canyon reward={sc['avg_reward']:9.1f} contain={sc['containment_rate']:.0%} "
              f"dom={sc['dominant_action']}({sc['dominant_action_fraction']:.1%})  |  "
              f"solved={st_solved}  zones_differ={zones_differ}  DIRECT_TEST_PASS={direct_test_pass}", flush=True)

    passing = [ep for ep, v in per_checkpoint.items() if v["direct_test_pass"]]
    print(f"\n=== DIRECT TEST: checkpoints where single_training is solved AND "
          f"stone_canyon's dominant zone differs: {passing if passing else 'NONE'} ===\n")

    # Top 5 by combined reward
    ranked = sorted(per_checkpoint.items(), key=lambda kv: kv[1]["combined_reward"], reverse=True)
    top5 = [ep for ep, _ in ranked[:5]]
    print(f"Top 5 checkpoints by combined (single_training + stone_canyon) reward: {top5}")
    for ep in top5:
        v = per_checkpoint[ep]
        print(f"  ep {ep}: combined={v['combined_reward']:.1f}  "
              f"single_training={v['single_training']['avg_reward']:.1f}/{v['single_training']['containment_rate']:.0%}  "
              f"stone_canyon={v['stone_canyon']['avg_reward']:.1f}/{v['stone_canyon']['containment_rate']:.0%}  "
              f"solved={v['single_training_solved']}  zones_differ={v['dominant_zones_differ']}")

    print(f"\n=== Held-out evaluation at top-5 checkpoints ===")
    heldout_results = {}
    for ep in top5:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt")
        model = InfernoModel(n_grid_channels=n_grid_channels).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        model.eval()
        r = eval_checkpoint(model, env, device, HELDOUT_SCENARIOS)
        heldout_results[ep] = r
        for name, res in r.items():
            baseline = HEURISTIC_BASELINE[name]
            print(f"  ep {ep} {name}: reward={res['avg_reward']:9.1f}  contain={res['containment_rate']:.0%}  "
                  f"destroyed={res['avg_buildings_destroyed']:.1f}  dom={res['dominant_action']}"
                  f"({res['dominant_action_fraction']:.1%})  vs heuristic {baseline:+.1f} "
                  f"-> delta={res['avg_reward']-baseline:+.1f}")

    out = {
        "checkpoints": CHECKPOINTS,
        "per_checkpoint": {str(k): v for k, v in per_checkpoint.items()},
        "direct_test_passing_checkpoints": passing,
        "top5_by_combined_reward": top5,
        "heldout_at_top5": {str(k): v for k, v in heldout_results.items()},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
