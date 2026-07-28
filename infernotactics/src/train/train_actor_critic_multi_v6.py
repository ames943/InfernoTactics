"""
Multi-ignition training, v6 -- multi-trajectory batched PPO, the next test after
v4/v5 exhausted the credit-assignment hypothesis's single-trajectory levers.

Full inherited rationale (GAE formulation, warm-start, reward shaping, continuous
ignition randomization, anchor replay, anchor-identity flag, KL/entropy early
stopping) is documented in train_actor_critic_multi_v4.py's module docstring and
is UNCHANGED here except for the one thing this file exists to test. Read that
file first if any of the constants/functions below look unfamiliar -- they're
copied verbatim.

Why this file exists: v1/v2/v3 (task-conflict fixes -- per-scenario return
normalization, scenario-ID input, separate value heads) all ended flat. v4/v5
(GAE + multi-epoch PPO, then a systematic sweep of PPO_EPOCHS 4/2/1, gradient
clip magnitude 0.5/0.05, learning rate 3e-4/3e-5, and advantage-estimate
clipping +-3std) ALL SIX variants still collapsed to near-zero entropy, ranging
from ep20 (tight grad clip) to ep170-189 (baseline) -- and the collapse happens
on single_training ALONE, no multi-scenario pressure required, so it isn't a
generalization artifact. Every v4/v5 lever changed HOW FAR a single episode's
gradient update could move the policy (epoch count, clip norm, step size,
advantage magnitude); none of them changed WHETHER one high-variance episode's
gradient could single-handedly dominate an update. v6 targets that directly.

THE CHANGE: instead of collecting one episode -> computing its GAE -> running a
PPO update on just those ~150 ticks -> repeat, this file collects
BATCH_SIZE_EPISODES independent episodes (different ignition/weather/seed
draws), computes GAE PER EPISODE (see below for why this must stay
per-episode), pools all episodes' ticks into one buffer, and runs ONE PPO
update over the pooled buffer. A single bad trajectory's ~150 ticks become at
most ~1/BATCH_SIZE_EPISODES of any minibatch on average (the existing
index-shuffle in ppo_update-equivalent code already mixes ticks from different
episodes into the same minibatch once given a pooled buffer -- no new shuffle
logic needed), and the advantage-standardization mean/std it's measured against
comes from the whole pooled batch instead of just its own episode.

TWO IMPLEMENTATION RISKS THIS FILE HANDLES EXPLICITLY (measured, not assumed):

1. GAE MUST stay per-episode. compute_gae()'s backward recursion
   (gae = delta + gamma*lambda*gae) assumes a single contiguous trajectory; if
   raw rewards/values from multiple episodes were concatenated and GAE run ONCE
   over the concatenation, advantage estimates would leak across the episode
   boundary (episode k's early ticks would get contaminated by episode k+1's
   signal). run_training_batch() below calls compute_gae() once per rollout and
   concatenates RESULTS, never raw trajectories.

2. The KL/entropy safety-net check (compute_batch_kl_and_entropy(), inherited
   from v4/v5 unchanged) rescans the ENTIRE steps buffer on every check. In
   v4/v5 that buffer was ~150 ticks (one episode); pooling to
   BATCH_SIZE_EPISODES episodes makes the buffer BATCH_SIZE_EPISODES times
   bigger AND produces BATCH_SIZE_EPISODES times more minibatches per update,
   so checking every single minibatch (as v4/v5 did) would make diagnostic cost
   grow QUADRATICALLY with batch size -- prohibitively slow well before memory
   becomes the bottleneck. ppo_update_batch() below checks only every
   `max(1, n_minibatches // SAFETY_CHECK_TARGET_COUNT)` minibatches instead,
   keeping total diagnostic cost roughly LINEAR in batch size again, at the
   honest cost of a coarser safety net (more minibatches of drift can
   accumulate between checks than v4/v5's every-minibatch version). Always
   force-checks the last minibatch of the batch so the final reported KL/entropy
   is never stale.

BATCH_SIZE_EPISODES=4 (not larger), chosen from a MEASURED constraint, not a
guess: each stored tick holds a full-resolution grid observation
((9,316,595) float32 = 6.77MB) plus a same-resolution classification target
(~1.5MB) -- roughly 8.3MB/tick, ~1.2GB for one 150-tick episode. This machine
has 19.3GB TOTAL RAM shared between CPU and GPU (Apple unified memory, so MPS
forward/backward competes for the same pool as the stored buffer). Batch=8
would mean ~10GB just for observations, too close to the ceiling alongside the
model/optimizer/env/OS. Batch=4 is ~5GB, leaving real headroom; can be raised
once actual peak memory during a run is confirmed safe.

reward_scale_normalizer.std() is read ONCE at the start of a batch and held
FROZEN across all BATCH_SIZE_EPISODES episodes in that batch (not re-read
between episodes), so every episode in a batch is scaled by the same factor --
re-reading mid-batch would let an early episode's update change the scale
applied to a later episode's GAE within the SAME update, breaking the "same
units" invariant GAE needs. The normalizer itself is updated once, with the
whole batch's pooled raw rewards, after all episodes' GAE is computed.

Smoke-test verification, matching prior rigor:
  - GAE correctness: unchanged from v4/v5 (compute_gae() itself is untouched,
    just called more times) -- src/train/test_gae.py still covers it, no new
    math introduced here.
  - Cross-episode minibatch mixing actually happening, not just assumed from
    the shuffle logic: each step is tagged with its episode_idx within the
    batch; every CHECKED minibatch records how many DISTINCT episode_idx
    values it contains, reported as mean_unique_episodes_per_checked_minibatch
    in the train log and end-of-run summary.
  - Same entropy/eval/grad-norm-spike signals as every prior smoke test,
    now logged per BATCH instead of per episode.

    python -m src.train.test_gae                                              # GAE unit test, run first
    INFERNO_N_EPISODES=50 INFERNO_BATCH_SIZE=4 python -m src.train.train_actor_critic_multi_v6  # smoke test
    python -m src.train.train_actor_critic_multi_v6                           # real run (after smoke test passes)
"""

import csv
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    MULTI_IGNITION_TRAINING_SCENARIO,
    RESOURCE_TYPES,
    SCALAR_KEYS,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402

# --- Hyperparameters ----------------------------------------------------------
N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 500))  # bounded follow-up, not a fresh full run
# BATCH_SIZE_EPISODES: the whole point of this file -- see module docstring's memory-bound
# rationale for why 4, not a larger number. N_EPISODES is still the total individual-episode
# target; the main loop groups them into batches of this size (last batch may be smaller if
# N_EPISODES isn't a multiple of this).
BATCH_SIZE_EPISODES = int(os.environ.get("INFERNO_BATCH_SIZE", 4))
# Target number of KL/entropy safety checks per batch update, REGARDLESS of batch size -- see
# module docstring's risk #2. Keeps diagnostic cost ~linear in batch size instead of quadratic.
SAFETY_CHECK_TARGET_COUNT = 15
BASE_SEED = 2000
LEARNING_RATE = float(os.environ.get("INFERNO_LEARNING_RATE", 3e-4))  # 3e-4 confirmed best-known (TEST 1 in v4/v5 made it worse)
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 1
PPO_CLIP_EPS = 0.2
MINIBATCH_SIZE = 16  # ticks per gradient step -- see module docstring's memory-footprint rationale (inherited from v1)
VALUE_LOSS_COEFF = 0.5
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.01
# GRAD_CLIP_NORM back to 0.5 (the original baseline), reverted from a 10x-tighter 0.05 test.
# That test found pre-clip other_grad_norm exceeded even the OLD 0.5 threshold in 195/195
# episodes -- clipping was already maximally aggressive at the vector-norm level -- and
# tightening it further collapsed FASTER and DEEPER (ep20-29 vs ep170), not slower: worse,
# not better. This rules out clip magnitude as the lever and points at update
# direction/Adam's per-parameter adaptivity (not bounded by vector-norm clipping) as the
# more likely amplifier. 0.5 is the confirmed best-known baseline for both remaining tests.
GRAD_CLIP_NORM = 0.5

# Advantage-estimate clipping for TEST 2 (see module docstring): independent of
# gradient-norm clipping, this bounds each INDIVIDUAL tick's standardized advantage
# (mean 0, std 1 over the episode's own batch, computed in ppo_update()) before it enters
# the surrogate loss -- rather than clipping the resulting gradient's aggregate norm AFTER
# one extreme tick has already skewed its direction. Diagnosed collapse pattern was a
# single high-variance trajectory producing a large, effectively unclippable-by-magnitude
# update (Part A: no detectable pre-collapse gradient buildup, confirmed twice) -- this
# targets that outlier tick directly at the source. ADV_CLIP_STD=3.0: for ~150 ticks/episode
# under an approximately normal advantage distribution, ~99.7% should fall within +-3 std;
# clipping there removes genuine tail outliers (the suspected collapse trigger) while
# leaving the bulk of the batch's signal untouched. Off by default (INFERNO_ADV_CLIP=1 to
# enable) so every other run/test (including TEST 1) is completely unaffected.
ADV_CLIP_ENABLED = os.environ.get("INFERNO_ADV_CLIP", "0") == "1"
ADV_CLIP_STD = float(os.environ.get("INFERNO_ADV_CLIP_STD", 3.0))

# Real-time gradient-norm spike detection (see ENTROPY_FLOOR's docstring for the
# ep100-103 buildup this is meant to surface earlier next time): flags when the
# current episode's pre-clip other_grad_norm exceeds GRAD_NORM_SPIKE_MULTIPLIER
# times the MEDIAN (not mean -- robust to a spike already sitting in the window)
# of the last GRAD_NORM_SPIKE_WINDOW episodes. 5x/20-episode window is a loose,
# exploratory threshold (not derived from a large sample of confirmed spikes --
# this run had exactly one) meant to catch a similar buildup with a few episodes
# of lead time, not a precisely calibrated statistical test.
GRAD_NORM_SPIKE_WINDOW = 20
GRAD_NORM_SPIKE_MULTIPLIER = 5.0

REWARD_SHAPING_ENABLED = os.environ.get("INFERNO_REWARD_SHAPING", "0") == "1"
SHAPING_COEFF = float(os.environ.get("INFERNO_SHAPING_COEFF", 0.02))
FIRE_CLASSES = (2, 3)  # Threat, Blaze -- see classification_head.fire_state_to_class

# Cold-start safety net for reward_scale_normalizer (see RunningMeanStd below): a
# 50-episode smoke test WITHOUT this found episode 1 alone drove value_loss from
# 7.23e5 to 3.6e8 and entropy to a hard 0 (policy collapse) within a single
# episode's PPO update. Root cause: RunningMeanStd defaults to std=1 until real
# data accumulates, but this env's per-tick rewards are already O(hundreds)
# (destroy_reward alone is -100 to -300ish/building) -- so episode 1's GAE
# targets were off by ~2-3 orders of magnitude from what the warm-started value
# head was calibrated to output. PPO's multi-epoch reuse of that SAME episode's
# data (4 epochs x ~10 minibatches = ~40 gradient steps, all clipped but all
# pointed the same "fix this huge error" direction) amplifies that cold-start
# mismatch far more than v1-v3's one-gradient-step-per-episode design ever
# could. Two independent fixes, both standard practice: (1) an informed prior
# std, order-of-magnitude reasoned from the reward function's own per-tick
# terms (not fit to this run's specific numbers), so episode 1 isn't starting
# from a blind std=1 guess; (2) hard reward clipping after scaling, the same
# trick the original PPO paper uses for Atari (clips raw reward to [-1,1] --
# here the SCALED reward is clipped instead, since rewards aren't already
# binary), as a safety net independent of how good the prior turns out to be.
INITIAL_REWARD_STD_PRIOR = 300.0
SCALED_REWARD_CLIP = 10.0

STATUS_EVERY = 20
CHECKPOINT_EVERY = 20
# Smoke-test regime (N_EPISODES<=100) evaluates every 10 episodes instead of every
# 50, specifically so single_training's reward trajectory can be watched across the
# WHOLE run, not just checked once at the end -- the smoke-test rerun's whole point
# is confirming KL early-stopping keeps the warm-started +32.1/100%-containment
# result from degrading at any point, not just recovering by ep 50 (see the
# conversation this fix came from: the first rerun's single ep-50 eval alone looked
# like "recovered", but had no visibility into how badly it dipped in between).
EVAL_EVERY = 10 if N_EPISODES <= 100 else 50
EVAL_EPISODES = 3
LOG_KL_EVERY_EPISODE = N_EPISODES <= 100  # smoke-test regime: log every episode; real run: every STATUS_EVERY

# Standard Spinning-Up-style PPO early stopping: after each epoch, if the
# approximate KL between the epoch's updated policy and the rollout's original
# policy exceeds KL_EARLY_STOP_THRESHOLD, stop taking further epochs on this
# episode's batch. Doesn't undo the epoch that tripped the threshold (standard
# practice doesn't either -- catching it before EVEN MORE epochs compound the
# same bad update is the point). TARGET_KL=0.02 is the common default (e.g.
# OpenAI Spinning Up's PPO); the 1.5x multiplier before stopping is also
# standard, giving a little slack for normal per-epoch noise. Added after the
# first v4 smoke test hit a KL of 10.5 on episode 3 (200-300x the healthy
# range) with FOUR full epochs run anyway, which correlates exactly with
# entropy nearly collapsing (0.0007) and the warm-started single_training
# result (+32.1, 100% containment) being destroyed (-119,815.7, 0% containment)
# by episode 50 -- a real causal story, not noise: one uncapped bad update.
TARGET_KL = 0.02
KL_EARLY_STOP_THRESHOLD = 1.5 * TARGET_KL

# Entropy-floor early stop: a real 500-episode run collapsed entropy to
# near-machine-epsilon (values like 1.9e-5, 7e-6, and literal 0.000000) for
# episodes 104-135+ straight, with ZERO recovery -- diagnosed from the raw
# per-episode CSV, not just the periodic status prints. Two hypotheses were
# tested against the data and BOTH were falsified: (1) it wasn't a slow
# decline that ep80-100's apparent "recovery" was already heading toward --
# entropy in that window was genuinely healthy, noisy variance (0.001-0.83
# across ep80-103); (2) it wasn't anchor-replay-specific -- anchor episodes
# (105,112,119,126,133) and non-anchor episodes (104,106-111,113...)
# collapsed together, at the same time, to the same degree, which rules out
# "the policy solving the anchor case is dragging global entropy down."
# The actual trigger: other_grad_norm (pre-clip) climbed sharply right
# before the collapse -- 28,261 (ep100) -> 32,097 (ep101) -> 16,805 (ep102)
# -> 80,483 (ep103) -- and ep104's entropy craters immediately after that
# spike and never recovers. Critically, from ep104 onward EVERY episode
# shows epochs_run=4, minibatches_run=40, early_stopped=False -- the
# KL-based safety net stopped firing entirely, right when it was needed
# most. This is not a bug in the KL check's implementation, it's a real
# mathematical blind spot: KL(old || new) is the divergence BETWEEN two
# distributions, and once both are already near-delta-functions pointing at
# the SAME action, that divergence is ~0 no matter how much the underlying
# logits keep growing -- softmax saturates, so "more confident" and
# "different" stop being the same thing. KL can catch a policy CHANGING
# direction; it cannot catch a policy that's already decided getting more
# extreme in the same direction. ENTROPY_FLOOR patches exactly this gap:
# checked alongside KL after every minibatch (see ppo_update()), it halts
# further epochs once the CURRENT entropy (not just KL) drops too low,
# independent of whether the policy is still "moving" in the KL sense.
# Value chosen from the empirical boundary in the same log: healthy noisy
# episodes dipped as low as ~0.0012-0.0023 (ep83, ep86) without collapsing
# further, while genuine collapse dropped below 1e-4 within one episode and
# never recovered. 5e-3 (~0.1% of the theoretical max entropy,
# log(4)+log(32)=4.852 for the 4-way resource/32-way zone factored
# distribution) sits comfortably above true collapse with margin, while
# only mildly conservative relative to the lowest healthy dips seen --
# erring toward catching collapse early, since the cost of a false-positive
# stop (one episode gets fewer epochs) is far lower than missing a real one.
ENTROPY_FLOOR = 5e-3

# Light anchor replay: after moving KL early-stopping to per-MINIBATCH granularity
# (see ppo_update()'s docstring), single_training was STILL destroyed by ep 10
# (-119,661.6, nearly identical to both earlier runs' ~-119,800) even though
# early-stopping now triggered in 10/10 episodes at an average of just 1.7
# minibatches -- i.e. updates were tightly controlled, not runaway. Directly
# re-evaluating the warm-started models/checkpoints/best.pt (no training at all)
# through this file's own eval path reproduced its known-good +32.1/100%
# containment exactly, ruling out a warm-start-loading bug. Conclusion: this
# isn't instability, it's ordinary catastrophic forgetting -- continuous
# ignition randomization structurally never resamples the exact Skull Rock
# point, so nothing ever reinforces that narrow, memorized-looking behavior
# while every other scenario's gradient signal pulls the shared weights
# elsewhere. No KL threshold fixes this, because it isn't an overshoot.
# ANCHOR_REPLAY_EVERY_N=7 (~14.3% of episodes) reignites TRAINING_IGNITION_POINT
# explicitly instead of a continuous-random point, keeping the anchor lightly
# reinforced so its trajectory is a real signal about whether the new GAE/PPO
# mechanism generalizes -- not an artifact of never revisiting it -- while the
# other ~85.7% of episodes still deliver the genuine broad-randomization
# pressure v4 is meant to test. Deterministic (ep % N == 0), so it needs no
# extra persisted RNG state and stays resumable for free.
ANCHOR_REPLAY_EVERY_N = 7

# Minimal scenario-identity signal (added after anchor replay ALONE was still not
# enough -- see below). With anchor replay live, single_training was STILL
# destroyed by ep 20 (-120,040.6, same degenerate number as every earlier
# config). A tick-by-tick action diagnostic comparing the original warm-started
# best.pt against v4's ep-20 checkpoint on the SAME deterministic single_training
# episode explained why: the original policy is itself a legitimate but narrow
# degenerate solution -- always dispatch helicopter to zone 18 (the one
# helicopter-only-reachable zone), extinguishing the fire in 4 ticks
# (resource_type logits [-0.35,-0.51,-0.77,0.76], helicopter wins by a modest
# margin). After just 20 episodes of v4 training, those SAME logits had blown
# up to [-0.64,25.25,25.61,5.1] -- trench_crew/rescue_vehicle now dominate by a
# huge margin, helicopter never gets picked. This isn't instability (each
# individual episode's update stayed within its own KL bound) and it isn't
# under-sampling (anchor replay was already firing every 7th episode) -- it's a
# genuine REPRESENTATIONAL conflict: with zero scenario-identity input, ONE
# shared set of actor logits has to serve every ignition point, and the optimal
# action for Skull Rock (fast, localized -> helicopter) is the OPPOSITE of
# what's broadly useful across the harder, far more common training
# distribution (slow, sprawling fires -> trench/rescue). ~6/7 episodes
# reinforce the "trench/rescue" direction against 1/7 reinforcing "helicopter,"
# so the majority signal wins on the shared weights regardless of how tightly
# any single episode's update is bounded. This is structurally the SAME class
# of problem v2/v3's scenario one-hot solved -- deliberately excluded from v4's
# first design to isolate the GAE/PPO/credit-assignment variable -- so the fix
# here is the SMALLEST possible version of that idea: a single binary scalar
# (1.0 if this episode's ignition is the exact anchor point, 0.0 otherwise)
# appended to the MLP's scalar input, giving the shared logits just enough
# signal to branch into two regimes. Deliberately NOT the full v2/v3 apparatus
# (no per-scenario value heads, no per-scenario return normalizers) -- if v4
# works with only this minimal addition, the win can still be attributed to
# GAE/PPO/credit-assignment, with the anchor flag documented as a small,
# necessary, minimal addition rather than a confound reopening the
# reward-scale-conflict question v2/v3 already closed.
ANCHOR_FLAG_DIM = 1
N_SCALARS = len(SCALAR_KEYS) + ANCHOR_FLAG_DIM

# INFERNO_RUN_TAG (see module docstring's TEST 1/TEST 2 section): suffixes every output
# path so independent hyperparameter tests never share a checkpoint dir / resume_state.pt /
# log file with each other OR with the untagged default -- critically, this keeps them from
# auto-resuming into the stale, incomplete (ep29, already-collapsing) 0.05-grad-clip run
# still sitting in the untagged models/checkpoints_multi_v6/, which is left untouched.
RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
_TAG_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_multi_v6{_TAG_SUFFIX}")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"train_log_multi_v6{_TAG_SUFFIX}.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, f"eval_log_multi_v6{_TAG_SUFFIX}.csv")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best.pt")  # ORIGINAL single-scenario best (ep 260)

EVAL_SCENARIOS = [
    ("single_training", TRAINING_IGNITION_POINT),
    ("topanga_ridge", MULTI_IGNITION_TRAINING_SCENARIO[0]),
    ("sullivan_canyon", MULTI_IGNITION_TRAINING_SCENARIO[1]),
    ("stone_canyon", MULTI_IGNITION_TRAINING_SCENARIO[2]),
] + list(VALIDATION_IGNITION_POINTS.items())
VALIDATION_NAMES = set(VALIDATION_IGNITION_POINTS.keys())

# One row per BATCH (not per episode -- see module docstring). Per-episode reward/ticks are
# aggregated (mean/min/max) since a batch pools BATCH_SIZE_EPISODES episodes into one update.
TRAIN_LOG_FIELDS = [
    "batch_idx", "episode_start", "episode_end", "n_episodes_in_batch", "device",
    "total_ticks", "mean_n_ticks", "mean_reward", "min_reward", "max_reward",
    "mean_buildings_destroyed", "contained_rate", "timeout_rate", "any_anchor_in_batch",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "approx_kl_first_epoch", "approx_kl_last_epoch", "epochs_run", "minibatches_run", "early_stopped",
    "stop_reason", "mean_unique_episodes_per_checked_minibatch",
    "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario_name", "avg_reward", "avg_buildings_destroyed",
    "avg_buildings_saved", "containment_rate",
]

# v5: re-baselined on the frozen 8-station real_depots.json environment (water_team 3,
# trench_crew 4, rescue_vehicle 3, helicopter 5) -- NOT the v1-v4 single-depot-per-type
# numbers. single_training/mandeville_canyon/getty_view_park re-run via
# heuristic_policy.py's own _run_baseline_table(); topanga_ridge/sullivan_canyon/
# stone_canyon computed individually (5 episodes each, same settings) since
# _run_baseline_table() only evaluates the 3-simultaneous-ignition aggregate, not
# each MULTI_IGNITION_TRAINING_SCENARIO point on its own -- what this per-scenario
# eval suite actually needs.
HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -19082.0, "avg_buildings_destroyed": 65.2, "containment_rate": 0.80},
    "topanga_ridge": {"avg_reward": -3446.3, "avg_buildings_destroyed": 12.8, "containment_rate": 0.80},
    "sullivan_canyon": {"avg_reward": -7821.5, "avg_buildings_destroyed": 31.6, "containment_rate": 0.80},
    "stone_canyon": {"avg_reward": -45379.9, "avg_buildings_destroyed": 167.0, "containment_rate": 0.60},
    "mandeville_canyon": {"avg_reward": 49.0, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
    "getty_view_park": {"avg_reward": 73.6, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
}


def build_scalars(obs_scalars, is_anchor):
    """flatten_scalars() plus one appended anchor-identity bit -- see
    ANCHOR_FLAG_DIM's docstring. is_anchor=True only for the exact
    TRAINING_IGNITION_POINT (real anchor-replay training episodes, and the
    single_training eval scenario); False everywhere else, including every
    other eval scenario -- topanga_ridge/sullivan_canyon/stone_canyon/
    mandeville_canyon/getty_view_park are NOT the anchor, even though some
    are training scenarios in the sense of being reachable by continuous
    random sampling."""
    return np.concatenate([flatten_scalars(obs_scalars), [1.0 if is_anchor else 0.0]]).astype(np.float32)


def build_model(n_grid_channels, device):
    """N_SCALARS = len(SCALAR_KEYS) + ANCHOR_FLAG_DIM (see that constant's
    docstring) means the MLP's first Linear layer has one more input column
    than models/checkpoints/best.pt was trained with, so this ISN'T a plain
    state_dict load anymore (unlike v4's original design) -- every OTHER
    parameter loads unchanged; only mlp.net.0.weight needs surgery: the
    original len(SCALAR_KEYS) columns are copied as-is, and the new anchor-
    flag column is zero-initialized so the warm-started model's behavior is
    UNCHANGED at load time (a zero-weighted input contributes nothing to the
    forward pass regardless of the flag's value) -- it only starts to matter
    once gradient updates give that column something to learn from the
    anchor-flag=1 episodes."""
    model = InfernoModel(n_grid_channels=n_grid_channels, n_scalars=N_SCALARS).to(device)
    if not os.path.exists(WARM_START_CKPT):
        print(f"[train-multi-v6] WARNING: {WARM_START_CKPT} not found -- falling back to random init.")
        return model
    warm_sd = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    new_sd = model.state_dict()
    for key, value in warm_sd.items():
        if key == "mlp.net.0.weight":
            new_sd[key][:, :len(SCALAR_KEYS)] = value
            new_sd[key][:, len(SCALAR_KEYS):] = 0.0
        else:
            new_sd[key] = value
    model.load_state_dict(new_sd)
    print(f"[train-multi-v6] Warm-started from {WARM_START_CKPT} (original single-scenario best.pt, ep 260) -- "
          f"all layers loaded unchanged except mlp.net.0.weight, which gained one zero-initialized column "
          f"for the anchor-identity flag (N_SCALARS={N_SCALARS}, was {len(SCALAR_KEYS)}) -- see ANCHOR_FLAG_DIM "
          f"docstring for why.")
    return model


class RunningMeanStd:
    """Welford's online mean/variance over each episode's raw per-tick
    rewards. Only .std() (SCALE, no mean subtraction) is used to normalize
    rewards before GAE -- see module docstring's "Reward/value scale"
    section for why mean-subtraction would break bootstrapping/shaping's
    zero-at-terminal invariant. epsilon-initialized count so episode 1
    doesn't divide by zero."""

    def __init__(self, epsilon=1e-4, prior_std=1.0):
        self.mean = 0.0
        self.var = prior_std ** 2
        self.count = epsilon

    def update(self, values):
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = len(values)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def std(self):
        return math.sqrt(self.var) + 1e-8


def compute_gae(rewards, values, bootstrap_value, gamma, lam):
    """Standard GAE. rewards/values/bootstrap_value must already be in the
    SAME scale (see RunningMeanStd above). Returns (advantages, returns),
    both length len(rewards), same order. See src/train/test_gae.py for a
    hand-computed correctness check."""
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def _is_mps_unimplemented_error(exc):
    msg = str(exc).lower()
    return "mps" in msg and "not implemented" in msg


def _extract_op_name(exc):
    msg = str(exc)
    match = re.search(r"operator '([^']+)'", msg)
    if match:
        return match.group(1)
    match = re.search(r"^([A-Za-z0-9_ ]+?MPS)[:,.]", msg)
    if match:
        return match.group(1).strip()
    return msg.splitlines()[0]


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _fire_area(grid_fire_channel):
    cls = fire_state_to_class(torch.from_numpy(grid_fire_channel).long())
    mask = (cls == FIRE_CLASSES[0]) | (cls == FIRE_CLASSES[1])
    return cls, int(mask.sum().item())


def collect_rollout(env, model, device, seed, ignition_point=None):
    """One full episode, current policy, NO gradients. ignition_point=None
    (the common case) lets InfernoEnv sample its own continuous-random
    point (see module docstring's change 4); passing an explicit point
    (only ever TRAINING_IGNITION_POINT, on ANCHOR_REPLAY_EVERY_N-th
    episodes -- see that constant) reignites the exact warm-started anchor
    scenario instead. Stores everything a PPO update needs: the RAW
    per-tick env reward, the OLD log-prob (the policy that actually
    sampled the action), and the model's own value estimate at that tick
    (used, after reward-scaling, as GAE's V(s_t))."""
    is_anchor = ignition_point is not None
    obs = env.reset(seed=seed, ignition_point=ignition_point)
    ignition_point = env.ignition_point
    steps = []
    total_reward = 0.0
    total_shaped = 0.0
    buildings_destroyed = 0
    done = False
    info = None

    _, prev_fire_area = _fire_area(obs["grid"][-1])

    with torch.no_grad():
        while not done:
            grid_t, _ = InfernoModel.obs_to_tensors(obs, device=device)
            scalars_np = build_scalars(obs["scalars"], is_anchor)
            scalars_t = torch.from_numpy(scalars_np).unsqueeze(0).to(device)
            action_logits, value, _classification_logits = model(grid_t, scalars_t)
            resource_dist = Categorical(logits=action_logits["resource_type"][0])
            zone_dist = Categorical(logits=action_logits["zone"][0])
            resource_idx = int(resource_dist.sample())
            zone_idx = int(zone_dist.sample())
            old_log_prob = float(
                resource_dist.log_prob(torch.tensor(resource_idx, device=device))
                + zone_dist.log_prob(torch.tensor(zone_idx, device=device))
            )
            fire_state_target = fire_state_to_class(torch.from_numpy(obs["grid"][-1]).long())

            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, reward, done, info = env.step(action)

            _, curr_fire_area = _fire_area(next_obs["grid"][-1])
            shaped_delta = SHAPING_COEFF * (prev_fire_area - curr_fire_area) if REWARD_SHAPING_ENABLED else 0.0
            prev_fire_area = curr_fire_area

            steps.append({
                "grid": obs["grid"],
                "scalars": scalars_np,
                "resource_idx": resource_idx,
                "zone_idx": zone_idx,
                "fire_state_target": fire_state_target,
                "raw_reward": reward + shaped_delta,
                "old_log_prob": old_log_prob,
                "raw_value": float(value.squeeze().item()),
            })
            total_reward += reward
            total_shaped += shaped_delta
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs

    if info["timeout"]:
        grid_t, _ = InfernoModel.obs_to_tensors(obs, device=device)
        scalars_t = torch.from_numpy(build_scalars(obs["scalars"], is_anchor)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, bootstrap_value, _ = model(grid_t, scalars_t)
        bootstrap_value_raw = float(bootstrap_value.squeeze().item())
    else:
        bootstrap_value_raw = 0.0

    return {
        "steps": steps,
        "total_reward": total_reward,
        "total_shaped": total_shaped,
        "buildings_destroyed": buildings_destroyed,
        "contained": info["contained"],
        "timeout": info["timeout"],
        "bootstrap_value_raw": bootstrap_value_raw,
        "ignition_point": ignition_point,
    }


def compute_batch_kl_and_entropy(model, steps, device):
    """Combined KL(old || new) and mean-entropy probe over the whole
    episode's batch, under the model's CURRENT weights -- merged from two
    formerly-separate functions (compute_batch_kl/compute_batch_entropy)
    that each did their OWN full forward pass over every step, i.e. every
    minibatch's safety check was silently doing 2x the necessary compute.
    Since KL needs new_log_prob and entropy needs the SAME distributions'
    .entropy(), both come from one forward pass per step -- this halves
    the dominant compute cost in ppo_update() (the per-minibatch safety
    check re-scans the WHOLE episode, not just that minibatch, so at
    MINIBATCH_SIZE=16 on a 150-tick episode this ran ~10x/episode; before
    this merge that was ~3000 extra forward passes/episode against ~450
    for rollout+training combined -- diagnostic checking, not training,
    was the actual bottleneck). Purely a redundant-computation removal:
    checking frequency, safety semantics, and computed values are all
    unchanged -- see ENTROPY_FLOOR's docstring for why entropy is checked
    at all (KL(old||new)~=0 for two near-delta-function distributions
    pointing at the same action, even as the underlying logits keep
    growing, so KL alone can't catch a policy that's already collapsed
    getting more extreme)."""
    with torch.no_grad():
        kl_total = 0.0
        entropy_total = 0.0
        for step in steps:
            grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
            action_logits, _, _ = model(grid_t, scalars_t)
            resource_dist = Categorical(logits=action_logits["resource_type"][0])
            zone_dist = Categorical(logits=action_logits["zone"][0])
            new_log_prob = float(
                resource_dist.log_prob(torch.tensor(step["resource_idx"], device=device))
                + zone_dist.log_prob(torch.tensor(step["zone_idx"], device=device))
            )
            kl_total += step["old_log_prob"] - new_log_prob
            entropy_total += float(resource_dist.entropy() + zone_dist.entropy())
    n = len(steps)
    return kl_total / n, entropy_total / n


def ppo_update_batch(model, optimizer, steps, advantages, returns, device, log_kl, seed, grad_norm_baseline):
    """Multi-trajectory analog of v4/v5's ppo_update() -- loss/clipping/optimizer-step
    logic is IDENTICAL, applied to a POOLED buffer of steps/advantages/returns spanning
    BATCH_SIZE_EPISODES independent episodes (see run_training_batch() and module
    docstring). PPO_EPOCHS epochs of shuffled MINIBATCH_SIZE-tick minibatches over the
    WHOLE POOLED buffer -- because the index shuffle operates over the full pooled `n`,
    a single episode's ~150 ticks land in many different minibatches mixed with other
    episodes' ticks, not all in the same one. old_log_prob (rollout-time) stays fixed;
    new_log_prob is recomputed every epoch. advantages are standardized (mean 0, std 1)
    over the WHOLE POOLED BATCH now, not one episode -- the intended generalization,
    not a special case: an outlier episode's advantages get measured against a much
    larger, more stable reference population instead of just its own ~150 ticks.
    returns (GAE's V-target) are NOT re-normalized here, already in reward-scaled units.

    ONE deliberate change from v4/v5's ppo_update(): the KL/entropy safety check
    (compute_batch_kl_and_entropy(), which rescans the ENTIRE steps list every call) is
    checked only every `max(1, n_minibatches_total // SAFETY_CHECK_TARGET_COUNT)`
    minibatches instead of every single one, with the batch's FINAL minibatch always
    force-checked. Checking every minibatch was fine at single-episode scale (~150
    steps, ~10 minibatches -- see v4/v5's own docstring on this). At batch scale,
    checking every minibatch makes total diagnostic cost grow QUADRATICALLY with batch
    size (more steps -> more minibatches -> each rescans more steps), which would make
    anything beyond a tiny batch prohibitively slow. Thinning the check frequency keeps
    total diagnostic cost roughly LINEAR again, at the honest cost of a coarser safety
    net than v4/v5 had (more minibatches of drift can accumulate between checks). This
    tradeoff is reported in the smoke test, not hidden.

    Also tracks, at every CHECKED minibatch, how many DISTINCT episodes (via each
    step's episode_idx, set by run_training_batch()) are represented in that
    minibatch -- direct evidence cross-episode mixing is actually happening, not just
    assumed from the shuffle logic.

    KL is checked after EVERY MINIBATCH's optimizer.step() (not just after
    every epoch -- see module docstring's "second smoke-test rerun" note:
    checking only at epoch boundaries was too coarse. A 150-tick episode
    with MINIBATCH_SIZE=16 packs ~10 gradient steps into a single "epoch";
    the smoke test's episode 1 alone produced a KL of 2.71 after just ONE
    epoch, and epoch-level early stopping (stopping AFTER that epoch
    completed) still let single_training's warm-started result get
    destroyed by ep 10 -- functionally identical to running all 4 epochs
    with no stopping at all, because all the damage happened inside that
    first epoch's ~10 minibatch steps, before the epoch-level check could
    ever fire. Checking after every minibatch instead means the update can
    be halted after just 1-2 minibatches when needed, well before a whole
    epoch's worth of drift accumulates. This is NOT gated by log_kl --
    log_kl only controls whether the full trace gets printed, not whether
    the safety check runs.

    ALSO checks current entropy after every minibatch, alongside KL -- but
    ONLY stops on low entropy if a gradient-norm spike was ALSO seen earlier
    in this SAME update (see ENTROPY_FLOOR's docstring for why entropy alone
    isn't enough: a real-500-episode run with an unconditional entropy-floor
    stop found the policy getting trapped at a low-entropy plateau for 19+
    consecutive episodes, each capped to just 1 minibatch, unable to
    accumulate enough gradient signal to recover -- the floor was correctly
    preventing further damage but had no way to distinguish "this update is
    pushing entropy down" from "this is routine single-episode noise that
    doesn't need braking." Corroborating with the grad-norm spike detector
    (GRAD_NORM_SPIKE_MULTIPLIER/GRAD_NORM_SPIKE_WINDOW, already validated as
    the earliest real warning signal in an earlier run's collapse) means
    isolated low-entropy readings are treated as normal variance, and only
    entropy drops that coincide with an actual anomalous gradient event get
    braked. grad_norm_baseline is the rolling median other_grad_norm from
    completed episodes BEFORE this one (None during the first
    GRAD_NORM_SPIKE_WINDOW episodes, before there's enough history --
    combined stopping simply can't trigger yet in that warmup window,
    matching how the standalone spike print behaves too).

    seed: shuffles minibatch order via a LOCAL random.Random(seed) instance,
    not the global random module -- added after a real run comparison
    turned out to be confounded by random.shuffle() using unseeded global
    state, meaning minibatch order (and therefore the exact SGD step
    sequence) was never reproducible run-to-run even with identical
    BASE_SEED+ep environment seeding. Ties this to the same seed=BASE_SEED+ep
    convention every other part of this file already uses.

    Whichever condition trips first stops the update; stop_reason records
    which one (or None if PPO_EPOCHS completed cleanly)."""
    n = len(steps)
    adv_mean = sum(advantages) / n
    adv_var = sum((a - adv_mean) ** 2 for a in advantages) / n
    adv_std = math.sqrt(adv_var) + 1e-8
    advantages_norm = [(a - adv_mean) / adv_std for a in advantages]
    if ADV_CLIP_ENABLED:
        advantages_norm = [max(-ADV_CLIP_STD, min(ADV_CLIP_STD, a)) for a in advantages_norm]

    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0
    n_updates = 0
    value_grad_norm_last = other_grad_norm_last = 0.0

    kl_first, entropy_first = compute_batch_kl_and_entropy(model, steps, device)
    kl_last, entropy_last = kl_first, entropy_first
    epochs_run = 0
    minibatches_run = 0
    early_stopped = False
    stop_reason = None
    saw_grad_norm_spike = False
    unique_episode_counts = []

    # See docstring: thins the per-minibatch safety check so total diagnostic cost stays
    # roughly linear in batch size instead of quadratic.
    n_minibatches_total = max(1, math.ceil(n / MINIBATCH_SIZE)) * PPO_EPOCHS
    safety_check_interval = max(1, n_minibatches_total // SAFETY_CHECK_TARGET_COUNT)

    shuffle_rng = random.Random(seed)
    indices = list(range(n))
    stop = False
    for epoch in range(PPO_EPOCHS):
        if stop:
            break
        shuffle_rng.shuffle(indices)
        for mb_start in range(0, n, MINIBATCH_SIZE):
            mb_idx = indices[mb_start:mb_start + MINIBATCH_SIZE]
            optimizer.zero_grad()
            for idx in mb_idx:
                step = steps[idx]
                grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
                scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
                action_logits, value, classification_logits = model(grid_t, scalars_t)

                resource_dist = Categorical(logits=action_logits["resource_type"][0])
                zone_dist = Categorical(logits=action_logits["zone"][0])
                new_log_prob = (
                    resource_dist.log_prob(torch.tensor(step["resource_idx"], device=device))
                    + zone_dist.log_prob(torch.tensor(step["zone_idx"], device=device))
                )
                entropy = resource_dist.entropy() + zone_dist.entropy()

                old_log_prob_t = torch.tensor(step["old_log_prob"], dtype=new_log_prob.dtype, device=device)
                ratio = torch.exp(new_log_prob - old_log_prob_t)
                adv = advantages_norm[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_scalar = value.squeeze()
                return_target = torch.tensor(returns[idx], dtype=value_scalar.dtype, device=device)
                value_loss = (value_scalar - return_target) ** 2

                classification_loss = F.cross_entropy(
                    classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
                )

                tick_loss = (
                    policy_loss
                    + VALUE_LOSS_COEFF * value_loss
                    + CLASSIFICATION_LOSS_COEFF * classification_loss
                    - ENTROPY_COEFF * entropy
                )
                tick_loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                classification_loss_sum += classification_loss.item()
                entropy_sum += entropy.item()
                n_updates += 1

            value_grad_norm_last = float(torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM))
            other_grad_norm_last = float(torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM))
            optimizer.step()
            minibatches_run += 1

            if grad_norm_baseline is not None and other_grad_norm_last > GRAD_NORM_SPIKE_MULTIPLIER * grad_norm_baseline:
                saw_grad_norm_spike = True

            is_last_minibatch_of_epoch = mb_start + MINIBATCH_SIZE >= n
            if minibatches_run % safety_check_interval == 0 or is_last_minibatch_of_epoch:
                unique_episode_counts.append(len({steps[i]["episode_idx"] for i in mb_idx}))
                kl_last, entropy_last = compute_batch_kl_and_entropy(model, steps, device)
                if abs(kl_last) > KL_EARLY_STOP_THRESHOLD:
                    early_stopped = True
                    stop_reason = "kl"
                    stop = True
                    break
                if entropy_last < ENTROPY_FLOOR and saw_grad_norm_spike:
                    early_stopped = True
                    stop_reason = "entropy_floor+grad_spike"
                    stop = True
                    break

        if not stop:
            epochs_run += 1

    mean_unique_episodes = (
        sum(unique_episode_counts) / len(unique_episode_counts) if unique_episode_counts else float("nan")
    )

    return (
        policy_loss_sum / n_updates, value_loss_sum / n_updates,
        classification_loss_sum / n_updates, entropy_sum / n_updates,
        value_grad_norm_last, other_grad_norm_last, kl_first, kl_last,
        epochs_run, minibatches_run, early_stopped, stop_reason, mean_unique_episodes,
    )


def run_training_batch(env, model, optimizer, device, episode_specs, reward_scale_normalizer,
                        log_kl, grad_norm_baseline):
    """episode_specs: list of (seed, anchor_point_or_None), one per episode in this batch
    (see module docstring). Collects len(episode_specs) independent rollouts, computes
    GAE INDEPENDENTLY per rollout using a SINGLE reward scale frozen from
    reward_scale_normalizer's state at the START of this batch (read once, before any
    episode in the batch runs -- NOT re-read between episodes, so every episode in the
    batch is scaled consistently; see module docstring for why re-reading mid-batch would
    break GAE's "same units" invariant across episodes in the same update). Pools all
    episodes' steps/advantages/returns into flat lists (concatenating GAE RESULTS, never
    raw trajectories -- see module docstring risk #1) and runs ONE ppo_update_batch() call
    over the whole pooled batch. The normalizer itself is updated once, with the whole
    batch's raw rewards, AFTER all episodes' GAE has been computed with the frozen scale."""
    scale = reward_scale_normalizer.std()

    all_steps = []
    all_advantages = []
    all_returns = []
    all_raw_rewards = []
    per_episode_results = []

    for episode_idx, (seed, anchor_point) in enumerate(episode_specs):
        rollout = collect_rollout(env, model, device, seed, ignition_point=anchor_point)
        steps = rollout["steps"]
        for step in steps:
            step["episode_idx"] = episode_idx  # diagnostic only -- see minibatch-mixing probe
        raw_rewards = [s["raw_reward"] for s in steps]
        scaled_rewards = [max(-SCALED_REWARD_CLIP, min(SCALED_REWARD_CLIP, r / scale)) for r in raw_rewards]
        values = [s["raw_value"] for s in steps]
        advantages, returns = compute_gae(scaled_rewards, values, rollout["bootstrap_value_raw"], GAMMA, GAE_LAMBDA)

        all_steps.extend(steps)
        all_advantages.extend(advantages)
        all_returns.extend(returns)
        all_raw_rewards.extend(raw_rewards)
        per_episode_results.append({
            "n_ticks": len(steps),
            "ignition_point": rollout["ignition_point"],
            "is_anchor": anchor_point is not None,
            "reward": rollout["total_reward"],
            "shaped_reward_total": rollout["total_shaped"],
            "buildings_destroyed": rollout["buildings_destroyed"],
            "contained": rollout["contained"],
            "timeout": rollout["timeout"],
        })

    reward_scale_normalizer.update(all_raw_rewards)  # AFTER all episodes' GAE, using the whole batch's pool

    (policy_loss, value_loss, classification_loss, entropy,
     value_grad_norm, other_grad_norm, kl_first, kl_last,
     epochs_run, minibatches_run, early_stopped, stop_reason,
     mean_unique_episodes) = ppo_update_batch(
        model, optimizer, all_steps, all_advantages, all_returns, device, log_kl,
        episode_specs[0][0], grad_norm_baseline
    )

    return {
        "per_episode": per_episode_results,
        "total_ticks": len(all_steps),
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "classification_loss": classification_loss,
        "entropy": entropy,
        "value_grad_norm": value_grad_norm,
        "other_grad_norm": other_grad_norm,
        "approx_kl_first_epoch": kl_first,
        "approx_kl_last_epoch": kl_last,
        "epochs_run": epochs_run,
        "minibatches_run": minibatches_run,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "mean_unique_episodes_per_checked_minibatch": mean_unique_episodes,
    }


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    results = {}
    for name, point in EVAL_SCENARIOS:
        is_anchor = (name == "single_training")
        result = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                              use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                              scalars_fn=lambda s, a=is_anchor: build_scalars(s, a))
        results[name] = result
        eval_writer.writerow({
            "episode": episode_num,
            "scenario_name": name,
            "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
        })
        eval_file.flush()
        kind = "VALIDATION" if name in VALIDATION_NAMES else "train"
        line = (f"    [eval @ ep {episode_num}] ({kind:10s}) {name}: avg_reward={result['avg_reward']:.1f}  "
                f"avg_buildings_destroyed={result['avg_buildings_destroyed']:.1f}  "
                f"containment_rate={result['containment_rate']:.0%}")
        baseline = HEURISTIC_BASELINE.get(name)
        if baseline is not None:
            delta = result["avg_reward"] - baseline["avg_reward"]
            line += f"  |  vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})"
        print(line, flush=True)
    return results


def save_resume_state(path, model, optimizer, reward_scale_normalizer, completed_episodes,
                       best_avg_eval_reward, best_episode, ignition_points_sampled,
                       episode_rewards, episode_losses, episode_wall_times):
    tmp_path = path + ".tmp"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward_scale_normalizer": {
            "mean": reward_scale_normalizer.mean, "var": reward_scale_normalizer.var,
            "count": reward_scale_normalizer.count,
        },
        "completed_episodes": completed_episodes,
        "best_avg_eval_reward": best_avg_eval_reward,
        "best_episode": best_episode,
        "ignition_points_sampled": ignition_points_sampled,
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "episode_wall_times": episode_wall_times,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_resume_state(path, model, optimizer, reward_scale_normalizer):
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    saved = state["reward_scale_normalizer"]
    reward_scale_normalizer.mean, reward_scale_normalizer.var, reward_scale_normalizer.count = (
        saved["mean"], saved["var"], saved["count"]
    )
    return {
        "completed_episodes": state["completed_episodes"],
        "best_avg_eval_reward": state["best_avg_eval_reward"],
        "best_episode": state["best_episode"],
        "ignition_points_sampled": state["ignition_points_sampled"],
        "episode_rewards": state["episode_rewards"],
        "episode_losses": state["episode_losses"],
        "episode_wall_times": state["episode_wall_times"],
    }


def _nearest_named_point_distance(point, named_points):
    r, c = point
    return min(math.hypot(r - nr, c - nc) for nr, nc in named_points)


def make_anchor_flag_probe(n_grid_channels, grid_shape, device):
    """A FIXED (seeded) random grid + base-scalars pair, reused every probe
    call -- see log_anchor_flag_engagement(). Independent of any real
    observation; its only job is holding everything ELSE constant so the
    anchor flag is the sole thing varying between the two forward passes."""
    g1 = torch.Generator().manual_seed(23)
    grid = torch.randn(1, n_grid_channels, *grid_shape, generator=g1).to(device)
    g2 = torch.Generator().manual_seed(29)
    base_scalars = torch.randn(1, len(SCALAR_KEYS), generator=g2).to(device)
    return grid, base_scalars


def log_anchor_flag_engagement(model, probe_grid, probe_base_scalars, device):
    """Confirms the anchor flag is actually being USED, not just present in
    the input -- the same rigor as v3's value-head divergence probe. Runs
    the SAME fixed grid/base-scalars through the model twice, only the
    anchor bit differs (0 vs 1); if resource_type logits come out identical,
    the flag isn't engaged yet (expected immediately after warm-start, since
    its weight column is zero-initialized -- see build_model()); growing
    max_abs_diff over training is the evidence it's actually learning to use
    the signal."""
    with torch.no_grad():
        zeros = torch.zeros(1, ANCHOR_FLAG_DIM, device=device)
        ones = torch.ones(1, ANCHOR_FLAG_DIM, device=device)
        scalars_0 = torch.cat([probe_base_scalars, zeros], dim=1)
        scalars_1 = torch.cat([probe_base_scalars, ones], dim=1)
        logits_0, _, _ = model(probe_grid, scalars_0)
        logits_1, _, _ = model(probe_grid, scalars_1)
        r0 = logits_0["resource_type"][0]
        r1 = logits_1["resource_type"][0]
        max_abs_diff = float((r1 - r0).abs().max())
    r0_str = [round(x, 3) for x in r0.tolist()]
    r1_str = [round(x, 3) for x in r1.tolist()]
    print(f"    [anchor-flag probe] resource_logits(flag=0)={r0_str}  resource_logits(flag=1)={r1_str}  "
          f"max_abs_diff={max_abs_diff:.4f}", flush=True)


def _crossed_multiple(prev_count, new_count, interval):
    """True if a multiple of `interval` lies in (prev_count, new_count]. Used to fire
    STATUS_EVERY/CHECKPOINT_EVERY/EVAL_EVERY exactly once per crossing, keyed to
    CUMULATIVE INDIVIDUAL EPISODE count -- so those constants keep the same
    cumulative-episode-count meaning as v4/v5's per-episode loop, regardless of how
    BATCH_SIZE_EPISODES relates to the interval (works even for a batch that straddles
    a boundary, e.g. batch covering episodes 18-21 crossing EVAL_EVERY=20)."""
    return new_count // interval > prev_count // interval


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = get_device()
    run_kind = "REAL TRAINING RUN" if N_EPISODES > 100 else "dry run / smoke test"
    tag_note = f" [RUN_TAG={RUN_TAG}]" if RUN_TAG else ""
    print(f"[train-multi-v6{tag_note}] {run_kind}: {N_EPISODES} episodes in batches of "
          f"{BATCH_SIZE_EPISODES}, initial device={device}, "
          f"reward_shaping={'ON' if REWARD_SHAPING_ENABLED else 'off'} (SHAPING_COEFF={SHAPING_COEFF}), "
          f"GAE(lambda={GAE_LAMBDA}, gamma={GAMMA}), PPO(epochs={PPO_EPOCHS}, "
          f"minibatch={MINIBATCH_SIZE}, clip_eps={PPO_CLIP_EPS}), "
          f"LEARNING_RATE={LEARNING_RATE:g}, GRAD_CLIP_NORM={GRAD_CLIP_NORM}, "
          f"adv_clip={'ON std=' + str(ADV_CLIP_STD) if ADV_CLIP_ENABLED else 'off'}, "
          f"SAFETY_CHECK_TARGET_COUNT={SAFETY_CHECK_TARGET_COUNT}")

    print("[train-multi-v6] Building InfernoEnv...")
    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)

    model = build_model(n_grid_channels=probe_obs["grid"].shape[0], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    reward_scale_normalizer = RunningMeanStd(prior_std=INITIAL_REWARD_STD_PRIOR)
    anchor_probe_grid, anchor_probe_scalars = make_anchor_flag_probe(
        probe_obs["grid"].shape[0], probe_obs["grid"].shape[1:], device
    )

    mps_errors = []
    # Window sized in BATCHES but targeting roughly STATUS_EVERY EPISODES of recent history,
    # so the "recent" window covers a similar amount of real training as v4/v5's per-episode
    # deque(maxlen=STATUS_EVERY) did, not STATUS_EVERY batches (which would be
    # STATUS_EVERY*BATCH_SIZE_EPISODES episodes -- much too wide a window).
    recent_window_batches = max(1, STATUS_EVERY // BATCH_SIZE_EPISODES)
    recent_wall_times = deque(maxlen=recent_window_batches)
    recent_rewards = deque(maxlen=recent_window_batches)
    last_eval_results = None
    best_avg_eval_reward = float("-inf")
    best_episode = None
    ignition_points_sampled = []
    episode_rewards = []  # one entry per BATCH now: that batch's mean per-episode reward
    episode_losses = []  # one entry per BATCH: (policy, value, classification, entropy)
    episode_wall_times = []  # one entry per BATCH: that batch's total wall time
    early_stop_minibatches = []  # minibatches_run for batches where early_stopped=True (see KL_EARLY_STOP_THRESHOLD)
    stop_reason_counts = Counter()  # "kl" vs "entropy_floor+grad_spike"
    grad_norm_history = []  # rolling window for real-time spike detection, see GRAD_NORM_SPIKE_WINDOW
    single_training_eval_history = []  # [(episode, avg_reward), ...] -- watch for mid-run collapse, not just the final number
    completed_episodes = 0
    batch_idx = 0
    interrupted = False

    resuming = os.path.exists(RESUME_STATE_PATH)
    if resuming:
        print(f"[train-multi-v6] Found {RESUME_STATE_PATH} -- resuming (not starting from episode 1).")
        resumed = load_resume_state(RESUME_STATE_PATH, model, optimizer, reward_scale_normalizer)
        completed_episodes = resumed["completed_episodes"]
        best_avg_eval_reward = resumed["best_avg_eval_reward"]
        best_episode = resumed["best_episode"]
        ignition_points_sampled = resumed["ignition_points_sampled"]
        episode_rewards = resumed["episode_rewards"]
        episode_losses = resumed["episode_losses"]
        episode_wall_times = resumed["episode_wall_times"]
        batch_idx = len(episode_wall_times)  # one entry per completed batch -- see save_resume_state
        print(f"[train-multi-v6] Resuming from episode {completed_episodes + 1}/{N_EPISODES} "
              f"(batch {batch_idx + 1}, best combined-avg eval reward so far: "
              f"{best_avg_eval_reward:.1f} at ep {best_episode})")
    start_episode = completed_episodes + 1
    if start_episode > N_EPISODES:
        print(f"[train-multi-v6] Resume state already shows {completed_episodes}/{N_EPISODES} episodes "
              f"complete -- nothing left to do. Delete {RESUME_STATE_PATH} to force a fresh run.")
        return

    log_mode = "a" if resuming else "w"
    with open(TRAIN_LOG_PATH, log_mode, newline="") as train_file, \
         open(EVAL_LOG_PATH, log_mode, newline="") as eval_file:
        train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
        if not resuming:
            train_writer.writeheader()
            eval_writer.writeheader()

        try:
            ep = start_episode
            while ep <= N_EPISODES:
                batch_t0 = time.perf_counter()
                batch_size_this = min(BATCH_SIZE_EPISODES, N_EPISODES - ep + 1)
                batch_last_ep = ep + batch_size_this - 1
                episode_specs = []
                for offset in range(batch_size_this):
                    this_ep = ep + offset
                    seed = BASE_SEED + this_ep
                    anchor_point = TRAINING_IGNITION_POINT if this_ep % ANCHOR_REPLAY_EVERY_N == 0 else None
                    episode_specs.append((seed, anchor_point))
                batch_idx += 1
                will_hit_status = _crossed_multiple(completed_episodes, batch_last_ep, STATUS_EVERY)
                log_kl = LOG_KL_EVERY_EPISODE or will_hit_status
                # Baseline from batches BEFORE this one only -- see ppo_update_batch()'s
                # docstring for why this gates the entropy-floor+grad-spike combined stop.
                grad_norm_baseline = (
                    sorted(grad_norm_history)[len(grad_norm_history) // 2]
                    if len(grad_norm_history) >= GRAD_NORM_SPIKE_WINDOW else None
                )

                try:
                    result = run_training_batch(env, model, optimizer, device, episode_specs,
                                                 reward_scale_normalizer, log_kl, grad_norm_baseline)
                except Exception as e:
                    if not _is_mps_unimplemented_error(e):
                        raise
                    op_name = _extract_op_name(e)
                    mps_errors.append(op_name)
                    print(f"[train-multi-v6] MPS op not implemented: '{op_name}' -- "
                          f"falling back to CPU for the rest of this run.")
                    device = torch.device("cpu")
                    model = model.to(device)
                    anchor_probe_grid, anchor_probe_scalars = anchor_probe_grid.to(device), anchor_probe_scalars.to(device)
                    result = run_training_batch(env, model, optimizer, device, episode_specs,
                                                 reward_scale_normalizer, log_kl, grad_norm_baseline)

                wall_s = time.perf_counter() - batch_t0
                per_ep = result["per_episode"]
                batch_rewards = [r["reward"] for r in per_ep]
                batch_mean_reward = sum(batch_rewards) / len(batch_rewards)
                batch_contained_rate = sum(1 for r in per_ep if r["contained"]) / len(per_ep)
                batch_timeout_rate = sum(1 for r in per_ep if r["timeout"]) / len(per_ep)
                batch_mean_destroyed = sum(r["buildings_destroyed"] for r in per_ep) / len(per_ep)
                batch_mean_ticks = sum(r["n_ticks"] for r in per_ep) / len(per_ep)
                any_anchor = any(r["is_anchor"] for r in per_ep)

                episode_wall_times.append(wall_s)
                episode_rewards.append(batch_mean_reward)
                episode_losses.append((result["policy_loss"], result["value_loss"],
                                        result["classification_loss"], result["entropy"]))
                recent_wall_times.append(wall_s)
                recent_rewards.append(batch_mean_reward)
                prev_completed = completed_episodes
                completed_episodes = batch_last_ep
                for r in per_ep:
                    ignition_points_sampled.append(r["ignition_point"])
                if result["early_stopped"]:
                    early_stop_minibatches.append(result["minibatches_run"])
                    stop_reason_counts[result["stop_reason"]] += 1

                train_writer.writerow({
                    "batch_idx": batch_idx, "episode_start": ep, "episode_end": batch_last_ep,
                    "n_episodes_in_batch": batch_size_this, "device": str(device),
                    "total_ticks": result["total_ticks"], "mean_n_ticks": batch_mean_ticks,
                    "mean_reward": batch_mean_reward, "min_reward": min(batch_rewards), "max_reward": max(batch_rewards),
                    "mean_buildings_destroyed": batch_mean_destroyed, "contained_rate": batch_contained_rate,
                    "timeout_rate": batch_timeout_rate, "any_anchor_in_batch": any_anchor,
                    "policy_loss": result["policy_loss"], "value_loss": result["value_loss"],
                    "classification_loss": result["classification_loss"], "entropy": result["entropy"],
                    "approx_kl_first_epoch": result["approx_kl_first_epoch"],
                    "approx_kl_last_epoch": result["approx_kl_last_epoch"],
                    "epochs_run": result["epochs_run"], "minibatches_run": result["minibatches_run"],
                    "early_stopped": result["early_stopped"], "stop_reason": result["stop_reason"] or "",
                    "mean_unique_episodes_per_checked_minibatch": result["mean_unique_episodes_per_checked_minibatch"],
                    "value_grad_norm": result["value_grad_norm"], "other_grad_norm": result["other_grad_norm"],
                    "wall_time_s": wall_s,
                })
                train_file.flush()

                # Real-time gradient-norm spike warning -- NOT gated by log_kl, printed the
                # instant it happens (see v4/v5's ENTROPY_FLOOR docstring for why this
                # signal mattered before). Baseline/threshold carried over unchanged from
                # v4/v5's single-episode calibration -- NOT re-derived for batch-pooled
                # gradients, which may have a different natural distribution; treat this as
                # a carried-over guess to sanity-check during the smoke test, not a
                # validated threshold for this file.
                grad_norm_history.append(result["other_grad_norm"])
                if len(grad_norm_history) > GRAD_NORM_SPIKE_WINDOW:
                    grad_norm_history.pop(0)
                if len(grad_norm_history) >= GRAD_NORM_SPIKE_WINDOW:
                    baseline = sorted(grad_norm_history)[len(grad_norm_history) // 2]  # median, robust to spikes in the window itself
                    if baseline > 0 and result["other_grad_norm"] > GRAD_NORM_SPIKE_MULTIPLIER * baseline:
                        print(f"    [grad-norm SPIKE] batch {batch_idx} (episodes {ep}-{batch_last_ep}): "
                              f"other_grad_norm={result['other_grad_norm']:.1f} vs recent median={baseline:.1f} "
                              f"({result['other_grad_norm'] / baseline:.1f}x) -- "
                              f"watch next few batches' entropy closely", flush=True)

                if log_kl:
                    stop_note = (f"  EARLY-STOPPED after {result['minibatches_run']} minibatch(es) "
                                 f"(epoch {result['epochs_run'] + 1}, reason={result['stop_reason']})") \
                        if result["early_stopped"] else ""
                    anchor_note = "  [ANCHOR IN BATCH]" if any_anchor else ""
                    print(f"    [ppo-kl] batch {batch_idx} (episodes {ep}-{batch_last_ep}, "
                          f"{batch_size_this} eps, {result['total_ticks']} ticks): "
                          f"approx_KL before epoch 1 = {result['approx_kl_first_epoch']:.5f}  "
                          f"final = {result['approx_kl_last_epoch']:.5f}  entropy={result['entropy']:.5f}  "
                          f"epochs_run={result['epochs_run']} minibatches_run={result['minibatches_run']}{stop_note}  "
                          f"mean_unique_eps/checked_mb={result['mean_unique_episodes_per_checked_minibatch']:.2f}"
                          f"{anchor_note}  other_grad_norm={result['other_grad_norm']:.1f}", flush=True)

                if will_hit_status:
                    window_batches_per_min = len(recent_wall_times) / (sum(recent_wall_times) / 60.0)
                    window_episodes_per_min = window_batches_per_min * BATCH_SIZE_EPISODES
                    window_avg_reward = sum(recent_rewards) / len(recent_rewards)
                    print(f"[status @ episode {completed_episodes:5d}/{N_EPISODES} (batch {batch_idx})] "
                          f"batches/min(last{len(recent_wall_times)})={window_batches_per_min:6.2f}  "
                          f"(~{window_episodes_per_min:.2f} episodes/min)  "
                          f"avg_batch_mean_reward(last{len(recent_rewards)})={window_avg_reward:10.1f}  "
                          f"entropy={result['entropy']:.3f}  "
                          f"policy_loss={result['policy_loss']:8.3f}  "
                          f"value_loss={result['value_loss']:8.3f}  "
                          f"class_loss={result['classification_loss']:.3f}  "
                          f"reward_scale_std={reward_scale_normalizer.std():.2f}  "
                          f"value_grad_norm={result['value_grad_norm']:.1f}  "
                          f"other_grad_norm={result['other_grad_norm']:.1f}  device={device}",
                          flush=True)
                    log_anchor_flag_engagement(model, anchor_probe_grid, anchor_probe_scalars, device)

                if _crossed_multiple(prev_completed, completed_episodes, CHECKPOINT_EVERY):
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{completed_episodes:04d}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    torch.save(model.state_dict(), latest_ckpt_path)
                    save_resume_state(RESUME_STATE_PATH, model, optimizer, reward_scale_normalizer,
                                       completed_episodes, best_avg_eval_reward, best_episode,
                                       ignition_points_sampled, episode_rewards, episode_losses, episode_wall_times)
                    print(f"  [checkpoint] saved {ckpt_path} (and latest.pt, resume_state.pt)", flush=True)

                if _crossed_multiple(prev_completed, completed_episodes, EVAL_EVERY):
                    last_eval_results = run_eval_suite(model, env, completed_episodes, eval_writer, eval_file, device)
                    avg_eval_reward = sum(r["avg_reward"] for r in last_eval_results.values()) / len(last_eval_results)
                    single_training_eval_history.append((completed_episodes, last_eval_results["single_training"]["avg_reward"]))
                    print(f"    [eval @ episode {completed_episodes}] combined avg reward across all "
                          f"{len(last_eval_results)} scenarios: {avg_eval_reward:.1f}", flush=True)
                    if avg_eval_reward > best_avg_eval_reward:
                        best_avg_eval_reward = avg_eval_reward
                        best_episode = completed_episodes
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"  [checkpoint] new best combined-avg eval reward ({avg_eval_reward:.1f}) "
                              f"at episode {completed_episodes} -- saved {best_ckpt_path}", flush=True)

                ep += batch_size_this
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[train-multi-v6] Caught interrupt (Ctrl+C or SIGTERM) after episode "
                  f"{completed_episodes}/{N_EPISODES} -- saving checkpoint + resume state before exit.")
            torch.save(model.state_dict(), latest_ckpt_path)
            save_resume_state(RESUME_STATE_PATH, model, optimizer, reward_scale_normalizer,
                               completed_episodes, best_avg_eval_reward, best_episode,
                               ignition_points_sampled, episode_rewards, episode_losses, episode_wall_times)
            print(f"  [checkpoint] saved {latest_ckpt_path} and {RESUME_STATE_PATH}")

        if completed_episodes > 0:
            torch.save(model.state_dict(), latest_ckpt_path)
            save_resume_state(RESUME_STATE_PATH, model, optimizer, reward_scale_normalizer,
                               completed_episodes, best_avg_eval_reward, best_episode,
                               ignition_points_sampled, episode_rewards, episode_losses, episode_wall_times)

    total_wall = sum(episode_wall_times)
    episodes_per_min = completed_episodes / (total_wall / 60.0) if total_wall > 0 else float("nan")
    policy_losses = [p for p, v, c, e in episode_losses]
    value_losses = [v for p, v, c, e in episode_losses]
    class_losses = [c for p, v, c, e in episode_losses]
    entropies = [e for p, v, c, e in episode_losses]

    def _nan_or_frozen(values):
        if not values:
            return "no data"
        if any(math.isnan(v) for v in values):
            return "NaN encountered"
        if len(set(round(v, 6) for v in values)) <= 1:
            return "frozen (never changed)"
        return f"moved (range {min(values):.3g} to {max(values):.3g})"

    label = f"INTERRUPTED at episode {completed_episodes}/{N_EPISODES} (batch {batch_idx})" if interrupted \
        else f"completed all {completed_episodes} episodes in {batch_idx} batches"
    print(f"\n=== Multi-ignition v6 (multi-trajectory batched PPO, BATCH_SIZE_EPISODES={BATCH_SIZE_EPISODES}) "
          f"summary ({label}) ===")
    print(f"Total wall time: {total_wall:.1f}s  Overall episodes/min: {episodes_per_min:.2f}  "
          f"(batches/min: {batch_idx / (total_wall / 60.0) if total_wall > 0 else float('nan'):.2f})")
    print(f"Policy loss: {_nan_or_frozen(policy_losses)}")
    print(f"Value loss: {_nan_or_frozen(value_losses)}")
    print(f"Classification loss: {_nan_or_frozen(class_losses)}")
    print(f"Entropy: {_nan_or_frozen(entropies)}")
    print(f"Final reward_scale_normalizer: mean={reward_scale_normalizer.mean:.2f} "
          f"std={reward_scale_normalizer.std():.2f} n={int(reward_scale_normalizer.count)}")

    n_early = len(early_stop_minibatches)
    if batch_idx > 0:
        avg_stop_mb = sum(early_stop_minibatches) / n_early if n_early else float("nan")
        print(f"Early-stopping: triggered in {n_early}/{batch_idx} batches "
              f"({n_early / batch_idx:.0%})" +
              (f", average stop after {avg_stop_mb:.1f} minibatch(es)" if n_early else "") +
              f"  -- by reason: {dict(stop_reason_counts)}  "
              f"(KL_threshold={KL_EARLY_STOP_THRESHOLD}, ENTROPY_FLOOR={ENTROPY_FLOOR}, "
              f"SAFETY_CHECK_TARGET_COUNT={SAFETY_CHECK_TARGET_COUNT})")

    if single_training_eval_history:
        print("single_training eval-reward trajectory across the run (watch for mid-run collapse, "
              "not just the final number -- warm-started value was +32.1, 100% containment):")
        for ep_num, avg_reward in single_training_eval_history:
            print(f"  ep {ep_num:5d}: avg_reward={avg_reward:.1f}")

    if ignition_points_sampled:
        unique_points = set(ignition_points_sampled)
        named_points = [TRAINING_IGNITION_POINT] + list(MULTI_IGNITION_TRAINING_SCENARIO)
        dists = [_nearest_named_point_distance(p, named_points) for p in ignition_points_sampled]
        print(f"Ignition variety check: {len(unique_points)} unique points sampled out of "
              f"{len(ignition_points_sampled)} episodes; distance-to-nearest-named-point "
              f"(grid cells): min={min(dists):.1f} mean={sum(dists) / len(dists):.1f} max={max(dists):.1f}")

    if episode_rewards:
        n_tail = min(20, len(episode_rewards))
        print(f"Stochastic rollout reward (raw, unshaped, per-batch MEAN across that batch's episodes), "
              f"first->last batch: {episode_rewards[0]:.1f} -> {episode_rewards[-1]:.1f}")
        print(f"Stochastic rollout reward, mean of first {n_tail} vs last {n_tail} batches: "
              f"{sum(episode_rewards[:n_tail]) / n_tail:.1f} vs "
              f"{sum(episode_rewards[-n_tail:]) / n_tail:.1f}")
    if last_eval_results:
        print(f"Final deterministic eval (all {len(last_eval_results)} scenarios) vs heuristic baseline:")
        for name, result in last_eval_results.items():
            kind = "VALIDATION" if name in VALIDATION_NAMES else "train"
            print(f"  ({kind:10s}) {name}: avg_reward={result['avg_reward']:.1f}  "
                  f"avg_buildings_destroyed={result['avg_buildings_destroyed']:.1f}  "
                  f"containment_rate={result['containment_rate']:.0%}")
            baseline = HEURISTIC_BASELINE.get(name)
            if baseline is not None:
                delta = result["avg_reward"] - baseline["avg_reward"]
                print(f"    heuristic:   avg_reward={baseline['avg_reward']:.1f}  "
                      f"avg_buildings_destroyed={baseline['avg_buildings_destroyed']:.1f}  "
                      f"containment_rate={baseline['containment_rate']:.0%}  "
                      f"(delta: {delta:+.1f})")
    else:
        print("No eval completed yet (fewer than EVAL_EVERY episodes ran).")
    if best_episode is not None:
        print(f"Best checkpoint: ep {best_episode}, combined avg eval reward = {best_avg_eval_reward:.1f} "
              f"-> {best_ckpt_path}")
    else:
        print("No best.pt saved yet (fewer than EVAL_EVERY episodes ran).")
    print(f"MPS errors encountered: {mps_errors if mps_errors else 'none'}")
    print(f"Final device: {device}")
    print(f"Latest checkpoint: {latest_ckpt_path}")


if __name__ == "__main__":
    main()
