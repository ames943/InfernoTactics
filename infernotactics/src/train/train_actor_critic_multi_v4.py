"""
Multi-ignition training, v4 -- GAE + multi-epoch PPO, replacing v1/v2/v3's
algorithm entirely, testing a NEW hypothesis after v1/v2/v3 all failed.

Why this file exists: v1 (no fixes), v2 (per-scenario return normalization +
scenario one-hot + curriculum warm-start), and v3 (+ separate value heads
per scenario) all ended flat on topanga_ridge/sullivan_canyon/stone_canyon/
mandeville_canyon/getty_view_park -- only single_training (the warm-started,
already-solved scenario) held steady. Three independent, principled fixes
targeting reward-scale/critic-sharing conflict in a row not moving anything
rules that theory out as the primary cause.

An audit of train_actor_critic.py (and confirmed identical in v2/v3) found
the algorithm has ALWAYS been plain single-episode REINFORCE with a learned
baseline, not PPO, despite loose "PPO-style" language elsewhere: the
"advantage" was G_t (a full-episode Monte-Carlo discounted return) minus
V(s_t), computed with exactly ONE gradient step per collected episode --
no stored old-policy log-prob, no importance-sampling ratio, no clip term,
no multi-epoch reuse of a batch, anywhere in the code. The original file's
own docstring says so explicitly ("chosen over PPO for this first version...
structured... to grow into PPO later" -- that never happened).

The same audit measured episode length per scenario directly from
train_log_multi.csv (v1) and train_log_multi_v2.csv: the three
never-improving scenarios almost ALWAYS run the full MAX_TICKS=150-tick cap
(stone_canyon 97-98%, topanga_ridge 100%, sullivan_canyon 93-99% of episodes
timed out rather than being contained), while single_training visibly
SHORTENED its own episodes as it learned (140.4 -> 80.6 mean ticks, 93% ->
52% timeout rate, first 100 vs last 100 episodes of its original 2000-episode
run). That's independent, measured evidence for a long-horizon (150-step),
sparse/delayed-reward credit-assignment problem -- exactly what plain
REINFORCE is known to struggle with, and what GAE + multi-epoch PPO exist to
address, independent of which scenario is involved.

Because this is a genuinely different hypothesis than v2/v3's, v4
deliberately does NOT carry forward their reward-scale-conflict machinery
(per-scenario return normalizers, scenario one-hot MLP input, separate value
heads) -- keeping that machinery would make it impossible to attribute any
v4 result to GAE/PPO specifically rather than to leftover v2/v3 fixes. v4
warm-starts directly from the ORIGINAL single-scenario checkpoint
(models/checkpoints/best.pt, ep 260) -- same architecture as that checkpoint
(n_scalars=len(SCALAR_KEYS), n_value_heads=1), so it's a plain,
unmodified state_dict load, no key surgery.

Four changes, all requested together, all present in this file:

1. GAE (Generalized Advantage Estimation), standard formulation:
     delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
     A_t = delta_t + gamma*lambda*A_{t+1}          (lambda=0.95, gamma=0.99)
   Bootstraps with V(s_T) (one extra no_grad forward pass on the episode's
   final next-observation) when the episode ends via MAX_TICKS timeout
   (info["timeout"]=True, a TRUNCATION, not a true terminal state) and with
   0 when it ends via real containment (info["contained"]=True, a true
   terminal state). v1-v3 never made this distinction -- they always
   returned a plain, un-bootstrapped Monte-Carlo sum, a source of bias
   specifically for the mostly-timed-out scenarios identified above.

   Reward/value scale: rather than mean-subtracting returns the way v1-v3
   did (which would break the "0 reward at a true terminal state" invariant
   GAE's bootstrapping and the reward-shaping potential both rely on), this
   file tracks a RunningMeanStd over each episode's raw per-tick rewards and
   divides (SCALE ONLY, no mean subtraction) by its running std before GAE.
   The value head then learns to predict values in this same scaled space,
   so value_loss = (V_pred - return_target)^2 is computed with both sides
   already in comparable, well-scaled units -- no separate return
   normalization step needed at loss time, and no re-derivation of the
   original value/policy loss-scale bug that motivated v1's fix.

2. Real multi-epoch PPO update: each episode's collected ticks are reused
   for PPO_EPOCHS=4 epochs (not 1), in shuffled minibatches of
   MINIBATCH_SIZE ticks, with a genuine importance-sampling ratio
   exp(new_log_prob - old_log_prob) clipped to [1-eps, 1+eps] (eps=0.2)
   against the (batch-standardized) advantage. old_log_prob is captured
   once at rollout time -- the policy that actually acted -- and held fixed
   for all 4 epochs of that episode's update, so the ratio has real policy
   drift to clip against. Verified via an approximate-KL probe
   (compute_batch_kl()) logged before epoch 1 and after every epoch: should
   read ~0 at the very start (ratio=1 everywhere, nothing has moved yet)
   and grow monotonically-ish across epochs 1->4 within one episode's
   update, then reset back near 0 at the start of the NEXT episode's
   rollout (new old_log_prob baseline). Logged every episode when
   LOG_KL_EVERY_EPISODE (the 50-episode smoke test); every STATUS_EVERY
   episodes on the real run, to bound the extra no_grad forward passes this
   costs.

3. Potential-based reward shaping, TOGGLEABLE (INFERNO_REWARD_SHAPING=1 env
   var; default OFF so v4 can be run both ways and compared): a small
   per-tick term SHAPING_COEFF * (fire_area_before_tick - fire_area_after_tick),
   where fire_area = count of grid cells classified Threat or Blaze (reusing
   the SAME classification target already computed for classification_loss,
   no env.py changes needed). Positive when the fire shrinks tick-to-tick.
   This is Ng et al. 1999 potential-based shaping with potential
   phi(s) = -fire_area(s): F(s,s') = phi(s') - phi(s) doesn't change the
   optimal policy, it only densifies the signal. SHAPING_COEFF defaults
   small (0.02/cell) specifically so a typical tick's shaping term stays
   well under the existing terminal rewards (+50 contained, ~-100/building).

4. Continuous ignition randomization, replacing v2/v3's 4-fixed-point
   curriculum entirely: every training episode calls env.reset(seed=seed)
   with NO ignition_point argument. InfernoEnv.reset()'s own default
   (scenario="single", ignition_point=None) already routes to
   _sample_ignition_point() -- the SAME "flammable, not water, within
   WUI_PROXIMITY_RADIUS_CELLS of a building" sampler originally used to
   pick TRAINING_IGNITION_POINT itself -- reseeded every episode
   (seed=BASE_SEED+ep, same convention as every prior script), so
   consecutive episodes draw genuinely different real fuel cells from
   across the whole grid, not a fixed rotation of 3-4 named points.
   mandeville_canyon and getty_view_park remain fixed, reserved,
   NEVER-sampled validation points, unchanged from every prior run --
   eval_policy() only ever ignites them explicitly; training never does.
   Because ignition point is now a pure function of each episode's seed
   (not a separate stateful RNG), resume support needs no extra RNG state
   beyond what v1 already saved.

   REFINEMENT (added after a smoke test showed pure continuous randomization
   destroys single_training within ~10 episodes -- see ANCHOR_REPLAY_EVERY_N
   below for the full diagnosis): every ANCHOR_REPLAY_EVERY_N=7th episode
   (~14.3%) explicitly reignites TRAINING_IGNITION_POINT instead of a
   continuous-random point. This was a deliberate, explicit design call (not
   a default) -- the alternative of letting single_training decay and judging
   v4 only by its final numbers would make it impossible to tell whether a
   mediocre single_training score reflects the new GAE/PPO mechanism working
   or not working, versus just never being trained on anymore; a fully
   dropped anchor also breaks side-by-side comparability with v1/v2/v3, which
   all explicitly tracked single_training holding its solved state. The
   other ~85.7% of episodes still deliver v4's genuine broad-randomization
   pressure.

KL early stopping (added after the FIRST smoke test rerun, not part of the
original four changes above): that run completed cleanly (no NaN/crash,
ignition variety confirmed excellent -- 50/50 unique points, mean 86.4 grid
cells from the nearest old named point) but silently ran all 4 epochs on
episode 3 despite a KL of 10.5 (~200-300x the healthy ~0.01-0.05 range),
and by ep 50 the warm-started single_training result (+32.1 reward, 100%
containment) had been destroyed (-119,815.7, 0% containment, worse than the
untrained heuristic) -- a real causal story (one uncapped bad update), not
noise, given entropy also nearly collapsed (0.0007) at the same point in the
run. Fix: ppo_update() now checks approx-KL after EVERY epoch
(unconditionally, not just when logging) and stops taking further epochs on
that episode's batch once |KL| > KL_EARLY_STOP_THRESHOLD -- the standard
Spinning-Up-style PPO early-stopping heuristic, TARGET_KL=0.02 with a 1.5x
slack multiplier before stopping. Doesn't undo the epoch that tripped it,
only prevents further compounding, matching standard practice. See the
KL_EARLY_STOP_THRESHOLD constant below for the full rationale.

Smoke-test verification (INFERNO_N_EPISODES=50), matching the rigor of
every prior run before trusting a real one:
  - GAE correctness: src/train/test_gae.py, a hand-computed 3-step
    known-values unit test (run standalone, BEFORE the smoke test:
    `python -m src.train.test_gae`).
  - PPO epochs actually moving the policy: compute_batch_kl()'s per-epoch
    approximate KL, logged every episode during the smoke test (see above).
  - Ignition variety: every episode's sampled (row, col) is written to
    TRAIN_LOG_PATH's ignition_row/ignition_col columns; the end-of-run
    summary reports the count of UNIQUE points sampled and each point's
    distance to the nearest of the 4 old named points, so clustering can be
    checked directly instead of assumed.

    python -m src.train.test_gae                                         # GAE unit test, run first
    INFERNO_N_EPISODES=50 python -m src.train.train_actor_critic_multi_v4 # smoke test
    python -m src.train.train_actor_critic_multi_v4                      # real 500-episode run
    INFERNO_REWARD_SHAPING=1 python -m src.train.train_actor_critic_multi_v4  # with shaping toggle on
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
BASE_SEED = 2000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_CLIP_EPS = 0.2
MINIBATCH_SIZE = 16  # ticks per gradient step -- see module docstring's memory-footprint rationale (inherited from v1)
VALUE_LOSS_COEFF = 0.5
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.01
GRAD_CLIP_NORM = 0.5

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

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v4")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train_log_multi_v4.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_log_multi_v4.csv")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best.pt")  # ORIGINAL single-scenario best (ep 260)

EVAL_SCENARIOS = [
    ("single_training", TRAINING_IGNITION_POINT),
    ("topanga_ridge", MULTI_IGNITION_TRAINING_SCENARIO[0]),
    ("sullivan_canyon", MULTI_IGNITION_TRAINING_SCENARIO[1]),
    ("stone_canyon", MULTI_IGNITION_TRAINING_SCENARIO[2]),
] + list(VALIDATION_IGNITION_POINTS.items())
VALIDATION_NAMES = set(VALIDATION_IGNITION_POINTS.keys())

TRAIN_LOG_FIELDS = [
    "episode", "device", "n_ticks", "ignition_row", "ignition_col",
    "reward", "shaped_reward_total", "buildings_destroyed", "contained", "timeout",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "is_anchor", "approx_kl_first_epoch", "approx_kl_last_epoch", "epochs_run", "minibatches_run", "early_stopped",
    "stop_reason", "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario_name", "avg_reward", "avg_buildings_destroyed",
    "avg_buildings_saved", "containment_rate",
]

HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -29410.9, "avg_buildings_destroyed": 98.4, "containment_rate": 0.80},
    "topanga_ridge": {"avg_reward": -5030.5, "avg_buildings_destroyed": 18.6, "containment_rate": 0.80},
    "sullivan_canyon": {"avg_reward": -5519.5, "avg_buildings_destroyed": 22.0, "containment_rate": 0.80},
    "stone_canyon": {"avg_reward": -300.3, "avg_buildings_destroyed": 1.0, "containment_rate": 1.00},
    "mandeville_canyon": {"avg_reward": 30.5, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
    "getty_view_park": {"avg_reward": 46.1, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
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
        print(f"[train-multi-v4] WARNING: {WARM_START_CKPT} not found -- falling back to random init.")
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
    print(f"[train-multi-v4] Warm-started from {WARM_START_CKPT} (original single-scenario best.pt, ep 260) -- "
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


def compute_batch_kl(model, steps, device):
    """Approximate KL(old || new) = mean(old_log_prob - new_log_prob) over
    the whole episode's batch, under the model's CURRENT weights. See
    module docstring's smoke-test verification section."""
    with torch.no_grad():
        total = 0.0
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
            total += step["old_log_prob"] - new_log_prob
    return total / len(steps)


def compute_batch_entropy(model, steps, device):
    """Mean (resource_type entropy + zone entropy) over the whole episode's
    batch, under the model's CURRENT weights -- the entropy-floor
    counterpart to compute_batch_kl(). See ENTROPY_FLOOR's docstring for
    why KL alone can't catch this: KL(old||new) is ~0 whenever both old and
    new are already near-delta-function distributions pointing at the same
    action, even while the underlying logits (and thus how "collapsed" the
    policy is) keep growing. This function measures collapse directly."""
    with torch.no_grad():
        total = 0.0
        for step in steps:
            grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
            action_logits, _, _ = model(grid_t, scalars_t)
            resource_dist = Categorical(logits=action_logits["resource_type"][0])
            zone_dist = Categorical(logits=action_logits["zone"][0])
            total += float(resource_dist.entropy() + zone_dist.entropy())
    return total / len(steps)


def ppo_update(model, optimizer, steps, advantages, returns, device, log_kl, seed, grad_norm_baseline):
    """PPO_EPOCHS epochs of shuffled MINIBATCH_SIZE-tick minibatches over
    ONE episode's collected steps. old_log_prob (rollout-time) stays fixed
    across all epochs; new_log_prob is recomputed every epoch as the model
    changes. advantages are standardized (mean 0, std 1) over this episode's
    own batch before use in the surrogate -- returns (GAE's V-target) are
    NOT re-normalized here, they're already in reward-scaled units (see
    module docstring).

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

    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0
    n_updates = 0
    value_grad_norm_last = other_grad_norm_last = 0.0

    kl_first = compute_batch_kl(model, steps, device)
    kl_last = kl_first
    entropy_first = compute_batch_entropy(model, steps, device)
    entropy_last = entropy_first
    epochs_run = 0
    minibatches_run = 0
    early_stopped = False
    stop_reason = None
    saw_grad_norm_spike = False

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

            kl_last = compute_batch_kl(model, steps, device)
            entropy_last = compute_batch_entropy(model, steps, device)
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

    return (
        policy_loss_sum / n_updates, value_loss_sum / n_updates,
        classification_loss_sum / n_updates, entropy_sum / n_updates,
        value_grad_norm_last, other_grad_norm_last, kl_first, kl_last,
        epochs_run, minibatches_run, early_stopped, stop_reason,
    )


def run_training_episode(env, model, optimizer, device, seed, reward_scale_normalizer, log_kl,
                          grad_norm_baseline, ignition_point=None):
    rollout = collect_rollout(env, model, device, seed, ignition_point=ignition_point)
    steps = rollout["steps"]
    raw_rewards = [s["raw_reward"] for s in steps]

    scale = reward_scale_normalizer.std()
    scaled_rewards = [max(-SCALED_REWARD_CLIP, min(SCALED_REWARD_CLIP, r / scale)) for r in raw_rewards]
    values = [s["raw_value"] for s in steps]
    bootstrap_value = rollout["bootstrap_value_raw"]
    advantages, returns = compute_gae(scaled_rewards, values, bootstrap_value, GAMMA, GAE_LAMBDA)
    reward_scale_normalizer.update(raw_rewards)

    (policy_loss, value_loss, classification_loss, entropy,
     value_grad_norm, other_grad_norm, kl_first, kl_last,
     epochs_run, minibatches_run, early_stopped, stop_reason) = ppo_update(
        model, optimizer, steps, advantages, returns, device, log_kl, seed, grad_norm_baseline
    )

    return {
        "n_ticks": len(steps),
        "ignition_point": rollout["ignition_point"],
        "is_anchor": ignition_point is not None,
        "reward": rollout["total_reward"],
        "shaped_reward_total": rollout["total_shaped"],
        "buildings_destroyed": rollout["buildings_destroyed"],
        "contained": rollout["contained"],
        "timeout": rollout["timeout"],
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
    print(f"[train-multi-v4] {run_kind}: {N_EPISODES} episodes, initial device={device}, "
          f"reward_shaping={'ON' if REWARD_SHAPING_ENABLED else 'off'} (SHAPING_COEFF={SHAPING_COEFF}), "
          f"GAE(lambda={GAE_LAMBDA}, gamma={GAMMA}), PPO(epochs={PPO_EPOCHS}, "
          f"minibatch={MINIBATCH_SIZE}, clip_eps={PPO_CLIP_EPS})")

    print("[train-multi-v4] Building InfernoEnv...")
    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)

    model = build_model(n_grid_channels=probe_obs["grid"].shape[0], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    reward_scale_normalizer = RunningMeanStd(prior_std=INITIAL_REWARD_STD_PRIOR)
    anchor_probe_grid, anchor_probe_scalars = make_anchor_flag_probe(
        probe_obs["grid"].shape[0], probe_obs["grid"].shape[1:], device
    )

    mps_errors = []
    recent_wall_times = deque(maxlen=STATUS_EVERY)
    recent_rewards = deque(maxlen=STATUS_EVERY)
    last_eval_results = None
    best_avg_eval_reward = float("-inf")
    best_episode = None
    ignition_points_sampled = []
    episode_rewards = []
    episode_losses = []  # (policy, value, classification, entropy)
    episode_wall_times = []
    early_stop_minibatches = []  # minibatches_run for episodes where early_stopped=True (see KL_EARLY_STOP_THRESHOLD)
    stop_reason_counts = Counter()  # "kl" vs "entropy_floor" -- see ENTROPY_FLOOR
    grad_norm_history = []  # rolling window for real-time spike detection, see GRAD_NORM_SPIKE_WINDOW
    single_training_eval_history = []  # [(episode, avg_reward), ...] -- watch for mid-run collapse, not just the final number
    completed_episodes = 0
    interrupted = False

    resuming = os.path.exists(RESUME_STATE_PATH)
    if resuming:
        print(f"[train-multi-v4] Found {RESUME_STATE_PATH} -- resuming (not starting from episode 1).")
        resumed = load_resume_state(RESUME_STATE_PATH, model, optimizer, reward_scale_normalizer)
        completed_episodes = resumed["completed_episodes"]
        best_avg_eval_reward = resumed["best_avg_eval_reward"]
        best_episode = resumed["best_episode"]
        ignition_points_sampled = resumed["ignition_points_sampled"]
        episode_rewards = resumed["episode_rewards"]
        episode_losses = resumed["episode_losses"]
        episode_wall_times = resumed["episode_wall_times"]
        print(f"[train-multi-v4] Resuming from episode {completed_episodes + 1}/{N_EPISODES} "
              f"(best combined-avg eval reward so far: {best_avg_eval_reward:.1f} at ep {best_episode})")
    start_episode = completed_episodes + 1
    if start_episode > N_EPISODES:
        print(f"[train-multi-v4] Resume state already shows {completed_episodes}/{N_EPISODES} episodes "
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
            for ep in range(start_episode, N_EPISODES + 1):
                ep_t0 = time.perf_counter()
                seed = BASE_SEED + ep
                log_kl = LOG_KL_EVERY_EPISODE or (ep % STATUS_EVERY == 0)
                anchor_point = TRAINING_IGNITION_POINT if ep % ANCHOR_REPLAY_EVERY_N == 0 else None
                # Baseline from episodes BEFORE this one only -- see ppo_update()'s docstring
                # for why this gates the entropy-floor+grad-spike combined stop.
                grad_norm_baseline = (
                    sorted(grad_norm_history)[len(grad_norm_history) // 2]
                    if len(grad_norm_history) >= GRAD_NORM_SPIKE_WINDOW else None
                )

                try:
                    result = run_training_episode(env, model, optimizer, device, seed,
                                                   reward_scale_normalizer, log_kl, grad_norm_baseline,
                                                   ignition_point=anchor_point)
                except Exception as e:
                    if not _is_mps_unimplemented_error(e):
                        raise
                    op_name = _extract_op_name(e)
                    mps_errors.append(op_name)
                    print(f"[train-multi-v4] MPS op not implemented: '{op_name}' -- "
                          f"falling back to CPU for the rest of this run.")
                    device = torch.device("cpu")
                    model = model.to(device)
                    anchor_probe_grid, anchor_probe_scalars = anchor_probe_grid.to(device), anchor_probe_scalars.to(device)
                    result = run_training_episode(env, model, optimizer, device, seed,
                                                   reward_scale_normalizer, log_kl, grad_norm_baseline,
                                                   ignition_point=anchor_point)

                wall_s = time.perf_counter() - ep_t0
                episode_wall_times.append(wall_s)
                episode_rewards.append(result["reward"])
                episode_losses.append((result["policy_loss"], result["value_loss"],
                                        result["classification_loss"], result["entropy"]))
                recent_wall_times.append(wall_s)
                recent_rewards.append(result["reward"])
                completed_episodes = ep
                ignition_points_sampled.append(result["ignition_point"])
                if result["early_stopped"]:
                    early_stop_minibatches.append(result["minibatches_run"])
                    stop_reason_counts[result["stop_reason"]] += 1

                train_writer.writerow({
                    "episode": ep, "device": str(device), "n_ticks": result["n_ticks"],
                    "ignition_row": result["ignition_point"][0], "ignition_col": result["ignition_point"][1],
                    "reward": result["reward"], "shaped_reward_total": result["shaped_reward_total"],
                    "buildings_destroyed": result["buildings_destroyed"],
                    "contained": result["contained"], "timeout": result["timeout"],
                    "is_anchor": result["is_anchor"],
                    "policy_loss": result["policy_loss"], "value_loss": result["value_loss"],
                    "classification_loss": result["classification_loss"], "entropy": result["entropy"],
                    "approx_kl_first_epoch": result["approx_kl_first_epoch"],
                    "approx_kl_last_epoch": result["approx_kl_last_epoch"],
                    "epochs_run": result["epochs_run"], "minibatches_run": result["minibatches_run"],
                    "early_stopped": result["early_stopped"], "stop_reason": result["stop_reason"] or "",
                    "value_grad_norm": result["value_grad_norm"], "other_grad_norm": result["other_grad_norm"],
                    "wall_time_s": wall_s,
                })
                train_file.flush()

                # Real-time gradient-norm spike warning -- NOT gated by log_kl, printed the
                # instant it happens. See ENTROPY_FLOOR's docstring: the ep100-103 buildup
                # (28k->32k->17k->80,483) was the earliest real signal ahead of that run's
                # entropy collapse, visible here days before entropy itself cratered.
                grad_norm_history.append(result["other_grad_norm"])
                if len(grad_norm_history) > GRAD_NORM_SPIKE_WINDOW:
                    grad_norm_history.pop(0)
                if len(grad_norm_history) >= GRAD_NORM_SPIKE_WINDOW:
                    baseline = sorted(grad_norm_history)[len(grad_norm_history) // 2]  # median, robust to spikes in the window itself
                    if baseline > 0 and result["other_grad_norm"] > GRAD_NORM_SPIKE_MULTIPLIER * baseline:
                        print(f"    [grad-norm SPIKE] ep {ep}: other_grad_norm={result['other_grad_norm']:.1f} "
                              f"vs recent median={baseline:.1f} ({result['other_grad_norm'] / baseline:.1f}x) -- "
                              f"watch next few episodes' entropy closely", flush=True)

                if log_kl:
                    stop_note = (f"  EARLY-STOPPED after {result['minibatches_run']} minibatch(es) "
                                 f"(epoch {result['epochs_run'] + 1}, reason={result['stop_reason']})") \
                        if result["early_stopped"] else ""
                    anchor_note = "  [ANCHOR REPLAY]" if result["is_anchor"] else ""
                    print(f"    [ppo-kl] ep {ep}: approx_KL before epoch 1 = {result['approx_kl_first_epoch']:.5f}  "
                          f"final = {result['approx_kl_last_epoch']:.5f}  entropy={result['entropy']:.5f}  "
                          f"epochs_run={result['epochs_run']} minibatches_run={result['minibatches_run']}{stop_note}  "
                          f"ignition=({result['ignition_point'][0]},{result['ignition_point'][1]})  "
                          f"n_ticks={result['n_ticks']}  timeout={result['timeout']}{anchor_note}  "
                          f"other_grad_norm={result['other_grad_norm']:.1f}", flush=True)

                if ep % STATUS_EVERY == 0:
                    window_eps_per_min = len(recent_wall_times) / (sum(recent_wall_times) / 60.0)
                    window_avg_reward = sum(recent_rewards) / len(recent_rewards)
                    print(f"[status @ ep {ep:5d}/{N_EPISODES}] "
                          f"episodes/min(last{STATUS_EVERY})={window_eps_per_min:6.2f}  "
                          f"avg_reward(last{STATUS_EVERY})={window_avg_reward:10.1f}  "
                          f"entropy={result['entropy']:.3f}  "
                          f"policy_loss={result['policy_loss']:8.3f}  "
                          f"value_loss={result['value_loss']:8.3f}  "
                          f"class_loss={result['classification_loss']:.3f}  "
                          f"reward_scale_std={reward_scale_normalizer.std():.2f}  "
                          f"value_grad_norm={result['value_grad_norm']:.1f}  "
                          f"other_grad_norm={result['other_grad_norm']:.1f}  device={device}",
                          flush=True)
                    log_anchor_flag_engagement(model, anchor_probe_grid, anchor_probe_scalars, device)

                if ep % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    torch.save(model.state_dict(), latest_ckpt_path)
                    save_resume_state(RESUME_STATE_PATH, model, optimizer, reward_scale_normalizer,
                                       completed_episodes, best_avg_eval_reward, best_episode,
                                       ignition_points_sampled, episode_rewards, episode_losses, episode_wall_times)
                    print(f"  [checkpoint] saved {ckpt_path} (and latest.pt, resume_state.pt)", flush=True)

                if ep % EVAL_EVERY == 0:
                    last_eval_results = run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                    avg_eval_reward = sum(r["avg_reward"] for r in last_eval_results.values()) / len(last_eval_results)
                    single_training_eval_history.append((ep, last_eval_results["single_training"]["avg_reward"]))
                    print(f"    [eval @ ep {ep}] combined avg reward across all "
                          f"{len(last_eval_results)} scenarios: {avg_eval_reward:.1f}", flush=True)
                    if avg_eval_reward > best_avg_eval_reward:
                        best_avg_eval_reward = avg_eval_reward
                        best_episode = ep
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"  [checkpoint] new best combined-avg eval reward ({avg_eval_reward:.1f}) "
                              f"at ep {ep} -- saved {best_ckpt_path}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[train-multi-v4] Caught interrupt (Ctrl+C or SIGTERM) after episode "
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

    label = f"INTERRUPTED at episode {completed_episodes}/{N_EPISODES}" if interrupted \
        else f"completed all {completed_episodes} episodes"
    print(f"\n=== Multi-ignition v4 (GAE + multi-epoch PPO + continuous ignition) summary ({label}) ===")
    print(f"Total wall time: {total_wall:.1f}s  Overall episodes/min: {episodes_per_min:.2f}")
    print(f"Policy loss: {_nan_or_frozen(policy_losses)}")
    print(f"Value loss: {_nan_or_frozen(value_losses)}")
    print(f"Classification loss: {_nan_or_frozen(class_losses)}")
    print(f"Entropy: {_nan_or_frozen(entropies)}")
    print(f"Final reward_scale_normalizer: mean={reward_scale_normalizer.mean:.2f} "
          f"std={reward_scale_normalizer.std():.2f} n={int(reward_scale_normalizer.count)}")

    n_early = len(early_stop_minibatches)
    if completed_episodes > 0:
        avg_stop_mb = sum(early_stop_minibatches) / n_early if n_early else float("nan")
        print(f"Early-stopping: triggered in {n_early}/{completed_episodes} episodes "
              f"({n_early / completed_episodes:.0%})" +
              (f", average stop after {avg_stop_mb:.1f} minibatch(es)" if n_early else "") +
              f"  -- by reason: {dict(stop_reason_counts)}  "
              f"(KL_threshold={KL_EARLY_STOP_THRESHOLD}, ENTROPY_FLOOR={ENTROPY_FLOOR})")

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
        print(f"Stochastic rollout reward (raw, unshaped), first->last episode: "
              f"{episode_rewards[0]:.1f} -> {episode_rewards[-1]:.1f}")
        print(f"Stochastic rollout reward, mean first {n_tail} vs last {n_tail}: "
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
