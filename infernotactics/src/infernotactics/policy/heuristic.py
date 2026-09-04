"""
Rule-based baseline policy -- NOT learned, with no training or gradients.
The preferred ``decide_actions`` interface returns the same list of dispatch
tuples consumed by InfernoEnv and allows a fair multi-dispatch comparison in
``eval_relative.eval_policy``. ``__call__`` is retained only for older tools.

Rule, evaluated fresh every tick, across the 4 resource types:
  - water_team / helicopter: dispatch to the nearest zone (by REAL road/air
    routing ETA -- self.zone_travel_time_s, the same per-zone travel times
    InfernoEnv itself computed from the real depots/road graph, cached from
    the env once at construction rather than any shortcut Euclidean metric)
    that currently contains an active (Threat/Blaze) fire cell. Straight
    suppression, wherever the fire actually is right now.
  - trench_crew: dispatch to the nearest zone (same real routing) that
    currently has NO active fire cell but DOES have a Fuel cell adjacent to
    one -- i.e. just ahead of the advancing front. Deliberately NOT "nearest
    zone touching the fire": inferno_env._apply_trench fails (0 cells
    affected) if the effect footprint contains any Threat/Blaze cell, and
    inferno_env._effect_target_point aims at the centroid of whatever active
    fire is in the target zone if there is any -- so sending trench_crew to
    a zone that already has active fire in it would aim the effect straight
    at the fire and guarantee failure. A zone with fire-adjacent Fuel but no
    fire yet is exactly where a real defensive line gets dug.
  - rescue_vehicle: dispatch to the zone maximizing
    (max population_density among its currently-threatened building cells)
    / travel_time_s -- population-weighted urgency (a denser, slightly
    farther zone can outrank a sparser, closer one), not pure nearest-
    neighbor, matching the project's population-aware reward design
    (inferno_env.RESCUE_PENALTY_REDUCTION_MIN/MAX).

The decision is recomputed after each selected unit, so multiple resource
units may be dispatched in one tick. If no resource qualifies, it returns an
empty list (a true no-dispatch tick).

    python -m infernotactics.policy.heuristic  # baseline table across all scenarios
"""

import math
import os

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from infernotactics.simulation.fire_sim import BLAZE, FUEL, THREAT
from infernotactics.operations.environment import (
    BUILDING_PRESENCE_THRESHOLD,
)
from infernotactics.domain.observations import SCALAR_KEYS  # noqa: E402
from infernotactics.domain.resources import RESOURCE_TYPES  # noqa: E402
from infernotactics.world.layers import LAYER_INDEX  # noqa: E402
from infernotactics.policy.models.classification_head import N_CLASSES

_DILATION_STRUCTURE = np.ones((3, 3), dtype=bool)  # 8-connected, matches fire_sim's Moore neighborhood


class HeuristicPolicy:
    """See module docstring for the rule. Constructed once per InfernoEnv
    (zone geometry + real per-zone travel times are static across resets,
    so they're cached here rather than recomputed every call)."""

    def __init__(self, env):
        self.zones = env.zones
        self.n_zones = env.n_zones
        self.zone_travel_time_s = env.zone_travel_time_s
        self.training = False  # duck-types InfernoModel's nn.Module surface, see module docstring

    def eval(self):
        pass

    def train(self, mode=True):
        self.training = mode

    def _decide(self, fire_state, building_density, population_density, available):
        active_mask = np.isin(fire_state, (THREAT, BLAZE))
        fuel_mask = fire_state == FUEL
        threatened_building_mask = (building_density > BUILDING_PRESENCE_THRESHOLD) & active_mask
        fuel_adjacent_to_fire = binary_dilation(active_mask, structure=_DILATION_STRUCTURE) & fuel_mask

        candidates = {}  # resource_type -> (zone_id, travel_time_s)

        for zone in self.zones:
            zid = zone["zone_id"]
            r0, r1 = zone["row_range"]
            c0, c1 = zone["col_range"]

            zone_active = active_mask[r0:r1, c0:c1].any()
            zone_fuel_adjacent = fuel_adjacent_to_fire[r0:r1, c0:c1].any()
            zone_threatened = threatened_building_mask[r0:r1, c0:c1]
            zone_has_threatened_building = zone_threatened.any()

            for rtype in ("water_team", "helicopter"):
                if available[rtype] < 1 or not zone_active:
                    continue
                t = self.zone_travel_time_s[rtype][zid]
                if math.isfinite(t) and (rtype not in candidates or t < candidates[rtype][1]):
                    candidates[rtype] = (zid, t)

            if available["trench_crew"] >= 1 and zone_fuel_adjacent and not zone_active:
                t = self.zone_travel_time_s["trench_crew"][zid]
                if math.isfinite(t) and ("trench_crew" not in candidates or t < candidates["trench_crew"][1]):
                    candidates["trench_crew"] = (zid, t)

            if available["rescue_vehicle"] >= 1 and zone_has_threatened_building:
                t = self.zone_travel_time_s["rescue_vehicle"][zid]
                if math.isfinite(t):
                    pop = float(population_density[r0:r1, c0:c1][zone_threatened].max())
                    score = pop / max(t, 1.0)
                    prev = candidates.get("rescue_vehicle")
                    prev_score = (prev[2] if prev is not None else -1.0)
                    if score > prev_score:
                        candidates["rescue_vehicle"] = (zid, t, score)

        if not candidates:
            return None, None

        rtype = min(candidates, key=lambda k: candidates[k][1])
        zone_id = candidates[rtype][0]
        return rtype, zone_id

    def decide_actions(self, grid_np, scalars_np):
        """Return a policy-decided list of dispatches for one simulation tick.

        Re-runs the existing greedy rule after each selected unit, decrementing
        that resource's local availability. The environment still validates
        the final list against the actual roster.
        """
        fire_state = grid_np[-1]
        building_density = grid_np[LAYER_INDEX["building_density"]]
        population_density = grid_np[LAYER_INDEX["population_density"]]
        available = {
            rtype: int(scalars_np[SCALAR_KEYS.index(f"{rtype}_available")])
            for rtype in RESOURCE_TYPES
        }
        actions = []
        for _ in range(sum(available.values())):
            rtype, zone_id = self._decide(
                fire_state, building_density, population_density, available
            )
            if rtype is None:
                break
            actions.append((rtype, zone_id))
            available[rtype] -= 1
        return actions

    def __call__(self, grid, scalars):
        """grid: (1, n_grid_channels, H, W), scalars: (1, n_scalars) -- same
        shapes InfernoModel.forward() takes. Returns (action_logits,
        value, classification_logits) with the exact same shapes
        InfernoModel returns, so eval_policy()'s deterministic argmax
        decoding (_select_action) picks out this rule's chosen action.
        value/classification_logits are unused by eval_policy() -- filled
        with zeros just to keep the interface shape-honest."""
        assert grid.shape[0] == 1, "HeuristicPolicy only supports batch size 1 (matches eval_policy()'s usage)"
        device = grid.device
        grid_np = grid[0].detach().cpu().numpy()
        scalars_np = scalars[0].detach().cpu().numpy()

        fire_state = grid_np[-1]
        building_density = grid_np[LAYER_INDEX["building_density"]]
        population_density = grid_np[LAYER_INDEX["population_density"]]
        available = {
            rtype: scalars_np[SCALAR_KEYS.index(f"{rtype}_available")] for rtype in RESOURCE_TYPES
        }

        rtype, zone_id = self._decide(fire_state, building_density, population_density, available)

        n_resource_types = len(RESOURCE_TYPES)
        resource_logits = torch.zeros(1, n_resource_types, device=device)
        zone_logits = torch.zeros(1, self.n_zones, device=device)
        if rtype is not None:
            resource_logits[0, RESOURCE_TYPES.index(rtype)] = 1.0
            zone_logits[0, zone_id] = 1.0
        # else: leave all-zero -- argmax picks index 0 for both heads, which
        # eval_policy's deterministic decoding will turn into a real dispatch
        # attempt at a possibly-empty zone rather than a true noop. InfernoEnv
        # has no noop-logit slot in its action factorization either (see
        # The learned policy has the same factorized-output limitation, so
        # this keeps the comparison fair rather than giving the heuristic
        # an escape hatch the learned policy does not have.

        h, w = fire_state.shape
        action_logits = {"resource_type": resource_logits, "zone": zone_logits}
        value = torch.zeros(1, 1, device=device)
        classification_logits = torch.zeros(1, N_CLASSES, h, w, device=device)
        return action_logits, value, classification_logits


def _run_baseline_table():
    from infernotactics.operations.environment import (
        MULTI_IGNITION_TRAINING_SCENARIO,
        TRAINING_IGNITION_POINT,
        VALIDATION_IGNITION_POINTS,
        InfernoEnv,
    )
    from infernotactics.training.evaluation import eval_policy

    print("Building InfernoEnv...")
    env = InfernoEnv(seed=0)
    env.reset(seed=0)  # warm up (loads routing) before constructing the policy
    policy = HeuristicPolicy(env)

    scenarios = [("single_training (Palisades Highlands / Skull Rock)", {"ignition_point": TRAINING_IGNITION_POINT})]
    for name, point in VALIDATION_IGNITION_POINTS.items():
        scenarios.append((f"validation:{name}", {"ignition_point": point}))
    scenarios.append(("multi_ignition (topanga/sullivan/stone canyon)",
                       {"ignition_points": MULTI_IGNITION_TRAINING_SCENARIO}))

    print(f"\n{'scenario':52s} {'avg_reward':>12s} {'avg_bldgs_destroyed':>20s} {'containment_rate':>18s}")
    print("-" * 106)
    results = {}
    for name, reset_kwargs in scenarios:
        result = eval_policy(policy, env, n_episodes=5, use_real_weather=True, deterministic=True,
                              seed=0, **reset_kwargs)
        results[name] = result
        print(f"{name:52s} {result['avg_reward']:12.1f} {result['avg_buildings_destroyed']:20.1f} "
              f"{result['containment_rate']:17.0%}")

    return results


if __name__ == "__main__":
    _run_baseline_table()
