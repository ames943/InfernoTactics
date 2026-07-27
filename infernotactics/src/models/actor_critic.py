"""
Actor-critic head (the RL piece of the 4-concept fusion): takes the fused
CNN+MLP feature vector and outputs an action distribution plus a state-value
estimate.

Action factorization
---------------------
InfernoEnv's action is (resource_type, target_zone). Rather than one flat
categorical over all 3*32=96 combinations, this factors the actor into two
independent categorical heads:

    p(resource_type, target_zone) = p(resource_type) * p(target_zone)

i.e. resource_type and target_zone are sampled independently, not
autoregressively (the zone logits do not condition on the sampled resource
type). This is the simpler of the two standard options for a
MultiDiscrete-style action space and a reasonable starting point -- revisit
only if training shows the two need to interact (e.g. a resource type
systematically wanting different zones than the others).

Note this head always proposes a (resource_type, zone) pair; it does not
have a separate "noop" logit even though InfernoEnv.step() accepts one --
matches the factorization specified in the project plan as-is.
"""

import torch.nn as nn

N_RESOURCE_TYPES = 4  # water_team, trench_crew, rescue_vehicle, helicopter
N_ZONES = 32
HIDDEN_DIM = 128


class ActorCritic(nn.Module):
    """n_value_heads=1 (default) is byte-for-byte the original architecture --
    a single self.value_head attribute, forward()'s value_head_idx argument
    unused -- so every existing caller (train_actor_critic.py,
    train_actor_critic_multi.py/v2, heuristic_policy.py, eval.py) is
    completely unaffected. n_value_heads>1 (added for
    train_actor_critic_multi_v3.py's separate-value-head-per-scenario
    experiment) instead builds self.value_heads, an nn.ModuleList, and
    forward()'s value_head_idx selects which one to use for a given
    call -- the shared trunk/actor heads are untouched either way, only the
    final value projection becomes scenario-specific. See
    train_actor_critic_multi_v3.py's module docstring for why (per-scenario
    return normalization alone, in v2, wasn't sufficient -- the shared value
    head's LAST layer was still one set of weights receiving gradient from
    every scenario's wildly different-scale value loss every episode)."""

    def __init__(self, in_features, n_resource_types=N_RESOURCE_TYPES, n_zones=N_ZONES, hidden_dim=HIDDEN_DIM,
                 n_value_heads=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resource_type_head = nn.Linear(hidden_dim, n_resource_types)
        self.zone_head = nn.Linear(hidden_dim, n_zones)
        self.n_value_heads = n_value_heads
        if n_value_heads == 1:
            self.value_head = nn.Linear(hidden_dim, 1)
        else:
            self.value_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_value_heads)])

    def forward(self, fused, value_head_idx=0):
        """fused: (B, in_features) -> (
            {'resource_type': (B, n_resource_types) logits, 'zone': (B, n_zones) logits},
            value: (B, 1)
        )
        value_head_idx is ignored when n_value_heads==1 (the original,
        single-shared-head behavior)."""
        h = self.trunk(fused)
        action_logits = {
            "resource_type": self.resource_type_head(h),
            "zone": self.zone_head(h),
        }
        value = self.value_head(h) if self.n_value_heads == 1 else self.value_heads[value_head_idx](h)
        return action_logits, value
