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

N_RESOURCE_TYPES = 3
N_ZONES = 32
HIDDEN_DIM = 128


class ActorCritic(nn.Module):
    def __init__(self, in_features, n_resource_types=N_RESOURCE_TYPES, n_zones=N_ZONES, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.resource_type_head = nn.Linear(hidden_dim, n_resource_types)
        self.zone_head = nn.Linear(hidden_dim, n_zones)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, fused):
        """fused: (B, in_features) -> (
            {'resource_type': (B, n_resource_types) logits, 'zone': (B, n_zones) logits},
            value: (B, 1)
        )"""
        h = self.trunk(fused)
        action_logits = {
            "resource_type": self.resource_type_head(h),
            "zone": self.zone_head(h),
        }
        value = self.value_head(h)
        return action_logits, value
