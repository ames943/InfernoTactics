"""Candidate PPO consumer contracts, not claims of learned-policy quality."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.world.contracts import Action
from src.world.learning import (
    ActorCritic, FEATURE_NAMES, HYPERPARAMETERS, candidate_features,
    checkpoint_metadata, compute_gae, load_policy, scenario_hashes, train,
)
from src.world.synthetic import make_synthetic
from src.world.world import World


def scenario_fixture():
    scenario = make_synthetic(size=9, duration_s=60)
    scenario.resources = []
    return scenario


def observed(scenario):
    world = World(scenario, seed=10000)
    ignition = scenario.provenance["ignition"]
    world.ignite(ignition["longitude"], ignition["latitude"])
    return world.observe()


def save_checkpoint(path, scenario, mutate=None):
    model = ActorCritic()
    # Equal scores make deterministic tie-breaking choose the first feasible action.
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    checkpoint = {"metadata": checkpoint_metadata(scenario, 7, [7]),
                  "state_dict": model.state_dict()}
    if mutate:
        mutate(checkpoint)
    torch.save(checkpoint, path)
    return path


def test_exact_architecture_and_fixed_hyperparameters():
    model = ActorCritic()
    for network in (model.actor, model.critic):
        assert [layer.out_features for layer in network if isinstance(layer, nn.Linear)] == [64, 64, 1]
        assert sum(isinstance(layer, nn.Tanh) for layer in network) == 2
    assert not ({id(item) for item in model.actor.parameters()} &
                {id(item) for item in model.critic.parameters()})
    assert HYPERPARAMETERS["learning_rate"] == 3e-4
    assert HYPERPARAMETERS["gamma"] == .99
    assert HYPERPARAMETERS["gae_lambda"] == .95
    assert HYPERPARAMETERS["clip"] == .2
    assert HYPERPARAMETERS["value_coefficient"] == .5
    assert HYPERPARAMETERS["entropy_coefficient"] == .01
    assert HYPERPARAMETERS["max_gradient_norm"] == .5
    assert HYPERPARAMETERS["transitions_per_update"] == 256
    assert HYPERPARAMETERS["epochs"] == 4
    assert HYPERPARAMETERS["minibatch_size"] == 64


def test_mask_excludes_infeasible_actions_and_padding_from_actor_and_critic():
    torch.manual_seed(3)
    model = ActorCritic()
    features = torch.rand(3, len(FEATURE_NAMES))
    mask = torch.tensor([True, False, True])
    distribution, value = model(features, mask)
    assert distribution.probs[1] == 0
    assert 1 not in distribution.sample((100,)).tolist()
    changed = features.clone()
    changed[1] = 1e6
    second, second_value = model(changed, mask)
    torch.testing.assert_close(distribution.probs, second.probs)
    torch.testing.assert_close(value, second_value)
    compact, compact_value = model(features[[0, 2]], torch.tensor([True, True]))
    torch.testing.assert_close(distribution.probs[[0, 2]], compact.probs)
    torch.testing.assert_close(value, compact_value)
    padded = torch.stack((features, changed))
    batched, batched_value = model(padded, mask.expand(2, -1))
    torch.testing.assert_close(batched.probs[0], distribution.probs)
    torch.testing.assert_close(batched_value, value.expand(2))
    with pytest.raises(ValueError, match="feasible"):
        model(features, torch.zeros(3, dtype=torch.bool))


def test_features_follow_metadata_order_and_fixed_bounds():
    scenario = make_synthetic(size=9, duration_s=120)
    obs = observed(scenario)
    unit = obs["units"][0]
    a, _, c, _, e, f = scenario.transform
    unit.update(x=c + 4.5 * a, y=f + 4.5 * e, capacity_l=1000., supply_l=250., status="available")
    target = Action(unit["id"], "water", c + 4.5 * a, f + 4.5 * e)
    obs["candidates"] = [Action(), target]
    obs["candidate_mask"] = [True, False]
    obs["candidate_metadata"] = [{"eta_s": None}, {"eta_s": 60.}]
    obs["time_s"] = 60
    obs["fire"]["fireline_intensity"] = obs["fire"]["fireline_intensity"].copy()
    obs["fire"]["fireline_intensity"][4, 4] = 10000.
    features, mask = candidate_features(obs, scenario)
    index = {name: pos for pos, name in enumerate(FEATURE_NAMES)}
    assert mask.tolist() == [True, False]
    assert features.shape == (2, len(FEATURE_NAMES))
    assert torch.all((features >= 0) & (features <= 1))
    assert features[1, index["operation_water"]] == 1
    assert features[1, index["unit_status_available"]] == 1
    assert features[1, index["unit_x"]] == .5
    assert features[1, index["unit_y"]] == .5
    assert features[1, index["unit_supply_fraction"]] == .25
    assert features[1, index["target_x"]] == .5
    assert features[1, index["target_fireline_intensity_fraction"]] == .5
    assert features[1, index["feasible_eta_fraction"]] == .5
    assert features[1, index["elapsed_time_fraction"]] == .5
    # Future weather/provenance is not a feature source.
    altered = copy.deepcopy(obs)
    altered["weather"] = {"future": "not observable policy input"}
    altered["provenance"] = {"hidden": 1234}
    second, _ = candidate_features(altered, scenario)
    torch.testing.assert_close(features, second)
    obs["candidate_mask"][0] = False
    with pytest.raises(ValueError, match="wait"):
        candidate_features(obs, scenario)


def test_gae_terminal_truncation_and_rollout_bootstrap_are_distinct():
    advantages, returns = compute_gae(
        rewards=[1., 2., 3., 4.], values=[.5, .6, .7, .8],
        next_values=[.6, 10., 100., 2.],
        terminated=[False, False, True, False],
        episode_ends=[False, True, True, False])
    # Truncated second transition bootstraps 10 but cannot see reset reward 3.
    expected_second = 2. + .99 * 10. - .6
    expected = [1. + .99 * .6 - .5 + .99 * .95 * expected_second,
                expected_second, 3. - .7, 4. + .99 * 2. - .8]
    np.testing.assert_allclose(advantages, expected, rtol=1e-6)
    np.testing.assert_allclose(returns, np.asarray(expected) + [.5, .6, .7, .8], rtol=1e-6)
    with pytest.raises(ValueError, match="end"):
        compute_gae([0], [0], [0], [True], [False])
    with pytest.raises(ValueError, match="vectors"):
        compute_gae([0, 1], [0], [0], [False], [False])


def test_checkpoint_loads_safely_and_evaluation_is_deterministic(tmp_path, monkeypatch):
    scenario = scenario_fixture()
    path = save_checkpoint(tmp_path / "policy.pt", scenario)
    original = torch.load
    safe_calls = []

    def safe_load(*args, **kwargs):
        safe_calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    policy = load_policy(path, scenario)
    obs = observed(scenario)
    assert policy.act(obs, scenario) == Action()
    assert policy.act(obs, scenario) == policy.act(obs, scenario)
    assert policy.training_seeds == (7,)
    assert policy.metadata["feature_ordering"] == list(FEATURE_NAMES)
    assert policy.metadata["input_hashes"]["scenario"] == scenario_hashes(scenario)["scenario"]
    assert safe_calls == [True]
    assert not policy.model.training


@pytest.mark.parametrize("field,value", [
    ("schema_version", 0), ("scenario_schema_version", 0),
    ("backend", "legacy"), ("engine_version", "old"),
    ("feature_ordering", list(reversed(FEATURE_NAMES))),
    ("hyperparameters", {"gamma": .5}), ("input_hashes", {}),
    ("training_seeds", [10000]), ("training_seeds", []),
])
def test_checkpoint_rejects_incompatible_metadata(tmp_path, field, value):
    scenario = scenario_fixture()
    path = save_checkpoint(tmp_path / "policy.pt", scenario,
                           lambda item: item["metadata"].update({field: value}))
    with pytest.raises(ValueError, match="incompatible"):
        load_policy(path, scenario)


def test_content_hashes_detect_input_changes_not_just_scenario_name(tmp_path):
    scenario = scenario_fixture()
    path = save_checkpoint(tmp_path / "policy.pt", scenario)
    changed = copy.deepcopy(scenario)
    changed.landscape["elevation"][0, 0] += 1.
    assert changed.name == scenario.name
    assert scenario_hashes(changed) != scenario_hashes(scenario)
    with pytest.raises(ValueError, match="input_hashes"):
        load_policy(path, changed)


@pytest.mark.parametrize("corruption", ["legacy", "shape", "nonfinite", "executable"])
def test_checkpoint_rejects_old_unsafe_or_invalid_weights(tmp_path, corruption):
    scenario = scenario_fixture()
    path = tmp_path / "policy.pt"
    if corruption == "legacy":
        torch.save({"model_state_dict": ActorCritic().state_dict()}, path)
    elif corruption == "executable":
        torch.save({"metadata": {"unsafe": Action()}, "state_dict": {}}, path)
    elif corruption == "shape":
        save_checkpoint(path, scenario,
                        lambda item: item["state_dict"].update({"actor.0.weight": torch.zeros(1, 1)}))
    else:
        save_checkpoint(path, scenario,
                        lambda item: item["state_dict"]["actor.0.weight"].fill_(float("nan")))
    with pytest.raises(ValueError, match="checkpoint"):
        load_policy(path, scenario)


@pytest.mark.parametrize("steps,seed", [(0, 7), (-1, 7), (True, 7), (1, -1), (1, 1000), (1, True)])
def test_training_rejects_invalid_bounds_and_evaluation_seeds(tmp_path, steps, seed):
    with pytest.raises(ValueError):
        train(scenario_fixture(), steps, seed, tmp_path / "invalid.pt")
    assert not (tmp_path / "invalid.pt").exists()


def test_training_exact_steps_seed_wrap_and_reproducible_checkpoint(tmp_path):
    scenario = scenario_fixture()
    first = train(scenario, 3, 999, tmp_path / "first.pt")
    second = train(scenario, 3, 999, tmp_path / "second.pt")
    assert first["steps"] == 3
    assert first["updates"] == 1
    assert first["training_seeds"] == [999, 0, 1]
    assert first["reward"] == second["reward"]
    left = load_policy(first["checkpoint"], scenario)
    right = load_policy(second["checkpoint"], scenario)
    for name, parameter in left.model.state_dict().items():
        assert torch.isfinite(parameter).all()
        torch.testing.assert_close(parameter, right.model.state_dict()[name], rtol=0, atol=0)
    assert left.act(observed(scenario), scenario) == Action()
    with pytest.raises(ValueError, match="exists"):
        train(scenario, 1, 7, first["checkpoint"])
