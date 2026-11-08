"""Consumer checks against actual small-world rollouts, not echoed mock scores."""
import json
from statistics import fmean, stdev

import pytest
import torch

from src.world.evaluation import DEFAULT_EVALUATION_SEEDS, PHYSICAL_METRICS, evaluate
from src.world.learning import ActorCritic, checkpoint_metadata
from src.world.rollout import run_episode
from src.world.synthetic import make_synthetic


def small_world(*, duration_s=120):
    scenario = make_synthetic(size=15, cell_size_m=30., duration_s=duration_s)
    # Keep candidate feasibility inexpensive while retaining real dispatch and
    # travel physics. Only the helicopter remains, not a fake policy/world.
    scenario.resources = [unit for unit in scenario.resources if unit['kind'] == 'helicopter']
    scenario.resources[0].update(mobilization_s=0., work_duration_s=30.)
    scenario.landscape['asset_value'][:] = 100.
    scenario.validate()
    return scenario


def save_wait_policy(scenario, path, *, training_seeds=(7,)):
    model = ActorCritic()
    # Equal candidate logits choose the first, feasible wait action at inference.
    # This checkpoint uses the real model and metadata compatibility path.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save({
        'metadata': checkpoint_metadata(scenario, seed=7, training_seeds=list(training_seeds)),
        'state_dict': model.state_dict(),
    }, path)
    return path


def test_paired_metrics_equal_same_seed_real_rollouts_and_persist(tmp_path):
    scenario = small_world()
    checkpoint = save_wait_policy(scenario, tmp_path / 'wait.pt')
    output = tmp_path / 'panel'
    seeds = [10000, 10003]
    result = evaluate(scenario, checkpoint, seeds, output)
    assert result['seeds'] == seeds
    assert result['training_seeds'] == [7]
    assert result['historical_validation'] == 'unvalidated'
    assert result['suppression_validation'] == 'experimental'
    assert json.loads((output / 'evaluation.json').read_text()) == result

    episodes = {(episode['seed'], episode['policy']): episode for episode in result['episodes']}
    assert len(episodes) == len(seeds) * 3
    for seed in seeds:
        for policy in ('none', 'heuristic', 'checkpoint'):
            episode = episodes[seed, policy]
            artifact = output / str(seed) / policy
            actual = json.loads((artifact / 'metrics.json').read_text())
            assert (artifact / 'events.jsonl').exists()
            assert episode['physical_outcomes'] == {metric: actual[metric] for metric in PHYSICAL_METRICS}
            assert episode['reward'] == actual['reward']
            assert 'reward' not in episode['physical_outcomes']
            assert episode['area_censored'] == actual['area_censored']
            # Independently consume the public rollout, including its actual
            # World, to reject accidental scenario/seed/policy mismatches.
            replay = run_episode(scenario, seed, policy,
                                 checkpoint=checkpoint if policy == 'checkpoint' else None)
            for metric in PHYSICAL_METRICS:
                assert episode['physical_outcomes'][metric] == replay[metric]
            assert episode['reward'] == replay['reward']
        assert episodes[seed, 'none']['physical_outcomes']['burned_area_m2'] > 0.
        assert episodes[seed, 'none']['physical_outcomes']['resource_time_s'] == 0.
        assert episodes[seed, 'heuristic']['physical_outcomes']['resource_time_s'] > 0.
        assert episodes[seed, 'checkpoint']['physical_outcomes'] == episodes[seed, 'none']['physical_outcomes']

    for comparison in result['paired_deltas'].values():
        policy, baseline = comparison['policy'], comparison['baseline']
        assert [pair['seed'] for pair in comparison['pairs']] == seeds
        for pair in comparison['pairs']:
            treatment = episodes[pair['seed'], policy]
            control = episodes[pair['seed'], baseline]
            for metric in PHYSICAL_METRICS:
                expected = treatment['physical_outcomes'][metric] - control['physical_outcomes'][metric]
                assert pair['physical_outcomes'][metric] == pytest.approx(expected)
            assert pair['reward'] == pytest.approx(treatment['reward'] - control['reward'])
            assert pair['area_censored'] == (treatment['area_censored'] or control['area_censored'])
        for metric in PHYSICAL_METRICS:
            values = [pair['physical_outcomes'][metric] for pair in comparison['pairs']
                      if metric != 'burned_area_m2' or not pair['area_censored']]
            aggregate = comparison['physical_outcomes'][metric]
            assert aggregate['count'] == len(values)
            assert aggregate['mean'] == (pytest.approx(fmean(values)) if values else None)
            assert aggregate['sample_std'] == (pytest.approx(stdev(values)) if len(values) > 1 else None)
        rewards = [pair['reward'] for pair in comparison['pairs']]
        assert comparison['reward']['mean'] == pytest.approx(fmean(rewards))
        assert comparison['reward']['sample_std'] == pytest.approx(stdev(rewards))


def test_default_panel_is_reserved_held_out_range_in_real_world(tmp_path):
    scenario = small_world(duration_s=60)
    checkpoint = save_wait_policy(scenario, tmp_path / 'wait.pt')
    result = evaluate(scenario, checkpoint)
    assert result['seeds'] == list(range(10000, 10032)) == list(DEFAULT_EVALUATION_SEEDS)
    for policy in ('none', 'heuristic', 'checkpoint'):
        assert [episode['seed'] for episode in result['episodes'] if episode['policy'] == policy] == result['seeds']
        assert result['summary'][policy]['reward']['count'] == 32


@pytest.mark.parametrize('seeds, message', [
    ([], 'at least one'), ([10000, 10000], 'unique'),
    ([-1], 'nonnegative integers'), ([True], 'nonnegative integers'),
    ([10000.5], 'nonnegative integers'), ([999], 'reserved training seeds'),
    ([7], 'overlap checkpoint training seeds'),
])
def test_invalid_or_training_seed_panel_rejected_before_output(tmp_path, seeds, message):
    scenario = small_world()
    checkpoint = save_wait_policy(scenario, tmp_path / 'wait.pt')
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError, match=message):
        evaluate(scenario, checkpoint, seeds, output)
    assert not output.exists()


def test_out_of_range_recorded_training_seeds_are_incompatible(tmp_path):
    scenario = small_world()
    checkpoint = save_wait_policy(scenario, tmp_path / 'wait.pt')
    payload = torch.load(checkpoint, weights_only=True)
    payload['metadata']['training_seeds'] = [10000]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='incompatible checkpoint training-seed metadata'):
        evaluate(scenario, checkpoint, [10000], tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_censored_real_world_deltas_are_flagged_and_not_area_averaged(tmp_path):
    scenario = make_synthetic(size=9, cell_size_m=5., duration_s=600)
    scenario.resources = []
    for key, value in (
        ('wind_speed_m_s', 8.), ('fuel_moisture_dead_1hr', .03),
        ('fuel_moisture_dead_10hr', .04), ('fuel_moisture_dead_100hr', .05),
        ('fuel_moisture_live_herbaceous', .30),
    ):
        scenario.weather[key][:] = value
    scenario.validate()
    checkpoint = save_wait_policy(scenario, tmp_path / 'wait.pt')
    result = evaluate(scenario, checkpoint, [10001])
    assert all(episode['area_censored'] and episode['domain_exit'] for episode in result['episodes'])
    for summary in result['summary'].values():
        assert summary['physical_outcomes']['burned_area_m2']['count'] == 0
        assert summary['physical_outcomes']['burned_area_m2']['mean'] is None
        assert summary['area_censored_count'] == 1
    for comparison in result['paired_deltas'].values():
        assert comparison['pairs'][0]['area_censored']
        assert comparison['pairs'][0]['domain_exit']
        assert comparison['physical_outcomes']['burned_area_m2']['count'] == 0
        assert comparison['physical_outcomes']['burned_area_m2']['mean'] is None
        assert comparison['reward']['count'] == 1


def test_existing_output_is_not_overwritten(tmp_path):
    scenario = small_world()
    checkpoint = save_wait_policy(scenario, tmp_path / 'wait.pt')
    output = tmp_path / 'existing'
    output.mkdir()
    sentinel = output / 'sentinel'
    sentinel.write_text('keep')
    with pytest.raises(ValueError, match='output already exists'):
        evaluate(scenario, checkpoint, [10000], output)
    assert sentinel.read_text() == 'keep'
