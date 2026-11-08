"""Held-out paired policy evaluation through the authoritative world rollout.

Seeds 0--999 are reserved for training; the default evaluation panel is
10000--10031. Deltas are treatment minus baseline, so negative area/exposure
and resource-time deltas mean smaller observed outcomes, not proven benefit.
Burned-area summaries exclude censored episodes/pairs. Other physical metrics
remain observed-at-stop outcomes and carry explicit censoring flags; neither
quiet fire nor an improved reward establishes containment or validation.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from statistics import fmean, stdev

from .contracts import Scenario
from .learning import load_policy
from .rollout import run_episode


DEFAULT_EVALUATION_SEEDS = tuple(range(10000, 10032))
POLICIES = ('none', 'heuristic', 'checkpoint')
PHYSICAL_METRICS = (
    'burned_area_m2', 'asset_exposure_m2', 'resource_time_s',
    'dispatch_count', 'time_s', 'no_new_growth_s',
)


def _statistics(values: list[float]) -> dict:
    """Keep absent uncensored evidence distinct from a zero outcome."""
    return {
        'count': len(values),
        'mean': fmean(values) if values else None,
        'sample_std': stdev(values) if len(values) > 1 else None,
        'min': min(values) if values else None,
        'max': max(values) if values else None,
    }


def _summary(episodes: list[dict]) -> dict:
    physical = {}
    for metric in PHYSICAL_METRICS:
        eligible = [episode['physical_outcomes'][metric] for episode in episodes
                    if metric != 'burned_area_m2' or not episode['area_censored']]
        physical[metric] = _statistics(eligible)
    return {
        'physical_outcomes': physical,
        'reward': _statistics([episode['reward'] for episode in episodes]),
        'area_censored_count': sum(episode['area_censored'] for episode in episodes),
        'domain_exit_count': sum(episode['domain_exit'] for episode in episodes),
    }


def evaluate(scenario: Scenario, checkpoint, seeds=None, out=None) -> dict:
    """Compare no response, nearest-feasible dispatch, and a learned checkpoint.

    Every policy receives the identical scenario and seed through ``run_episode``.
    Explicit evaluation seeds must be unique nonnegative integers outside the
    reserved training range and the checkpoint's recorded training panel.
    ``out``, when supplied, must be a new directory: it receives ``evaluation.json``
    plus full rollout artifacts in ``<seed>/<policy>/``. A failed panel never
    writes a successful aggregate report. Checkpoint compatibility and seed
    leakage are checked before any episode or output directory is created.
    """
    selected = list(DEFAULT_EVALUATION_SEEDS if seeds is None else seeds)
    if not selected:
        raise ValueError('evaluation requires at least one seed')
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
           for seed in selected):
        raise ValueError('evaluation seeds must be nonnegative integers')
    selected = [int(seed) for seed in selected]
    if len(set(selected)) != len(selected):
        raise ValueError('evaluation seeds must be unique')
    learned = load_policy(checkpoint, scenario)
    training_seeds = tuple(learned.training_seeds)
    overlap = sorted(set(selected).intersection(training_seeds))
    if overlap:
        raise ValueError(f'evaluation seeds overlap checkpoint training seeds: {overlap}')
    reserved = sorted(seed for seed in selected if seed < 1000)
    if reserved:
        raise ValueError(f'evaluation seeds use reserved training seeds 0--999: {reserved}')

    destination = Path(out) if out is not None else None
    if destination is not None:
        if destination.exists():
            raise ValueError(f'output already exists: {destination}')
        destination.mkdir(parents=True)

    episodes = []
    for seed in selected:
        for policy in POLICIES:
            episode = run_episode(
                scenario, seed, policy, checkpoint=checkpoint if policy == 'checkpoint' else None,
                out=destination / str(seed) / policy if destination is not None else None,
            )
            episodes.append({
                'seed': seed,
                'policy': policy,
                'physical_outcomes': {metric: episode[metric] for metric in PHYSICAL_METRICS},
                'reward': episode['reward'],
                'area_censored': bool(episode['area_censored']),
                'domain_exit': bool(episode['domain_exit']),
            })

    by_key = {(episode['seed'], episode['policy']): episode for episode in episodes}
    paired = {}
    for policy, baseline in (('heuristic', 'none'), ('checkpoint', 'none'),
                             ('checkpoint', 'heuristic')):
        pairs = []
        for seed in selected:
            treatment, control = by_key[seed, policy], by_key[seed, baseline]
            pairs.append({
                'seed': seed,
                'physical_outcomes': {
                    metric: treatment['physical_outcomes'][metric] - control['physical_outcomes'][metric]
                    for metric in PHYSICAL_METRICS
                },
                'reward': treatment['reward'] - control['reward'],
                'area_censored': treatment['area_censored'] or control['area_censored'],
                'domain_exit': treatment['domain_exit'] or control['domain_exit'],
            })
        paired[f'{policy}_minus_{baseline}'] = {
            'policy': policy, 'baseline': baseline, 'pairs': pairs, **_summary(pairs),
        }

    report = {
        'scenario': scenario.name,
        'checkpoint': str(checkpoint),
        'seeds': selected,
        'training_seeds': list(training_seeds),
        'observation_mode': 'oracle_current',
        'suppression_validation': 'experimental',
        'historical_validation': 'unvalidated',
        'delta_convention': 'policy minus baseline; negative means smaller observed outcome',
        'area_summary': 'burned-area aggregates exclude area-censored episodes and pairs',
        'outcome_scope': 'observed at episode stop, not proof of containment or response effectiveness',
        'episodes': episodes,
        'summary': {
            policy: _summary([episode for episode in episodes if episode['policy'] == policy])
            for policy in POLICIES
        },
        'paired_deltas': paired,
    }
    if destination is not None:
        (destination / 'evaluation.json').write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report
