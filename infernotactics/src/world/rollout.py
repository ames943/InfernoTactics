"""One authoritative rollout path for command line and paired evaluation."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .contracts import Action, Scenario
from .world import World


def run_episode(scenario: Scenario, seed: int, policy='none', *, checkpoint=None,
                longitude=None, latitude=None, out: Path | None = None) -> dict:
    world = World(scenario, seed)
    ignition = scenario.provenance.get('ignition', {})
    if (longitude is None) != (latitude is None):
        raise ValueError('longitude and latitude overrides must be supplied together')
    longitude = ignition.get('longitude') if longitude is None else longitude
    latitude = ignition.get('latitude') if latitude is None else latitude
    if longitude is None or latitude is None:
        raise ValueError('scenario has no ignition; supply --longitude and --latitude')
    world.ignite(longitude, latitude)
    if policy not in ('none', 'heuristic', 'checkpoint'):
        raise ValueError(f'unknown policy: {policy}')
    learned = None
    if policy == 'checkpoint':
        if checkpoint is None:
            raise ValueError('checkpoint policy requires --checkpoint')
        from .learning import load_policy
        learned = load_policy(checkpoint, scenario)
    total_reward = 0.0
    busy_s = 0.0
    previous = np.zeros_like(scenario.landscape['valid'])
    dispatch_count = 0
    snapshots = []
    if out is not None:
        out = Path(out)
        if out.exists():
            raise ValueError(f'output already exists: {out}')
        out.mkdir(parents=True)
    obs = world.observe()
    try:
        while world.time_s < scenario.duration_s:
            if policy == 'none':
                action = Action()
            elif policy == 'heuristic':
                from .actions import heuristic_action
                action = heuristic_action(obs, scenario)
            else:
                action = learned.act(obs, scenario)
            obs, reward, done, info = world.step([action])
            total_reward += float(reward)
            busy_s += info['resource_time_s']
            dispatch_count += action.operation in ('water', 'line')
            snap = world.fire.snapshot()
            previous |= snap['burned']
            if out is not None:
                np.savez_compressed(out / f'snapshot-{world.time_s:06d}.npz',
                                    time_s=np.asarray(world.time_s), **snap)
            snapshots.append({'time_s': world.time_s, 'burned_area_m2': int(previous.sum()) * scenario.transform[0] ** 2})
            if done:
                break
        burned = world.fire.snapshot()['burned'] & scenario.landscape['valid']
        metrics = {
            'scenario': scenario.name, 'seed': seed, 'policy': policy,
            'time_s': world.time_s, 'burned_area_m2': float(burned.sum() * scenario.transform[0] ** 2),
            'asset_exposure_m2': float(scenario.landscape['asset_value'][burned].sum()),
            'resource_time_s': busy_s, 'dispatch_count': dispatch_count,
            'reward': total_reward, 'no_new_growth_s': world.no_new_growth_s,
            'area_censored': bool(world.fire.area_censored),
            'domain_exit': bool(world.fire.domain_exited),
            'units': world.resources.snapshot(), 'trajectory': snapshots,
            'observation_mode': 'oracle_current', 'suppression_validation': 'experimental',
            'historical_validation': 'unvalidated',
        }
        if out is not None:
            (out / 'events.jsonl').write_text(''.join(json.dumps(asdict(e), sort_keys=True, allow_nan=False) + '\n' for e in world.events))
            (out / 'metrics.json').write_text(json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + '\n')
        return metrics
    except Exception:
        # A partial output has snapshots but never a success metrics report.
        if out is not None:
            (out / 'metrics.json').unlink(missing_ok=True)
        raise
