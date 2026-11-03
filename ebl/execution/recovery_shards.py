"""Deterministic native recovery work units and completed-bundle dependencies."""
import json
from pathlib import Path

from core.artifacts import content_hash, sha256_file


def screens(settings):
    if settings.learning_rate is not None:
        return []
    return [{'key': f'lr{i}-{mask}-b{budget:g}', 'learning_rate': lr,
             'mask': mask, 'allowance': budget}
            for i, lr in enumerate(settings.learning_rates)
            for mask in ('stable', 'random') for budget in settings.write_budgets]


def trajectories(settings):
    result = []
    scenarios = [(rho, False) for rho in settings.persistent_fractions]
    if settings.no_drift_control: scenarios.append((.9, True))
    for rho, no_drift in scenarios:
        scenario = 'no_drift' if no_drift else f'rho{rho:g}'
        for arm in settings.arms:
            if no_drift and arm not in ('untreated', 'stable_learning', 'random_learning'): continue
            if arm == 'oracle_learning' and (rho != .9 or no_drift): continue
            for allowance in settings.write_budgets if arm.endswith('_learning') else (None,):
                if arm == 'oracle_learning' and allowance != max(settings.write_budgets): continue
                for protocol in settings.protocols:
                    if arm == 'oracle_learning' and protocol != 'single': continue
                    result.append({'key': f'{scenario}-{arm}-{protocol}-b{allowance}',
                        'rho': rho, 'no_drift': no_drift, 'arm': arm,
                        'allowance': allowance, 'protocol': protocol})
    return result


def read_artifact(run, name):
    """Require a completed native bundle and the declared artifact digest."""
    run = Path(run)
    result = json.loads((run/'result.json').read_text())
    # Native result status is checked as well as the artifact itself.
    status = json.loads((run/'status.json').read_text())
    if status.get('state', status.get('status')) != 'complete':
        raise ValueError(f'Incomplete native dependency: {run}')
    records = [r for r in result['artifacts'] if r['path'] == name]
    if len(records) != 1 or sha256_file(run/name) != records[0]['sha256']:
        raise ValueError(f'Invalid native dependency artifact: {run/name}')
    return json.loads((run/name).read_text())


def select_learning_rate(settings, prepared, screen_runs):
    if settings.learning_rate is not None:
        if screen_runs: raise ValueError('An explicit LR cannot consume development screens.')
        return settings.learning_rate, []
    expected = {s['key']: s for s in screens(settings)}
    seen, rows = set(), []
    for run in screen_runs:
        result = read_artifact(run, 'screen.json')
        if result['preparation_sha256'] != prepared or result['settings_sha256'] != content_hash(settings.science_dict()):
            raise ValueError('Development screen belongs to a different preparation or protocol.')
        key = result['key']
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected development screen.')
        row = result['row']
        if any(row[k] != expected[key][k] for k in ('learning_rate', 'mask', 'allowance')):
            raise ValueError('Development screen metadata mismatch.')
        seen.add(key)
        rows.append(row)
    if seen != set(expected): raise ValueError('Incomplete development screen coverage.')
    lr = min(settings.learning_rates, key=lambda v: sum(r['clean_hwa_kl'] for r in rows if r['learning_rate'] == v))
    return lr, rows


if __name__ == '__main__':
    import sys
    from methods.maintenance.config import DriftRecovery
    settings = DriftRecovery.parse(json.loads(Path(sys.argv[1]).read_text())['drift_recovery'])
    result = {'screens': screens(settings), 'trajectories': []}
    for t in trajectories(settings):
        events = settings.intervention_times(t['protocol'])
        observations = []
        for seconds in settings.times(t['protocol']):
            observations.append([seconds, 'before' if seconds in events else 'observation'])
            if seconds in events and t['arm'] != 'untreated': observations.append([seconds, 'after'])
        result['trajectories'].append(t | {'observations': observations,
            'events': len(events) if t['arm'] != 'untreated' else 0})
    print(json.dumps(result))
