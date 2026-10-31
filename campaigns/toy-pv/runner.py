"""TOY conductance programming and drift simulator for testing the experiment loop.

Not device evidence. Conductance is normalized to [0, 1] and clamped there. The
first read is at T0 = 1 s; drift is log-time with a conductance-dependent mean
and independent random increments.
"""
import argparse
import json
from pathlib import Path

import numpy as np

T0 = 1.0
TIMES_S = [1, 10, 60, 600, 3600]
SIGMA_WRITE = 0.03
DRIFT_SD_PER_DECADE = 0.006


def drift_mean_per_decade(g):
    return -0.01 - 0.02 * g


def program(rng, aim, max_retries, tol):
    g = np.clip(aim + rng.normal(0, SIGMA_WRITE, aim.shape), 0, 1)
    writes = np.ones_like(aim)
    for _ in range(max_retries):
        bad = np.abs(g - aim) > tol
        if not bad.any():
            break
        g[bad] = np.clip(aim[bad] + rng.normal(0, SIGMA_WRITE, bad.sum()), 0, 1)
        writes[bad] += 1
    return g, writes


def run_arm(seed, k, target, arm):
    rng = np.random.default_rng([seed, k])
    aim = target
    if arm.get('precomp_t_ref_s') is not None:
        aim = np.clip(target - drift_mean_per_decade(target) * np.log10(arm['precomp_t_ref_s'] / T0), 0, 1)
    g0, writes = program(rng, aim, arm['max_retries'], arm['tol'])
    rms, g, prev = {}, g0.copy(), 0.0
    for t in TIMES_S:
        d = np.log10(t / T0)
        g = np.clip(g + drift_mean_per_decade(g0) * (d - prev) + rng.normal(0, DRIFT_SD_PER_DECADE * np.sqrt(d - prev), g.shape), 0, 1)
        prev = d
        rms[str(t)] = float(np.sqrt(np.mean((g - target) ** 2)))
    return rms, float(writes.mean() - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    per_seed = {arm['name']: [] for arm in cfg['arms']}
    for seed in range(cfg['seeds']):
        target = np.random.default_rng(seed).uniform(0, 1, cfg['n_devices'])
        for k, arm in enumerate(cfg['arms']):
            rms, extra = run_arm(seed, k, target, arm)
            per_seed[arm['name']].append({'rms_err': rms, 'extra_writes_per_device': extra})

    def stat(xs):
        return {'mean': float(np.mean(xs)), 'sd': float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0}

    arms = {name: {'rms_err': {str(t): stat([r['rms_err'][str(t)] for r in rows]) for t in TIMES_S},
                   'extra_writes_per_device': stat([r['extra_writes_per_device'] for r in rows])}
            for name, rows in per_seed.items()}
    out = {'toy': True, 'units': 'normalized conductance', 'times_s': TIMES_S, 'config': cfg,
           'arms': arms, 'per_seed': per_seed}
    Path(a.out, 'metrics.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
