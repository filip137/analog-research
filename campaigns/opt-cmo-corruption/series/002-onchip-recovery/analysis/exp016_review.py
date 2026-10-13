"""Validate exp-016 random3 collections and apply the frozen R0–R6 rules.

CPU-only analysis of completed native evidence. No dispatch or model fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'campaigns/opt-cmo-corruption/launch')]
import exp16_pair_rewrite as experiment
from campaigns.recovery_collect import checked

DOMAINS = ('transfer_generated', 'transfer_wikitext')


def decisions(summary):
    """Classify each source/domain separately; undefined fault gaps stay inconclusive."""
    if (summary['phase'] != 'random3' or not summary['coverage_complete']
            or summary['starting_conditions'] != 18 or summary['endpoints'] != 90):
        raise ValueError('Expected complete random3 coverage')
    units = {(u['source'], u['array_seed'], u['domain'], u['arm']): u for u in summary['units']}
    rows = []
    for tag in experiment.TAGS:
        for domain in DOMAINS:
            pairs = []
            for array in experiment.ARRAYS:
                arms = {name: units[tag, array, domain, name] for name in (
                    'active_rewrite_oracle', 'pair_rewrite_oracle', 'pair_rewrite_verify',
                    'pair_rewrite_verify_recovery')}
                pairs.append(arms)
            defined = all(a[n]['gap_closed'] is not None for a in pairs for n in a)
            verify = [a['pair_rewrite_verify'] for a in pairs]
            delta = [a['pair_rewrite_verify_recovery']['X'] - a['pair_rewrite_verify']['X'] for a in pairs]
            rows.append({'source': tag, 'domain': domain,
                'R1': all(u['gap_closed'] >= .90 for u in verify) if defined else None,
                'R2': all(a['pair_rewrite_verify']['gap_closed'] >= a['pair_rewrite_oracle']['gap_closed'] - .02
                          for a in pairs) if defined else None,
                'R3': all(a['pair_rewrite_oracle']['gap_closed'] - a['active_rewrite_oracle']['gap_closed'] >= .20
                          for a in pairs) if defined else None,
                'R4': all(u['X'] < u['exp013_' + cap] for u in verify for cap in ('uncapped', 'cap4')),
                'R6': 'adds' if all(d < 0 for d in delta) else 'harms' if all(d > 0 for d in delta) else 'mixed',
                'pair_verify_gap_closed': [u['gap_closed'] for u in verify],
                'learning_kl_change': delta})
    best = {}
    for domain in DOMAINS:
        winners = [tag for tag in experiment.TAGS if all(
            units[tag, array, domain, 'pair_rewrite_verify']['X']
            < units[other, array, domain, 'pair_rewrite_verify']['X']
            for other in experiment.TAGS if other != tag for array in experiment.ARRAYS)]
        best[domain] = winners[0] if len(winners) == 1 else 'not established'
    held = {(r['tag'], r['array_seed']): r for r in summary['rows'] if r['case'] == 'nominal' and r['arm'] == 'none'}
    healthy = []
    for row in summary['rows']:
        if row['case'] != 'nominal' or row['arm'] != 'pair_rewrite_verify':
            continue
        original = held[row['tag'], row['array_seed']]
        changes = {s: (row['final'][s]['teacher_kl'] - original['final'][s]['teacher_kl'])
                      / original['final'][s]['teacher_kl'] for s in experiment.SPLITS}
        healthy.append({'source': row['tag'], 'array_seed': row['array_seed'],
                        'false_positive_devices': row['fault_rewrite']['analysis']['false_positive_devices'],
                        'relative_kl_changes': changes, 'R0_flag': any(abs(v) > 1e-4 for v in changes.values())})
    return {'scope': 'Three frozen HWA sources, three arrays, random 3% only; model-based.',
            'assignments': list(experiment.ARRAYS), 'R1_R4_R6': rows, 'R5': best, 'R0': healthy}


def review(evidence):
    hosts = [json.loads(p.read_text()) for p in (evidence / 'prepared').glob('*.json')]
    placements = {(h['host'], c['id']): c['placement'] for h in hosts for c in h['cases']}
    reports, hashes = [], {}
    for label in experiment.labels('random3'):
        report = json.loads((evidence / 'reviews' / (label + '.json')).read_text())
        path = Path(report['collection']); folder = path.parent
        receipt = json.loads(path.read_text())
        for name, digest in receipt['files'].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Changed collected artifact: ' + str(folder / name))
        checked(json.loads((folder / 'export.json').read_text()),
                placements[receipt['target'], receipt['job']]['recovery_input_contract'])
        rows = json.loads((folder / 'measurements.json').read_text())
        records = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines() if s.strip()]
        if experiment.checked_report(label, rows, records, path) != report:
            raise ValueError('Review differs from its collected native evidence')
        reports.append(report)
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = experiment.summarize(reports, evidence.parents[1] / 'opt125m-cmo-l6-exp7-recovery-regimes', 'random3')
    if summary != json.loads((evidence / 'reviews/summary.json').read_text()):
        raise ValueError('Summary differs from its validated per-job reports')
    output = evidence / 'analysis'; output.mkdir(exist_ok=True)
    verdict = decisions(summary)
    verdict['collection_sha256'] = hashes
    (output / 'decision.json').write_text(json.dumps(verdict, indent=2, allow_nan=False) + '\n')
    units = {(u['source'], u['array_seed'], u['domain'], u['arm']): u for u in summary['units']}
    rows = []
    for row in summary['rows']:
        rewrite = row['fault_rewrite'] or {}
        for split, metrics in row['final'].items():
            unit = units.get((row['tag'], row['array_seed'], split, row['arm'])) if row['case'] != 'nominal' else None
            rows.append({'source': row['tag'], 'array_seed': row['array_seed'], 'case': row['case'],
                'arm': row['arm'], 'domain': split, 'teacher_kl': metrics['teacher_kl'],
                'perplexity': metrics['perplexity'], 'gap_closed': unit['gap_closed'] if unit else None,
                'rewrite_device_programs': rewrite.get('cost', {}).get('device_programs', 0),
                'verify_reads': rewrite.get('cost', {}).get('verify_reads', 0),
                'recovery_device_writes': row['cost'].get('device_writes', 0),
                'recovery_stuck_attempts': row['cost'].get('stuck_write_attempts', 0),
                'active_program_requests': rewrite.get('controller', {}).get('programmed_active', 0),
                'reference_program_requests': rewrite.get('controller', {}).get('programmed_reference', 0),
                'clipped_targets': rewrite.get('controller', {}).get('clipped_targets', 0),
                'both_stuck': rewrite.get('analysis', {}).get('stuck_both', 0),
                'false_positive_devices': rewrite.get('analysis', {}).get('false_positive_devices', 0),
                'false_negative_devices': rewrite.get('analysis', {}).get('false_negative_devices', 0)})
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    from statistics import mean
    lines = ['# exp-016: healthy and random 3% first phase', '',
        'Complete validated coverage: three frozen HWA sources, assignments 271002–271004, '
        '18 starting conditions and 90 endpoints. Six analog OPT-125M MLPs contain 28,311,552 logical weights. '
        'The broader rates/kinds comparison remains deferred.', '',
        'The following means summarize the three arrays; decisions below require every array separately. '
        'Costs exclude initial deployment and separate partner programs from subsequent recovery writes.', '',
        '| Source | Arm | WikiText KL, mean (range) | Rewrite programs, mean | Recovery writes, mean |',
        '| --- | --- | ---: | ---: | ---: |']
    for tag in experiment.TAGS:
        for arm in [a.arm_id for a in experiment.lifecycle(tag, 'random3').onchip.arms]:
            selected = [r for r in rows if r['source'] == tag and r['case'] == 'random_30000ppm'
                        and r['arm'] == arm and r['domain'] == 'transfer_wikitext']
            values = [r['teacher_kl'] for r in selected]
            lines.append(f'| {tag} | {arm} | {mean(values):.6f} ({min(values):.6f}–{max(values):.6f}) | '
                         f'{mean(r["rewrite_device_programs"] for r in selected):,.0f} | '
                         f'{mean(r["recovery_device_writes"] for r in selected):,.0f} |')
    lines += ['', '| Source | Held-out domain | R1: ≥90% closed | R2: detection | R3: role swap | R4: beats exp-013 | R6: learning after rewrite |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in verdict['R1_R4_R6']:
        labels = ['inconclusive' if row[k] is None else 'pass' if row[k] else 'fail' for k in ('R1','R2','R3','R4')]
        lines.append('| ' + ' | '.join([row['source'], row['domain'], *labels, row['R6']]) + ' |')
    lines += ['', 'R5, lowest compensated KL on every assignment: ' + '; '.join(
        f'{domain}: {tag}' for domain, tag in verdict['R5'].items()) + '.', '',
        f'R0 flags a healthy-array relative KL change above 1e-4 in '
        f'{sum(r["R0_flag"] for r in verdict["R0"])} of nine source/array pairs. '
        'False detections and signed changes are retained in decision.json.', '',
        'R1 failing means the predeclared 90% gap-closure criterion is not established for that source/domain; '
        'it does not imply that rewriting provides no benefit. R6 compares learning after the rewrite with '
        'that same rewritten state. The source comparison is conditional on this fault regime.', '',
        'These are model-based results with synthetic permanent faults, independently programmable references, '
        'a fitted Gaussian P&V endpoint law, and no read noise or drift. Programs are not measured pulse counts. '
        'One HWA training seed and reused test cohorts limit generalization. No conclusions cover other fault rates or kinds.', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.evidence.resolve()), indent=2, allow_nan=False))
