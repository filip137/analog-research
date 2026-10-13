"""CPU-only descriptive figures for the frozen exp-013 comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

SPLITS = ('development', 'transfer_generated', 'transfer_wikitext')
TAGS = ('std3', 'cdt3', 'std9')
CAPS = ('cap2', 'cap4', 'uncapped')
LOGICAL_WEIGHTS = 28_311_552


def write_statistics(cost):
    # Uncapped native writers retain successful writes and stuck attempts but
    # do not allocate the capped writer's per-device attempted-write counters.
    total = cost['device_writes'] + cost['stuck_write_attempts']
    if cost.get('issued_write_attempts', total) != total:
        raise ValueError('Issued attempts disagree with successful and stuck writes')
    return {
        'attempts_per_active_cell': total/LOGICAL_WEIGHTS,
        'writes_per_active_cell': cost['device_writes']/LOGICAL_WEIGHTS,
        'maximum_attempts_per_active_cell': cost.get('max_onchip_device_write_attempts'),
        'maximum_writes_per_active_cell': cost['max_onchip_device_writes'],
        'attempt_total_source': 'recorded' if 'issued_write_attempts' in cost else 'successful_plus_stuck',
    }


def render(summary, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not summary['coverage_complete'] or len(summary['rows']) != 162:
        raise ValueError('Figures require complete declared coverage')
    output.mkdir(parents=True, exist_ok=True)
    flat, aggregates = [], []
    for row in summary['rows']:
        for split in SPLITS:
            effect = row['effects'][split]
            flat.append({
                'source': row['tag'], 'array_seed': row['array_seed'],
                'rate_ppm': row['rate_ppm'], 'cap': row['cap'], 'domain': split,
                'deployment_kl': effect['deployment']['teacher_kl'],
                'recovered_kl': effect['onchip']['teacher_kl'],
                'kl_reduction': effect['kl_reduction'],
                'deployment_ppl': effect['deployment']['perplexity'],
                'recovered_ppl': effect['onchip']['perplexity'],
                'ppl_reduction': effect['deployment']['perplexity'] - effect['onchip']['perplexity'],
                'retained_gain': row['retained_gain'][split],
                'gain_over_best_hwa_only': row['vs_best_hwa_only'][split]['kl_reduction'],
                **write_statistics(row['cost']),
            })
    fields = ('deployment_kl', 'recovered_kl', 'kl_reduction',
              'deployment_ppl', 'recovered_ppl', 'ppl_reduction', 'retained_gain',
              'gain_over_best_hwa_only', 'attempts_per_active_cell', 'writes_per_active_cell')
    for tag in TAGS:
        for split in SPLITS:
            for cap in CAPS:
                for rate in sorted({r['rate_ppm'] for r in flat}):
                    group = [r for r in flat if (r['source'], r['domain'], r['cap'], r['rate_ppm']) == (tag, split, cap, rate)]
                    if len(group) != 3 or len({r['array_seed'] for r in group}) != 3:
                        raise ValueError('Expected three distinct assignments in each comparison')
                    aggregate = {'source': tag, 'domain': split, 'cap': cap, 'rate_ppm': rate,
                                 'improved_assignments': sum(r['kl_reduction'] > 0 for r in group),
                                 'ppl_improved_assignments': sum(r['ppl_reduction'] > 0 for r in group),
                                 'assignments': [r['array_seed'] for r in group]}
                    for field in fields:
                        values = [r[field] for r in group if r[field] is not None]
                        aggregate[field] = {'n': len(values), 'mean': mean(values) if values else None,
                                            'min': min(values) if values else None,
                                            'max': max(values) if values else None}
                    aggregates.append(aggregate)
    with (output/'endpoints.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    (output/'grouped.json').write_text(json.dumps(aggregates, indent=2, allow_nan=False)+'\n')
    colors = dict(zip(CAPS, ('tab:blue', 'tab:orange', 'tab:green')))
    source_names = {'std3': 'Standard 3% Gaussian HWA', 'cdt3': 'Corruption-aware 3% HWA',
                    'std9': 'Standard 9% Gaussian HWA'}
    split_names = {'development': 'Development', 'transfer_generated': 'Generated test',
                   'transfer_wikitext': 'WikiText test'}
    for metric, label, suffix in (('kl', 'Teacher KL', ''), ('ppl', 'Perplexity', '-ppl')):
        # PPL is the recorded exp(cross-entropy), averaged across arrays, never exp(KL).
        deployed, recovered, reduction = f'deployment_{metric}', f'recovered_{metric}', f'{metric}_reduction'
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True,
                                 sharey='row' if metric == 'ppl' else False)
        costs, cost_axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True,
                                       sharey='row' if metric == 'ppl' else False)
        for i, split in enumerate(SPLITS):
            for j, tag in enumerate(TAGS):
                ax, cost_ax = axes[i, j], cost_axes[i, j]
                selected = [g for g in aggregates if g['source'] == tag and g['domain'] == split]
                baseline = [g for g in selected if g['cap'] == 'uncapped']
                x = [g['rate_ppm']/10000 for g in baseline]
                ax.plot(x, [g[deployed]['mean'] for g in baseline], 'k--', label='HWA deployment')
                ax.fill_between(x, [g[deployed]['min'] for g in baseline],
                                [g[deployed]['max'] for g in baseline], color='black', alpha=.08)
                for cap in CAPS:
                    group = [g for g in selected if g['cap'] == cap]
                    x = [g['rate_ppm']/10000 for g in group]
                    ax.plot(x, [g[recovered]['mean'] for g in group], marker='o', color=colors[cap], label=cap)
                    ax.fill_between(x, [g[recovered]['min'] for g in group],
                                    [g[recovered]['max'] for g in group], color=colors[cap], alpha=.12)
                    for row in flat:
                        if (row['source'], row['domain'], row['cap']) == (tag, split, cap):
                            ax.scatter(row['rate_ppm']/10000, row[recovered], color=colors[cap], s=8, alpha=.5)
                for rate in sorted({g['rate_ppm'] for g in selected}):
                    group = [next(g for g in selected if g['rate_ppm'] == rate and g['cap'] == cap) for cap in CAPS]
                    cost_ax.plot([g['attempts_per_active_cell']['mean'] for g in group],
                                 [g[reduction]['mean'] for g in group], 'o-', label=f'{rate/10000:g}% faults')
                title = f'{source_names[tag]}\n{split_names[split]}'
                ax.set(title=title, xlabel='Actual corruption (%)', ylabel=f'{label} (lower is better)')
                cost_ax.set(title=title, xlabel='Mean issued attempts per active cell',
                            ylabel=f'{label} reduction from own deployment')
                cost_ax.axhline(0, color='black', linewidth=.6)
                ax.grid(alpha=.2); cost_ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8); cost_axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{label}: arithmetic means, ranges and all three arrays; caps count attempts/device')
        costs.suptitle(f'{label} reduction versus idealized write attempts: cap 2, cap 4, uncapped; means of three arrays')
        for figure, name in ((fig, 'recovery-versus-corruption'), (costs, 'recovery-versus-writes')):
            for extension in ('png', 'pdf'): figure.savefig(output/f'{name}{suffix}.{extension}', dpi=160)
            plt.close(figure)
    return {'endpoints': len(summary['rows']), 'domain_rows': len(flat), 'groups': len(aggregates)}
