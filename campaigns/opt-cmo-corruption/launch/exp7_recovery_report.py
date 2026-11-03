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


def render_teacher_comparison(summary, reference, output: Path):
    """Compare the matched digital teacher, deployment and alternative recovery caps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not summary['coverage_complete'] or len(summary['rows']) != 162:
        raise ValueError('Figures require complete declared coverage')
    output.mkdir(parents=True, exist_ok=True)
    rows = summary['rows']
    index = {(r['tag'], r['array_seed'], r['rate_ppm'], r['cap']): r for r in rows}
    if len(index) != 162:
        raise ValueError('Duplicate recovery endpoints')
    arrays = sorted({r['array_seed'] for r in rows})
    if len(arrays) != 3:
        raise ValueError('Expected three independent arrays')
    names = {'std3': 'Standard 3% Gaussian HWA', 'cdt3': 'Corruption-aware 3% HWA',
             'std9': 'Standard 9% Gaussian HWA'}
    domains = {'development': 'Development', 'transfer_generated': 'Generated test',
               'transfer_wikitext': 'WikiText test'}
    colors = dict(zip(TAGS, ('tab:blue', 'tab:orange', 'tab:purple')))
    comparisons = []
    for split in SPLITS:
        teacher = reference['metrics'][split]['perplexity']
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True, constrained_layout=True)
        for ax, rate in zip(axes.flat, (0, 10000, 20000, 30000, 40000, 50000)):
            for tag in TAGS:
                group = [index[tag, seed, rate, 'cap4'] for seed in arrays]
                if any(r['effects'][split]['deployment']['scored_tokens'] !=
                       reference['metrics'][split]['scored_tokens'] for r in group):
                    raise ValueError('Teacher and deployment token counts differ')
                stages = [[r['effects'][split]['deployment']['perplexity'] for r in group]]
                for cap in CAPS:
                    stages.append([index[tag, seed, rate, cap]['effects'][split]['onchip']['perplexity']
                                   for seed in arrays])
                means = [mean(values) for values in stages]
                ax.plot(range(5), [teacher] + means, 'o-', color=colors[tag], label=names[tag], markersize=4)
                ax.fill_between(range(1, 5), [min(v) for v in stages], [max(v) for v in stages],
                                color=colors[tag], alpha=.12)
                for values, x in zip(stages, range(1, 5)):
                    ax.scatter([x] * 3, values, color=colors[tag], s=9, alpha=.6)
                for before, after, before_values, after_values in (
                    ('cap2', 'cap4', stages[1], stages[2]),
                    ('cap4', 'uncapped', stages[2], stages[3]),
                ):
                    gain = [a - b for a, b in zip(before_values, after_values)]
                    comparisons.append({'domain': split, 'source': tag, 'rate_ppm': rate,
                                        'from_cap': before, 'to_cap': after,
                                        'mean_ppl_reduction': mean(gain), 'min': min(gain), 'max': max(gain),
                                        'improved_arrays': sum(x > 0 for x in gain), 'arrays': len(gain)})
            ax.axhline(teacher, color='black', linestyle=':', linewidth=1)
            ax.scatter([0], [teacher], color='black', marker='D', s=25, zorder=5)
            ax.set(title=f'{rate/10000:g}% actual corruption', ylabel='Perplexity (lower is better)',
                   xticks=range(5), xticklabels=['Teacher', 'Deployed', '≤2 attempts', '≤4 attempts', 'Uncapped'])
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(axis='y', alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{domains[split]}: teacher, HWA deployment and on-chip recovery'
                     f'\nTeacher PPL = {teacher:.3f}; means, ranges and three individual arrays'
                     '\nEach recovery arm restarts from deployment; same 2.1M-token budget; caps are per device')
        for extension in ('png', 'pdf'):
            fig.savefig(output/f'ppl-teacher-deployment-recovery-{split}.{extension}', dpi=160)
        plt.close(fig)
    with (output/'ppl-incremental-write-gains.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    return comparisons


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
                'ppl_reduction_percent': 100 * (effect['deployment']['perplexity'] - effect['onchip']['perplexity'])
                                         / effect['deployment']['perplexity'],
                'retained_gain': row['retained_gain'][split],
                'gain_over_best_hwa_only': row['vs_best_hwa_only'][split]['kl_reduction'],
                **write_statistics(row['cost']),
            })
    fields = ('deployment_kl', 'recovered_kl', 'kl_reduction',
              'deployment_ppl', 'recovered_ppl', 'ppl_reduction', 'ppl_reduction_percent', 'retained_gain',
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
    gain_views = (
        ('ppl-gain-versus-corruption', 'WikiText PPL gain from on-chip recovery',
         [('transfer_wikitext', 'ppl_reduction', 'PPL reduction (points)'),
          ('transfer_wikitext', 'ppl_reduction_percent', 'PPL reduction (%)')]),
        ('ppl-gain-versus-corruption-all-domains', 'Relative PPL gain from on-chip recovery',
         [(split, 'ppl_reduction_percent', 'PPL reduction (%)') for split in SPLITS]),
    )
    for filename, title, panels in gain_views:
        fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.4 * len(panels)),
                                 sharex=True, sharey='row', constrained_layout=True)
        for i, (split, field, ylabel) in enumerate(panels):
            for j, tag in enumerate(TAGS):
                ax = axes[i, j]
                for cap, marker, style, legend in zip(
                    CAPS, ('^', 'o', 's'), ('-', '-', '--'),
                    ('At most 2 attempts/device', 'At most 4 attempts/device', 'Uncapped'),
                ):
                    group = [g for g in aggregates if (g['source'], g['domain'], g['cap']) == (tag, split, cap)]
                    x = [g['rate_ppm']/10000 for g in group]
                    ax.plot(x, [g[field]['mean'] for g in group], marker=marker,
                            linestyle=style, color=colors[cap], label=legend, markersize=5)
                    ax.fill_between(x, [g[field]['min'] for g in group],
                                    [g[field]['max'] for g in group], color=colors[cap], alpha=.12)
                    for row in flat:
                        if (row['source'], row['domain'], row['cap']) == (tag, split, cap):
                            ax.scatter(row['rate_ppm']/10000, row[field], color=colors[cap], s=8, alpha=.5)
                ax.axhline(0, color='black', linewidth=.8)
                ax.set(title=f'{source_names[tag]}\n{split_names[split]}',
                       xlabel='Actual corruption (%)', ylabel=ylabel, xticks=range(6))
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title + '\nGain = HWA deployment PPL − recovered PPL; percent = 100 × gain / deployment PPL'
                     '\nPositive is better; means and ranges of three paired arrays')
        for extension in ('png', 'pdf'): fig.savefig(output/f'{filename}.{extension}', dpi=160)
        plt.close(fig)
    return {'endpoints': len(summary['rows']), 'domain_rows': len(flat), 'groups': len(aggregates)}
