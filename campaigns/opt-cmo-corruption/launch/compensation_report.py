"""Compare all predeclared compensators using collected held-out metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

SOURCES = {"standard_hwa_model": "std3", "standard_hwa_9pct_model": "std9", "cdt_random_3pct_model": "cdt3"}
METHODS = {"global": "Global gain", "tile_global": "Tile gains", "row": "Input-row gains",
           "column": "Output-column gains", "row_column": "Row + column LS",
           "global_shift": "Global + mean shift", "global_moments": "Global + mean/variance"}
MODELS = {"reram_cmo": "CMO ReRAM", "pcm": "PCM", "reram_wan2022": "Wan ReRAM"}
CASES = {"nominal": "Healthy", "random_30000ppm": "3% faults"}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.collected / "compensation.json"
    rows = json.loads(source.read_text())
    assert len(rows) == 378
    index = {(r['hwa_arm'], r['case'], r['model'], r['seconds'], r['method']): r for r in rows}
    expected = {(s, c, m, t, a) for s in SOURCES for c in CASES for m in MODELS for t in (1, 86400, 172800) for a in METHODS}
    assert set(index) == expected and len(index) == len(rows)
    assert all(not r['duplicate_coverage'] and r['replay_exact'] and r['state_unchanged'] for r in rows)
    assert all(r['array_seed'] == 271002 and r['cost_device_writes'] == 0 for r in rows)
    final = [dict(source=SOURCES[r['hwa_arm']], **r) for r in rows if r['seconds'] == 172800]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "two_day_recovery.csv", final)
    (args.output / "two_day_recovery.json").write_text(json.dumps(final, indent=2) + '\n')
    cost_rows = []
    for method in METHODS:
        row = next(r for r in final if r['method'] == method)
        costs = {k.removeprefix('cost_'): v for k,v in row.items() if k.startswith('cost_')}
        assert all({k.removeprefix('cost_'):v for k,v in r.items() if k.startswith('cost_')} == costs
                   for r in final if r['method'] == method)
        cost_rows.append({'method':method, **costs})
    write_csv(args.output / 'calibration_cost.csv', cost_rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'savefig.dpi': 180})
    conditions = [(s, c) for c in CASES for s in SOURCES]
    figures = []
    for split, domain in [('transfer_wikitext', 'WikiText'), ('transfer_generated', 'Generated text')]:
        for metric in ['aging_kl_regained_percent', 'teacher_kl']:
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.9), layout='constrained')
            row_labels = list(METHODS.values()) if metric != 'teacher_kl' else ['Programmed baseline', 'Uncompensated'] + list(METHODS.values())
            all_values = []
            for model in MODELS:
                values = []
                if metric == 'teacher_kl':
                    for state in ['baseline', 'uncompensated']:
                        values.append([index[s,c,model,172800,'global'][f'{state}_{split}_teacher_kl'] for s,c in conditions])
                for method in METHODS:
                    key = f'{split}_{metric}' if metric != 'teacher_kl' else f'compensated_{split}_teacher_kl'
                    values.append([index[s,c,model,172800,method][key] for s,c in conditions])
                all_values.append(np.array(values, dtype=float))
            norm = LogNorm(vmin=min(v.min() for v in all_values), vmax=max(v.max() for v in all_values)) if metric == 'teacher_kl' else None
            for i, ((model, title), values) in enumerate(zip(MODELS.items(), all_values)):
                ax = axes[i]
                if metric == 'teacher_kl':
                    im = ax.imshow(values, cmap='viridis_r', norm=norm, aspect='auto')
                else:
                    im = ax.imshow(values, cmap='RdYlGn', vmin=-100, vmax=100, aspect='auto')
                ax.set_title(title, fontsize=12)
                ax.set_xticks(range(6), [f'{SOURCES[s]}\n{CASES[c]}' for s,c in conditions], fontsize=8)
                ax.set_yticks(range(len(row_labels)), row_labels if i == 0 else [''] * len(row_labels))
                ax.axvline(2.5, color='white', linewidth=2)
                for y in range(values.shape[0]):
                    for x in range(values.shape[1]):
                        v = values[y,x]
                        text = f'{v:.3f}' if metric == 'teacher_kl' else (f'{v:.1f}%' if abs(v) < 10000 else f'{v:.1e}%')
                        rgba = im.cmap(im.norm(v))
                        color = 'white' if sum(a*b for a,b in zip(rgba[:3], [.2126,.7152,.0722])) < .5 else 'black'
                        ax.text(x, y, text, ha='center', va='center', color=color, fontsize=8)
            fig.colorbar(im, ax=list(axes), shrink=.8, label='Teacher KL · nats/token · log colors' if metric == 'teacher_kl' else 'Aging KL regained (%) · colors saturate at ±100%')
            fig.suptitle(f'{domain} · two-day retention · all predeclared digital compensators', fontsize=14)
            fig.supxlabel('Frozen OPT-125M, six analog MLPs; one matched assignment; zero device writes.\n'
                          + ('Small CMO aging gaps make percentage recovery sensitive; inspect absolute KL alongside it.' if metric != 'teacher_kl' else 'Calibration uses training tokens or one-hot probes; evaluation cohorts remain held out.'), fontsize=9)
            stem=f'{split}_{metric}'
            for ext in ('png','pdf'): fig.savefig(args.output / f'{stem}.{ext}')
            plt.close(fig); figures.append(stem)
    # The PCM age plot shows the dominant loss directly, including the baseline.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for iy, (case, case_label) in enumerate(CASES.items()):
        for ix, (src, src_label) in enumerate(SOURCES.items()):
            ax = axes[iy, ix]
            ages = (1, 86400, 172800)
            ax.axhline(index[src,case,'pcm',1,'global']['baseline_transfer_wikitext_perplexity'], color='gray', linestyle=':', label='Programmed baseline')
            ax.plot(range(3), [index[src,case,'pcm',t,'global']['uncompensated_transfer_wikitext_perplexity'] for t in ages], color='black', marker='o', label='Uncompensated')
            for k,(method,label) in enumerate(METHODS.items()):
                ax.plot(range(3), [index[src,case,'pcm',t,method]['compensated_transfer_wikitext_perplexity'] for t in ages], color=colors[k], marker='.', label=label, linewidth=1.4)
            ax.set_xticks(range(3), ['1 s','1 day','2 days']); ax.set_title(f'{src_label} · {case_label}')
            ax.grid(alpha=.2)
            if ix == 0: ax.set_ylabel('WikiText perplexity')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('PCM drift compensation · identical array snapshots across methods', fontsize=14)
    for ext in ('png','pdf'): fig.savefig(args.output / f'pcm_perplexity_over_time.{ext}')
    plt.close(fig); figures.append('pcm_perplexity_over_time')
    receipt={'rows':len(rows), 'two_day_rows':len(final), 'figures':figures,
             'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'report_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'selection':'all predeclared methods; no held-out selection', 'independent_assignments':1}
    (args.output/'report_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
