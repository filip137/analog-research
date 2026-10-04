"""Audit completed paired runs and generate standalone CIFAR-10 figures."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.artifacts import atomic_write_json, sha256_file
from campaigns.cifar10_crossbar import STUDY, TEACHER, TEACHER_SHA256, read

METHODS = ('direct', 'hwa', 'open_loop', 'closed_loop_pv')
LABELS = ('Direct', 'HWA only', 'HWA + open loop', 'HWA + closed loop')
COLORS = ('#999999', '#3977b9', '#c87930', '#27916a')


def records(run, stage):
    return [r for line in (run / 'metrics.jsonl').read_text().splitlines()
            if (r := json.loads(line)).get('stage') == stage]


def rank(metric, epoch):
    return metric['kl_teacher_student'], -metric['accuracy'], epoch


def collect():
    assert sha256_file(TEACHER) == TEACHER_SHA256
    plan = read(STUDY / 'study.json')
    runs, sources, metrics, histories = {}, {}, {}, {}
    for arm in plan['arms']:
        name = arm['arm_id']
        pointer = read(STUDY / 'runs' / name / 'completed_run.json')
        # The immutable pointer records its original host. Preserve its exact
        # run ID while reading the corresponding bundle in this worktree.
        run = STUDY / 'runs' / name / Path(pointer['run_dir']).name
        result = read(run / 'result.json')
        assert result['status'] == 'complete', name
        config = read(run / 'config.resolved.json')
        assert config['maximum_batches'] is None and config['evaluation_limit'] is None
        assert result['metrics']['teacher_sha256'] == TEACHER_SHA256
        runs[name], metrics[name] = run, result['metrics']
        histories[name] = records(run, config['stage'])
        sources[name] = dict(run=str(run), result_sha256=sha256_file(run / 'result.json'),
                             config_sha256=sha256_file(run / 'config.resolved.json'))
    assert len(runs) == 22
    teacher_metrics = read(TEACHER.parent.parent / 'result.json')['metrics']
    assert teacher_metrics['acceptance_passed']
    for name, run in runs.items():
        assert read(run / 'artifacts/data_receipt.json') == teacher_metrics['data_receipt'], name
    training_populations, development_populations = set(), set()
    for name in ('hwa_n0', 'hwa_n025', 'hwa_n05', 'hwa_n1'):
        for role, target in (('training', training_populations), ('development', development_populations)):
            receipt = read(runs[name] / f'artifacts/{role}_sampled_population.receipt.json')
            assert receipt['aihwkit_version'] == '1.1.0' and receipt['num_cells'] == 788992
            target.add(receipt['population_fingerprint'])
    assert len(training_populations) == len(development_populations) == 1
    assert training_populations.isdisjoint(development_populations)
    deployment_populations = set()
    for array in ('dev', 'a1', 'a2'):
        for role in ('healthy', 'published'):
            direct = read(runs[f'direct_{array}'] / f'artifacts/{role}_sampled_population.receipt.json')
            hwa = read(runs[f'hwa_{array}'] / f'artifacts/{role}_sampled_population.receipt.json')
            assert direct['population_fingerprint'] == hwa['population_fingerprint']
            if role == 'healthy':
                deployment_populations.add(hwa['population_fingerprint'])
    assert len(deployment_populations) == 3
    assert training_populations.isdisjoint(deployment_populations)
    selection = read(STUDY / 'hwa_selection.json')
    assert not selection['test_used']
    expected = min(selection['candidates'], key=lambda c: (*rank(c['validation'], c['epoch']), c['noise_multiplier']))
    assert expected == selection['selected']
    orders = {}
    for name, history in histories.items():
        if not history:
            continue
        stage = history[0]['stage']
        assert len(history) == (10 if stage == 'hwa' else 30), name
        assert all(h['train_examples'] == 45000 for h in history), name
        permutations = [h['order_sha256'] for h in history]
        if stage in orders:
            assert orders[stage] == permutations, name
        orders[stage] = permutations
        candidates = [(rank(h['validation'], h['epoch']), h['epoch']) for h in history]
        if stage == 'recover':
            candidates.append((rank(metrics[name]['initial_validation'], 0), 0))
        assert min(candidates)[1] == metrics[name]['selected']['epoch'], name
    assert orders['hwa'] == orders['recover'][:10]
    assert metrics['hwa_dev']['programming']['apparent_sha256'] == metrics[expected['arm']]['selected']['programming']['apparent_sha256']
    assert metrics['hwa_dev']['healthy']['apparent']['validation'] == expected['validation']
    rows, effects = [], []
    for array in ('dev', 'a1', 'a2'):
        for condition in ('healthy', 'faulted'):
            base = metrics[f'hwa_{array}'][condition]['apparent']['test']
            initialization_hashes = []
            for method in METHODS:
                if method in ('direct', 'hwa'):
                    m = metrics[f'{method}_{array}']
                    test = m[condition]['apparent']['test']
                    persistent = m[condition]['persistent']['test']
                    selected_epoch, pulses, verify, final_pulses, final_accuracy = 0, 0, 0, 0, test['accuracy']
                    final_kl = test['kl_teacher_student']
                else:
                    name = f'recover_{array}_{condition}_{method}'
                    m = metrics[name]
                    assert m['initial_test'] == base, name
                    assert m['initial_validation'] == metrics[f'hwa_{array}'][condition]['apparent']['validation'], name
                    initialization_hashes.append((m['initial_apparent_sha256'], m['initial_persistent_sha256'], m['deployment_sha256']))
                    test, persistent = m['selected_test'], m['selected_persistent_test']
                    selected_epoch = m['selected']['epoch']
                    pulses, verify = m['selected_cost']['pulses'], m['selected_cost']['verify_reads']
                    final_pulses, final_accuracy = m['final_cost']['pulses'], m['final_test']['accuracy']
                    final_kl = m['final_test']['kl_teacher_student']
                    changes = m['prediction_changes']
                    assert changes['incorrect_to_correct'] - changes['correct_to_incorrect'] == test['correct'] - base['correct']
                    effects.append(dict(array=array, role='development' if array == 'dev' else 'held_out',
                        condition=condition, writer=method, accuracy_delta_pp=100 * (test['accuracy'] - base['accuracy']),
                        kl_reduction_percent=100 * (1 - test['kl_teacher_student'] / base['kl_teacher_student']),
                        selected_epoch=selected_epoch, selected_pulses_per_cell=pulses / 788992,
                        final_pulses_per_cell=final_pulses / 788992, verify_reads=verify))
                assert test['examples'] == 10000
                rows.append(dict(array=array, role='development' if array == 'dev' else 'held_out', condition=condition,
                    method=method, accuracy_percent=100 * test['accuracy'], kl=test['kl_teacher_student'],
                    teacher_agreement_percent=100 * test['teacher_agreement'], selected_epoch=selected_epoch,
                    selected_pulses_per_cell=pulses / 788992, selected_verify_reads=verify,
                    final_pulses_per_cell=final_pulses / 788992, final_accuracy_percent=100 * final_accuracy,
                    final_kl=final_kl,
                    persistent_accuracy_percent=100 * persistent['accuracy'], persistent_kl=persistent['kl_teacher_student']))
            assert len(set(initialization_hashes)) == 1, (array, condition)
        assert metrics[f'direct_{array}']['controls']['ideal_requested']['test']['predictions_sha256'] == teacher_metrics['test']['predictions_sha256']
    return rows, effects, histories, metrics, sources, selection, teacher_metrics


def figures(output, rows, histories, metrics, teacher_accuracy):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, condition in enumerate(('healthy', 'faulted')):
        for col, (field, label) in enumerate((('accuracy_percent', 'Test accuracy (%)'), ('kl', 'KL(teacher || student)'))):
            ax = axes[row, col]
            for ai, array in enumerate(('a1', 'a2')):
                values = [next(r[field] for r in rows if r['array'] == array and r['condition'] == condition and r['method'] == m) for m in METHODS]
                for i, value in enumerate(values):
                    ax.scatter(i + (-.055, .055)[ai], value, color=COLORS[i], marker=('o', 's')[ai], s=42, zorder=3,
                               label=f'Held-out array {ai + 1}' if i == 0 else None)
            means = [mean(r[field] for r in rows if r['role'] == 'held_out' and r['condition'] == condition and r['method'] == m) for m in METHODS]
            for i, value in enumerate(means):
                ax.hlines(value, i - .14, i + .14, color=COLORS[i], linewidth=1.5)
                ax.annotate(f'{value:.2f}' if col == 0 else f'{value:.4f}', (i, value), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
            if col == 0:
                ax.axhline(teacher_accuracy, color='#333333', linestyle=':', linewidth=1, label='Digital source')
            ax.set_title(f'{condition.capitalize()} arrays')
            ax.set_ylabel(label)
            ax.set_xticks(range(4), ('Direct', 'HWA', '+ open\nloop', '+ closed\nloop'))
            ax.grid(axis='y', alpha=.2)
            ax.margins(y=.18)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='best')
    fig.suptitle('CIFAR-10: recovery beyond HWA on two held-out IBM OM arrays\n3072–256–10 ReLU • held apparent state • recovery arms fork from identical HWA deployments', fontsize=12)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'comparison.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, constrained_layout=True)
    for row, condition in enumerate(('healthy', 'faulted')):
        for col, array in enumerate(('dev', 'a1', 'a2')):
            ax = axes[row, col]
            for method, color in zip(METHODS[2:], COLORS[2:]):
                name = f'recover_{array}_{condition}_{method}'
                history, m = histories[name], metrics[name]
                ys = [m['initial_validation']['kl_teacher_student']] + [h['validation']['kl_teacher_student'] for h in history]
                ax.plot(range(31), ys, color=color, label='Open loop' if method == 'open_loop' else 'Closed loop')
                e = m['selected']['epoch']
                ax.scatter(e, ys[e], color=color, marker='*', s=75, zorder=3)
            ax.set_title(f'{array}: {condition}')
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel('Validation teacher KL')
            if row == 1:
                ax.set_xlabel('Recovery epoch')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Full recovery trajectories; stars mark selected epochs', fontsize=12)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'recovery_curves.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY / 'analysis/cifar10')
    args = parser.parse_args()
    rows, effects, histories, metrics, sources, selection, teacher = collect()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('results', rows), ('paired_effects', effects)]:
        atomic_write_json(output / f'{name}.json', values)
        with (output / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    atomic_write_json(output / 'verification.json', dict(status='passed', native_runs=22, matched_recovery_pairs=6,
                      full_training_examples_per_epoch=45000, full_test_examples=10000,
                      common_epoch_permutations=True, selected_hwa_development_replay_exact=True,
                      distinct_deployment_assignments=3, independent_hwa_training_population=True,
                      matched_direct_hwa_populations=True, identical_data_receipts=True,
                      recovery_p0_hashes_match=True, selection_uses_validation_only=True,
                      teacher_sha256=TEACHER_SHA256, sources=sources))
    figures(output, rows, histories, metrics, teacher['test']['accuracy'] * 100)
    held = [r for r in rows if r['role'] == 'held_out']
    text = ['# CIFAR-10: HWA versus physical recovery', '',
        f"Digital source: **{teacher['test']['accuracy'] * 100:.2f}%** test accuracy, bias-free 3072–256–10 ReLU, trained locally. Validation CE selected epoch {teacher['selected']['epoch']} from 100 epochs.", '',
        'All 22 declared hardware runs completed. Results below average two held-out physical array assignments; the development array is reported separately. This is exploratory evidence from the AIHWKit 1.1.0 IBM OM model.', '',
        '| Condition | Direct deployment | HWA only | HWA + open loop | HWA + closed loop |',
        '|---|---:|---:|---:|---:|']
    for condition in ('healthy', 'faulted'):
        values = [mean(r['accuracy_percent'] for r in held if r['condition'] == condition and r['method'] == m) for m in METHODS]
        text.append(f"| {condition.capitalize()} | " + ' | '.join(f'{v:.2f}%' for v in values) + ' |')
    text.append('')
    for condition in ('healthy', 'faulted'):
        group = [e for e in effects if e['role'] == 'held_out' and e['condition'] == condition and e['writer'] == 'closed_loop_pv']
        wins = sum(e['accuracy_delta_pp'] > 0 for e in group)
        text.append(f"Closed-loop recovery improves test accuracy on **{wins}/2 {condition} arrays**, with a mean change of **{mean(e['accuracy_delta_pp'] for e in group):+.3f} percentage points** beyond HWA.")
        text.append('')
    text += ['', 'Accuracy effects and KL reductions are paired against each array’s exact HWA-only starting state:', '',
             '| Condition | Writer | Accuracy change (pp), mean [range] | Teacher-KL reduction, mean | Selected pulses/cell | Full-budget pulses/cell | Selected verify reads (billions) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for condition in ('healthy', 'faulted'):
        for method in METHODS[2:]:
            group = [e for e in effects if e['role'] == 'held_out' and e['condition'] == condition and e['writer'] == method]
            delta = [e['accuracy_delta_pp'] for e in group]
            text.append(f"| {condition} | {method} | {mean(delta):+.3f} [{min(delta):+.2f}, {max(delta):+.2f}] | {mean(e['kl_reduction_percent'] for e in group):.1f}% | {mean(e['selected_pulses_per_cell'] for e in group):.3f} | {mean(e['final_pulses_per_cell'] for e in group):.3f} | {mean(e['verify_reads'] for e in group) / 1e9:.3f} |")
    programming = [metrics[f'hwa_{a}']['programming'] for a in ('a1', 'a2')]
    text += ['', f"For context, initial HWA programming averages {mean(p['mean_programming_pulses_per_cell'] for p in programming):.2f} pulses/cell and {mean(p['verify_reads'] for p in programming) / 1e6:.2f} million verify observations per array, starting from the fully RESET state. Recovery costs in the table are additional; conditioning, digital computation and physical energy are not included."]
    text += ['', '![Held-out comparison](comparison.png)', '', '## Selection and interpretation', '',
        f"HWA selected noise multiplier **{selection['selected']['noise_multiplier']}**, epoch **{selection['selected']['epoch']}**, using healthy development validation KL. Tested multipliers were 0, 0.25, 0.5 and 1; each had ten epochs. No test result entered selection.", '',
        'The no-noise HWA arm still adapts to one training population’s support. Direct deployment remains a separate control. Recovery measures the incremental effect after selected HWA; it does not compare all possible HWA optimizers or budgets.', '',
        'The faulted scenario injects the matched published-companion stuck pattern after healthy programming (configured probability 13.48%). This tests adaptation to deployment-specific faults unknown during HWA. It does not compare against off-chip retraining with knowledge of each target array’s faults.', '',
        '## Individual arrays', '',
        '| Array | Condition | Method | Test accuracy | Teacher KL | Agreement | Selected epoch | Fixed-final accuracy | Persistent accuracy |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['array']} | {r['condition']} | {r['method']} | {r['accuracy_percent']:.2f}% | {r['kl']:.6f} | {r['teacher_agreement_percent']:.2f}% | {r['selected_epoch']} | {r['final_accuracy_percent']:.2f}% | {r['persistent_accuracy_percent']:.2f}% |")
    text += ['', 'For direct/HWA rows, epoch 0 denotes the deployed state before recovery. The HWA source epoch is specified above. Persistent accuracy is a diagnostic, not the primary forward.', '',
        '![Recovery curves](recovery_curves.png)', '', '## Costs and limits', '',
        '- Six 512×256 input tiles plus one 256×10 output tile: 788,992 active programmable states. Layer scales are frozen from the digital teacher.',
        '- Both recovery writers use LR 1e-5 in q, batch 64 and 30 epochs. These are exploratory transferred settings, not a CIFAR-10 optimum. Both use digital gradients and Adam memories.',
        '- HWA uses support clipping plus additive apparent noise. It does not train with the complete P&V-conditioned endpoint distribution or unseen target fault maps. Its ten-epoch budget also differs from recovery’s thirty epochs.',
        '- Recovery selects by validation teacher KL, including epoch zero. Accuracy can therefore fall despite improved teacher KL. Fixed-final accuracy is reported separately.',
        '- Selected endpoints are retrospective simulation checkpoints. Physically returning to an earlier state would require additional programming, excluded from the selected cost. Full-budget pulse costs include later discarded updates.',
        '- Verify reads count cell observations, not hardware time or energy. ADC/DAC quantization, wire resistance and physical conductance calibration are absent.',
        '- Two held-out arrays, one digital training seed and one programming realization per array establish a directional result, not statistical significance or performance on convolutional CIFAR-10 networks.',
        '', '## Reproducibility', '',
        'The protocol is [cifar10_crossbar_protocol.md](../../../../docs/cifar10_crossbar_protocol.md). '
        'Full tables: [results.csv](results.csv), [paired_effects.csv](paired_effects.csv). '
        'Verification: [verification.json](verification.json). Native study coverage is in [summary.json](../summary.json).', '',
        'This extends the matched-fork comparison proposed in the [earlier MNIST evaluation](/home/filip/server_code/.codex/worktrees/artifacts/onchip_vs_hwa_20260917/report.md). Differences in architecture, task difficulty and update exposure preclude interpreting the change in effect size as a controlled dataset-only effect.', '',
        'Scientific interpretation and formal manifest finalization remain pending human review under the repository study workflow.', '']
    hwa_diagnostic = []
    for condition in ('healthy', 'faulted'):
        direct = [r for r in held if r['condition'] == condition and r['method'] == 'direct']
        hwa = [r for r in held if r['condition'] == condition and r['method'] == 'hwa']
        hwa_diagnostic.append(f"On {condition} held-out arrays, HWA changes accuracy by {mean(r['accuracy_percent'] for r in hwa) - mean(r['accuracy_percent'] for r in direct):+.3f} pp versus direct deployment, while mean teacher KL changes from {mean(r['kl'] for r in direct):.6f} to {mean(r['kl'] for r in hwa):.6f}.")
    persistent = [r for r in held if r['method'] == 'closed_loop_pv']
    hwa_diagnostic += ['', f"The selected closed-loop endpoints have mean apparent accuracy {mean(r['accuracy_percent'] for r in persistent):.2f}%, but mean hidden-persistent accuracy {mean(r['persistent_accuracy_percent'] for r in persistent):.2f}% across the two conditions and held-out arrays. This observed gap matters: the primary result relies on the model’s held post-write apparent state. It is not evidence of equally accurate hidden persistent weights, retention robustness, or robustness after an independent noise refresh."]
    insertion = text.index('## Individual arrays')
    text[insertion:insertion] = hwa_diagnostic + ['']
    (output / 'report.md').write_text('\n'.join(text))
    print(output / 'report.md')


if __name__ == '__main__':
    main()
