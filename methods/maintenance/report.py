"""Pure analysis of custom-recovery trajectories; no model execution."""
import math


def flat_rows(rows, run_dir):
    result = []
    for row in rows:
        base = {k: v for k, v in row.items() if k not in ('metrics', 'baseline')}
        for split, metrics in row['metrics'].items():
            fresh = row['baseline'][split]
            result.append({**base, 'run_dir': str(run_dir), 'split': split, **metrics,
                'fresh_perplexity': fresh['perplexity'],
                'perplexity_ratio': metrics['perplexity']/fresh['perplexity']})
    return result


def durability(rows, threshold=1.10):
    """First observed crossing after each repair, with honest sampling brackets."""
    groups = {}
    for row in flat_rows(rows, ''):
        groups.setdefault((row['trajectory'], row['split']), []).append(row)
    result = []
    for (trajectory, split), samples in groups.items():
        samples.sort(key=lambda r: (r['seconds'], r['phase'] == 'after'))
        windows, window = [], []
        for sample in samples:
            if sample['phase'] == 'after' and window:
                windows.append(window)
                window = []
            window.append(sample)
        if window: windows.append(window)
        for window in windows:
            previous = None
            crossing = None
            for sample in window:
                if sample['perplexity_ratio'] > threshold:
                    crossing = {'lower_seconds': None if previous is None else previous['seconds'],
                                'upper_seconds': sample['seconds'],
                                'already_above_at_window_start': previous is None}
                    break
                previous = sample
            result.append({'trajectory': trajectory, 'split': split,
                'window_start_seconds': window[0]['seconds'], 'window_end_seconds': window[-1]['seconds'],
                'threshold_ratio': threshold, 'first_crossing': crossing,
                'right_censored': crossing is None})
    return result
