"""Apply Experiment 1c's frozen decision rules (exp-003) to whatever parts have been collected.

usage: python exp1c_decision.py [results root, default results/lifecycles]

Stdlib only. Reads <root>/<lifecycle>/collected/deployments.json for Experiment 1b
(opt125m-cmo-l6-exp1b), parts A (opt125m-cmo-l6-exp1c), B (-decay), C (opt125m-cmo-l3-exp1b,
paired with Experiment 1 L3) and D (-seed2); a missing part is reported as not collected.
"A beats B at p": lower KL in >= 4 of 5 assignments and a mean relative KL reduction >= 10 %.
"""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

RATES = (0, 10000, 20000, 30000, 40000, 50000)
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/lifecycles")


def load(lifecycle, prefix=""):
    path = ROOT / lifecycle / "collected/deployments.json"
    if not path.exists():
        return None
    by = defaultdict(dict)
    for row in json.load(open(path)):
        if row["duplicate_coverage"]:
            print(f"WARNING: {lifecycle}: duplicated coverage in {row['hwa_arm']} {row['case']}")
        by[(prefix + row["hwa_arm"], row["defect_rate_ppm"])][row["array_seed"]] = row["deployed_teacher_kl"]
    return by


def compare(a, b):
    seeds = sorted(set(a) & set(b))
    wins = sum(a[s] < b[s] for s in seeds)
    relative = statistics.mean((b[s] - a[s]) / b[s] for s in seeds)
    return wins, len(seeds), relative, len(seeds) == 5 and wins >= 4 and relative >= 0.10


def verdict(by, a, b, rate):
    x, y = by[(a, rate)], by[(b, rate)]
    forward, backward = compare(x, y), compare(y, x)
    label = f"{a} BEATS {b}" if forward[3] else f"{b} BEATS {a}" if backward[3] else "no material difference"
    return f"  {rate / 1e4:.0f} %: {a} vs {b}: {forward[0]}/{forward[1]}, {forward[2]:+.1%} -> {label}"


def means(by, source):
    return " ".join(f"{statistics.mean(by[(source, r)].values()):.4f}" for r in RATES)


def best_sets(by, sources):
    for rate in RATES:
        avg = {s: statistics.mean(by[(s, rate)].values()) for s in sources}
        best = min(avg, key=avg.get)
        tied = [s for s in sources if s != best and not compare(by[(best, rate)], by[(s, rate)])[3]]
        print(f"  {rate / 1e4:.0f} %: best {best} ({avg[best]:.4f}); not beaten by it: {tied or 'none'}")


def merged(*tables):
    out = defaultdict(dict)
    for table in tables:
        for key, value in table.items():
            out[key] = value
    return out


def main() -> int:
    b1 = load("opt125m-cmo-l6-exp1b")
    if b1 is None:
        print("Experiment 1b not collected; nothing to compare against")
        return 1
    print("KL means per rate (0..5 %)")
    parts = {
        "A": load("opt125m-cmo-l6-exp1c"),
        "B": load("opt125m-cmo-l6-exp1c-decay", "decay_"),
        "C": load("opt125m-cmo-l3-exp1b", "l3b_"),
        "D": load("opt125m-cmo-l6-exp1c-seed2", "seed2_"),
    }
    exp1_l3 = load("opt125m-cmo-l3-exp1", "l3exp1_")
    for name, table in parts.items():
        print(f"\n=== part {name}: {'collected' if table else 'NOT COLLECTED'}")
        if not table:
            continue
        for source in sorted({k[0] for k in table}):
            print(f"  {source:28s} {means(table, source)}")
        if name == "A":
            by = merged(b1, table)
            for a, b in (
                ("standard_hwa_9pct", "standard_hwa_6pct"),
                ("standard_hwa_9pct", "standard_hwa"),
                ("cdt_random_mixed", "cdt_random_3pct"),
                ("cdt_random_mixed", "cdt_random_5pct"),
                ("cdt_random_mixed", "standard_hwa"),
            ):
                print()
                for rate in RATES:
                    print(verdict(by, a, b, rate))
            print("\n  per-rate best set with A added:")
            sources = sorted({k[0] for k in by})
            best_sets(by, sources)
            for source in ("standard_hwa_9pct", "cdt_random_mixed"):
                covers = all(
                    source == min(sources, key=lambda s: statistics.mean(by[(s, r)].values()))
                    or not compare(
                        by[(min(sources, key=lambda s: statistics.mean(by[(s, r)].values())), r)], by[(source, r)]
                    )[3]
                    for r in RATES[1:]
                ) and not compare(by[("standard_hwa", 0)], by[(source, 0)])[3]
                print(f"  {source} covers 0-5 %: {covers}")
        if name == "B":
            by = merged(b1, table)
            for arm in ("standard_hwa", "cdt_random_3pct"):
                print()
                for rate in RATES:
                    print(verdict(by, f"decay_{arm}", arm, rate))
        if name == "C":
            by = merged(table, exp1_l3 or {})
            pairs = [
                ("l3b_standard_hwa_6pct", "l3b_standard_hwa"),
                ("l3b_cdt_random_5pct", "l3b_cdt_random_3pct"),
                ("l3b_cdt_random_3pct", "l3b_standard_hwa"),
            ]
            if exp1_l3:
                pairs += [("l3b_standard_hwa", "l3exp1_standard_hwa"), ("l3b_cdt_random_3pct", "l3exp1_cdt_random_3pct")]
            for a, b in pairs:
                print()
                for rate in RATES:
                    print(verdict(by, a, b, rate))
            print("\n  per-rate best set (L3, Experiment 1b arms):")
            best_sets(table, sorted({k[0] for k in table}))
        if name == "D":
            by = merged(b1, table)
            for arm in ("standard_hwa", "cdt_random_3pct"):
                print(f"\n  {arm}: seed-2 vs Experiment 1b, relative difference of means | array spread (max/min - 1) of each")
                for rate in RATES:
                    first, second = by[(arm, rate)], by[(f"seed2_{arm}", rate)]
                    m1, m2 = statistics.mean(first.values()), statistics.mean(second.values())
                    spread = [max(v.values()) / min(v.values()) - 1 for v in (first, second)]
                    print(f"    {rate / 1e4:.0f} %: {m2 / m1 - 1:+.2%} | {spread[0]:.2%}, {spread[1]:.2%} | {verdict(by, f'seed2_{arm}', arm, rate).split('-> ')[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
