"""Apply Experiment 1b's frozen decision rule (exp-002) to collected deployment rows.

usage: python exp1b_decision.py <exp1b deployments.json> <exp1 L6 deployments.json> [<hwa-metrics dir>]

Stdlib only. Rule, as frozen in exp-002:

- Unit: one assignment x one stuck-device rate; never pooled across rates.
- A beats B at rate p: KL_A < KL_B in >= 4 of 5 assignments AND the mean over
  assignments of (KL_B - KL_A) / KL_B is >= 10 %.
- Comparisons: data effect (1b vs Experiment 1 for standard_hwa and
  cdt_random_3pct, paired per array), stronger noise (standard_hwa_6pct vs
  standard_hwa), overestimation (cdt_random_5pct vs cdt_random_3pct), and
  every trained source vs pretrained.
- Per-rate best set: the lowest-mean-KL source plus every source it does not beat.
- Reproduction guard: 1b pretrained KL equals Experiment 1 L6 pretrained KL to 1e-6 relative.
- Convergence (from the training metrics, if given): selected epoch, 40->50 change of the
  healthy and 3 % development components, development/training ratio at epochs 10 and 50.
"""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

SOURCES = ("pretrained", "standard_hwa", "standard_hwa_6pct", "cdt_random_3pct", "cdt_random_5pct")
TRAINED = SOURCES[1:]
RATES = (0, 10000, 20000, 30000, 40000, 50000)


def compare(a, b):
    """a, b: {seed: KL}. Returns (wins, n, mean relative reduction, beats)."""

    seeds = sorted(set(a) & set(b))
    wins = sum(a[s] < b[s] for s in seeds)
    relative = statistics.mean((b[s] - a[s]) / b[s] for s in seeds)
    return wins, len(seeds), relative, len(seeds) == 5 and wins >= 4 and relative >= 0.10


def table(path):
    rows = json.load(open(path))
    duplicated = [row for row in rows if row["duplicate_coverage"]]
    if duplicated:
        print(f"WARNING: {path}: {len(duplicated)} rows have duplicated coverage (retries); inspect before reading on")
    by = defaultdict(dict)  # (source, rate) -> {seed: row}
    for row in rows:
        by[(row["hwa_arm"], row["defect_rate_ppm"])][row["array_seed"]] = row
    return by


def kl(by, source, rate):
    return {seed: row["deployed_teacher_kl"] for seed, row in by[(source, rate)].items()}


def verdict(a_name, a, b_name, b, rate):
    forward, backward = compare(a, b), compare(b, a)
    if forward[3]:
        label = f"{a_name} BEATS {b_name}"
    elif backward[3]:
        label = f"{b_name} BEATS {a_name}"
    else:
        label = "no material difference"
    return f"  {rate / 1e4:.0f} %: {a_name} vs {b_name}: {forward[0]}/{forward[1]}, {forward[2]:+.1%} -> {label}"


def convergence(directory):
    print("\nconvergence (development teacher KL; training loss = mean of batches 0/100/200)")
    for source in TRAINED:
        path = Path(directory) / f"{source}.jsonl"
        if not path.exists():
            print(f"  {source}: no metrics")
            continue
        records = [json.loads(line) for line in path.open()]
        selection = {r["epoch"]: r for r in records if r["stage"] == "hwa_selection"}
        loss = defaultdict(list)
        for r in records:
            if r["stage"] == "hwa" and "loss" in r:
                loss[r["epoch"]].append(r["loss"])
        best = min(selection.values(), key=lambda r: r["teacher_kl"])

        def component(epoch, case):
            return {c["case"]: c["teacher_kl"] for c in selection[epoch]["cases"]}[case]

        def window(epochs, case):
            values = [component(e, case) for e in epochs if e in selection]
            return statistics.mean(values)

        changes = {
            case: window(range(46, 51), case) / window(range(36, 41), case) - 1
            for case in ("nominal", "random_30000ppm")
        }
        # Development component matching the arm's training condition (healthy for standard, 3 % for CDT).
        case = "nominal" if source.startswith("standard") else "random_30000ppm"
        ratios = {e: component(e, case) / statistics.mean(loss[e]) for e in (10, 50) if e in selection and loss[e]}
        print(
            f"  {source}: selected epoch {best['epoch']} (score {best['teacher_kl']:.4f}); "
            f"healthy {component(50, 'nominal'):.4f}, 3 % {component(50, 'random_30000ppm'):.4f} at epoch 50; "
            f"40->50 healthy {changes['nominal']:+.1%}, 3 % {changes['random_30000ppm']:+.1%}; "
            f"dev({case})/train ratio " + ", ".join(f"e{e} {v:.2f}" for e, v in ratios.items())
        )


def main() -> int:
    new, old = table(sys.argv[1]), table(sys.argv[2])
    teacher = statistics.mean(row["clean_perplexity"] for row in new[("pretrained", 0)].values())
    print(f"digital teacher perplexity (pretrained source, clean, test): {teacher:.3f}")

    worst = max(
        abs(new[("pretrained", r)][s]["deployed_teacher_kl"] / old[("pretrained", r)][s]["deployed_teacher_kl"] - 1)
        for r in RATES
        for s in new[("pretrained", r)]
    )
    print(f"reproduction: max relative |KL_1b / KL_exp1 - 1| over pretrained rows = {worst:.2e} "
          f"-> {'PASS' if worst <= 1e-6 else 'FAIL: investigate before interpreting'}")

    print("\ndeployed teacher KL, mean [min, max] over assignments | mean perplexity | n")
    for rate in RATES:
        print(f"  {rate / 1e4:.0f} %:")
        for source in SOURCES:
            group = new[(source, rate)].values()
            values = [r["deployed_teacher_kl"] for r in group]
            ppl = [r["deployed_perplexity"] for r in group]
            print(f"    {source:18s} {statistics.mean(values):.4f} [{min(values):.4f}, {max(values):.4f}] | "
                  f"{statistics.mean(ppl):8.2f} | {len(values)}")
        for source in ("standard_hwa", "cdt_random_3pct"):
            values = list(kl(old, source, rate).values())
            print(f"    exp1 {source:13s} {statistics.mean(values):.4f} [{min(values):.4f}, {max(values):.4f}]")

    for title, pairs in (
        ("data effect (1b vs Experiment 1, paired per array)", [("standard_hwa", "standard_hwa"), ("cdt_random_3pct", "cdt_random_3pct")]),
    ):
        print(f"\n{title}")
        for rate in RATES:
            for a, b in pairs:
                print(verdict(f"1b {a}", kl(new, a, rate), f"exp1 {b}", kl(old, b, rate), rate))
    for title, (a, b) in (
        ("stronger noise", ("standard_hwa_6pct", "standard_hwa")),
        ("overestimated training fault rate", ("cdt_random_5pct", "cdt_random_3pct")),
        ("corruption-aware vs standard (3 % noise)", ("cdt_random_3pct", "standard_hwa")),
        ("5 % corruption-aware vs 6 % noise", ("cdt_random_5pct", "standard_hwa_6pct")),
    ):
        print(f"\n{title}")
        for rate in RATES:
            print(verdict(a, kl(new, a, rate), b, kl(new, b, rate), rate))
    print("\ntrained sources vs pretrained")
    for rate in RATES:
        for source in TRAINED:
            print(verdict(source, kl(new, source, rate), "pretrained", kl(new, "pretrained", rate), rate))

    print("\nper-rate best set (lowest mean KL plus every source it does not beat)")
    for rate in RATES:
        means = {s: statistics.mean(kl(new, s, rate).values()) for s in SOURCES}
        best = min(means, key=means.get)
        tied = [s for s in SOURCES if s != best and not compare(kl(new, best, rate), kl(new, s, rate))[3]]
        print(f"  {rate / 1e4:.0f} %: best {best} ({means[best]:.4f}); not beaten by it: {tied or 'none'}")
        if all(
            sum(r["deployed_perplexity"] > 2 * teacher for r in new[(s, rate)].values()) >= 4 for s in SOURCES
        ):
            print(f"  {rate / 1e4:.0f} %: DESTRUCTIVE (every source > 2x teacher perplexity in >= 4/5 assignments)")
    flags = {s: next(iter(new[(s, 0)].values()))["hwa_convergence_review_required"] for s in SOURCES}
    print("\nconvergence_review_required:", flags)
    if len(sys.argv) > 3:
        convergence(sys.argv[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
