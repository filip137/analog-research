"""Apply Experiment 1's frozen decision rule to collected deployment rows.

usage: python exp1_decision.py <deployments.json> [<deployments.json> ...]

The input is the `deployments.json` written by `python -m ebl lifecycle collect` for
each Experiment 1 lifecycle (stdlib only). Rule, as frozen in exp-001:

- Unit: one assignment x one stuck-device rate; never pooled across rates or sizes.
- A beats B at rate p: KL_A < KL_B in >= 4 of 5 assignments AND the mean over
  assignments of (KL_B - KL_A) / KL_B is >= 10 %.
- Fidelity cost on healthy hardware: the same rule at 0 %.
- Destructive rate: every source's deployed perplexity > 2x the digital teacher's
  (the pretrained source's clean perplexity) in >= 4 of 5 assignments.
- Carry forward: every source not beaten at any rate in 1-5 % by another source.
"""

from collections import defaultdict
import json
import statistics
import sys

SOURCES = ("pretrained", "standard_hwa", "cdt_random_3pct")
RATES = (0, 10000, 20000, 30000, 40000, 50000)
PAIRS = (
    ("cdt_random_3pct", "standard_hwa"),
    ("standard_hwa", "cdt_random_3pct"),
    ("standard_hwa", "pretrained"),
    ("cdt_random_3pct", "pretrained"),
    ("pretrained", "standard_hwa"),
    ("pretrained", "cdt_random_3pct"),
)


def compare(a, b):
    """a, b: {seed: KL}. Returns (wins, n, mean relative reduction, beats)."""

    seeds = sorted(set(a) & set(b))
    wins = sum(a[s] < b[s] for s in seeds)
    relative = statistics.mean((b[s] - a[s]) / b[s] for s in seeds)
    return wins, len(seeds), relative, len(seeds) == 5 and wins >= 4 and relative >= 0.10


def main() -> int:
    rows = [row for path in sys.argv[1:] for row in json.load(open(path))]
    duplicated = [row for row in rows if row["duplicate_coverage"]]
    if duplicated:
        print(f"WARNING: {len(duplicated)} rows have duplicated coverage (retries); inspect before reading on")
    by = defaultdict(dict)  # (lifecycle, source, rate) -> {seed: row}
    for row in rows:
        by[(row["lifecycle_id"], row["hwa_arm"], row["defect_rate_ppm"])][row["array_seed"]] = row
    for lifecycle in sorted({row["lifecycle_id"] for row in rows}):
        print(f"\n=== {lifecycle}")
        teacher = statistics.mean(r["clean_perplexity"] for r in by[(lifecycle, "pretrained", 0)].values())
        print(f"digital teacher perplexity (pretrained source, clean, test): {teacher:.3f}")
        print("\ndeployed teacher KL, mean [min, max] over assignments | mean perplexity | n")
        for rate in RATES:
            cells = []
            for source in SOURCES:
                group = by[(lifecycle, source, rate)].values()
                kl = [r["deployed_teacher_kl"] for r in group]
                ppl = [r["deployed_perplexity"] for r in group]
                cells.append(f"{source}: {statistics.mean(kl):.4f} [{min(kl):.4f}, {max(kl):.4f}] | {statistics.mean(ppl):.2f} | {len(kl)}")
            print(f"  {rate / 1e4:.0f} %: " + "; ".join(cells))
        print("\npairwise rule (wins/n, mean relative KL reduction)")
        beaten = defaultdict(list)
        for rate in RATES:
            kl = {s: {seed: r["deployed_teacher_kl"] for seed, r in by[(lifecycle, s, rate)].items()} for s in SOURCES}
            for a, b in PAIRS:
                wins, n, relative, verdict = compare(kl[a], kl[b])
                if verdict and rate:
                    beaten[b].append((rate / 1e4, a))
                if verdict or {a, b} == {"cdt_random_3pct", "standard_hwa"}:
                    label = "BEATS" if verdict else "no material difference"
                    print(f"  {rate / 1e4:.0f} %: {a} vs {b}: {wins}/{n}, {relative:+.1%} -> {label}")
            if all(
                sum(r["deployed_perplexity"] > 2 * teacher for r in by[(lifecycle, s, rate)].values()) >= 4
                for s in SOURCES
            ):
                print(f"  {rate / 1e4:.0f} %: DESTRUCTIVE (every source > 2x teacher perplexity in >= 4/5 assignments)")
        print("\ncarry forward:", [s for s in SOURCES if not beaten[s]], "| beaten:", dict(beaten))
        flags = {s: next(iter(by[(lifecycle, s, 0)].values()))["hwa_convergence_review_required"] for s in SOURCES}
        print("convergence_review_required:", flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
