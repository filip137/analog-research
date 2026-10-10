"""Apply Experiment 1e's frozen decision rule (exp-005) to the collected deployment rows.

usage: python exp1e_decision.py [results root, default results/lifecycles]

Stdlib only. Reads <root>/opt125m-cmo-l6-exp1e/collected/deployments.json (both test domains:
`deployed_teacher_kl` on WikiText-2, `deployed_transfer_generated_teacher_kl` on held-out generated
text) and, for the reproduction guard, the Experiment 1b and 1c parts B/F rows.
"A beats B at p": lower KL in >= 4 of 5 assignments and a mean relative KL reduction >= 10 %,
per rate and per test domain, never pooled.
"""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/lifecycles")
RATES = (0, 10000, 20000, 30000, 40000, 50000)
DOMAINS = {"wikitext": "deployed_teacher_kl", "generated": "deployed_transfer_generated_teacher_kl"}
BASES = ("standard_hwa", "standard_hwa_9pct", "cdt_random_3pct")


def rows(lifecycle):
    path = ROOT / lifecycle / "collected/deployments.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = json.load(open(path))
    duplicated = [r for r in data if r["duplicate_coverage"]]
    if duplicated:
        print(f"WARNING: {lifecycle}: {len(duplicated)} rows with duplicated coverage; inspect before reading on")
    return data


def table(data, column):
    by = defaultdict(dict)  # (arm, rate) -> {seed: KL}
    for r in data:
        by[(r["hwa_arm"], r["defect_rate_ppm"])][r["array_seed"]] = r[column]
    return by


def compare(a, b):
    seeds = sorted(set(a) & set(b))
    wins = sum(a[s] < b[s] for s in seeds)
    relative = statistics.mean((b[s] - a[s]) / b[s] for s in seeds)
    return wins, len(seeds), relative, len(seeds) == 5 and wins >= 4 and relative >= 0.10


def verdict(by, a, b, rate):
    forward, backward = compare(by[(a, rate)], by[(b, rate)]), compare(by[(b, rate)], by[(a, rate)])
    label = f"{a} BEATS {b}" if forward[3] else f"{b} BEATS {a}" if backward[3] else "no material difference"
    return f"    {rate / 1e4:.0f} %: {a} vs {b}: {forward[0]}/{forward[1]}, {forward[2]:+.1%} -> {label}"


def mean(by, arm, rate):
    return statistics.mean(by[(arm, rate)].values())


def main() -> int:
    data = rows("opt125m-cmo-l6-exp1e")
    arms = sorted({r["hwa_arm"] for r in data})
    model = {base: next(a for a in arms if a.startswith(f"{base}_model_")) for base in BASES}
    tags = {name.rsplit("_model_", 1)[1] for name in model.values()}
    print(f"sources: {arms}\nfrozen digital learning-rate tag: {tags}")

    # Reproduction guard: WikiText rows of pretrained and the reused analog sources are unchanged.
    reference = {
        "pretrained": table(rows("opt125m-cmo-l6-exp1b"), DOMAINS["wikitext"]),
        "standard_hwa": table(rows("opt125m-cmo-l6-exp1c-decay"), DOMAINS["wikitext"]),
        "cdt_random_3pct": table(rows("opt125m-cmo-l6-exp1c-decay"), DOMAINS["wikitext"]),
        "standard_hwa_9pct": table(rows("opt125m-cmo-l6-exp1c-decay-9pct"), DOMAINS["wikitext"]),
    }
    wikitext = table(data, DOMAINS["wikitext"])
    worst = max(
        abs(wikitext[(arm, r)][s] / ref[(arm, r)][s] - 1) for arm, ref in reference.items() for r in RATES for s in ref[(arm, r)]
    )
    print(f"reproduction: max relative KL difference against earlier rows = {worst:.2e} -> "
          f"{'PASS' if worst <= 1e-6 else 'FAIL: investigate before interpreting'}")

    for domain, column in DOMAINS.items():
        by = table(data, column)
        print(f"\n===== {domain} test: KL means per rate (0..5 %)")
        for arm in arms:
            print(f"  {arm:34s}", " ".join(f"{mean(by, arm, r):.4f}" for r in RATES))
        print("\n  1. scope effect (whole-model vs analog-only, paired per array)")
        for base in BASES:
            for r in RATES:
                print(verdict(by, model[base], base, r))
        print("\n  2. recipe ordering under each scope")
        for scope in ("analog", "model"):
            name = (lambda b: b) if scope == "analog" else (lambda b: model[b])
            for a, b in (("cdt_random_3pct", "standard_hwa"), ("standard_hwa_9pct", "cdt_random_3pct")):
                for r in RATES:
                    print(verdict(by, name(a), name(b), r))
        print("\n  4. per-rate best set (lowest mean KL plus every source it does not beat)")
        for r in RATES:
            means = {a: mean(by, a, r) for a in arms}
            best = min(means, key=means.get)
            tied = [a for a in arms if a != best and not compare(by[(best, r)], by[(a, r)])[3]]
            print(f"    {r / 1e4:.0f} %: best {best} ({means[best]:.4f}); not beaten by it: {tied or 'none'}")

    generated = table(data, DOMAINS["generated"])
    print("\n===== 3. specialization: out-of-domain (WikiText) KL / in-domain (generated) KL at 0 %")
    for base in BASES:
        for arm in (base, model[base]):
            print(f"  {arm:34s} {mean(wikitext, arm, 0) / mean(generated, arm, 0):.2f}x")
    flags = {r["hwa_arm"]: r["hwa_convergence_review_required"] for r in data}
    print("\nconvergence_review_required:", flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
