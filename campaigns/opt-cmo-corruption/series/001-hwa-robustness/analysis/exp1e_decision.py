"""Apply Experiment 1e's decision rule after strict deployment coverage checks.

The default ``transfer-tests-v1`` contract reads both final test domains from
explicit transfer columns; ``deployed_teacher_kl`` is DEVELOPMENT and never a
surrogate for WikiText. Historical input requires ``--contract legacy-v1`` and
an explicit ``--legacy-model-tag``. Only the unchanged pretrained source must
reproduce Experiment 1b; fresh analog-only controls are newly trained evidence.

"A beats B at p": lower KL in >= 4 of 5 assignments and mean per-assignment
relative KL reduction >= 10%, per defect rate and test domain, never pooled.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys

RATES = (0, 10000, 20000, 30000, 40000, 50000)
SEEDS = (271001, 271002, 271003, 271004, 271005)
DOMAINS = {
    "wikitext": "deployed_transfer_wikitext_teacher_kl",
    "generated": "deployed_transfer_generated_teacher_kl",
}
LEGACY_DOMAINS = {"wikitext": "deployed_teacher_kl", "generated": "deployed_transfer_generated_teacher_kl"}
BASES = ("standard_hwa", "standard_hwa_9pct", "cdt_random_3pct")
MAIN = "opt125m-cmo-l6-exp1e"
REFERENCE = "opt125m-cmo-l6-exp1b"


def rows(root, lifecycle):
    path = root / lifecycle / "collected/deployments.json"
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected collected deployment rows: {path}")
    return data


def validate_rows(data, arms, domains, *, lifecycle=MAIN, evaluation="development"):
    """Reject partial or ambiguous evidence before computing any verdict."""
    expected = {(arm, rate, seed) for arm in arms for rate in RATES for seed in SEEDS}
    seen, flags = set(), defaultdict(set)
    for row in data:
        if row.get("duplicate_coverage") is not False:
            raise ValueError("Duplicate coverage or missing duplicate-coverage receipt")
        key = (row["hwa_arm"], row["defect_rate_ppm"], row["array_seed"])
        if key in seen:
            raise ValueError(f"Duplicate deployment key: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected deployment key: {key}")
        seen.add(key)
        if row.get("lifecycle_id") != lifecycle or row.get("evaluation") != evaluation:
            raise ValueError(f"Deployment identity/evaluation contract mismatch: {key}")
        if row.get("technology") != "cmo":
            raise ValueError(f"Unexpected technology: {key}")
        if row.get("defect_kind") != ("none" if key[1] == 0 else "random"):
            raise ValueError(f"Unexpected defect kind: {key}")
        for field in ("defect_die_concentration", "defect_array_concentration"):
            if field not in row or row[field] is not None:
                raise ValueError(f"Expected independent fault placement ({field}): {key}")
        for column in domains.values():
            value = row.get(column)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing or invalid positive test KL in {column}: {key}")
        flag = row.get("hwa_convergence_review_required")
        if type(flag) is not bool:
            raise ValueError(f"Missing convergence-review receipt: {key}")
        flags[key[0]].add(flag)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Incomplete deployment coverage: {len(seen)}/{len(expected)}; missing {missing[:6]}")
    if any(len(values) != 1 for values in flags.values()):
        raise ValueError("Inconsistent convergence-review flags for one source")
    return {arm: next(iter(values)) for arm, values in flags.items()}


def table(data, column):
    by = defaultdict(dict)
    for row in data:
        key, seed = (row["hwa_arm"], row["defect_rate_ppm"]), row["array_seed"]
        if row.get("duplicate_coverage") is not False or seed in by[key]:
            raise ValueError(f"Duplicate deployment key: {key}, {seed}")
        by[key][seed] = row[column]
    return by


def compare(a, b):
    if set(a) != set(SEEDS) or set(b) != set(SEEDS):
        raise ValueError("Comparison requires all five matched assignments")
    if any(not math.isfinite(value) or value <= 0 for source in (a, b) for value in source.values()):
        raise ValueError("Comparison requires finite positive KL values")
    wins = sum(a[seed] < b[seed] for seed in SEEDS)
    relative = statistics.mean((b[seed] - a[seed]) / b[seed] for seed in SEEDS)
    return wins, len(SEEDS), relative, wins >= 4 and relative >= 0.10


def verdict(by, a, b, rate):
    forward, backward = compare(by[(a, rate)], by[(b, rate)]), compare(by[(b, rate)], by[(a, rate)])
    label = f"{a} BEATS {b}" if forward[3] else f"{b} BEATS {a}" if backward[3] else "no material difference"
    return f"    {rate / 1e4:.0f} %: {a} vs {b}: {forward[0]}/{forward[1]}, {forward[2]:+.1%} -> {label}"


def mean(by, arm, rate):
    return statistics.mean(by[(arm, rate)].values())


def analyze(root, *, contract="transfer-tests-v1", legacy_model_tag=None):
    if contract == "transfer-tests-v1":
        if legacy_model_tag is not None:
            raise ValueError("A legacy model tag requires the explicit legacy-v1 contract")
        domains, evaluation = DOMAINS, "development"
        model = {base: f"{base}_model" for base in BASES}
    elif contract == "legacy-v1":
        if legacy_model_tag is None or not re.fullmatch(r"[A-Za-z0-9_]+", legacy_model_tag):
            raise ValueError("legacy-v1 requires an explicit --legacy-model-tag")
        domains, evaluation = LEGACY_DOMAINS, "test"
        model = {base: f"{base}_model_{legacy_model_tag}" for base in BASES}
    else:
        raise ValueError(f"Unsupported explicit analysis contract: {contract!r}")
    arms = sorted(("pretrained", *BASES, *model.values()))
    data = rows(root, MAIN)
    flags = validate_rows(data, arms, domains, evaluation=evaluation)
    reference = [row for row in rows(root, REFERENCE) if row["hwa_arm"] == "pretrained"]
    validate_rows(reference, ("pretrained",), {"wikitext": "deployed_teacher_kl"},
                  lifecycle=REFERENCE, evaluation="test")
    old = table(reference, "deployed_teacher_kl")
    wikitext = table(data, domains["wikitext"])
    worst = max(abs(wikitext[("pretrained", rate)][seed] / old[("pretrained", rate)][seed] - 1)
                for rate in RATES for seed in SEEDS)
    if worst > 1e-6:
        raise ValueError(f"Pretrained reproduction FAIL: maximum relative KL difference {worst:.2e}")
    print(f"analysis contract: {contract}\nsources: {arms}")
    print(f"pretrained reproduction: maximum relative KL difference = {worst:.2e} -> PASS")
    for domain, column in domains.items():
        by = table(data, column)
        print(f"\n===== {domain} test: KL means per rate (0..5 %)")
        for arm in arms:
            print(f"  {arm:34s}", " ".join(f"{mean(by, arm, rate):.4f}" for rate in RATES))
        print("\n  1. scope effect (whole-model vs analog-only, paired per array)")
        for base in BASES:
            for rate in RATES:
                print(verdict(by, model[base], base, rate))
        print("\n  2. recipe ordering under each scope")
        for scope in ("analog", "model"):
            names = {base: base if scope == "analog" else model[base] for base in BASES}
            for a, b in (("cdt_random_3pct", "standard_hwa"), ("standard_hwa_9pct", "cdt_random_3pct")):
                for rate in RATES:
                    print(verdict(by, names[a], names[b], rate))
        print("\n  4. per-rate best set (lowest mean KL plus every source it does not beat)")
        for rate in RATES:
            means = {arm: mean(by, arm, rate) for arm in arms}
            best = min(means, key=means.get)
            tied = [arm for arm in arms if arm != best and not compare(by[(best, rate)], by[(arm, rate)])[3]]
            print(f"    {rate / 1e4:.0f} %: best {best} ({means[best]:.4f}); not beaten by it: {tied or 'none'}")
    generated = table(data, domains["generated"])
    print("\n===== 3. specialization at 0 %: WikiText KL / generated KL, with absolute domain changes")
    for base in BASES:
        for arm in (base, model[base]):
            print(f"  {arm:34s} {mean(wikitext, arm, 0) / mean(generated, arm, 0):.2f}x; "
                  f"WikiText KL {mean(wikitext, arm, 0):.6g}, generated KL {mean(generated, arm, 0):.6g}")
        print(f"    absolute KL change (whole-model minus analog): "
              f"WikiText {mean(wikitext, model[base], 0) - mean(wikitext, base, 0):+.6g}; "
              f"generated {mean(generated, model[base], 0) - mean(generated, base, 0):+.6g}")
    print("\nconvergence_review_required:", flags)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/lifecycles"))
    parser.add_argument("--contract", choices=("transfer-tests-v1", "legacy-v1"), default="transfer-tests-v1")
    parser.add_argument("--legacy-model-tag")
    args = parser.parse_args(argv)
    try:
        return analyze(args.root, contract=args.contract, legacy_model_tag=args.legacy_model_tag)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Experiment 1e validation FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
