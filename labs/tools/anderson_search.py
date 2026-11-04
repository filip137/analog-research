#!/usr/bin/env python3
"""
Iterative Anderson parameter search:
1) run linspace simulation
2) compute error vs spice
3) decide next (heuristic) + log to markdown
4) repeat
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from ebl.registry import parse_experiment_config
from ebl.schema.base import RunMode


EXPERIMENT_ID = "small_drn.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _compare_script_path() -> Path:
    return Path(__file__).resolve().with_name("compare_npz.sh")


def _validate_base_config(payload: dict[str, Any]) -> None:
    """Validate the complete versioned document with the public registry."""

    definition, document = parse_experiment_config(payload)
    if definition.experiment_id != EXPERIMENT_ID:
        raise ValueError(
            f"Expected --config to select {EXPERIMENT_ID!r}. "
            f"Provided value: {definition.experiment_id!r}."
        )
    spec = definition.resolve(document, RunMode.LINSPACE)
    if not spec.settings.record_states:
        raise ValueError(
            "Expected config.modes.linspace.record_states to be true so the "
            "search can compare settled states. Provided value: false."
        )


def _load_base_config(config_path: Path) -> dict[str, Any]:
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            "Expected --config to reference a readable JSON file. "
            f"Provided value: {str(config_path)!r}. {error}"
        ) from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Expected --config to contain valid JSON. "
            f"Provided value: {str(config_path)!r} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected --config to contain a JSON object. "
            f"Provided value: {payload!r}."
        )
    _validate_base_config(payload)
    return payload


def _resolve_states_artifact(output_root: Path) -> tuple[Path, Path]:
    """Resolve the states artifact from exactly one completed EBL run."""

    result_paths = sorted(output_root.glob("*/result.json"))
    if len(result_paths) != 1:
        raise ValueError(
            "Expected the step output directory to contain exactly one "
            "completed run result.json. "
            f"Provided value: {[str(path) for path in result_paths]!r}."
        )

    result_path = result_paths[0]
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Expected the completed run result.json to be readable valid "
            f"JSON. Provided value: {str(result_path)!r}. {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        status = payload.get("status") if isinstance(payload, dict) else payload
        raise ValueError(
            "Expected result.json status to be 'complete'. "
            f"Provided value: {status!r}."
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(
            "Expected result.json artifacts to be a JSON array. "
            f"Provided value: {artifacts!r}."
        )
    state_records = [
        record
        for record in artifacts
        if isinstance(record, dict) and record.get("kind") == "states"
    ]
    if len(state_records) != 1:
        raise ValueError(
            "Expected result.json artifacts to contain exactly one artifact "
            "with kind 'states'. "
            f"Provided value: {state_records!r}."
        )
    relative_path = state_records[0].get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(
            "Expected the states artifact path to be a non-empty relative "
            f"string. Provided value: {relative_path!r}."
        )

    run_dir = result_path.parent.resolve()
    states_path = (run_dir / relative_path).resolve()
    try:
        states_path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError(
            "Expected the states artifact to remain inside its run "
            f"directory. Provided value: {str(states_path)!r}."
        ) from error
    if not states_path.is_file():
        raise ValueError(
            "Expected the states artifact to reference an existing file. "
            f"Provided value: {str(states_path)!r}."
        )
    return run_dir, states_path


def _load_percentile(summary_path: Path, metric: str) -> float | None:
    data = json.loads(summary_path.read_text())
    percentiles = data.get("node_weighted_rel_l1_percentiles", {})
    value = percentiles.get(metric)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return math.nan
    return value


def _write_markdown_header(md_path: Path) -> None:
    if md_path.exists():
        return
    md_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Anderson Search Log\n\n"
        "Run log for iterative Anderson sweeps (linspace + error vs SPICE).\n\n"
        "| step | m | omega | reg | run_dir | metric | value | decision |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    md_path.write_text(header, encoding="utf-8")


def _append_markdown_row(
    md_path: Path,
    *,
    step: int,
    m: int,
    omega: float,
    reg: float,
    run_dir: Path | None,
    metric: str,
    value: float | None,
    decision: str,
) -> None:
    run_dir_text = str(run_dir) if run_dir else "n/a"
    value_text = (
        "NaN"
        if value is not None and math.isnan(value)
        else (f"{value:.6g}" if value is not None else "n/a")
    )
    line = (
        f"| {step} | {m} | {omega:.3g} | {reg:.3g} | {run_dir_text} | "
        f"{metric} | {value_text} | {decision} |\n"
    )
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _prepare_config(
    base_cfg: dict[str, Any],
    *,
    m: int,
    omega: float,
    reg: float,
    batch_size: int,
    linspace_samples: int,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["solver"]["minimizer_impl"] = "anderson"
    cfg["solver"]["anderson"]["memory"] = int(m)
    cfg["solver"]["anderson"]["omega"] = float(omega)
    cfg["solver"]["anderson"]["regularization"] = float(reg)
    cfg["data"]["batch_size"] = int(batch_size)
    cfg["modes"]["linspace"]["samples"] = int(linspace_samples)
    _validate_base_config(cfg)
    return cfg


def _run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd,
    )


def _run_linspace(
    *,
    config_path: Path,
    output_dir: Path,
    weights_path: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "ebl",
        "linspace",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--weights",
        str(weights_path),
    ]
    return _run_command(command, env=env, cwd=_repo_root())


def _decide_next(
    *,
    current_m: int,
    current_omega: float,
    current_reg: float,
    metric_value: float | None,
    best_value: float | None,
    min_m: int,
    max_m: int,
    min_omega: float,
    max_omega: float,
) -> tuple[int, float, float, str, float | None]:
    # Heuristic:
    # - If NaN: reduce omega, reduce m.
    # - If improved: keep m, slightly increase omega.
    # - If worse: reduce omega; if already low, reduce m.
    decision = ""
    next_m = current_m
    next_omega = current_omega
    next_reg = current_reg
    next_best = best_value

    if metric_value is None or math.isnan(metric_value):
        next_omega = max(min_omega, current_omega * 0.5)
        next_m = max(min_m, current_m - 1)
        decision = "NaN -> reduce omega, reduce m"
        return next_m, next_omega, next_reg, decision, next_best

    if best_value is None or metric_value < best_value:
        next_best = metric_value
        next_omega = min(max_omega, current_omega * 1.25)
        decision = "improved -> try higher omega"
        return next_m, next_omega, next_reg, decision, next_best

    # worse
    if current_omega > min_omega * 1.1:
        next_omega = max(min_omega, current_omega * 0.7)
        decision = "worse -> reduce omega"
    else:
        next_m = max(min_m, current_m - 1)
        decision = "worse -> reduce m"
    return next_m, next_omega, next_reg, decision, next_best


def _param_key(
    m: int,
    omega: float,
    reg: float,
    *,
    omega_digits: int = 6,
    reg_digits: int = 12,
) -> tuple[int, float, float]:
    return (
        int(m),
        round(float(omega), omega_digits),
        round(float(reg), reg_digits),
    )


def _ensure_unique_params(
    *,
    m: int,
    omega: float,
    reg: float,
    seen: set[tuple[int, float, float]],
    min_m: int,
    max_m: int,
    min_omega: float,
    max_omega: float,
    max_attempts: int = 50,
) -> tuple[int, float, float, str]:
    key = _param_key(m, omega, reg)
    if key not in seen:
        return m, omega, reg, ""
    for _ in range(max_attempts):
        m = random.randint(min_m, max_m)
        omega = random.uniform(min_omega, max_omega)
        key = _param_key(m, omega, reg)
        if key not in seen:
            return m, omega, reg, "duplicate -> random restart"
    raise SystemExit(
        "Unable to sample unique (m, omega, reg) after 50 attempts."
    )


def _random_restart(
    *,
    min_m: int,
    max_m: int,
    min_omega: float,
    max_omega: float,
    reg: float,
) -> tuple[int, float, float, str]:
    new_m = random.randint(min_m, max_m)
    new_omega = random.uniform(min_omega, max_omega)
    decision = "repeat detected -> random restart"
    return new_m, new_omega, reg, decision


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Iterative Anderson parameter search."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Strict nested small_drn.v1 base config JSON.",
    )
    p.add_argument("--weights", required=True, help="Named weights path.")
    p.add_argument("--spice-npz", required=True, help="Reference SPICE npz.")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Operational root for this search's versioned EBL runs.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--metric",
        default="p95",
        help="Percentile key to optimize (for example p90, p95, or p99).",
    )
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--m-start", type=int, default=None)
    p.add_argument("--omega-start", type=float, default=None)
    p.add_argument("--reg", type=float, default=None)
    p.add_argument(
        "--linspace-samples",
        type=int,
        default=None,
        help="Override modes.linspace.samples (per-axis grid count).",
    )
    p.add_argument("--m-min", type=int, default=1)
    p.add_argument("--m-max", type=int, default=8)
    p.add_argument("--omega-min", type=float, default=0.1)
    p.add_argument("--omega-max", type=float, default=1.0)
    p.add_argument("--md-log", default=None, help="Markdown log path.")
    args = p.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    spice_path = Path(args.spice_npz).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if not weights_path.is_file():
        raise SystemExit(
            "Expected --weights to reference an existing file. "
            f"Provided value: {str(weights_path)!r}."
        )
    if not spice_path.is_file():
        raise SystemExit(
            "Expected --spice-npz to reference an existing file. "
            f"Provided value: {str(spice_path)!r}."
        )
    try:
        base_cfg = _load_base_config(config_path)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    anderson = base_cfg["solver"]["anderson"]
    m = (
        int(args.m_start)
        if args.m_start is not None
        else int(anderson["memory"])
    )
    omega = (
        float(args.omega_start)
        if args.omega_start is not None
        else float(anderson["omega"])
    )
    reg = (
        float(args.reg)
        if args.reg is not None
        else float(anderson["regularization"])
    )
    linspace_samples = (
        int(args.linspace_samples)
        if args.linspace_samples is not None
        else int(base_cfg["modes"]["linspace"]["samples"])
    )

    compare_script = _compare_script_path()
    if not compare_script.is_file():
        raise SystemExit(
            "Expected compare_npz.sh beside anderson_search.py. "
            f"Provided value: {str(compare_script)!r}."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    search_root = Path(
        tempfile.mkdtemp(prefix="anderson-search-", dir=output_root)
    )
    config_root = search_root / "configs"
    config_root.mkdir()
    md_path = (
        Path(args.md_log).expanduser().resolve()
        if args.md_log
        else search_root / "anderson_search.md"
    )
    _write_markdown_header(md_path)

    env = os.environ.copy()
    env.setdefault("KMP_DISABLE_SHM", "1")
    env.setdefault("KMP_SHM_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")

    best_value: float | None = None
    repeat_value: float | None = None
    repeat_count = 0
    repeat_eps = 1e-9
    seen_params: set[tuple[int, float, float]] = set()
    for step in range(1, args.steps + 1):
        m, omega, reg, pre_decision = _ensure_unique_params(
            m=m,
            omega=omega,
            reg=reg,
            seen=seen_params,
            min_m=args.m_min,
            max_m=args.m_max,
            min_omega=args.omega_min,
            max_omega=args.omega_max,
        )
        seen_params.add(_param_key(m, omega, reg))

        run_cfg = _prepare_config(
            base_cfg,
            m=m,
            omega=omega,
            reg=reg,
            batch_size=args.batch_size,
            linspace_samples=linspace_samples,
        )
        tmp_cfg = config_root / f"step-{step:03d}.json"
        tmp_cfg.write_text(
            json.dumps(run_cfg, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )

        step_output = search_root / "runs" / f"step-{step:03d}"
        res = _run_linspace(
            config_path=tmp_cfg,
            output_dir=step_output,
            weights_path=weights_path,
            env=env,
        )
        if res.returncode != 0:
            decision = f"run failed (code {res.returncode})"
            _append_markdown_row(
                md_path,
                step=step,
                m=m,
                omega=omega,
                reg=reg,
                run_dir=None,
                metric=args.metric,
                value=None,
                decision=decision,
            )
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
            break

        try:
            run_dir, run_npz = _resolve_states_artifact(step_output)
        except ValueError as error:
            decision = f"run artifact resolution failed: {error}"
            _append_markdown_row(
                md_path,
                step=step,
                m=m,
                omega=omega,
                reg=reg,
                run_dir=None,
                metric=args.metric,
                value=None,
                decision=decision,
            )
            print(error)
            break

        error_dir = run_dir / "error_npz"
        compare_cmd = [
            str(compare_script),
            str(run_npz),
            str(spice_path),
            str(error_dir),
        ]
        res_cmp = _run_command(
            compare_cmd,
            env=env,
            cwd=_repo_root(),
        )
        if res_cmp.returncode != 0:
            decision = f"error compare failed (code {res_cmp.returncode})"
            _append_markdown_row(
                md_path,
                step=step,
                m=m,
                omega=omega,
                reg=reg,
                run_dir=run_dir,
                metric=args.metric,
                value=None,
                decision=decision,
            )
            print(res_cmp.stdout)
            print(res_cmp.stderr)
            break

        summary_path = (
            error_dir
            / "cross_layer_rel_l1_percentiles_node_weighted.json"
        )
        metric_value = _load_percentile(summary_path, args.metric)

        next_m, next_omega, next_reg, decision, best_value = _decide_next(
            current_m=m,
            current_omega=omega,
            current_reg=reg,
            metric_value=metric_value,
            best_value=best_value,
            min_m=args.m_min,
            max_m=args.m_max,
            min_omega=args.omega_min,
            max_omega=args.omega_max,
        )

        if metric_value is not None and not math.isnan(metric_value):
            if (
                repeat_value is not None
                and abs(metric_value - repeat_value) <= repeat_eps
            ):
                repeat_count += 1
            else:
                repeat_value = metric_value
                repeat_count = 1

            if repeat_count >= 5:
                (
                    next_m,
                    next_omega,
                    next_reg,
                    restart_decision,
                ) = _random_restart(
                    min_m=args.m_min,
                    max_m=args.m_max,
                    min_omega=args.omega_min,
                    max_omega=args.omega_max,
                    reg=reg,
                )
                decision = f"{decision}; {restart_decision}"
                repeat_count = 0
                repeat_value = None
        if pre_decision:
            decision = f"{pre_decision}; {decision}"

        _append_markdown_row(
            md_path,
            step=step,
            m=m,
            omega=omega,
            reg=reg,
            run_dir=run_dir,
            metric=args.metric,
            value=metric_value,
            decision=decision,
        )

        m, omega, reg = next_m, next_omega, next_reg

    print(f"Wrote log to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
