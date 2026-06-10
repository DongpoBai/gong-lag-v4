#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gong_lag_v4_q1_splitting_validation.py

WMLTP-Q1: splitting-corrected m-averaging validation for GONG-Lag v4.

Scope:
  - Does NOT run the AB science pipeline.
  - Does NOT implement FFT, score/Fisher, lag analysis, leakage model,
    temporal-window convolution, NK asymmetry, or negative-m reconstruction.
  - Generates Discovery command templates with --make-commands.
  - Compares already-existing no-splitting and splitting-corrected AB run
    directories with --compare-only.

Q1 question answered:
  Does WMLTP Legendre splitting correction improve stored-nonnegative-m
  GONG MRVMT shift-and-stack and downstream score/Fisher diagnostics relative
  to a no-splitting baseline?

Expected AB run products:
  run_dir/
    tables/audit.csv
    tables/window_index.csv
    tables/task_errors.csv
    tables/mode_diagnostics.csv
    tables/bin_time_series.csv
    tables/stack_diagnostics.csv
    checkpoints/*.json
    metadata.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Q1 validation") from exc

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for Q1 validation") from exc


DEFAULT_Q1_ROOT = "/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation"
DEFAULT_AB_SCRIPT = "/scratch1/dongpoba/gong_lag_v4/scripts/gong_lag_v4_mvp_ab.py"
DEFAULT_MRVMT_ROOT = "/scratch1/dongpoba/gong_mrvmt/data"
DEFAULT_MODE_TABLE = "/scratch1/dongpoba/gong_mrvmt/mrv1f_m0_target_l.csv"
DEFAULT_SPLITTING_FILE = (
    "/project2/erhodes_44/rcf-04/jreiter/wmltp/mdi61d/DFLS/"
    "MDI.960523to0724r.045to1000.m.ns12.5mHz.mlh.lkm61.61.asym.dr.100.dist2.spl.rev2.dat"
)
ACCEPTED_BASELINE = "/scratch1/dongpoba/gong_lag_v4/runs/mvp_ab_fixed_science_nosplit_all"

Q1_SMALL_L_LIST = "7,10,14,18,22,27,32,39,48,58,70"
Q1_LOW_L_LIST = "0,2,4,7,10"
Q1_PRODUCTION_L_LIST = "0,2,4,7,10,14,18,22,27,32,39,48,58,70,86,107,135"

TABLE_NAMES = [
    "audit",
    "window_index",
    "task_errors",
    "mode_diagnostics",
    "bin_time_series",
    "stack_diagnostics",
]

ESSENTIAL_COMPARE_TABLES = {
    "window_index",
    "stack_diagnostics",
    "mode_diagnostics",
    "bin_time_series",
}

HARD_FAILURE_FLAGS = {
    "parsing_failed",
    "bad_fits_shape",
    "bad_window_row",
    "task_exception",
}

SEVERE_DIAGNOSTIC_FLAGS = {
    "fft_failed",
    "fisher_singular",
    "bad_reference_model",
    "no_valid_windows",
}

EXPECTED_NONHARD_FLAGS = {
    "missing_splitting",
    "fisher_ill_conditioned",
    "no_modes_in_bin",
    "low_fisher_nu",
    "low_fisher_gamma",
    "edge_of_fit_band",
}


@dataclass
class RunData:
    label: str
    run_dir: Path
    tables: Dict[str, pd.DataFrame]
    table_paths: Dict[str, str]
    checkpoints: List[Dict[str, Any]]
    checkpoint_paths: List[str]
    metadata: Dict[str, Any]
    warnings: List[str]


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    return str(obj)


def setup_logging(outdir: Path, log_level: str) -> logging.Logger:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("gong_lag_v4_q1")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = lambda *args: _dt.datetime.utcnow().timetuple()

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(getattr(logging, log_level.upper()))
    stream.setFormatter(fmt)

    log_path = log_dir / f"q1_validation_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.info("Q1 logging initialized: %s", log_path)
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GONG-Lag v4 WMLTP-Q1 splitting validation command generator and run comparator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--baseline-run", default="", help="No-splitting baseline AB run directory")
    parser.add_argument("--split-run", default="", help="Splitting-corrected AB run directory")
    parser.add_argument("--outdir", default=DEFAULT_Q1_ROOT, help="Q1 output directory")
    parser.add_argument("--make-commands", action="store_true", help="Generate Discovery command templates only")
    parser.add_argument("--compare-only", action="store_true", help="Compare existing baseline and split run directories only")

    parser.add_argument("--ab-script", default=DEFAULT_AB_SCRIPT, help="Existing AB script path")
    parser.add_argument("--mrvmt-root", default=DEFAULT_MRVMT_ROOT, help="MRVMT input root for generated AB commands")
    parser.add_argument("--mode-table", default=DEFAULT_MODE_TABLE, help="Mode table for generated AB commands")
    parser.add_argument("--splitting-file", default=DEFAULT_SPLITTING_FILE, help="WMLTP Legendre splitting file for generated AB commands")
    parser.add_argument("--target-l-list", default=Q1_SMALL_L_LIST, help="Default l list; command templates use fixed Q1 designs")
    parser.add_argument("--max-gms", type=int, default=3, help="Default max_gms for generated ad-hoc commands")
    parser.add_argument("--nproc", type=int, default=8, help="nproc for generated AB commands")
    parser.add_argument("--window-days", type=float, default=9.0)
    parser.add_argument("--stride-days", type=float, default=3.0)
    parser.add_argument("--fill-min", type=float, default=0.40)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting Q1 outputs and include --overwrite in generated AB commands")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)

    if args.make_commands and args.compare_only:
        parser.error("--make-commands and --compare-only are mutually exclusive")
    if not args.make_commands and not args.compare_only:
        parser.error("Select exactly one mode: --make-commands or --compare-only")
    if args.compare_only and (not args.baseline_run or not args.split_run):
        parser.error("--compare-only requires --baseline-run and --split-run")
    if args.nproc < 1:
        parser.error("--nproc must be >= 1")
    if args.window_days <= 0:
        parser.error("--window-days must be > 0")
    if args.stride_days <= 0:
        parser.error("--stride-days must be > 0")
    if not (0.0 <= args.fill_min <= 1.0):
        parser.error("--fill-min must be in [0, 1]")

    return args


def prepare_outdir(outdir: Path, overwrite: bool, mode_name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sentinel = outdir / "q1_acceptance_summary.json"
    command_sentinel = outdir / "q1_run_commands.sh"
    if not overwrite:
        if mode_name == "compare" and sentinel.exists():
            raise SystemExit(f"Refusing to overwrite existing {sentinel}; add --overwrite")
        if mode_name == "commands" and command_sentinel.exists():
            raise SystemExit(f"Refusing to overwrite existing {command_sentinel}; add --overwrite")
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)


def shell_quote(path_or_value: Any) -> str:
    return shlex.quote(str(path_or_value))


def with_overwrite_flag(args: argparse.Namespace) -> str:
    return " \\\n  --overwrite" if args.overwrite else ""


def make_ab_command(
    args: argparse.Namespace,
    target_l_list: str,
    max_gms: int,
    splitting_file: str,
    shift_sign: str,
    outdir: str,
) -> str:
    return (
        f"python {shell_quote(args.ab_script)} \\\n"
        f"  --mrvmt-root {shell_quote(args.mrvmt_root)} \\\n"
        f"  --mode-table {shell_quote(args.mode_table)} \\\n"
        f"  --splitting-file {shell_quote(splitting_file)} \\\n"
        f"  --shift-sign {shell_quote(shift_sign)} \\\n"
        f"  --target-l-list {shell_quote(target_l_list)} \\\n"
        f"  --max-gms {int(max_gms)} \\\n"
        f"  --nproc {int(args.nproc)} \\\n"
        f"  --window-days {args.window_days:g} \\\n"
        f"  --stride-days {args.stride_days:g} \\\n"
        f"  --fill-min {args.fill_min:g} \\\n"
        f"  --outdir {shell_quote(outdir)}"
        f"{with_overwrite_flag(args)}"
    )


def make_compare_command(
    q1_script: str,
    baseline_run: str,
    split_run: str,
    outdir: str,
    overwrite: bool,
) -> str:
    ow = " \\\n  --overwrite" if overwrite else ""
    return (
        f"python {shell_quote(q1_script)} \\\n"
        f"  --compare-only \\\n"
        f"  --baseline-run {shell_quote(baseline_run)} \\\n"
        f"  --split-run {shell_quote(split_run)} \\\n"
        f"  --outdir {shell_quote(outdir)}"
        f"{ow}"
    )


def write_q1_commands(args: argparse.Namespace, logger: logging.Logger) -> None:
    q1_root = Path(args.outdir)
    runs = q1_root / "runs"
    comparisons = q1_root / "comparisons"
    q1_script = "/scratch1/dongpoba/gong_lag_v4/scripts/gong_lag_v4_q1_splitting_validation.py"

    small_nosplit = str(runs / "q1_small_nosplit")
    small_split = str(runs / "q1_small_split_auto")
    low_l_split = str(runs / "q1_low_l_split_auto_all")
    prod_nosplit = str(runs / "q1_prod_candidate_nosplit_all")
    prod_split = str(runs / "q1_prod_candidate_split_auto_all")

    commands: List[Tuple[str, str]] = [
        (
            "Q1-small no-splitting AB run",
            make_ab_command(
                args=args,
                target_l_list=Q1_SMALL_L_LIST,
                max_gms=3,
                splitting_file="none",
                shift_sign="none",
                outdir=small_nosplit,
            ),
        ),
        (
            "Q1-small split-auto AB run",
            make_ab_command(
                args=args,
                target_l_list=Q1_SMALL_L_LIST,
                max_gms=3,
                splitting_file=args.splitting_file,
                shift_sign="auto",
                outdir=small_split,
            ),
        ),
        (
            "Q1-small compare-only validation",
            make_compare_command(
                q1_script=q1_script,
                baseline_run=small_nosplit,
                split_run=small_split,
                outdir=str(comparisons / "q1_small_split_auto_vs_nosplit"),
                overwrite=args.overwrite,
            ),
        ),
        (
            "Q1-low-l full split-auto AB run for comparison against accepted no-splitting baseline",
            make_ab_command(
                args=args,
                target_l_list=Q1_LOW_L_LIST,
                max_gms=0,
                splitting_file=args.splitting_file,
                shift_sign="auto",
                outdir=low_l_split,
            ),
        ),
        (
            "Q1-low-l full compare-only validation against accepted baseline",
            make_compare_command(
                q1_script=q1_script,
                baseline_run=ACCEPTED_BASELINE,
                split_run=low_l_split,
                outdir=str(comparisons / "q1_low_l_split_auto_vs_accepted_nosplit"),
                overwrite=args.overwrite,
            ),
        ),
        (
            "Q1-production candidate no-splitting AB run; do not treat as accepted before Q1-small passes",
            make_ab_command(
                args=args,
                target_l_list=Q1_PRODUCTION_L_LIST,
                max_gms=0,
                splitting_file="none",
                shift_sign="none",
                outdir=prod_nosplit,
            ),
        ),
        (
            "Q1-production candidate split-auto AB run; do not treat as accepted before Q1-small passes",
            make_ab_command(
                args=args,
                target_l_list=Q1_PRODUCTION_L_LIST,
                max_gms=0,
                splitting_file=args.splitting_file,
                shift_sign="auto",
                outdir=prod_split,
            ),
        ),
        (
            "Q1-production candidate compare-only validation; run only after both candidate AB runs finish",
            make_compare_command(
                q1_script=q1_script,
                baseline_run=prod_nosplit,
                split_run=prod_split,
                outdir=str(comparisons / "q1_prod_candidate_split_auto_vs_nosplit"),
                overwrite=args.overwrite,
            ),
        ),
    ]

    command_path = q1_root / "q1_run_commands.sh"
    readme_path = q1_root / "README_Q1_commands.md"

    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# WMLTP-Q1 command templates only.",
        "# This file does not submit jobs by itself unless you execute the commands.",
        "# It does not modify /project2; /project2 is used only as read-only input.",
        "",
        "export OPENBLAS_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "",
        f"mkdir -p {shell_quote(str(runs))} {shell_quote(str(comparisons))}",
        "",
    ]
    for title, command in commands:
        script_lines.append(f"# --- {title} ---")
        script_lines.append(command)
        script_lines.append("")

    command_path.write_text("\n".join(script_lines), encoding="utf-8")
    command_path.chmod(0o755)

    readme_lines = [
        "# WMLTP-Q1 splitting validation commands",
        "",
        "This directory contains static command templates for Discovery.",
        "",
        "Q1 scope:",
        "",
        "- Run paired AB comparisons with and without WMLTP Legendre splitting correction.",
        "- Use the accepted AB pipeline as-is.",
        "- Do not run HMI degradation, lag analysis, leakage modeling, temporal-window convolution, NK asymmetry, or negative-m reconstruction.",
        "- Do not treat production-candidate results as accepted until Q1-small passes.",
        "",
        "Generated commands:",
        "",
    ]
    for title, command in commands:
        readme_lines.append(f"## {title}")
        readme_lines.append("")
        readme_lines.append("```bash")
        readme_lines.append(command)
        readme_lines.append("```")
        readme_lines.append("")

    readme_lines.extend(
        [
            "Acceptance categories are computed by `--compare-only` and written to:",
            "",
            "- `q1_acceptance_summary.json`",
            "- `Q1_REPORT.md`",
            "- `tables/*.csv`",
            "- `plots/*.png` when matplotlib is available",
            "",
            "This is not a lag result.",
        ]
    )
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    logger.info("Wrote command templates: %s", command_path)
    logger.info("Wrote command README: %s", readme_path)


def find_table_path(run_dir: Path, table_name: str) -> Optional[Path]:
    candidates = [
        run_dir / "tables" / f"{table_name}.csv",
        run_dir / f"{table_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_csv_table(run_dir: Path, table_name: str, warnings_out: List[str]) -> Tuple[pd.DataFrame, str]:
    path = find_table_path(run_dir, table_name)
    if path is None:
        warnings_out.append(f"missing_table:{table_name}")
        return pd.DataFrame(), ""
    try:
        return pd.read_csv(path, low_memory=False), str(path)
    except Exception as exc:
        warnings_out.append(f"read_failed:{table_name}:{exc}")
        return pd.DataFrame(), str(path)


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {"_json_value": value}
    except Exception as exc:
        return {"_read_error": str(exc)}


def read_metadata(run_dir: Path) -> Dict[str, Any]:
    for name in ["metadata.json", "metadata_failed.json"]:
        path = run_dir / name
        if path.exists():
            meta = read_json_file(path)
            meta["_metadata_path"] = str(path)
            return meta
    return {}


def read_checkpoints(run_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return [], []
    checkpoint_paths = sorted(set(checkpoint_dir.glob("*.json")))
    out: List[Dict[str, Any]] = []
    paths: List[str] = []
    for path in checkpoint_paths:
        payload = read_json_file(path)
        payload["_checkpoint_path"] = str(path)
        out.append(payload)
        paths.append(str(path))
    return out, paths


def read_run(run_dir: str, label: str, logger: logging.Logger) -> RunData:
    root = Path(run_dir)
    warnings_out: List[str] = []
    if not root.exists():
        warnings_out.append(f"run_dir_missing:{root}")

    tables: Dict[str, pd.DataFrame] = {}
    table_paths: Dict[str, str] = {}
    for table_name in TABLE_NAMES:
        df, path = read_csv_table(root, table_name, warnings_out)
        tables[table_name] = df
        table_paths[table_name] = path

    checkpoints, checkpoint_paths = read_checkpoints(root)
    metadata = read_metadata(root)

    logger.info(
        "Loaded %s run %s: tables=%s checkpoint_count=%d warnings=%s",
        label,
        root,
        {k: int(len(v)) for k, v in tables.items()},
        len(checkpoints),
        warnings_out,
    )

    return RunData(
        label=label,
        run_dir=root,
        tables=tables,
        table_paths=table_paths,
        checkpoints=checkpoints,
        checkpoint_paths=checkpoint_paths,
        metadata=metadata,
        warnings=warnings_out,
    )


def parse_flags(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def count_quality_flags(df: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    if df is None or df.empty or "quality_flags" not in df.columns:
        return counter
    for value in df["quality_flags"].dropna().tolist():
        counter.update(parse_flags(value))
    return counter


def rows_with_flag(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    if df is None or df.empty or "quality_flags" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    mask = df["quality_flags"].fillna("").astype(str).map(lambda x: flag in parse_flags(x))
    return df[mask].copy()


def rows_with_any_flag(df: pd.DataFrame) -> int:
    if df is None or df.empty or "quality_flags" not in df.columns:
        return 0
    return int(df["quality_flags"].fillna("").astype(str).map(lambda x: len(parse_flags(x)) > 0).sum())


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["true", "t", "1", "yes", "y"]
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def finite_median(values: Any) -> float:
    arr = finite_array(values)
    return float(np.nanmedian(arr)) if arr.size else math.nan


def finite_mean(values: Any) -> float:
    arr = finite_array(values)
    return float(np.nanmean(arr)) if arr.size else math.nan


def finite_std(values: Any) -> float:
    arr = finite_array(values)
    return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else math.nan


def finite_fraction(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.isfinite(arr).sum() / arr.size)


def safe_ratio(num: Any, den: Any) -> np.ndarray:
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.full_like(n, np.nan, dtype=float)
    mask = np.isfinite(n) & np.isfinite(d) & (d != 0.0)
    out[mask] = n[mask] / d[mask]
    return out


def safe_corr(x: Any, y: Any) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def counter_json(counter: Mapping[str, int]) -> str:
    return json.dumps({str(k): int(v) for k, v in counter.items()}, sort_keys=True)


def health_summary_for_run(run: RunData) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, Any]] = []
    hard_failures: List[str] = []

    def add(metric: str, value: Any, hard_fail: bool = False, details: Any = "") -> None:
        rows.append(
            {
                "run_label": run.label,
                "run_dir": str(run.run_dir),
                "metric": metric,
                "value": value,
                "hard_fail": bool(hard_fail),
                "details": details if isinstance(details, str) else json.dumps(details, default=json_default, sort_keys=True),
            }
        )
        if hard_fail:
            hard_failures.append(f"{run.label}:{metric}:{value}")

    add("run_dir_exists", run.run_dir.exists(), not run.run_dir.exists())

    for table_name in TABLE_NAMES:
        df = run.tables.get(table_name, pd.DataFrame())
        missing = run.table_paths.get(table_name, "") == ""
        is_hard_missing = table_name in ESSENTIAL_COMPARE_TABLES or table_name in {"audit", "task_errors"}
        add(
            f"{table_name}.rows",
            int(len(df)),
            bool(missing and is_hard_missing),
            {"path": run.table_paths.get(table_name, ""), "missing": missing},
        )

    task_errors = run.tables.get("task_errors", pd.DataFrame())
    task_error_rows = int(len(task_errors))
    add("task_errors.rows", task_error_rows, task_error_rows > 0)

    task_error_text = ""
    if task_error_rows > 0:
        task_error_text = " ".join(
            task_errors.head(10000).astype(str).fillna("").to_numpy().ravel().tolist()
        )
    lower_text = task_error_text.lower()
    add("task_errors.contains_parsing_failed", "parsing_failed" in lower_text, "parsing_failed" in lower_text)
    add("task_errors.contains_status_failed", "status=failed" in lower_text or ",failed" in lower_text or " failed " in lower_text, " failed " in lower_text)
    add("task_errors.contains_traceback", "traceback" in lower_text, "traceback" in lower_text)

    status_counter = Counter(str(c.get("status", "missing_status")) for c in run.checkpoints)
    checkpoint_count = len(run.checkpoints)
    add("checkpoint.count", checkpoint_count, checkpoint_count <= 0)
    add("checkpoint.status_distribution", counter_json(status_counter), any(k != "done" for k in status_counter.keys()))

    audit = run.tables.get("audit", pd.DataFrame())
    if audit.empty:
        add("audit.present", False, True)
    else:
        if "shape_ok" in audit.columns:
            shape_bool = bool_series(audit["shape_ok"])
            frac = float(shape_bool.mean()) if len(shape_bool) else math.nan
            add("audit.shape_ok_fraction", frac, not bool(shape_bool.all()))
        else:
            add("audit.shape_ok_column_present", False, True)

        if "window_binary" in audit.columns:
            window_bool = bool_series(audit["window_binary"])
            frac = float(window_bool.mean()) if len(window_bool) else math.nan
            add("audit.window_binary_fraction", frac, not bool(window_bool.all()))
        else:
            add("audit.window_binary_column_present", False, True)

        add("audit.quality_flags_counts", counter_json(count_quality_flags(audit)))

    window_index = run.tables.get("window_index", pd.DataFrame())
    add("window_index.quality_flags_counts", counter_json(count_quality_flags(window_index)))
    if "accepted" in window_index.columns and not window_index.empty:
        accepted = bool_series(window_index["accepted"])
        add("window_index.accepted_rows", int(accepted.sum()), int(accepted.sum()) <= 0)

    all_flag_counts: Counter = Counter()
    by_table_counts: Dict[str, Dict[str, int]] = {}
    for table_name, df in run.tables.items():
        c = count_quality_flags(df)
        by_table_counts[table_name] = dict(c)
        all_flag_counts.update(c)

    for flag in sorted(HARD_FAILURE_FLAGS):
        count = int(all_flag_counts.get(flag, 0))
        add(f"hard_flag_count.{flag}", count, count > 0)

    add("all_quality_flags_by_table", json.dumps(by_table_counts, sort_keys=True, default=json_default))

    for warning in run.warnings:
        add("loader_warning", warning, warning.startswith("read_failed"))

    return pd.DataFrame(rows), hard_failures


def normalize_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "<NA>"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isfinite(x):
            return f"{x:.12g}"
        return "<NA>"
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return str(int(float(text)))
    return text


def make_key_counter(df: pd.DataFrame, key_cols: Sequence[str]) -> Counter:
    if df.empty or not key_cols:
        return Counter()
    key_series = None
    for col in key_cols:
        part = df[col].map(normalize_cell)
        key_series = part if key_series is None else key_series + "|" + part
    return Counter(key_series.tolist())


def compare_key_sets(
    baseline: pd.DataFrame,
    split: pd.DataFrame,
    table_name: str,
    comparison_name: str,
    candidate_cols: Sequence[str],
    min_cols: int,
    hard_if_mismatch: bool,
) -> Dict[str, Any]:
    key_cols = [c for c in candidate_cols if c in baseline.columns and c in split.columns]
    if len(key_cols) < min_cols:
        return {
            "comparison_name": comparison_name,
            "table": table_name,
            "key_columns": ",".join(key_cols),
            "baseline_rows": int(len(baseline)),
            "split_rows": int(len(split)),
            "baseline_unique_keys": 0,
            "split_unique_keys": 0,
            "matching_keys": 0,
            "baseline_only_keys": 0,
            "split_only_keys": 0,
            "baseline_duplicate_key_rows": 0,
            "split_duplicate_key_rows": 0,
            "comparable": False,
            "hard_failure": bool(hard_if_mismatch),
            "examples_baseline_only": "",
            "examples_split_only": "",
            "details": f"insufficient_key_columns; required>={min_cols}; present={key_cols}",
        }

    base_counter = make_key_counter(baseline, key_cols)
    split_counter = make_key_counter(split, key_cols)
    base_keys = set(base_counter)
    split_keys = set(split_counter)
    baseline_only = sorted(base_keys - split_keys)[:10]
    split_only = sorted(split_keys - base_keys)[:10]
    matching = base_keys & split_keys
    duplicate_base = sum(max(v - 1, 0) for v in base_counter.values())
    duplicate_split = sum(max(v - 1, 0) for v in split_counter.values())
    comparable = (
        len(base_keys - split_keys) == 0
        and len(split_keys - base_keys) == 0
        and duplicate_base == 0
        and duplicate_split == 0
        and len(base_keys) > 0
        and len(split_keys) > 0
    )
    return {
        "comparison_name": comparison_name,
        "table": table_name,
        "key_columns": ",".join(key_cols),
        "baseline_rows": int(len(baseline)),
        "split_rows": int(len(split)),
        "baseline_unique_keys": int(len(base_keys)),
        "split_unique_keys": int(len(split_keys)),
        "matching_keys": int(len(matching)),
        "baseline_only_keys": int(len(base_keys - split_keys)),
        "split_only_keys": int(len(split_keys - base_keys)),
        "baseline_duplicate_key_rows": int(duplicate_base),
        "split_duplicate_key_rows": int(duplicate_split),
        "comparable": bool(comparable),
        "hard_failure": bool((not comparable) and hard_if_mismatch),
        "examples_baseline_only": json.dumps(baseline_only, ensure_ascii=False),
        "examples_split_only": json.dumps(split_only, ensure_ascii=False),
        "details": "",
    }


def grid_consistency(baseline: RunData, split: RunData) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grid_specs = [
        (
            "audit",
            "audit_gm_l_grid",
            ["gm_id", "gmonth_label", "l_requested"],
            2,
            True,
        ),
        (
            "window_index",
            "window_time_l_grid",
            ["gm_id", "gmonth", "gmonth_label", "l", "window_id", "window_index", "t_mid", "start_sample", "end_sample"],
            4,
            True,
        ),
        (
            "stack_diagnostics",
            "stack_window_l_grid",
            ["gm_id", "gmonth", "gmonth_label", "l", "window_id", "t_mid"],
            4,
            True,
        ),
        (
            "mode_diagnostics",
            "mode_window_l_n_grid",
            ["gm_id", "gmonth", "gmonth_label", "l", "n", "mode_id", "window_id", "t_mid"],
            5,
            True,
        ),
        (
            "bin_time_series",
            "bin_time_frequency_grid",
            ["gm_id", "gmonth", "t_mid", "freq_bin_low", "freq_bin_high", "bin_width"],
            5,
            True,
        ),
        (
            "bin_time_series",
            "frequency_bin_grid_only",
            ["freq_bin_low", "freq_bin_high", "freq_bin_center", "bin_width"],
            3,
            False,
        ),
        (
            "bin_time_series",
            "time_grid_only",
            ["gm_id", "gmonth", "t_mid"],
            2,
            False,
        ),
    ]

    for table_name, comparison_name, cols, min_cols, hard in grid_specs:
        base_df = baseline.tables.get(table_name, pd.DataFrame())
        split_df = split.tables.get(table_name, pd.DataFrame())

        if table_name == "audit" and "l_requested" not in base_df.columns:
            cols = ["gm_id", "gmonth_label", "l"]
        if table_name == "audit" and "l_requested" not in split_df.columns:
            cols = ["gm_id", "gmonth_label", "l"]

        rows.append(
            compare_key_sets(
                baseline=base_df,
                split=split_df,
                table_name=table_name,
                comparison_name=comparison_name,
                candidate_cols=cols,
                min_cols=min_cols,
                hard_if_mismatch=hard,
            )
        )

    return pd.DataFrame(rows)


def merge_on_common_keys(
    baseline: pd.DataFrame,
    split: pd.DataFrame,
    candidate_cols: Sequence[str],
    min_cols: int,
) -> Tuple[pd.DataFrame, List[str]]:
    key_cols = [c for c in candidate_cols if c in baseline.columns and c in split.columns]
    if len(key_cols) < min_cols:
        return pd.DataFrame(), key_cols
    merged = baseline.merge(
        split,
        on=key_cols,
        how="inner",
        suffixes=("_baseline", "_split"),
    )
    return merged, key_cols


def stack_sharpness_summaries(baseline: RunData, split: RunData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = baseline.tables.get("stack_diagnostics", pd.DataFrame())
    spl = split.tables.get("stack_diagnostics", pd.DataFrame())

    merged, keys = merge_on_common_keys(
        base,
        spl,
        ["gm_id", "gmonth", "gmonth_label", "l", "window_id", "t_mid"],
        4,
    )

    if merged.empty:
        empty_by_l = pd.DataFrame()
        empty_overall = pd.DataFrame()
        empty_sign = pd.DataFrame()
        return empty_by_l, empty_overall, empty_sign

    for col in [
        "sharpness_before",
        "sharpness_after_plus",
        "sharpness_after_minus",
        "sharpness_chosen",
    ]:
        for suffix in ["baseline", "split"]:
            full = f"{col}_{suffix}"
            if full in merged.columns:
                merged[full] = pd.to_numeric(merged[full], errors="coerce")
            else:
                merged[full] = np.nan

    merged["l_numeric"] = pd.to_numeric(merged["l"], errors="coerce")
    merged["ratio_chosen_to_before"] = safe_ratio(
        merged["sharpness_chosen_split"].to_numpy(),
        merged["sharpness_before_split"].to_numpy(),
    )
    merged["ratio_split_to_baseline_chosen"] = safe_ratio(
        merged["sharpness_chosen_split"].to_numpy(),
        merged["sharpness_chosen_baseline"].to_numpy(),
    )
    merged["plus_minus_delta"] = (
        merged["sharpness_after_plus_split"] - merged["sharpness_after_minus_split"]
    )

    def preference_label(row: pd.Series) -> str:
        plus = row.get("sharpness_after_plus_split", np.nan)
        minus = row.get("sharpness_after_minus_split", np.nan)
        if not (math.isfinite(float(plus)) and math.isfinite(float(minus))):
            return "nonfinite"
        if np.isclose(float(plus), float(minus), rtol=1e-12, atol=1e-12):
            return "tie"
        return "plus_better" if float(plus) > float(minus) else "minus_better"

    merged["plus_minus_preference"] = merged.apply(preference_label, axis=1)

    def summarize(sub: pd.DataFrame, scope: str, l_value: Any = "") -> Dict[str, Any]:
        ratio = sub["ratio_chosen_to_before"].to_numpy(dtype=float)
        ratio_base = sub["ratio_split_to_baseline_chosen"].to_numpy(dtype=float)
        chosen_col = "chosen_shift_sign_split"
        chosen_counts = Counter(sub[chosen_col].fillna("").astype(str).tolist()) if chosen_col in sub.columns else Counter()
        pref_counts = Counter(sub["plus_minus_preference"].fillna("").astype(str).tolist())
        finite_ratio = finite_array(ratio)
        finite_ratio_base = finite_array(ratio_base)
        return {
            "scope": scope,
            "l": l_value,
            "n_pairs": int(len(sub)),
            "median_sharpness_before": finite_median(sub["sharpness_before_split"]),
            "median_sharpness_after_plus": finite_median(sub["sharpness_after_plus_split"]),
            "median_sharpness_after_minus": finite_median(sub["sharpness_after_minus_split"]),
            "median_sharpness_chosen": finite_median(sub["sharpness_chosen_split"]),
            "baseline_median_sharpness_chosen": finite_median(sub["sharpness_chosen_baseline"]),
            "median_ratio_chosen_to_before": finite_median(ratio),
            "median_ratio_split_to_baseline_chosen": finite_median(ratio_base),
            "fraction_ratio_chosen_to_before_gt_1": float(np.mean(finite_ratio > 1.0)) if finite_ratio.size else math.nan,
            "fraction_ratio_chosen_to_before_gt_1p05": float(np.mean(finite_ratio > 1.05)) if finite_ratio.size else math.nan,
            "fraction_ratio_chosen_to_before_lt_0p95": float(np.mean(finite_ratio < 0.95)) if finite_ratio.size else math.nan,
            "fraction_split_to_baseline_gt_1": float(np.mean(finite_ratio_base > 1.0)) if finite_ratio_base.size else math.nan,
            "fraction_split_to_baseline_lt_0p95": float(np.mean(finite_ratio_base < 0.95)) if finite_ratio_base.size else math.nan,
            "chosen_shift_sign_distribution": counter_json(chosen_counts),
            "plus_minus_preference_counts": counter_json(pref_counts),
            "plus_better_count": int(pref_counts.get("plus_better", 0)),
            "minus_better_count": int(pref_counts.get("minus_better", 0)),
            "tie_count": int(pref_counts.get("tie", 0)),
            "nonfinite_preference_count": int(pref_counts.get("nonfinite", 0)),
        }

    by_l_rows: List[Dict[str, Any]] = []
    for l_value, sub in merged.groupby("l_numeric", dropna=False):
        by_l_rows.append(summarize(sub, "by_l", int(l_value) if pd.notna(l_value) else ""))

    overall_rows = [
        summarize(merged, "all", "all"),
        summarize(merged[merged["l_numeric"] > 0], "l_gt_0", "l>0"),
        summarize(merged[merged["l_numeric"] == 0], "l_eq_0", "l=0"),
    ]

    sign_rows: List[Dict[str, Any]] = []
    chosen_col = "chosen_shift_sign_split"
    if chosen_col in merged.columns:
        for l_value, sub in merged.groupby("l_numeric", dropna=False):
            total = len(sub)
            counts = Counter(sub[chosen_col].fillna("").astype(str).tolist())
            for sign, count in sorted(counts.items()):
                sign_rows.append(
                    {
                        "l": int(l_value) if pd.notna(l_value) else "",
                        "chosen_shift_sign": sign,
                        "count": int(count),
                        "fraction": float(count / total) if total else math.nan,
                    }
                )

    return pd.DataFrame(by_l_rows), pd.DataFrame(overall_rows), pd.DataFrame(sign_rows)


def bin_time_merge(baseline: RunData, split: RunData) -> Tuple[pd.DataFrame, List[str]]:
    base = baseline.tables.get("bin_time_series", pd.DataFrame())
    spl = split.tables.get("bin_time_series", pd.DataFrame())
    merged, keys = merge_on_common_keys(
        base,
        spl,
        ["gm_id", "gmonth", "t_mid", "freq_bin_low", "freq_bin_high", "freq_bin_center", "bin_width"],
        5,
    )
    return merged, keys


def add_ratio_columns_for_bin_merge(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    for col in ["fisher_trace", "fisher_det", "fisher_cond", "sigma_nu", "sigma_lngamma"]:
        bcol = f"{col}_baseline"
        scol = f"{col}_split"
        if bcol not in out.columns:
            out[bcol] = np.nan
        if scol not in out.columns:
            out[scol] = np.nan
        out[f"{col}_ratio_split_over_baseline"] = safe_ratio(
            pd.to_numeric(out[scol], errors="coerce").to_numpy(),
            pd.to_numeric(out[bcol], errors="coerce").to_numpy(),
        )

    for col in ["n_modes_used", "n_modes_total"]:
        bcol = f"{col}_baseline"
        scol = f"{col}_split"
        if bcol not in out.columns:
            out[bcol] = np.nan
        if scol not in out.columns:
            out[scol] = np.nan
        out[f"{col}_diff_split_minus_baseline"] = (
            pd.to_numeric(out[scol], errors="coerce") - pd.to_numeric(out[bcol], errors="coerce")
        )

    for col in ["delta_nu", "delta_lngamma"]:
        for suffix in ["baseline", "split"]:
            full = f"{col}_{suffix}"
            if full not in out.columns:
                out[full] = np.nan
            out[f"{full}_finite"] = np.isfinite(pd.to_numeric(out[full], errors="coerce").to_numpy())

    return out


def frequency_center(df: pd.DataFrame) -> pd.Series:
    if "freq_bin_center" in df.columns:
        return pd.to_numeric(df["freq_bin_center"], errors="coerce")
    low = pd.to_numeric(df.get("freq_bin_low", np.nan), errors="coerce")
    high = pd.to_numeric(df.get("freq_bin_high", np.nan), errors="coerce")
    return 0.5 * (low + high)


def summarize_bin_subset(sub: pd.DataFrame, scope: str, bin_width_label: Any, low: Any = "", high: Any = "", center: Any = "") -> Dict[str, Any]:
    return {
        "source": "bin_time_series",
        "summary_scope": scope,
        "bin_width": bin_width_label,
        "freq_bin_low": low,
        "freq_bin_high": high,
        "freq_bin_center": center,
        "n_pairs": int(len(sub)),
        "median_fisher_trace_baseline": finite_median(sub.get("fisher_trace_baseline", pd.Series(dtype=float))),
        "median_fisher_trace_split": finite_median(sub.get("fisher_trace_split", pd.Series(dtype=float))),
        "median_fisher_trace_ratio": finite_median(sub.get("fisher_trace_ratio_split_over_baseline", pd.Series(dtype=float))),
        "median_fisher_det_ratio": finite_median(sub.get("fisher_det_ratio_split_over_baseline", pd.Series(dtype=float))),
        "median_fisher_cond_ratio": finite_median(sub.get("fisher_cond_ratio_split_over_baseline", pd.Series(dtype=float))),
        "median_sigma_nu_ratio": finite_median(sub.get("sigma_nu_ratio_split_over_baseline", pd.Series(dtype=float))),
        "median_sigma_lngamma_ratio": finite_median(sub.get("sigma_lngamma_ratio_split_over_baseline", pd.Series(dtype=float))),
        "median_n_modes_used_diff": finite_median(sub.get("n_modes_used_diff_split_minus_baseline", pd.Series(dtype=float))),
        "median_n_modes_total_diff": finite_median(sub.get("n_modes_total_diff_split_minus_baseline", pd.Series(dtype=float))),
        "baseline_fraction_finite_delta_nu": float(np.mean(sub.get("delta_nu_baseline_finite", pd.Series(dtype=bool)))) if len(sub) else math.nan,
        "split_fraction_finite_delta_nu": float(np.mean(sub.get("delta_nu_split_finite", pd.Series(dtype=bool)))) if len(sub) else math.nan,
        "baseline_fraction_finite_delta_lngamma": float(np.mean(sub.get("delta_lngamma_baseline_finite", pd.Series(dtype=bool)))) if len(sub) else math.nan,
        "split_fraction_finite_delta_lngamma": float(np.mean(sub.get("delta_lngamma_split_finite", pd.Series(dtype=bool)))) if len(sub) else math.nan,
    }


def bin_fisher_comparison(baseline: RunData, split: RunData) -> pd.DataFrame:
    merged, keys = bin_time_merge(baseline, split)
    if merged.empty:
        return pd.DataFrame()

    merged = add_ratio_columns_for_bin_merge(merged)
    merged["freq_center_for_range"] = frequency_center(merged)
    if "bin_width" not in merged.columns:
        merged["bin_width"] = np.nan

    rows: List[Dict[str, Any]] = []

    group_cols = [c for c in ["bin_width", "freq_bin_low", "freq_bin_high", "freq_bin_center"] if c in merged.columns]
    for key, sub in merged.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(group_cols, key))
        rows.append(
            summarize_bin_subset(
                sub=sub,
                scope="frequency_bin",
                bin_width_label=values.get("bin_width", ""),
                low=values.get("freq_bin_low", ""),
                high=values.get("freq_bin_high", ""),
                center=values.get("freq_bin_center", ""),
            )
        )

    widths = ["all"] + sorted(
        [float(x) for x in pd.to_numeric(merged["bin_width"], errors="coerce").dropna().unique()]
    )
    ranges = [
        ("range_all", -math.inf, math.inf),
        ("range_2000_4500", 2000.0, 4500.0),
        ("range_3000_3500", 3000.0, 3500.0),
    ]
    for scope, lo, hi in ranges:
        for width in widths:
            sub = merged.copy()
            if width != "all":
                sub = sub[np.isclose(pd.to_numeric(sub["bin_width"], errors="coerce"), float(width))]
            center = pd.to_numeric(sub["freq_center_for_range"], errors="coerce")
            sub = sub[(center >= lo) & (center < hi)]
            if len(sub) == 0:
                continue
            rows.append(
                summarize_bin_subset(
                    sub=sub,
                    scope=scope,
                    bin_width_label=width,
                    low=lo if math.isfinite(lo) else "",
                    high=hi if math.isfinite(hi) else "",
                    center="",
                )
            )

    return pd.DataFrame(rows)


def mode_fisher_comparison(baseline: RunData, split: RunData) -> pd.DataFrame:
    base = baseline.tables.get("mode_diagnostics", pd.DataFrame())
    spl = split.tables.get("mode_diagnostics", pd.DataFrame())
    merged, keys = merge_on_common_keys(
        base,
        spl,
        ["gm_id", "gmonth", "gmonth_label", "l", "n", "mode_id", "window_id", "t_mid"],
        5,
    )
    if merged.empty:
        return pd.DataFrame()

    for suffix in ["baseline", "split"]:
        for col in ["J_nu_nu", "J_lngamma_lngamma", "fisher_cond_mode", "n_freq_used"]:
            full = f"{col}_{suffix}"
            if full not in merged.columns:
                merged[full] = np.nan
            merged[full] = pd.to_numeric(merged[full], errors="coerce")
        merged[f"J_trace_mode_{suffix}"] = merged[f"J_nu_nu_{suffix}"] + merged[f"J_lngamma_lngamma_{suffix}"]

    merged["J_trace_mode_ratio"] = safe_ratio(
        merged["J_trace_mode_split"].to_numpy(),
        merged["J_trace_mode_baseline"].to_numpy(),
    )
    merged["n_freq_used_diff"] = merged["n_freq_used_split"] - merged["n_freq_used_baseline"]

    nu_col = "nu_ref_split" if "nu_ref_split" in merged.columns else "nu_ref_baseline"
    merged["nu_ref_for_range"] = pd.to_numeric(merged.get(nu_col, np.nan), errors="coerce")
    merged["l_numeric"] = pd.to_numeric(merged["l"], errors="coerce")

    rows: List[Dict[str, Any]] = []

    def summarize(sub: pd.DataFrame, scope: str, l_value: Any = "", lo: Any = "", hi: Any = "") -> Dict[str, Any]:
        return {
            "source": "mode_diagnostics",
            "summary_scope": scope,
            "l": l_value,
            "freq_low": lo,
            "freq_high": hi,
            "n_pairs": int(len(sub)),
            "median_J_trace_mode_ratio": finite_median(sub["J_trace_mode_ratio"]),
            "median_fisher_cond_mode_baseline": finite_median(sub["fisher_cond_mode_baseline"]),
            "median_fisher_cond_mode_split": finite_median(sub["fisher_cond_mode_split"]),
            "median_n_freq_used_diff": finite_median(sub["n_freq_used_diff"]),
            "baseline_mode_used_fraction": float(bool_series(sub["mode_used_baseline"]).mean()) if "mode_used_baseline" in sub.columns and len(sub) else math.nan,
            "split_mode_used_fraction": float(bool_series(sub["mode_used_split"]).mean()) if "mode_used_split" in sub.columns and len(sub) else math.nan,
        }

    for l_value, sub in merged.groupby("l_numeric", dropna=False):
        rows.append(summarize(sub, "by_l", int(l_value) if pd.notna(l_value) else ""))

    for scope, lo, hi in [
        ("range_all", -math.inf, math.inf),
        ("range_2000_4500", 2000.0, 4500.0),
        ("range_3000_3500", 3000.0, 3500.0),
    ]:
        center = merged["nu_ref_for_range"]
        sub = merged[(center >= lo) & (center < hi)]
        if len(sub):
            rows.append(summarize(sub, scope, "all", lo if math.isfinite(lo) else "", hi if math.isfinite(hi) else ""))

    return pd.DataFrame(rows)


def quality_flag_comparison(baseline: RunData, split: RunData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    for table_name in TABLE_NAMES:
        base_df = baseline.tables.get(table_name, pd.DataFrame())
        split_df = split.tables.get(table_name, pd.DataFrame())
        base_counter = count_quality_flags(base_df)
        split_counter = count_quality_flags(split_df)
        flags = sorted(set(base_counter) | set(split_counter))
        rows.append(
            {
                "table": table_name,
                "scope": "all",
                "l": "",
                "n": "",
                "quality_flag": "__rows_with_any_flag__",
                "baseline_count": rows_with_any_flag(base_df),
                "split_count": rows_with_any_flag(split_df),
                "delta_split_minus_baseline": rows_with_any_flag(split_df) - rows_with_any_flag(base_df),
                "baseline_rows": int(len(base_df)),
                "split_rows": int(len(split_df)),
            }
        )
        for flag in flags:
            rows.append(
                {
                    "table": table_name,
                    "scope": "all",
                    "l": "",
                    "n": "",
                    "quality_flag": flag,
                    "baseline_count": int(base_counter.get(flag, 0)),
                    "split_count": int(split_counter.get(flag, 0)),
                    "delta_split_minus_baseline": int(split_counter.get(flag, 0) - base_counter.get(flag, 0)),
                    "baseline_rows": int(len(base_df)),
                    "split_rows": int(len(split_df)),
                }
            )

    missing_rows: List[Dict[str, Any]] = []
    base_mode = baseline.tables.get("mode_diagnostics", pd.DataFrame())
    split_mode = split.tables.get("mode_diagnostics", pd.DataFrame())

    def missing_counts_by_l_n(df: pd.DataFrame) -> Counter:
        if df.empty or not {"l", "n", "quality_flags"}.issubset(df.columns):
            return Counter()
        sub = rows_with_flag(df, "missing_splitting")
        out: Counter = Counter()
        for _, row in sub.iterrows():
            out[(normalize_cell(row.get("l", "")), normalize_cell(row.get("n", "")))] += 1
        return out

    base_missing = missing_counts_by_l_n(base_mode)
    split_missing = missing_counts_by_l_n(split_mode)
    for key in sorted(set(base_missing) | set(split_missing)):
        l_value, n_value = key
        rec = {
            "table": "mode_diagnostics",
            "scope": "missing_splitting_by_l_n",
            "l": l_value,
            "n": n_value,
            "quality_flag": "missing_splitting",
            "baseline_count": int(base_missing.get(key, 0)),
            "split_count": int(split_missing.get(key, 0)),
            "delta_split_minus_baseline": int(split_missing.get(key, 0) - base_missing.get(key, 0)),
            "baseline_rows": int(len(base_mode)),
            "split_rows": int(len(split_mode)),
        }
        rows.append(rec)
        missing_rows.append(rec)

    base_stack = baseline.tables.get("stack_diagnostics", pd.DataFrame())
    split_stack = split.tables.get("stack_diagnostics", pd.DataFrame())

    def missing_counts_by_l(df: pd.DataFrame) -> Counter:
        if df.empty or not {"l", "quality_flags"}.issubset(df.columns):
            return Counter()
        sub = rows_with_flag(df, "missing_splitting")
        out: Counter = Counter()
        for _, row in sub.iterrows():
            out[normalize_cell(row.get("l", ""))] += 1
        return out

    base_stack_missing = missing_counts_by_l(base_stack)
    split_stack_missing = missing_counts_by_l(split_stack)
    for l_value in sorted(set(base_stack_missing) | set(split_stack_missing)):
        rows.append(
            {
                "table": "stack_diagnostics",
                "scope": "missing_splitting_by_l",
                "l": l_value,
                "n": "",
                "quality_flag": "missing_splitting",
                "baseline_count": int(base_stack_missing.get(l_value, 0)),
                "split_count": int(split_stack_missing.get(l_value, 0)),
                "delta_split_minus_baseline": int(split_stack_missing.get(l_value, 0) - base_stack_missing.get(l_value, 0)),
                "baseline_rows": int(len(base_stack)),
                "split_rows": int(len(split_stack)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(missing_rows)


def time_series_difference_summary(baseline: RunData, split: RunData) -> pd.DataFrame:
    merged, keys = bin_time_merge(baseline, split)
    if merged.empty:
        return pd.DataFrame()

    for col in ["delta_nu", "delta_lngamma"]:
        for suffix in ["baseline", "split"]:
            full = f"{col}_{suffix}"
            if full not in merged.columns:
                merged[full] = np.nan
            merged[full] = pd.to_numeric(merged[full], errors="coerce")
        merged[f"{col}_diff_split_minus_baseline"] = merged[f"{col}_split"] - merged[f"{col}_baseline"]

    merged["freq_center_for_range"] = frequency_center(merged)
    if "bin_width" not in merged.columns:
        merged["bin_width"] = np.nan

    rows: List[Dict[str, Any]] = []

    def summarize(sub: pd.DataFrame, scope: str, bin_width_label: Any, low: Any = "", high: Any = "", center: Any = "") -> Dict[str, Any]:
        delta_nu_diff = sub["delta_nu_diff_split_minus_baseline"].to_numpy(dtype=float)
        delta_lngamma_diff = sub["delta_lngamma_diff_split_minus_baseline"].to_numpy(dtype=float)
        return {
            "summary_scope": scope,
            "bin_width": bin_width_label,
            "freq_bin_low": low,
            "freq_bin_high": high,
            "freq_bin_center": center,
            "n_pairs": int(len(sub)),
            "delta_nu_diff_median": finite_median(delta_nu_diff),
            "delta_nu_diff_mean": finite_mean(delta_nu_diff),
            "delta_nu_diff_std": finite_std(delta_nu_diff),
            "delta_nu_diff_mad": finite_median(np.abs(delta_nu_diff - finite_median(delta_nu_diff))) if finite_array(delta_nu_diff).size else math.nan,
            "delta_lngamma_diff_median": finite_median(delta_lngamma_diff),
            "delta_lngamma_diff_mean": finite_mean(delta_lngamma_diff),
            "delta_lngamma_diff_std": finite_std(delta_lngamma_diff),
            "delta_lngamma_diff_mad": finite_median(np.abs(delta_lngamma_diff - finite_median(delta_lngamma_diff))) if finite_array(delta_lngamma_diff).size else math.nan,
            "corr_delta_nu_split_vs_baseline": safe_corr(sub["delta_nu_split"], sub["delta_nu_baseline"]),
            "corr_delta_lngamma_split_vs_baseline": safe_corr(sub["delta_lngamma_split"], sub["delta_lngamma_baseline"]),
            "baseline_nonfinite_delta_nu_count": int((~np.isfinite(sub["delta_nu_baseline"].to_numpy(dtype=float))).sum()),
            "split_nonfinite_delta_nu_count": int((~np.isfinite(sub["delta_nu_split"].to_numpy(dtype=float))).sum()),
            "baseline_nonfinite_delta_lngamma_count": int((~np.isfinite(sub["delta_lngamma_baseline"].to_numpy(dtype=float))).sum()),
            "split_nonfinite_delta_lngamma_count": int((~np.isfinite(sub["delta_lngamma_split"].to_numpy(dtype=float))).sum()),
        }

    group_cols = [c for c in ["bin_width", "freq_bin_low", "freq_bin_high", "freq_bin_center"] if c in merged.columns]
    for key, sub in merged.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(group_cols, key))
        rows.append(
            summarize(
                sub=sub,
                scope="frequency_bin",
                bin_width_label=values.get("bin_width", ""),
                low=values.get("freq_bin_low", ""),
                high=values.get("freq_bin_high", ""),
                center=values.get("freq_bin_center", ""),
            )
        )

    widths = ["all"] + sorted(
        [float(x) for x in pd.to_numeric(merged["bin_width"], errors="coerce").dropna().unique()]
    )
    for scope, lo, hi in [
        ("range_all", -math.inf, math.inf),
        ("range_2000_4500", 2000.0, 4500.0),
        ("range_3000_3500", 3000.0, 3500.0),
    ]:
        for width in widths:
            sub = merged.copy()
            if width != "all":
                sub = sub[np.isclose(pd.to_numeric(sub["bin_width"], errors="coerce"), float(width))]
            center = pd.to_numeric(sub["freq_center_for_range"], errors="coerce")
            sub = sub[(center >= lo) & (center < hi)]
            if len(sub):
                rows.append(
                    summarize(
                        sub=sub,
                        scope=scope,
                        bin_width_label=width,
                        low=lo if math.isfinite(lo) else "",
                        high=hi if math.isfinite(hi) else "",
                    )
                )

    return pd.DataFrame(rows)


def write_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def make_plots(
    outdir: Path,
    stack_by_l: pd.DataFrame,
    shift_summary: pd.DataFrame,
    fisher: pd.DataFrame,
    ts_diff: pd.DataFrame,
    logger: logging.Logger,
) -> Dict[str, Any]:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {"matplotlib_available": False, "plots": [], "warnings": []}

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning = f"matplotlib not available; plots skipped: {exc}"
        logger.warning(warning)
        status["warnings"].append(warning)
        return status

    status["matplotlib_available"] = True

    def save_current(name: str) -> None:
        path = plot_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        status["plots"].append(str(path))
        logger.info("Wrote plot: %s", path)

    try:
        if not stack_by_l.empty and {"l", "median_ratio_chosen_to_before"}.issubset(stack_by_l.columns):
            sub = stack_by_l.copy()
            sub["l_numeric"] = pd.to_numeric(sub["l"], errors="coerce")
            sub = sub[np.isfinite(sub["l_numeric"])]
            sub = sub.sort_values("l_numeric")
            plt.figure()
            plt.plot(sub["l_numeric"], pd.to_numeric(sub["median_ratio_chosen_to_before"], errors="coerce"), marker="o")
            plt.axhline(1.0, linestyle="--")
            plt.xlabel("l")
            plt.ylabel("median sharpness_chosen / sharpness_before")
            plt.title("Q1 stack sharpness ratio by l")
            save_current("q1_stack_sharpness_ratio_by_l.png")
    except Exception as exc:
        status["warnings"].append(f"stack sharpness plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    try:
        if not shift_summary.empty and {"l", "chosen_shift_sign", "fraction"}.issubset(shift_summary.columns):
            pivot = shift_summary.pivot_table(
                index="l",
                columns="chosen_shift_sign",
                values="fraction",
                aggfunc="sum",
                fill_value=0.0,
            )
            pivot = pivot.sort_index()
            ax = pivot.plot(kind="bar", stacked=True)
            ax.set_xlabel("l")
            ax.set_ylabel("fraction")
            ax.set_title("Q1 chosen shift sign distribution by l")
            plt.tight_layout()
            path = plot_dir / "q1_chosen_shift_sign_distribution_by_l.png"
            plt.savefig(path, dpi=160)
            plt.close()
            status["plots"].append(str(path))
            logger.info("Wrote plot: %s", path)
    except Exception as exc:
        status["warnings"].append(f"shift sign plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    def plot_fisher_metric(metric: str, ylabel: str, filename: str) -> None:
        if fisher.empty:
            return
        sub = fisher[(fisher.get("source") == "bin_time_series") & (fisher.get("summary_scope") == "frequency_bin")].copy()
        if sub.empty or metric not in sub.columns:
            return
        sub["freq_bin_center_numeric"] = pd.to_numeric(sub.get("freq_bin_center", np.nan), errors="coerce")
        sub["bin_width_numeric"] = pd.to_numeric(sub.get("bin_width", np.nan), errors="coerce")
        sub = sub[np.isfinite(sub["freq_bin_center_numeric"])]
        plt.figure()
        for width, g in sub.groupby("bin_width_numeric", dropna=False):
            g = g.sort_values("freq_bin_center_numeric")
            label = f"bin_width={width:g}" if pd.notna(width) else "bin_width=NA"
            plt.plot(g["freq_bin_center_numeric"], pd.to_numeric(g[metric], errors="coerce"), marker="o", label=label)
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("frequency bin center [uHz]")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        save_current(filename)

    try:
        plot_fisher_metric("median_fisher_trace_ratio", "median fisher_trace ratio split/baseline", "q1_fisher_trace_ratio_by_frequency_bin.png")
    except Exception as exc:
        status["warnings"].append(f"fisher trace plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    try:
        plot_fisher_metric("median_sigma_nu_ratio", "median sigma_nu ratio split/baseline", "q1_sigma_nu_ratio_by_frequency_bin.png")
    except Exception as exc:
        status["warnings"].append(f"sigma_nu plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    try:
        plot_fisher_metric("median_sigma_lngamma_ratio", "median sigma_lngamma ratio split/baseline", "q1_sigma_lngamma_ratio_by_frequency_bin.png")
    except Exception as exc:
        status["warnings"].append(f"sigma_lngamma plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    def plot_ts_metric(metric: str, ylabel: str, filename: str) -> None:
        if ts_diff.empty:
            return
        sub = ts_diff[ts_diff.get("summary_scope") == "frequency_bin"].copy()
        if sub.empty or metric not in sub.columns:
            return
        sub["freq_bin_center_numeric"] = pd.to_numeric(sub.get("freq_bin_center", np.nan), errors="coerce")
        sub["bin_width_numeric"] = pd.to_numeric(sub.get("bin_width", np.nan), errors="coerce")
        sub = sub[np.isfinite(sub["freq_bin_center_numeric"])]
        plt.figure()
        for width, g in sub.groupby("bin_width_numeric", dropna=False):
            g = g.sort_values("freq_bin_center_numeric")
            label = f"bin_width={width:g}" if pd.notna(width) else "bin_width=NA"
            plt.plot(g["freq_bin_center_numeric"], pd.to_numeric(g[metric], errors="coerce"), marker="o", label=label)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("frequency bin center [uHz]")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        save_current(filename)

    try:
        plot_ts_metric("delta_nu_diff_median", "median(delta_nu_split - delta_nu_baseline)", "q1_delta_nu_difference_by_frequency_bin.png")
    except Exception as exc:
        status["warnings"].append(f"delta_nu difference plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    try:
        plot_ts_metric("delta_lngamma_diff_median", "median(delta_lngamma_split - delta_lngamma_baseline)", "q1_delta_lngamma_difference_by_frequency_bin.png")
    except Exception as exc:
        status["warnings"].append(f"delta_lngamma difference plot failed: {exc}")
        logger.warning(status["warnings"][-1])

    return status


def get_single_row_value(df: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if df.empty or column not in df.columns:
        return math.nan
    sub = df[mask]
    if sub.empty:
        return math.nan
    return float(pd.to_numeric(sub.iloc[0:1][column], errors="coerce").iloc[0])


def quality_count(quality_df: pd.DataFrame, table: str, scope: str, flag: str, count_col: str) -> int:
    if quality_df.empty:
        return 0
    sub = quality_df[
        (quality_df["table"].astype(str) == table)
        & (quality_df["scope"].astype(str) == scope)
        & (quality_df["quality_flag"].astype(str) == flag)
    ]
    if sub.empty or count_col not in sub.columns:
        return 0
    return int(pd.to_numeric(sub[count_col], errors="coerce").fillna(0).sum())


def decide_acceptance(
    health_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    stack_overall: pd.DataFrame,
    fisher_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> Dict[str, Any]:
    reasons_fail: List[str] = []
    reasons_warn: List[str] = []
    metrics: Dict[str, Any] = {}

    if not health_df.empty:
        hard_health = health_df[health_df.get("hard_fail", False).astype(bool)]
        if len(hard_health):
            reasons_fail.append(f"hard run-health failures: {len(hard_health)}")
            metrics["hard_health_failures"] = hard_health[["run_label", "metric", "value", "details"]].to_dict(orient="records")

    if not grid_df.empty:
        hard_grid = grid_df[grid_df.get("hard_failure", False).astype(bool)]
        if len(hard_grid):
            reasons_fail.append(f"grid mismatch prevents comparison: {len(hard_grid)}")
            metrics["grid_failures"] = hard_grid[["comparison_name", "table", "details", "baseline_only_keys", "split_only_keys"]].to_dict(orient="records")
    else:
        reasons_fail.append("grid consistency table is empty")

    if stack_overall.empty:
        reasons_fail.append("stack sharpness comparison is empty")
    else:
        lgt = stack_overall[stack_overall["scope"] == "l_gt_0"]
        if lgt.empty or int(pd.to_numeric(lgt["n_pairs"], errors="coerce").fillna(0).iloc[0]) <= 0:
            reasons_warn.append("no l>0 stack-sharpness pairs available")
        else:
            ratio = float(pd.to_numeric(lgt["median_ratio_chosen_to_before"], errors="coerce").iloc[0])
            ratio_base = float(pd.to_numeric(lgt["median_ratio_split_to_baseline_chosen"], errors="coerce").iloc[0])
            frac_lt = float(pd.to_numeric(lgt["fraction_ratio_chosen_to_before_lt_0p95"], errors="coerce").iloc[0])
            metrics["l_gt_0_median_ratio_chosen_to_before"] = ratio
            metrics["l_gt_0_median_ratio_split_to_baseline_chosen"] = ratio_base
            metrics["l_gt_0_fraction_ratio_lt_0p95"] = frac_lt

            if math.isfinite(ratio) and ratio < 0.98:
                reasons_fail.append(f"stack sharpness systematically worsened for l>0: median chosen/before={ratio:.4g}")
            elif not math.isfinite(ratio) or ratio <= 1.0:
                reasons_warn.append(f"weak stack sharpness improvement for l>0: median chosen/before={ratio}")
            if math.isfinite(ratio_base) and ratio_base < 0.98:
                reasons_fail.append(f"split stack is below no-splitting baseline for l>0: median split/baseline={ratio_base:.4g}")
            elif not math.isfinite(ratio_base) or ratio_base <= 1.0:
                reasons_warn.append(f"split stack is not clearly above baseline for l>0: median split/baseline={ratio_base}")
            if math.isfinite(frac_lt) and frac_lt > 0.50:
                reasons_fail.append(f"more than half of l>0 stack pairs have chosen/before < 0.95: fraction={frac_lt:.3f}")

    if fisher_df.empty:
        reasons_fail.append("bin Fisher comparison is empty")
    else:
        f2000 = fisher_df[
            (fisher_df.get("source") == "bin_time_series")
            & (fisher_df.get("summary_scope") == "range_2000_4500")
            & (fisher_df.get("bin_width").astype(str) == "all")
        ]
        if f2000.empty:
            reasons_warn.append("missing 2000-4500 uHz all-bin Fisher summary")
        else:
            row = f2000.iloc[0]
            trace_ratio = float(pd.to_numeric(pd.Series([row.get("median_fisher_trace_ratio", np.nan)]), errors="coerce").iloc[0])
            sigma_nu_ratio = float(pd.to_numeric(pd.Series([row.get("median_sigma_nu_ratio", np.nan)]), errors="coerce").iloc[0])
            sigma_gamma_ratio = float(pd.to_numeric(pd.Series([row.get("median_sigma_lngamma_ratio", np.nan)]), errors="coerce").iloc[0])
            finite_nu = float(pd.to_numeric(pd.Series([row.get("split_fraction_finite_delta_nu", np.nan)]), errors="coerce").iloc[0])
            finite_gamma = float(pd.to_numeric(pd.Series([row.get("split_fraction_finite_delta_lngamma", np.nan)]), errors="coerce").iloc[0])
            metrics["range_2000_4500_median_fisher_trace_ratio"] = trace_ratio
            metrics["range_2000_4500_median_sigma_nu_ratio"] = sigma_nu_ratio
            metrics["range_2000_4500_median_sigma_lngamma_ratio"] = sigma_gamma_ratio
            metrics["range_2000_4500_split_fraction_finite_delta_nu"] = finite_nu
            metrics["range_2000_4500_split_fraction_finite_delta_lngamma"] = finite_gamma

            if math.isfinite(trace_ratio) and trace_ratio < 0.80:
                reasons_fail.append(f"Fisher information strongly degraded in 2000-4500 uHz: trace ratio={trace_ratio:.4g}")
            elif not math.isfinite(trace_ratio) or trace_ratio < 0.95:
                reasons_warn.append(f"Fisher information not clearly preserved in 2000-4500 uHz: trace ratio={trace_ratio}")

            if math.isfinite(sigma_nu_ratio) and sigma_nu_ratio > 1.25:
                reasons_fail.append(f"sigma_nu strongly worsened in 2000-4500 uHz: ratio={sigma_nu_ratio:.4g}")
            elif math.isfinite(sigma_nu_ratio) and sigma_nu_ratio > 1.10:
                reasons_warn.append(f"sigma_nu worsened in 2000-4500 uHz: ratio={sigma_nu_ratio:.4g}")

            if math.isfinite(sigma_gamma_ratio) and sigma_gamma_ratio > 1.25:
                reasons_fail.append(f"sigma_lngamma strongly worsened in 2000-4500 uHz: ratio={sigma_gamma_ratio:.4g}")
            elif math.isfinite(sigma_gamma_ratio) and sigma_gamma_ratio > 1.10:
                reasons_warn.append(f"sigma_lngamma worsened in 2000-4500 uHz: ratio={sigma_gamma_ratio:.4g}")

            if math.isfinite(finite_nu) and finite_nu < 0.50:
                reasons_fail.append(f"delta_nu mostly unusable in split run, 2000-4500 uHz: finite fraction={finite_nu:.3f}")
            elif not math.isfinite(finite_nu) or finite_nu < 0.80:
                reasons_warn.append(f"delta_nu finite fraction is weak in split run, 2000-4500 uHz: {finite_nu}")

            if math.isfinite(finite_gamma) and finite_gamma < 0.50:
                reasons_fail.append(f"delta_lngamma mostly unusable in split run, 2000-4500 uHz: finite fraction={finite_gamma:.3f}")
            elif not math.isfinite(finite_gamma) or finite_gamma < 0.80:
                reasons_warn.append(f"delta_lngamma finite fraction is weak in split run, 2000-4500 uHz: {finite_gamma}")

        f3000 = fisher_df[
            (fisher_df.get("source") == "bin_time_series")
            & (fisher_df.get("summary_scope") == "range_3000_3500")
            & (fisher_df.get("bin_width").astype(str) == "all")
        ]
        if f3000.empty:
            reasons_warn.append("missing 3000-3500 uHz all-bin Fisher summary")
        else:
            row = f3000.iloc[0]
            finite_nu = float(pd.to_numeric(pd.Series([row.get("split_fraction_finite_delta_nu", np.nan)]), errors="coerce").iloc[0])
            finite_gamma = float(pd.to_numeric(pd.Series([row.get("split_fraction_finite_delta_lngamma", np.nan)]), errors="coerce").iloc[0])
            metrics["range_3000_3500_split_fraction_finite_delta_nu"] = finite_nu
            metrics["range_3000_3500_split_fraction_finite_delta_lngamma"] = finite_gamma
            if math.isfinite(min(finite_nu, finite_gamma)) and min(finite_nu, finite_gamma) < 0.50:
                reasons_fail.append(f"3000-3500 uHz mostly unusable: finite fractions nu={finite_nu:.3f}, lngamma={finite_gamma:.3f}")
            elif not math.isfinite(min(finite_nu, finite_gamma)) or min(finite_nu, finite_gamma) < 0.80:
                reasons_warn.append(f"3000-3500 uHz finite usability is weak: nu={finite_nu}, lngamma={finite_gamma}")

    severe_increases: List[Dict[str, Any]] = []
    if not quality_df.empty:
        for flag in sorted(SEVERE_DIAGNOSTIC_FLAGS):
            split_count = int(quality_df.loc[quality_df["quality_flag"] == flag, "split_count"].sum())
            base_count = int(quality_df.loc[quality_df["quality_flag"] == flag, "baseline_count"].sum())
            if split_count > base_count:
                severe_increases.append({"flag": flag, "baseline": base_count, "split": split_count})
        if severe_increases:
            reasons_warn.append(f"new/increased severe diagnostic flags: {severe_increases}")

        base_missing = quality_count(quality_df, "mode_diagnostics", "all", "missing_splitting", "baseline_count")
        split_missing = quality_count(quality_df, "mode_diagnostics", "all", "missing_splitting", "split_count")
        metrics["missing_splitting_mode_diagnostics_baseline"] = base_missing
        metrics["missing_splitting_mode_diagnostics_split"] = split_missing
        if base_missing > 0 and split_missing >= base_missing:
            reasons_warn.append(
                "missing_splitting was not reduced in mode_diagnostics; this may be expected for l=0 or unavailable coefficients, but Q1 should inspect q1_missing_splitting_by_l_n.csv"
            )

    if reasons_fail:
        category = "FAIL"
    elif reasons_warn:
        category = "WARN"
    else:
        category = "PASS"

    return {
        "acceptance_category": category,
        "fail_reasons": reasons_fail,
        "warn_reasons": reasons_warn,
        "metrics": metrics,
        "note": "This Q1 result is not a lag result and is not a complete WMLTP validation.",
    }


def write_report(
    outdir: Path,
    acceptance: Mapping[str, Any],
    baseline: RunData,
    split: RunData,
    output_paths: Mapping[str, str],
    plot_status: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# WMLTP-Q1 splitting validation report")
    lines.append("")
    lines.append(f"Generated: `{utc_now_iso()}`")
    lines.append("")
    lines.append(f"Acceptance category: **{acceptance.get('acceptance_category', 'UNKNOWN')}**")
    lines.append("")
    lines.append("## Run directories")
    lines.append("")
    lines.append(f"- baseline: `{baseline.run_dir}`")
    lines.append(f"- split: `{split.run_dir}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report compares an existing no-splitting AB run with an existing splitting-corrected AB run.")
    lines.append("It does not run AB, does not submit jobs, and does not perform lag inference.")
    lines.append("")
    lines.append("## Acceptance reasoning")
    lines.append("")
    fail_reasons = list(acceptance.get("fail_reasons", []))
    warn_reasons = list(acceptance.get("warn_reasons", []))
    if fail_reasons:
        lines.append("### Fail reasons")
        lines.append("")
        for reason in fail_reasons:
            lines.append(f"- {reason}")
        lines.append("")
    if warn_reasons:
        lines.append("### Warning reasons")
        lines.append("")
        for reason in warn_reasons:
            lines.append(f"- {reason}")
        lines.append("")
    if not fail_reasons and not warn_reasons:
        lines.append("No fail or warning reasons were triggered by the scripted criteria.")
        lines.append("")

    lines.append("## Key metrics")
    lines.append("")
    metrics = acceptance.get("metrics", {})
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No scalar metrics available.")
    lines.append("")

    lines.append("## Output tables")
    lines.append("")
    for name, path in output_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")

    lines.append("## Plot status")
    lines.append("")
    lines.append(f"- matplotlib_available: `{plot_status.get('matplotlib_available')}`")
    for warning in plot_status.get("warnings", []):
        lines.append(f"- warning: {warning}")
    for path in plot_status.get("plots", []):
        lines.append(f"- plot: `{path}`")
    lines.append("")

    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("This is a Q1 shift-and-stack / score-Fisher diagnostics validation only.")
    lines.append("Do not interpret `delta_nu` / `delta_lngamma` differences here as a lag result.")
    lines.append("Do not claim independent GONG verification from this report.")

    report_path = outdir / "Q1_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def compare_runs(args: argparse.Namespace, logger: logging.Logger) -> None:
    outdir = Path(args.outdir)
    table_dir = outdir / "tables"

    baseline = read_run(args.baseline_run, "baseline", logger)
    split = read_run(args.split_run, "split", logger)

    health_base, _ = health_summary_for_run(baseline)
    health_split, _ = health_summary_for_run(split)
    health_df = pd.concat([health_base, health_split], ignore_index=True)

    grid_df = grid_consistency(baseline, split)
    stack_by_l, stack_overall, shift_summary = stack_sharpness_summaries(baseline, split)
    fisher_df = bin_fisher_comparison(baseline, split)
    mode_fisher_df = mode_fisher_comparison(baseline, split)
    quality_df, missing_l_n_df = quality_flag_comparison(baseline, split)
    ts_diff_df = time_series_difference_summary(baseline, split)

    output_paths: Dict[str, str] = {}
    output_paths["q1_run_health_summary.csv"] = write_df(health_df, table_dir / "q1_run_health_summary.csv")
    output_paths["q1_grid_consistency.csv"] = write_df(grid_df, table_dir / "q1_grid_consistency.csv")
    output_paths["q1_stack_sharpness_by_l.csv"] = write_df(stack_by_l, table_dir / "q1_stack_sharpness_by_l.csv")
    output_paths["q1_stack_sharpness_overall.csv"] = write_df(stack_overall, table_dir / "q1_stack_sharpness_overall.csv")
    output_paths["q1_bin_fisher_comparison.csv"] = write_df(fisher_df, table_dir / "q1_bin_fisher_comparison.csv")
    output_paths["q1_mode_fisher_comparison.csv"] = write_df(mode_fisher_df, table_dir / "q1_mode_fisher_comparison.csv")
    output_paths["q1_quality_flag_comparison.csv"] = write_df(quality_df, table_dir / "q1_quality_flag_comparison.csv")
    output_paths["q1_time_series_difference_summary.csv"] = write_df(ts_diff_df, table_dir / "q1_time_series_difference_summary.csv")
    output_paths["q1_shift_sign_summary.csv"] = write_df(shift_summary, table_dir / "q1_shift_sign_summary.csv")
    output_paths["q1_missing_splitting_by_l_n.csv"] = write_df(missing_l_n_df, table_dir / "q1_missing_splitting_by_l_n.csv")

    plot_status = make_plots(
        outdir=outdir,
        stack_by_l=stack_by_l,
        shift_summary=shift_summary,
        fisher=fisher_df,
        ts_diff=ts_diff_df,
        logger=logger,
    )

    acceptance = decide_acceptance(
        health_df=health_df,
        grid_df=grid_df,
        stack_overall=stack_overall,
        fisher_df=fisher_df,
        quality_df=quality_df,
    )

    summary = {
        **acceptance,
        "timestamp": utc_now_iso(),
        "baseline_run": str(baseline.run_dir),
        "split_run": str(split.run_dir),
        "baseline_table_paths": baseline.table_paths,
        "split_table_paths": split.table_paths,
        "baseline_checkpoint_count": len(baseline.checkpoints),
        "split_checkpoint_count": len(split.checkpoints),
        "baseline_metadata_path": baseline.metadata.get("_metadata_path", ""),
        "split_metadata_path": split.metadata.get("_metadata_path", ""),
        "outputs": output_paths,
        "plot_status": plot_status,
        "q1_question": "Does WMLTP Legendre splitting correction improve stored-nonnegative-m GONG MRVMT shift-and-stack and downstream score/Fisher diagnostics relative to no-splitting baseline?",
    }

    summary_path = outdir / "q1_acceptance_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=json_default)
    output_paths["q1_acceptance_summary.json"] = str(summary_path)

    report_path = write_report(
        outdir=outdir,
        acceptance=acceptance,
        baseline=baseline,
        split=split,
        output_paths=output_paths,
        plot_status=plot_status,
    )
    output_paths["Q1_REPORT.md"] = report_path

    logger.info("Q1 acceptance category: %s", acceptance.get("acceptance_category"))
    logger.info("Q1 acceptance summary: %s", summary_path)
    logger.info("Q1 report: %s", report_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    prepare_outdir(outdir, overwrite=args.overwrite, mode_name="commands" if args.make_commands else "compare")
    logger = setup_logging(outdir, args.log_level)

    logger.info("Q1 command line: %s", " ".join([sys.argv[0]] + sys.argv[1:]))
    logger.info("Q1 mode: %s", "make-commands" if args.make_commands else "compare-only")

    try:
        if args.make_commands:
            write_q1_commands(args, logger)
        elif args.compare_only:
            compare_runs(args, logger)
        logger.info("Q1 script completed")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Q1 fatal error: %s", exc)
        logger.exception("Traceback:")
        failed_path = outdir / "q1_acceptance_summary.failed.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "acceptance_category": "FAIL",
                    "fatal_error": str(exc),
                    "timestamp": utc_now_iso(),
                    "note": "The Q1 validation script failed before completing comparison outputs.",
                },
                f,
                indent=2,
                sort_keys=True,
                default=json_default,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
