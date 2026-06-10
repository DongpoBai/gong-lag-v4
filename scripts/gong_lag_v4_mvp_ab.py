#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gong_lag_v4_mvp_ab.py

Self-contained command-line entrypoint for GONG-Lag v4 MVP-A/B.

Scope implemented here:
  MVP-A:
    - stored-nonnegative-m extraction
    - shift-and-stack diagnostic
    - symmetric Lorentzian reference
    - score/Fisher bin-level delta_nu and delta_lngamma

  MVP-B:
    - 9-day windows
    - 3-day stride
    - FITS audit
    - checkpoint/restart
    - metadata
    - diagnostics

Not implemented here:
    - HMI-to-GONG degradation
    - complete Paper-1 lag analysis
    - cross-spectral phase
    - GPCC
    - full WMLTP leakage model
    - temporal-window convolution
    - negative-m reconstruction
    - Nigam-Kosovichev asymmetry in the estimator

Important HPC recommendation before running:
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

Shift sign convention used in this script:
    raw_shift = L * sum_k a_k * P_k(m / L), where L^2 = l(l+1).
    For a target central frequency nu0, "plus" evaluates P_m at
        nu_grid + raw_shift
    before averaging over stored m, which corresponds to corrected frequency
        nu_corrected = nu_observed - raw_shift.
    "minus" evaluates P_m at nu_grid - raw_shift.
    "none" does no frequency shift.

This is an MVP score/Fisher pipeline, not the Rhodes/WMLTP Fortran production
peak-fitting pipeline.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import concurrent.futures
import datetime as _dt
import importlib
import json
import logging
import math
import platform
import re
import socket
import sys
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for gong_lag_v4_mvp_ab.py") from exc

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


QUALITY_FLAGS = [
    "bad_fits_shape",
    "bad_window_row",
    "low_fill",
    "missing_mode_table",
    "missing_splitting",
    "bad_reference_model",
    "fisher_singular",
    "fisher_ill_conditioned",
    "low_fisher_nu",
    "low_fisher_gamma",
    "edge_of_fit_band",
    "no_modes_in_bin",
    "no_valid_windows",
    "fft_failed",
    "parsing_failed",
    "task_exception",
]

AUDIT_COLUMNS = [
    "gm_id", "gmonth_label", "l_requested", "path", "exists", "file_size_bytes",
    "shape", "expected_shape", "shape_ok", "naxis", "bitpix", "l_header",
    "lmax", "cd1_1", "ctype1", "ts_start", "ts_end", "gmonth_header",
    "dtype", "mtype", "origin", "ts_neff", "gts_neff", "window_component",
    "window_binary", "window_fill", "window_unique_values", "n_time", "cadence_s",
    "quality_flags", "rejection_reason",
]

WINDOW_COLUMNS = [
    "gm_id", "gmonth", "gmonth_label", "l", "path", "window_id", "window_index",
    "start_sample", "end_sample", "t_start", "t_mid", "t_end", "window_days",
    "stride_days", "cadence_s", "n_samples", "n_good", "fill", "accepted",
    "quality_flags",
]

MODE_DIAG_COLUMNS = [
    "gm_id", "gmonth", "gmonth_label", "l", "n", "mode_id", "window_id",
    "t_start", "t_mid", "t_end", "fill", "m_count", "m_min", "m_max",
    "m_weighting", "splitting_file", "splitting_k_used", "shift_sign_policy",
    "chosen_shift_sign", "sharpness_before", "sharpness_after_plus",
    "sharpness_after_minus", "sharpness_chosen", "stack_sharpness", "nu_ref",
    "gamma_ref", "H_ref_used", "B_ref_used", "H_source", "B_source", "q_nu",
    "q_lngamma", "J_nu_nu", "J_nu_lngamma", "J_lngamma_lngamma",
    "fisher_cond_mode", "n_freq_used", "mode_used", "quality_flags",
]

STACK_DIAG_COLUMNS = [
    "gm_id", "gmonth", "gmonth_label", "l", "window_id", "t_mid",
    "n_modes_for_sign", "shift_sign_policy", "chosen_shift_sign", "sharpness_before",
    "sharpness_after_plus", "sharpness_after_minus", "sharpness_chosen", "splitting_file",
    "splitting_missing", "quality_flags",
]

EXAMPLE_STACK_COLUMNS = [
    "gm_id", "gmonth", "gmonth_label", "l", "n", "window_id", "t_mid",
    "chosen_shift_sign", "freq_uhz", "power",
]

BIN_COLUMNS = [
    "t_start", "t_mid", "t_end", "gm_id", "gmonth", "l_list", "freq_bin_low",
    "freq_bin_high", "freq_bin_center", "bin_width", "delta_nu", "delta_lngamma",
    "cov_nu_nu", "cov_lngamma_lngamma", "cov_nu_lngamma", "sigma_nu",
    "sigma_lngamma", "corr_nu_lngamma", "n_modes_total", "n_modes_used",
    "n_modes_rejected", "fisher_trace", "fisher_det", "fisher_cond", "fill_median",
    "fill_min", "quality_flags",
]

TASK_ERROR_COLUMNS = [
    "timestamp", "gm_id", "gmonth_label", "l", "path", "stage", "status",
    "exception_type", "exception_message", "traceback", "error_file",
    "checkpoint", "quality_flags",
]


@dataclass
class SplittingData:
    path: str = ""
    parsed: bool = False
    records: Dict[Tuple[int, int], Dict[int, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def time_gmtime(*args):
    import time
    return time.gmtime(*args)


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def quality_join(flags: Iterable[str]) -> str:
    return ";".join(sorted(set(str(f).strip() for f in flags if str(f).strip())))


def quality_split(flag_string: Any) -> List[str]:
    if flag_string is None:
        return []
    text = str(flag_string).strip()
    if not text or text.lower() == "nan":
        return []
    return [x for x in text.split(";") if x]


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        value = float(x)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        return int(float(x))
    except Exception:
        return default


def parse_datetime_utc(value: Any) -> Optional[_dt.datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if pd is not None:
        try:
            t = pd.to_datetime(text, utc=True, errors="coerce")
            if pd.notna(t):
                return t.to_pydatetime()
        except Exception:
            pass
    for candidate in [text, text.replace("Z", "+00:00"), text.replace("/", "-").replace("Z", "+00:00")]:
        try:
            dt = _dt.datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return dt.astimezone(_dt.timezone.utc)
        except Exception:
            continue
    return None


def format_datetime(dt: Optional[_dt.datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc).isoformat()


def parse_l_list(spec: str) -> List[int]:
    if spec is None:
        raise ValueError("--target-l-list is required")
    out: List[int] = []
    for token in re.split(r"[,\s]+", str(spec).strip()):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(token))
    unique_sorted = sorted(set(out))
    if not unique_sorted:
        raise ValueError("target_l_list parsed to an empty list")
    for ell in unique_sorted:
        if ell < 0:
            raise ValueError(f"negative l is invalid: {ell}")
    return unique_sorted


def parse_bin_widths(spec: str) -> List[float]:
    widths = []
    for token in re.split(r"[,\s]+", str(spec).strip()):
        if not token:
            continue
        width = float(token)
        if width <= 0:
            raise ValueError(f"bin width must be positive: {token}")
        widths.append(width)
    return widths or [250.0, 500.0]


def require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for table I/O in this MVP script")


def get_fits_module():
    try:
        from astropy.io import fits
        return fits
    except Exception as exc:
        raise RuntimeError("astropy is required for FITS I/O. Load/install astropy on Discovery before running.") from exc


def package_versions() -> Dict[str, str]:
    modules = ["numpy", "pandas", "astropy", "matplotlib", "pyarrow", "fastparquet"]
    versions: Dict[str, str] = {}
    for name in modules:
        try:
            mod = importlib.import_module(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "not_available"
    return versions


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    return str(obj)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GONG-Lag v4 MVP-A/B MRVMT -> score/Fisher bin time series entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mrvmt-root", required=True, help="MRVMT root directory")
    parser.add_argument("--mode-table", required=True, help="CSV/parquet mode table")
    parser.add_argument("--splitting-file", required=True, help="WMLTP-style splitting file; use 'none' to force shift_sign=none")
    parser.add_argument("--target-l-list", required=True, help="Comma/range list, e.g. '50,75,100-105'")
    parser.add_argument("--max-gms", type=int, default=0, help="Maximum GM directories to process; 0 means all discovered")
    parser.add_argument("--nproc", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--window-days", type=float, default=9.0)
    parser.add_argument("--stride-days", type=float, default=3.0)
    parser.add_argument("--fill-min", type=float, default=0.40)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite checkpoints and outputs")
    parser.add_argument("--dry-run", action="store_true", help="Plan work without FITS array reads or FFT")
    parser.add_argument("--audit-only", action="store_true", help="Read FITS headers/window rows and write audit/window tables only")
    parser.add_argument("--shift-sign", choices=["auto", "plus", "minus", "none"], default="auto")
    parser.add_argument("--bin-widths", default="250,500", help="Comma-separated bin widths in microHz")
    parser.add_argument("--freq-min", type=float, default=1000.0)
    parser.add_argument("--freq-max", type=float, default=4500.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)
    if args.dry_run and args.audit_only:
        parser.error("--dry-run and --audit-only are mutually exclusive")
    if args.nproc < 1:
        parser.error("--nproc must be >= 1")
    if args.window_days <= 0:
        parser.error("--window-days must be > 0")
    if args.stride_days <= 0:
        parser.error("--stride-days must be > 0")
    if not (0.0 <= args.fill_min <= 1.0):
        parser.error("--fill-min must be in [0, 1]")
    if args.freq_max <= args.freq_min:
        parser.error("--freq-max must exceed --freq-min")
    parse_l_list(args.target_l_list)
    parse_bin_widths(args.bin_widths)
    return args


def setup_logging(outdir: str, log_level: str = "INFO") -> Tuple[logging.Logger, str]:
    log_dir = Path(outdir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"gong_lag_v4_mvp_ab_{stamp}.log"

    logger = logging.getLogger("gong_lag_v4_mvp_ab")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(processName)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time_gmtime
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, log_level.upper()))
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.info("Logging initialized: %s", log_path)
    return logger, str(log_path)


def write_metadata(args: argparse.Namespace, outdir: str, log_path: str, extra: Optional[Mapping[str, Any]] = None, filename: str = "metadata.json") -> str:
    metadata: Dict[str, Any] = {
        "command_line": " ".join([sys.argv[0]] + sys.argv[1:]),
        "timestamp": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "package_versions": package_versions(),
        "mrvmt_root": args.mrvmt_root,
        "mode_table": args.mode_table,
        "splitting_file": args.splitting_file,
        "target_l_list": parse_l_list(args.target_l_list),
        "max_gms": args.max_gms,
        "nproc": args.nproc,
        "all_m_policy": "stored_nonnegative",
        "negative_m_reconstructed": False,
        "window_days": args.window_days,
        "stride_days": args.stride_days,
        "fill_min": args.fill_min,
        "shift_sign_policy": args.shift_sign,
        "chosen_shift_signs": {},
        "profile": "symmetric_lorentzian",
        "leakage_enabled": False,
        "temporal_window_convolution_enabled": False,
        "estimator": "score_fisher_exponential_periodogram",
        "m_weighting": "uniform",
        "bin_widths": parse_bin_widths(args.bin_widths),
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "dry_run": bool(args.dry_run),
        "audit_only": bool(args.audit_only),
        "overwrite": bool(args.overwrite),
        "log_path": log_path,
        "notes": "MVP-A/B only; not full WMLTP; not final lag verification",
    }
    if extra:
        metadata.update(dict(extra))
    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=json_default)
    return str(out_path)


def discover_mrvmt_files(mrvmt_root: str, target_l_list: Sequence[int], max_gms: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    root = Path(mrvmt_root)
    tasks: List[Dict[str, Any]] = []
    if not root.exists():
        logger.warning("MRVMT root does not exist on this runtime: %s", root)
        return tasks
    gm_dirs = sorted([p for p in root.iterdir() if p.is_dir() and re.match(r"^mrvmt\d{6,8}$", p.name)], key=lambda p: p.name)
    if max_gms and max_gms > 0:
        gm_dirs = gm_dirs[:max_gms]
    if not gm_dirs:
        logger.warning("No mrvmtYYMMDD directories discovered under %s", root)
        return tasks
    for gm_dir in gm_dirs:
        match = re.match(r"^mrvmt(\d{6,8})$", gm_dir.name)
        if not match:
            continue
        label = match.group(1)
        gm_id = gm_dir.name
        for ell in target_l_list:
            path = gm_dir / f"mrvmt{label}d{ell:03d}.fits"
            if path.exists():
                tasks.append({"gm_id": gm_id, "gmonth_label": label, "l": int(ell), "path": str(path)})
            else:
                logger.warning("Missing MRVMT file for gm=%s l=%03d: %s", gm_id, ell, path)
    logger.info("Discovered %d existing MRVMT FITS files across %d GM directories for %d requested l values", len(tasks), len(gm_dirs), len(target_l_list))
    return tasks


def _normalize_column_name(name: Any) -> str:
    text = str(name).strip().replace("ℓ", "l").replace("ν", "nu").replace("Γ", "gamma")
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def _infer_no_header_mode_table(df: "pd.DataFrame") -> "pd.DataFrame":
    ncol = df.shape[1]
    if ncol < 3:
        raise ValueError("no-header mode table needs at least 3 columns")
    out = df.copy()
    if ncol >= 6:
        out = out.iloc[:, :6].copy()
        out.columns = ["l", "n", "nu_ref", "gamma_ref", "dnu_dl", "dlnA_dl"]
        return out
    c0 = pd.to_numeric(out.iloc[:, 0], errors="coerce")
    c1 = pd.to_numeric(out.iloc[:, 1], errors="coerce")
    c0max = float(c0.max(skipna=True)) if c0.notna().any() else math.nan
    c1max = float(c1.max(skipna=True)) if c1.notna().any() else math.nan
    first_two = ["l", "n"] if math.isfinite(c0max) and math.isfinite(c1max) and c0max > c1max and c0max > 60 else ["n", "l"]
    if ncol >= 4:
        out = out.iloc[:, :4].copy()
        out.columns = first_two + ["nu_ref", "gamma_ref"]
    else:
        out = out.iloc[:, :3].copy()
        out.columns = first_two + ["nu_ref"]
    return out


def _normalize_mode_dataframe(df: "pd.DataFrame") -> "pd.DataFrame":
    require_pandas()
    if all(isinstance(c, int) for c in df.columns):
        work = _infer_no_header_mode_table(df)
    else:
        work = df.copy()
        normalized = {_normalize_column_name(c): c for c in work.columns}
        aliases = {
            "n": ["n", "radial_order", "order", "n_order", "nmode"],
            "l": ["l", "ell", "degree", "l_value", "target_l", "harmonic_degree"],
            "m": ["m", "azimuthal_order", "m_value"],
            "nu_ref": ["nu_ref", "freq", "frequency", "nu", "freq_uhz", "nu_uhz", "nu0", "nu0_uhz"],
            "gamma_ref": ["gamma_ref", "width", "gamma", "fwhm", "w", "width_uhz", "line_width"],
            "H_ref": ["h_ref", "height", "h", "amp", "amplitude", "a_ref"],
            "B_ref": ["b_ref", "background", "bkg", "bg", "background_ref"],
            "bad": ["bad", "flag", "quality_flag", "reject", "rejected"],
        }
        rename: Dict[Any, str] = {}
        for target, names in aliases.items():
            for name in names:
                key = _normalize_column_name(name)
                if key in normalized:
                    rename[normalized[key]] = target
                    break
        work = work.rename(columns=rename)
        if not {"n", "l", "nu_ref"}.issubset(set(work.columns)):
            raise ValueError("mode table is missing required n/l/nu columns")
    if "gamma_ref" not in work.columns:
        raise ValueError("mode table is missing gamma_ref/width/fwhm column")
    original_rows = int(len(work))
    for col in ["n", "l", "m", "bad", "nu_ref", "gamma_ref", "H_ref", "B_ref"]:
        if col not in work.columns:
            work[col] = np.nan
    for col in ["n", "l", "m", "bad"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
    for col in ["nu_ref", "gamma_ref", "H_ref", "B_ref"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # A common input table contains columns like
    #     gm_label,n,l,m,nu,dnu,fwhm,...,bad
    # i.e. many epochs for the same (n,l).  This script needs a reference
    # mode list, not thousands of epoch duplicates.  Keep the zonal reference
    # row when an m column is present, drop explicit bad rows when available,
    # then collapse duplicates by (l,n) with numeric medians.
    if work["m"].notna().any():
        work = work[(work["m"].isna()) | (work["m"] == 0)].copy()
    if work["bad"].notna().any():
        work = work[(work["bad"].isna()) | (work["bad"] == 0)].copy()
    work = work.dropna(subset=["n", "l", "nu_ref", "gamma_ref"]).copy()
    work["n"] = work["n"].astype(int)
    work["l"] = work["l"].astype(int)
    work = work[(work["l"] >= 0) & (work["l"] <= 10000) & (work["n"] >= 0)].copy()
    grouped = (
        work.groupby(["l", "n"], as_index=False, dropna=False)
        .agg({"nu_ref": "median", "gamma_ref": "median", "H_ref": "median", "B_ref": "median"})
    )
    grouped = grouped.dropna(subset=["nu_ref", "gamma_ref"]).copy()
    grouped["n"] = grouped["n"].astype(int)
    grouped["l"] = grouped["l"].astype(int)
    grouped.attrs["normalization"] = {
        "input_rows": original_rows,
        "after_m_bad_valid_filter_rows": int(len(work)),
        "unique_l_n_rows": int(len(grouped)),
        "collapsed_duplicate_l_n_rows": int(max(len(work) - len(grouped), 0)),
        "dedupe_policy": "median_by_l_n_after_optional_m0_and_bad0_filters",
    }
    return grouped[["n", "l", "nu_ref", "gamma_ref", "H_ref", "B_ref"]].sort_values(["l", "nu_ref", "n"])


def filter_mode_table_for_targets(mode_table: "pd.DataFrame", target_l_list: Sequence[int], freq_min: float, freq_max: float, logger: logging.Logger) -> "pd.DataFrame":
    require_pandas()
    target_set = {int(x) for x in target_l_list}
    work = mode_table.copy()
    work["l"] = pd.to_numeric(work["l"], errors="coerce").astype("Int64")
    work["nu_ref"] = pd.to_numeric(work["nu_ref"], errors="coerce")
    work = work.dropna(subset=["l", "nu_ref"]).copy()
    work["l"] = work["l"].astype(int)
    filt = work[
        work["l"].isin(target_set)
        & (work["nu_ref"] >= float(freq_min) - 500.0)
        & (work["nu_ref"] <= float(freq_max) + 500.0)
    ].copy()
    observed_l = sorted(set(int(x) for x in filt["l"].dropna().tolist())) if len(filt) else []
    unexpected = sorted(set(observed_l) - target_set)
    if unexpected:
        raise ValueError(f"target-l filter failed: unexpected l values after filter: {unexpected}")
    counts = {int(k): int(v) for k, v in filt.groupby("l", dropna=False).size().to_dict().items()} if len(filt) else {}
    missing = sorted(target_set - set(observed_l))
    logger.info("Mode table after target-l/frequency filter: rows=%d counts_by_l=%s missing_l=%s", len(filt), counts, missing)
    if len(filt) > max(100 * max(len(target_set), 1), 1000):
        logger.warning("Mode table still has many rows after l/frequency filtering (%d rows for %d target l values). Check whether the input table has non-mode duplicates.", len(filt), len(target_set))
    return filt.sort_values(["l", "nu_ref", "n"]).reset_index(drop=True)


def load_mode_table(mode_table_path: str, logger: logging.Logger) -> "pd.DataFrame":
    require_pandas()
    path = Path(mode_table_path)
    if not path.exists():
        raise FileNotFoundError(f"missing_mode_table: {path}")
    attempts: List[Tuple[str, Dict[str, Any]]] = []
    if path.suffix.lower() in [".parquet", ".pq"]:
        attempts.append(("parquet", {}))
    attempts.extend([
        ("csv_default", {"comment": "#"}),
        ("csv_comma", {"sep": ",", "comment": "#"}),
        ("csv_whitespace_header", {"sep": r"\s+", "engine": "python", "comment": "#"}),
        ("csv_whitespace_no_header", {"sep": r"\s+", "engine": "python", "comment": "#", "header": None}),
    ])
    errors: List[str] = []
    for label, kwargs in attempts:
        try:
            df = pd.read_parquet(path) if label == "parquet" else pd.read_csv(path, **kwargs)
            norm = _normalize_mode_dataframe(df)
            if len(norm) == 0:
                raise ValueError("normalized mode table is empty")
            meta = norm.attrs.get("normalization", {}) if hasattr(norm, "attrs") else {}
            logger.info("Loaded mode table using %s: %s normalized (l,n) rows from %s; normalization=%s", label, len(norm), path, meta)
            missing_h = int(norm["H_ref"].isna().sum())
            missing_b = int(norm["B_ref"].isna().sum())
            if missing_h or missing_b:
                logger.warning("Mode table lacks H_ref/B_ref for some modes: H missing=%d, B missing=%d; local stacked spectra will estimate missing values.", missing_h, missing_b)
            return norm
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise ValueError("missing_mode_table/parsing_failed: " + " | ".join(errors))


def _parse_numeric_tokens(line: str) -> Optional[List[float]]:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    text = text.split("#", 1)[0].strip().replace("D", "E").replace("d", "E")
    values = []
    for token in text.split():
        try:
            values.append(float(token))
        except Exception:
            return None
    return values or None


def _is_nearly_int(value: float, tol: float = 1e-6) -> bool:
    return math.isfinite(value) and abs(value - round(value)) < tol


def load_splitting_file(splitting_file: str, logger: logging.Logger) -> SplittingData:
    data = SplittingData(path=splitting_file)
    if splitting_file is None or str(splitting_file).strip().lower() in {"", "none", "null"}:
        data.quality_flags = ["missing_splitting"]
        data.warnings.append("No splitting file was provided; shift_sign will be none.")
        logger.warning(data.warnings[-1])
        return data
    path = Path(splitting_file)
    if not path.exists():
        data.quality_flags = ["missing_splitting"]
        data.warnings.append(f"Splitting file does not exist: {path}")
        logger.warning(data.warnings[-1])
        return data
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception as exc:
        data.quality_flags = ["missing_splitting", "parsing_failed"]
        data.warnings.append(f"Could not read splitting file {path}: {exc}")
        logger.warning(data.warnings[-1])
        return data
    numeric_lines = [(idx + 1, values) for idx, line in enumerate(lines) if (values := _parse_numeric_tokens(line)) is not None]
    records: Dict[Tuple[int, int], Dict[int, float]] = {}
    i = 1 if numeric_lines and len(numeric_lines[0][1]) == 1 else 0
    while i <= len(numeric_lines) - 3:
        _, header = numeric_lines[i]
        _, coeffs = numeric_lines[i + 1]
        _, errs = numeric_lines[i + 2]
        if len(header) >= 4 and _is_nearly_int(header[0]) and _is_nearly_int(header[1]) and _is_nearly_int(header[2]):
            ell = int(round(header[0]))
            n = int(round(header[1]))
            kmax = int(round(header[2]))
            if kmax > 0 and len(coeffs) >= kmax and len(errs) >= min(kmax, len(errs)):
                records[(ell, n)] = {k: float(coeffs[k - 1]) for k in range(1, kmax + 1)}
                i += 3
                continue
        i += 1
    if not records and numeric_lines:
        width_counts = Counter(len(vals) for _, vals in numeric_lines)
        common_width, _ = width_counts.most_common(1)[0]
        rows = [vals for _, vals in numeric_lines if len(vals) == common_width]
        if common_width >= 5 and rows:
            for vals in rows:
                if not (_is_nearly_int(vals[0]) and _is_nearly_int(vals[1])):
                    continue
                c0 = int(round(vals[0])); c1 = int(round(vals[1]))
                ell, n = (c0, c1) if c0 > c1 and c0 > 60 else (c1, c0)
                coeff_values = vals[2:]
                rec = {1: coeff_values[0], 3: coeff_values[1], 5: coeff_values[2]} if len(coeff_values) == 3 else {k + 1: coeff_values[k] for k in range(len(coeff_values))}
                records[(ell, n)] = {int(k): float(v) for k, v in rec.items()}
    if records:
        max_abs = max(abs(v) for rec in records.values() for v in rec.values())
        data.parsed = True
        data.records = records
        data.metadata = {"n_records": len(records), "k_values_seen": sorted(set(k for rec in records.values() for k in rec.keys())), "max_abs_coefficient_assumed_uhz": max_abs}
        if max_abs > 100:
            data.warnings.append("Splitting coefficients have large absolute values; script assumes coefficients are already in microHz. Verify units before science use.")
            logger.warning(data.warnings[-1])
        logger.info("Parsed splitting file %s with %d mode records; k seen=%s", path, len(records), data.metadata["k_values_seen"])
        return data
    data.quality_flags = ["missing_splitting", "parsing_failed"]
    data.warnings.append(f"Could not parse any splitting coefficients from {path}")
    logger.warning(data.warnings[-1])
    return data


def legendre_p(k: int, x: np.ndarray) -> np.ndarray:
    if k == 0:
        return np.ones_like(x, dtype=np.float64)
    if k == 1:
        return x.astype(np.float64)
    p0 = np.ones_like(x, dtype=np.float64)
    p1 = x.astype(np.float64)
    for n in range(1, k):
        p2 = ((2 * n + 1) * x * p1 - n * p0) / (n + 1)
        p0, p1 = p1, p2
    return p1


def get_splitting_coeffs(splitting: SplittingData, ell: int, n: int) -> Tuple[Dict[int, float], List[int], List[str]]:
    if not splitting.parsed or not splitting.records:
        return {}, [], ["missing_splitting"]
    rec = splitting.records.get((int(ell), int(n)))
    if rec is None:
        return {}, [], ["missing_splitting"]
    k_used = [k for k in [1, 3, 5] if k in rec and math.isfinite(float(rec[k]))]
    return rec, k_used, [] if k_used else ["missing_splitting"]


def compute_splitting_shift(ell: int, m_values: np.ndarray, coeffs: Mapping[int, float], k_used: Sequence[int]) -> np.ndarray:
    if ell <= 0 or not k_used:
        return np.zeros_like(m_values, dtype=np.float64)
    L = math.sqrt(float(ell) * float(ell + 1))
    x = m_values.astype(np.float64) / L
    shift = np.zeros_like(x, dtype=np.float64)
    for k in k_used:
        shift += float(coeffs[k]) * legendre_p(int(k), x)
    return L * shift


def _shape_from_header(header: Mapping[str, Any]) -> Optional[Tuple[int, int, int]]:
    try:
        return (int(header.get("NAXIS3")), int(header.get("NAXIS2")), int(header.get("NAXIS1"))) if int(header.get("NAXIS", 0)) == 3 else None
    except Exception:
        return None


def _safe_header_value(header: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = header.get(key, default)  # type: ignore[attr-defined]
        return default if value is None else value
    except Exception:
        return default


def _repair_known_mrvmt_header_cards(header: Any) -> List[str]:
    """Repair known non-standard MRVMT cards in memory only.

    Some MRVMT files contain an unparsable MTYPE card.  Astropy can read much
    of the header, but later header/data access can raise VerifyError unless
    the card is fixed in the in-memory Header.  This does not modify the FITS
    file on disk.
    """
    repaired: List[str] = []
    try:
        from astropy.io.fits import Card
    except Exception:
        Card = None  # type: ignore[assignment]
    for key, value in [("MTYPE", "all")]:
        try:
            _ = header.get(key)
            continue
        except Exception:
            pass
        try:
            header[key] = value
            repaired.append(key)
            continue
        except Exception:
            pass
        if Card is None:
            continue
        try:
            cards = getattr(header, "_cards", None)
            if cards is None:
                continue
            replaced = False
            for idx, card in enumerate(cards):
                keyword = ""
                try:
                    keyword = str(getattr(card, "keyword", "")).strip()
                except Exception:
                    try:
                        keyword = str(card)[:8].strip()
                    except Exception:
                        keyword = ""
                if keyword == key:
                    cards[idx] = Card(key, value)
                    repaired.append(key)
                    replaced = True
                    break
            if not replaced:
                cards.append(Card(key, value))
                repaired.append(key)
        except Exception:
            continue
    for option in ["fix", "silentfix"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                header.verify(option)
            break
        except Exception:
            continue
    return sorted(set(repaired))


def _open_mrvmt_hdul(path: str, memmap: bool = True):
    fits = get_fits_module()
    kwargs: Dict[str, Any] = {"memmap": memmap, "do_not_scale_image_data": True, "ignore_missing_end": True}
    try:
        hdul = fits.open(path, **kwargs)
    except TypeError:
        kwargs.pop("ignore_missing_end", None)
        hdul = fits.open(path, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for option in ["fix", "silentfix"]:
            try:
                hdul.verify(option)
                break
            except Exception:
                continue
        try:
            _repair_known_mrvmt_header_cards(hdul[0].header)
        except Exception:
            pass
    return hdul


def _binary_mask_score(values: np.ndarray) -> Tuple[bool, str, float]:
    arr = np.asarray(values)
    finite = np.isfinite(arr)
    if arr.size == 0 or finite.sum() == 0:
        return False, "", math.nan
    arr = arr[finite].astype(np.float64)
    rounded = np.rint(arr)
    binary = np.all(np.abs(arr - rounded) < 1e-5) and set(np.unique(rounded).astype(int)).issubset({0, 1})
    uniq = np.unique(rounded if binary else arr)
    uniq_text = f"{uniq[:12].tolist()}..." if uniq.size > 12 else str(uniq.tolist())
    fill = float(np.mean(rounded > 0.5)) if binary else math.nan
    return bool(binary), uniq_text, fill


def _select_window_row(data: np.ndarray, ell: int) -> Tuple[Optional[np.ndarray], str, bool, str, float, List[str]]:
    if data is None or data.ndim != 3:
        return None, "", False, "", math.nan, ["bad_window_row"]
    row_index = int(ell) + 1
    if data.shape[0] < 1 or data.shape[1] <= row_index:
        return None, "", False, "", math.nan, ["bad_window_row"]
    candidates = []
    for comp in range(min(2, data.shape[0])):
        try:
            values = np.asarray(data[comp, row_index, :])
            binary, uniq_text, fill = _binary_mask_score(values)
            candidates.append((comp, values, binary, uniq_text, fill))
        except Exception:
            continue
    if not candidates:
        return None, "", False, "", math.nan, ["bad_window_row"]
    binary_candidates = [c for c in candidates if c[2]]
    if binary_candidates:
        chosen = next((c for c in binary_candidates if c[0] == 0), binary_candidates[0])
        comp, values, binary, uniq_text, fill = chosen
        mask = (np.rint(values).astype(np.int8) > 0).astype(np.int8)
        return mask, f"axis0={comp}", bool(binary), uniq_text, float(fill), []
    comp, values, binary, uniq_text, fill = candidates[0]
    finite = np.isfinite(values)
    mask = (values > 0.5).astype(np.int8) if finite.any() else None
    fill = float(np.mean(mask > 0)) if mask is not None else math.nan
    return mask, f"axis0={comp}", False, uniq_text, fill, ["bad_window_row"]


def audit_fits_file(path: str, ell: int, gm_id: str = "", gmonth_label: str = "", read_window: bool = True, return_window: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]:
    p = Path(path)
    flags: List[str] = []
    rejection: List[str] = []
    row: Dict[str, Any] = {col: "" for col in AUDIT_COLUMNS}
    row.update({"gm_id": gm_id, "gmonth_label": gmonth_label, "l_requested": int(ell), "path": str(p), "exists": p.exists(), "file_size_bytes": p.stat().st_size if p.exists() else 0, "expected_shape": str((2, int(ell) + 2, 51840)), "shape_ok": False, "window_binary": False, "window_fill": math.nan, "cadence_s": 60.0})
    header_info: Dict[str, Any] = {"cadence_s": 60.0, "n_time": 51840, "ts_start": "", "ts_end": "", "gmonth": "", "l_header": ""}
    if not p.exists():
        flags.append("parsing_failed"); rejection.append("file_missing")
        row["quality_flags"] = quality_join(flags); row["rejection_reason"] = quality_join(rejection)
        return row, header_info, None
    window_mask: Optional[np.ndarray] = None
    try:
        with _open_mrvmt_hdul(str(p), memmap=True) as hdul:
            hdu = hdul[0]
            header = hdu.header
            data = None
            shape = _shape_from_header(header)
            if read_window or return_window:
                data = hdu.data
                shape = tuple(data.shape) if data is not None else shape
            for key, col in [("NAXIS", "naxis"), ("BITPIX", "bitpix"), ("L_VALUE", "l_header"), ("LMAX", "lmax"), ("CD1_1", "cd1_1"), ("CTYPE1", "ctype1"), ("TS_START", "ts_start"), ("TS_END", "ts_end"), ("GMONTH", "gmonth_header"), ("DTYPE", "dtype"), ("MTYPE", "mtype"), ("ORIGIN", "origin"), ("TS_NEFF", "ts_neff"), ("GTS_NEFF", "gts_neff")]:
                row[col] = _safe_header_value(header, key, "")
            row["shape"] = str(shape)
            cadence_s = safe_float(_safe_header_value(header, "CD1_1", 60.0), 60.0)
            n_time = int(shape[2]) if shape and len(shape) == 3 else 0
            row["n_time"] = n_time; row["cadence_s"] = cadence_s
            header_info.update({"cadence_s": cadence_s, "n_time": n_time, "ts_start": row["ts_start"], "ts_end": row["ts_end"], "gmonth": row["gmonth_header"], "l_header": row["l_header"]})
            expected_shape = (2, int(ell) + 2, 51840)
            row["shape_ok"] = shape == expected_shape
            if shape != expected_shape:
                flags.append("bad_fits_shape"); rejection.append("shape_mismatch")
            if not row["ts_start"] or not row["ts_end"]:
                flags.append("parsing_failed"); rejection.append("missing_TS_START_or_TS_END")
            l_header = safe_int(row["l_header"], None)
            if l_header is not None and l_header != int(ell):
                flags.append("bad_fits_shape"); rejection.append("L_VALUE_mismatch")
            if read_window and data is not None and shape is not None and len(shape) == 3:
                window_mask, component, binary, unique_values, fill, window_flags = _select_window_row(data, int(ell))
                flags.extend(window_flags)
                row["window_component"] = component; row["window_binary"] = bool(binary); row["window_fill"] = fill; row["window_unique_values"] = unique_values
                if not binary:
                    rejection.append("non_binary_window_row")
    except Exception as exc:
        flags.append("parsing_failed"); rejection.append(f"fits_open_or_read_failed:{exc}")
    row["quality_flags"] = quality_join(flags); row["rejection_reason"] = quality_join(rejection)
    if not return_window:
        window_mask = None
    return row, header_info, window_mask


def build_window_index(gm_id: str, gmonth: Any, gmonth_label: str, ell: int, path: str, ts_start: Any, n_time: int, cadence_s: float, window_mask: Optional[np.ndarray], window_days: float, stride_days: float, fill_min: float) -> "pd.DataFrame":
    require_pandas()
    rows: List[Dict[str, Any]] = []
    flags_base: List[str] = []
    if n_time <= 0:
        n_time = 51840; flags_base.append("parsing_failed")
    if cadence_s <= 0 or not math.isfinite(cadence_s):
        cadence_s = 60.0; flags_base.append("parsing_failed")
    if window_mask is None or len(window_mask) < n_time:
        flags_base.append("bad_window_row"); mask = np.zeros(n_time, dtype=np.int8)
    else:
        mask = np.asarray(window_mask[:n_time], dtype=np.int8)
    window_samples = int(round(window_days * 86400.0 / cadence_s))
    stride_samples = int(round(stride_days * 86400.0 / cadence_s))
    if window_samples <= 0:
        window_samples = int(round(9.0 * 86400.0 / cadence_s)); flags_base.append("parsing_failed")
    if stride_samples <= 0:
        stride_samples = int(round(3.0 * 86400.0 / cadence_s)); flags_base.append("parsing_failed")
    t0 = parse_datetime_utc(ts_start)
    if t0 is None:
        flags_base.append("parsing_failed")
    idx = 0
    start = 0
    while start + window_samples <= n_time:
        end = start + window_samples
        seg = mask[start:end]
        n_good = int(np.sum(seg > 0))
        fill = float(n_good / max(window_samples, 1))
        flags = list(flags_base)
        if fill < fill_min:
            flags.append("low_fill")
        dt_start = t0 + _dt.timedelta(seconds=start * cadence_s) if t0 is not None else None
        dt_end = t0 + _dt.timedelta(seconds=end * cadence_s) if t0 is not None else None
        dt_mid = t0 + _dt.timedelta(seconds=(start + end) * 0.5 * cadence_s) if t0 is not None else None
        rows.append({"gm_id": gm_id, "gmonth": gmonth, "gmonth_label": gmonth_label, "l": int(ell), "path": path, "window_id": f"w{idx:02d}", "window_index": idx, "start_sample": start, "end_sample": end, "t_start": format_datetime(dt_start), "t_mid": format_datetime(dt_mid), "t_end": format_datetime(dt_end), "window_days": window_days, "stride_days": stride_days, "cadence_s": cadence_s, "n_samples": window_samples, "n_good": n_good, "fill": fill, "accepted": not any(f in {"bad_window_row", "low_fill"} for f in flags), "quality_flags": quality_join(flags)})
        idx += 1; start += stride_samples
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def read_mrvmt_fits(path: str, ell: int, gm_id: str, gmonth_label: str) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], np.ndarray]:
    with _open_mrvmt_hdul(path, memmap=True) as hdul:
        header = hdul[0].header
        data = np.array(hdul[0].data, copy=True)
        shape = tuple(data.shape)
        flags: List[str] = []
        rejection: List[str] = []
        audit_row: Dict[str, Any] = {col: "" for col in AUDIT_COLUMNS}
        audit_row.update({"gm_id": gm_id, "gmonth_label": gmonth_label, "l_requested": int(ell), "path": path, "exists": Path(path).exists(), "file_size_bytes": Path(path).stat().st_size if Path(path).exists() else 0, "expected_shape": str((2, int(ell) + 2, 51840)), "shape_ok": False, "window_binary": False, "window_fill": math.nan, "cadence_s": 60.0})
        for key, col in [("NAXIS", "naxis"), ("BITPIX", "bitpix"), ("L_VALUE", "l_header"), ("LMAX", "lmax"), ("CD1_1", "cd1_1"), ("CTYPE1", "ctype1"), ("TS_START", "ts_start"), ("TS_END", "ts_end"), ("GMONTH", "gmonth_header"), ("DTYPE", "dtype"), ("MTYPE", "mtype"), ("ORIGIN", "origin"), ("TS_NEFF", "ts_neff"), ("GTS_NEFF", "gts_neff")]:
            audit_row[col] = _safe_header_value(header, key, "")
        audit_row["shape"] = str(shape)
        expected_shape = (2, int(ell) + 2, 51840)
        if shape != expected_shape:
            flags.append("bad_fits_shape"); rejection.append("shape_mismatch")
        audit_row["shape_ok"] = shape == expected_shape
        l_header = safe_int(audit_row.get("l_header"), None)
        if l_header is not None and l_header != int(ell):
            flags.append("bad_fits_shape"); rejection.append("L_VALUE_mismatch")
        window_mask, component, binary, unique_values, fill, window_flags = _select_window_row(data, int(ell))
        flags.extend(window_flags)
        audit_row["window_component"] = component; audit_row["window_binary"] = bool(binary); audit_row["window_fill"] = fill; audit_row["window_unique_values"] = unique_values
        if window_mask is None:
            flags.append("bad_window_row"); window_mask = np.zeros(shape[2], dtype=np.int8)
        cadence_s = safe_float(_safe_header_value(header, "CD1_1", 60.0), 60.0)
        header_info = {"cadence_s": cadence_s, "n_time": int(shape[2]), "ts_start": _safe_header_value(header, "TS_START", ""), "ts_end": _safe_header_value(header, "TS_END", ""), "gmonth": _safe_header_value(header, "GMONTH", ""), "l_header": _safe_header_value(header, "L_VALUE", "")}
        audit_row["n_time"] = int(shape[2]); audit_row["cadence_s"] = cadence_s
        if not header_info["ts_start"] or not header_info["ts_end"]:
            flags.append("parsing_failed"); rejection.append("missing_TS_START_or_TS_END")
        audit_row["quality_flags"] = quality_join(flags); audit_row["rejection_reason"] = quality_join(rejection)
    return data, header_info, audit_row, np.asarray(window_mask, dtype=np.int8)


def extract_windowed_m_timeseries(data: np.ndarray, ell: int, start_sample: int, end_sample: int, window_mask: np.ndarray) -> np.ndarray:
    m_count = int(ell) + 1
    expected_shape = (2, int(ell) + 2, 51840)
    if data.ndim != 3 or data.shape[0] != 2 or data.shape[1] < int(ell) + 2:
        raise ValueError(f"bad_fits_shape: expected MRVMT axes compatible with {expected_shape}, got {tuple(data.shape)}")
    if start_sample < 0 or end_sample > data.shape[2] or end_sample <= start_sample:
        raise ValueError(f"bad window sample bounds: start={start_sample} end={end_sample} n_time={data.shape[2]}")
    real = np.asarray(data[0, :m_count, start_sample:end_sample], dtype=np.float32)
    imag = np.asarray(data[1, :m_count, start_sample:end_sample], dtype=np.float32)
    ts = real.astype(np.complex64) + 1j * imag.astype(np.complex64)
    mask = np.asarray(window_mask[start_sample:end_sample], dtype=np.float32)
    if mask.shape[0] != ts.shape[1]:
        raise ValueError("window mask length does not match extracted time series")
    ts *= mask[None, :]
    return np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)


def compute_fft_power(time_series: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        n_time = int(time_series.shape[1])
        fft = np.fft.fft(time_series, axis=1)
        freq_uhz = np.fft.fftfreq(n_time, d=float(cadence_s)) * 1.0e6
        keep = freq_uhz >= 0.0
        power = (np.abs(fft[:, keep]) ** 2) / max(n_time, 1)
        return freq_uhz[keep].astype(np.float64), power.astype(np.float64)
    except Exception as exc:
        raise RuntimeError(f"fft_failed: {exc}") from exc


def shift_and_stack_spectra(freq_uhz: np.ndarray, power_by_m: np.ndarray, m_values: np.ndarray, shifts_uhz: Optional[np.ndarray], shift_sign: str) -> np.ndarray:
    if power_by_m.size == 0:
        return np.array([], dtype=np.float64)
    if shift_sign == "none" or shifts_uhz is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(power_by_m, axis=0)
    sign = 1.0 if shift_sign == "plus" else -1.0
    shifted_rows = []
    for i, _m in enumerate(m_values):
        sample_freq = freq_uhz + sign * float(shifts_uhz[i])
        shifted_rows.append(np.interp(sample_freq, freq_uhz, power_by_m[i], left=np.nan, right=np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.vstack(shifted_rows), axis=0)


def compute_stack_sharpness(freq_uhz: np.ndarray, power: np.ndarray, nu0: float, gamma: float) -> float:
    if freq_uhz.size == 0 or power.size == 0:
        return math.nan
    df = float(np.nanmedian(np.diff(freq_uhz))) if freq_uhz.size > 2 else 1.0
    if not math.isfinite(df) or df <= 0:
        df = 1.0
    g = max(float(gamma), df)
    band_half = max(5.0 * g, 20.0, 8.0 * df)
    peak_half = max(0.75 * g, 2.0 * df)
    band = (freq_uhz >= nu0 - band_half) & (freq_uhz <= nu0 + band_half)
    peak = (freq_uhz >= nu0 - peak_half) & (freq_uhz <= nu0 + peak_half)
    side = band & ~peak
    if peak.sum() < 1 or side.sum() < 4:
        return math.nan
    p_peak = power[peak]; p_side = power[side]
    finite_peak = np.isfinite(p_peak); finite_side = np.isfinite(p_side)
    if finite_peak.sum() < 1 or finite_side.sum() < 4:
        return math.nan
    peak_max = float(np.nanmax(p_peak[finite_peak])); side_med = float(np.nanmedian(p_side[finite_side]))
    side_mad = float(np.nanmedian(np.abs(p_side[finite_side] - side_med)))
    return (peak_max - side_med) / max(side_mad, abs(side_med) * 1e-6, 1e-30)


def estimate_reference_height_background(freq_uhz: np.ndarray, power: np.ndarray, nu0: float, gamma: float) -> Tuple[float, float, List[str]]:
    flags: List[str] = []
    if freq_uhz.size == 0 or power.size == 0:
        return math.nan, math.nan, ["bad_reference_model"]
    df = float(np.nanmedian(np.diff(freq_uhz))) if freq_uhz.size > 2 else 1.0
    if not math.isfinite(df) or df <= 0:
        df = 1.0
    g = max(float(gamma), df)
    band_half = max(6.0 * g, 25.0, 10.0 * df)
    center_half = max(1.0 * g, 2.0 * df)
    band = (freq_uhz >= nu0 - band_half) & (freq_uhz <= nu0 + band_half)
    center = (freq_uhz >= nu0 - center_half) & (freq_uhz <= nu0 + center_half)
    side = band & ~center
    p_band = power[band]; p_center = power[center]; p_side = power[side]
    p_band = p_band[np.isfinite(p_band)]; p_center = p_center[np.isfinite(p_center)]; p_side = p_side[np.isfinite(p_side)]
    if p_band.size < 8 or p_center.size < 1:
        return math.nan, math.nan, ["bad_reference_model"]
    B = float(np.nanmedian(p_side)) if p_side.size >= 4 else float(np.nanpercentile(p_band, 20.0))
    peak = float(np.nanmax(p_center))
    floor = max(float(np.nanmedian(p_band)) * 1e-12, 1e-30)
    B = max(B, floor); H = max(peak - B, floor)
    if not (math.isfinite(H) and H > 0 and math.isfinite(B) and B > 0):
        flags.append("bad_reference_model")
    return H, B, flags


def lorentzian_model(freq_uhz: np.ndarray, nu0: float, gamma: float, H: float, B: float) -> np.ndarray:
    g = max(float(gamma), 1.0e-6); h = max(float(H), 1.0e-300); b = max(float(B), 1.0e-300)
    a = 0.5 * g
    x = freq_uhz - float(nu0)
    A = (a * a) / (x * x + a * a)
    return np.maximum(h * A + b, 1.0e-300)


def lorentzian_log_derivatives(freq_uhz: np.ndarray, nu0: float, gamma: float, H: float, B: float) -> np.ndarray:
    g = max(float(gamma), 1.0e-6); h = max(float(H), 1.0e-300); b = max(float(B), 1.0e-300)
    a = 0.5 * g
    x = freq_uhz - float(nu0)
    D = x * x + a * a
    A = (a * a) / D
    M = np.maximum(h * A + b, 1.0e-300)
    dM_dnu0 = h * (2.0 * x * a * a) / (D * D)
    dM_dlngamma = h * (2.0 * a * a * x * x) / (D * D)
    dM_dlnH = h * A
    dM_dlnB = np.full_like(freq_uhz, b, dtype=np.float64)
    return np.vstack([dM_dnu0 / M, dM_dlngamma / M, dM_dlnH / M, dM_dlnB / M]).T


def profile_nuisance_parameters(score: np.ndarray, fisher: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    flags: List[str] = []
    score = np.asarray(score, dtype=np.float64); fisher = np.asarray(fisher, dtype=np.float64)
    s_alpha = score[:2]; s_beta = score[2:]
    I_aa = fisher[:2, :2]; I_ab = fisher[:2, 2:]; I_ba = fisher[2:, :2]; I_bb = fisher[2:, 2:]
    try:
        cond_bb = float(np.linalg.cond(I_bb))
    except Exception:
        cond_bb = math.inf
    try:
        if not np.all(np.isfinite(I_bb)) or abs(float(np.linalg.det(I_bb))) < 1e-30:
            raise np.linalg.LinAlgError("I_bb singular")
        inv_bb = np.linalg.inv(I_bb)
    except Exception:
        flags.append("fisher_singular")
        inv_bb = np.linalg.pinv(I_bb, rcond=1e-12)
    if not math.isfinite(cond_bb) or cond_bb > 1.0e12:
        flags.append("fisher_ill_conditioned")
    q = s_alpha - I_ab @ inv_bb @ s_beta
    J = I_aa - I_ab @ inv_bb @ I_ba
    return q, 0.5 * (J + J.T), flags, cond_bb


def score_fisher_for_mode(freq_uhz: np.ndarray, stacked_power: np.ndarray, mode: Mapping[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    nu0 = safe_float(mode.get("nu_ref"), math.nan); gamma = safe_float(mode.get("gamma_ref"), math.nan)
    H = safe_float(mode.get("H_ref"), math.nan); B = safe_float(mode.get("B_ref"), math.nan)
    H_source = "mode_table"; B_source = "mode_table"
    if not (math.isfinite(nu0) and math.isfinite(gamma) and gamma > 0):
        flags.append("bad_reference_model")
        return {"q": np.zeros(2), "J": np.zeros((2, 2)), "flags": flags, "mode_used": False, "H_ref_used": H, "B_ref_used": B, "H_source": H_source, "B_source": B_source, "fisher_cond_mode": math.inf, "n_freq_used": 0}
    df = float(np.nanmedian(np.diff(freq_uhz))) if freq_uhz.size > 2 else 1.0
    if not math.isfinite(df) or df <= 0:
        df = 1.0
    band_half = max(6.0 * gamma, 25.0, 10.0 * df)
    if nu0 - band_half < float(np.nanmin(freq_uhz)) or nu0 + band_half > float(np.nanmax(freq_uhz)):
        flags.append("edge_of_fit_band")
    fit = (freq_uhz >= nu0 - band_half) & (freq_uhz <= nu0 + band_half)
    if int(np.sum(fit)) < 8:
        flags.append("bad_reference_model")
        return {"q": np.zeros(2), "J": np.zeros((2, 2)), "flags": flags, "mode_used": False, "H_ref_used": H, "B_ref_used": B, "H_source": H_source, "B_source": B_source, "fisher_cond_mode": math.inf, "n_freq_used": int(np.sum(fit))}
    f = freq_uhz[fit]; P = stacked_power[fit]
    valid = np.isfinite(f) & np.isfinite(P)
    f = f[valid]; P = P[valid]
    if f.size < 8:
        flags.append("bad_reference_model")
        return {"q": np.zeros(2), "J": np.zeros((2, 2)), "flags": flags, "mode_used": False, "H_ref_used": H, "B_ref_used": B, "H_source": H_source, "B_source": B_source, "fisher_cond_mode": math.inf, "n_freq_used": int(f.size)}
    if not (math.isfinite(H) and H > 0 and math.isfinite(B) and B > 0):
        H_est, B_est, est_flags = estimate_reference_height_background(freq_uhz, stacked_power, nu0, gamma)
        flags.extend(est_flags)
        if not (math.isfinite(H) and H > 0):
            H = H_est; H_source = "estimated_local_stack"
        if not (math.isfinite(B) and B > 0):
            B = B_est; B_source = "estimated_local_stack"
    if not (math.isfinite(H) and H > 0 and math.isfinite(B) and B > 0):
        flags.append("bad_reference_model")
        return {"q": np.zeros(2), "J": np.zeros((2, 2)), "flags": flags, "mode_used": False, "H_ref_used": H, "B_ref_used": B, "H_source": H_source, "B_source": B_source, "fisher_cond_mode": math.inf, "n_freq_used": int(f.size)}
    M = lorentzian_model(f, nu0, gamma, H, B)
    D = lorentzian_log_derivatives(f, nu0, gamma, H, B)
    valid = np.all(np.isfinite(D), axis=1) & np.isfinite(M) & (M > 0) & np.isfinite(P)
    if valid.sum() < 8:
        flags.append("bad_reference_model")
        return {"q": np.zeros(2), "J": np.zeros((2, 2)), "flags": flags, "mode_used": False, "H_ref_used": H, "B_ref_used": B, "H_source": H_source, "B_source": B_source, "fisher_cond_mode": math.inf, "n_freq_used": int(valid.sum())}
    P = P[valid]; M = M[valid]; D = D[valid, :]
    score = np.sum(((P / M) - 1.0)[:, None] * D, axis=0)
    fisher = D.T @ D
    q, J, profile_flags, _ = profile_nuisance_parameters(score, fisher)
    flags.extend(profile_flags)
    try:
        fisher_cond_mode = float(np.linalg.cond(J))
    except Exception:
        fisher_cond_mode = math.inf
    if not np.all(np.isfinite(J)):
        flags.append("fisher_singular")
    if not math.isfinite(fisher_cond_mode) or fisher_cond_mode > 1.0e12:
        flags.append("fisher_ill_conditioned")
    if not (math.isfinite(J[0, 0]) and J[0, 0] > 0):
        flags.append("low_fisher_nu")
    if not (math.isfinite(J[1, 1]) and J[1, 1] > 0):
        flags.append("low_fisher_gamma")
    mode_used = not any(flag in {"bad_reference_model", "fisher_singular", "low_fisher_nu", "low_fisher_gamma"} for flag in flags)
    return {"q": q, "J": J, "flags": flags, "mode_used": bool(mode_used), "H_ref_used": H, "B_ref_used": B, "H_source": H_source, "B_source": B_source, "fisher_cond_mode": fisher_cond_mode, "n_freq_used": int(valid.sum())}


def choose_shift_sign_for_window(freq_uhz: np.ndarray, power_by_m: np.ndarray, m_values: np.ndarray, mode_records: Sequence[Mapping[str, Any]], splitting: SplittingData, ell: int, shift_policy: str) -> Dict[str, Any]:
    flags: List[str] = []
    sharpness: Dict[str, List[float]] = {"none": [], "plus": [], "minus": []}
    n_diag_modes = 0; n_with_splitting = 0
    sorted_modes = sorted(mode_records, key=lambda m: (abs(safe_float(m.get("nu_ref"), 0.0) - 3250.0), safe_int(m.get("n"), 9999)))
    stack_none_cache = None
    for mode in sorted_modes[: min(12, len(sorted_modes))]:
        n = int(mode["n"]); nu0 = safe_float(mode.get("nu_ref"), math.nan); gamma = safe_float(mode.get("gamma_ref"), math.nan)
        if not (math.isfinite(nu0) and math.isfinite(gamma) and gamma > 0):
            continue
        n_diag_modes += 1
        if stack_none_cache is None:
            stack_none_cache = shift_and_stack_spectra(freq_uhz, power_by_m, m_values, None, "none")
        sharpness["none"].append(compute_stack_sharpness(freq_uhz, stack_none_cache, nu0, gamma))
        coeffs, k_used, split_flags = get_splitting_coeffs(splitting, ell, n)
        if split_flags:
            continue
        shifts = compute_splitting_shift(ell, m_values, coeffs, k_used)
        n_with_splitting += 1
        for sign in ["plus", "minus"]:
            stack = shift_and_stack_spectra(freq_uhz, power_by_m, m_values, shifts, sign)
            sharpness[sign].append(compute_stack_sharpness(freq_uhz, stack, nu0, gamma))
    def med(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=np.float64); arr = arr[np.isfinite(arr)]
        return float(np.nanmedian(arr)) if arr.size else math.nan
    sharp_before = med(sharpness["none"]); sharp_plus = med(sharpness["plus"]); sharp_minus = med(sharpness["minus"])
    splitting_missing = n_with_splitting == 0
    if splitting_missing:
        flags.append("missing_splitting")
    if shift_policy == "none":
        chosen = "none"
    elif shift_policy in {"plus", "minus"}:
        chosen = shift_policy if not splitting_missing else "none"
    else:
        if splitting_missing or (not math.isfinite(sharp_plus) and not math.isfinite(sharp_minus)):
            chosen = "none"
        elif not math.isfinite(sharp_minus):
            chosen = "plus"
        elif not math.isfinite(sharp_plus):
            chosen = "minus"
        else:
            chosen = "plus" if sharp_plus >= sharp_minus else "minus"
    sharp_chosen = sharp_plus if chosen == "plus" else sharp_minus if chosen == "minus" else sharp_before
    return {"chosen_shift_sign": chosen, "sharpness_before": sharp_before, "sharpness_after_plus": sharp_plus, "sharpness_after_minus": sharp_minus, "sharpness_chosen": sharp_chosen, "n_modes_for_sign": n_diag_modes, "splitting_missing": splitting_missing, "quality_flags": flags}


def make_task_key(gm_id: str, ell: int) -> str:
    return f"{re.sub(r'[^0-9A-Za-z_.-]+', '_', str(gm_id))}_l{int(ell):03d}"


def checkpoint_path(checkpoint_dir: Path, gm_id: str, ell: int) -> Path:
    return checkpoint_dir / f"{make_task_key(gm_id, ell)}.json"


def checkpoint_done(checkpoint_dir: Path, gm_id: str, ell: int, payload: Mapping[str, Any]) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(checkpoint_dir, gm_id, ell)
    content = dict(payload); content["checkpoint_timestamp"] = utc_now_iso(); content["gm_id"] = gm_id; content["l"] = int(ell)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, sort_keys=True, default=json_default)
    return str(path)


def load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_df_csv(df: "pd.DataFrame", path: Path, columns: Sequence[str]) -> str:
    require_pandas(); path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame(columns=list(columns))
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[list(columns)]
    df.to_csv(path, index=False)
    return str(path)


def _empty_df(columns: Sequence[str]) -> "pd.DataFrame":
    require_pandas(); return pd.DataFrame(columns=list(columns))


def make_task_error_row(
    gm_id: str,
    gmonth_label: str,
    ell: int,
    path: str,
    stage: str,
    status: str,
    exc: Optional[BaseException] = None,
    tb: str = "",
    error_file: str = "",
    checkpoint: str = "",
    quality_flags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "gm_id": gm_id,
        "gmonth_label": gmonth_label,
        "l": int(ell),
        "path": path,
        "stage": stage,
        "status": status,
        "exception_type": type(exc).__name__ if exc is not None else "",
        "exception_message": str(exc) if exc is not None else "",
        "traceback": tb or "",
        "error_file": error_file,
        "checkpoint": checkpoint,
        "quality_flags": quality_join(quality_flags or ["task_exception"]),
    }


def _failed_mode_row(
    gm_id: str,
    gmonth: Any,
    gmonth_label: str,
    ell: int,
    n: int,
    mode: Mapping[str, Any],
    wrow: Mapping[str, Any],
    window_flags: Sequence[str],
    split_flags: Sequence[str],
    splitting: SplittingData,
    shift_sign_policy: Any,
    chosen_sign: str,
    sign_info: Mapping[str, Any],
    stage_flag: str = "task_exception",
) -> Dict[str, Any]:
    return {"gm_id": gm_id, "gmonth": gmonth, "gmonth_label": gmonth_label, "l": ell, "n": n, "mode_id": f"n{n}_l{ell}", "window_id": wrow.get("window_id", ""), "t_start": wrow.get("t_start", ""), "t_mid": wrow.get("t_mid", ""), "t_end": wrow.get("t_end", ""), "fill": safe_float(wrow.get("fill"), math.nan), "m_count": ell + 1, "m_min": 0, "m_max": ell, "m_weighting": "uniform", "splitting_file": splitting.path, "splitting_k_used": "", "shift_sign_policy": shift_sign_policy, "chosen_shift_sign": chosen_sign, "sharpness_before": sign_info.get("sharpness_before", math.nan), "sharpness_after_plus": sign_info.get("sharpness_after_plus", math.nan), "sharpness_after_minus": sign_info.get("sharpness_after_minus", math.nan), "sharpness_chosen": sign_info.get("sharpness_chosen", math.nan), "stack_sharpness": math.nan, "nu_ref": safe_float(mode.get("nu_ref"), math.nan), "gamma_ref": safe_float(mode.get("gamma_ref"), math.nan), "H_ref_used": math.nan, "B_ref_used": math.nan, "H_source": "", "B_source": "", "q_nu": 0.0, "q_lngamma": 0.0, "J_nu_nu": 0.0, "J_nu_lngamma": 0.0, "J_lngamma_lngamma": 0.0, "fisher_cond_mode": math.inf, "n_freq_used": 0, "mode_used": False, "quality_flags": quality_join(list(window_flags) + list(split_flags) + ["parsing_failed", stage_flag])}


def process_gm_l_task(task: Mapping[str, Any], args_dict: Mapping[str, Any], mode_records: Sequence[Mapping[str, Any]], splitting: SplittingData) -> Dict[str, Any]:
    require_pandas()
    gm_id = str(task["gm_id"]); gmonth_label = str(task["gmonth_label"]); ell = int(task["l"]); path = str(task["path"])
    outdir = Path(str(args_dict["outdir"])); task_dir = outdir / "task_outputs"; checkpoint_dir = outdir / "checkpoints"; task_dir.mkdir(parents=True, exist_ok=True)
    task_key = make_task_key(gm_id, ell)
    audit_csv = task_dir / f"{task_key}__audit.csv"; windows_csv = task_dir / f"{task_key}__windows.csv"; mode_diag_csv = task_dir / f"{task_key}__mode_diagnostics.csv"; stack_diag_csv = task_dir / f"{task_key}__stack_diagnostics.csv"; example_csv = task_dir / f"{task_key}__example_stacked_spectra.csv"; task_errors_csv = task_dir / f"{task_key}__task_errors.csv"; error_txt = task_dir / f"{task_key}__error.txt"
    output_paths = {"audit": str(audit_csv), "window_index": str(windows_csv), "mode_diagnostics": str(mode_diag_csv), "stack_diagnostics": str(stack_diag_csv), "example_stacked_spectra": str(example_csv), "task_errors": str(task_errors_csv)}
    audit_rows: List[Dict[str, Any]] = []; mode_rows: List[Dict[str, Any]] = []; stack_rows: List[Dict[str, Any]] = []; example_rows: List[Dict[str, Any]] = []; task_error_rows: List[Dict[str, Any]] = []
    window_df = _empty_df(WINDOW_COLUMNS)
    try:
        data, header_info, audit_row, window_mask = read_mrvmt_fits(path, ell, gm_id, gmonth_label)
        audit_rows.append(audit_row)
        gmonth = header_info.get("gmonth", ""); cadence_s = float(header_info.get("cadence_s", 60.0)); n_time = int(header_info.get("n_time", data.shape[2]))
        window_df = build_window_index(gm_id, gmonth, gmonth_label, ell, path, header_info.get("ts_start", ""), n_time, cadence_s, window_mask, float(args_dict["window_days"]), float(args_dict["stride_days"]), float(args_dict["fill_min"]))
        # Persist ingestion diagnostics before any FFT/score path can fail.
        save_df_csv(pd.DataFrame(audit_rows), audit_csv, AUDIT_COLUMNS)
        save_df_csv(window_df, windows_csv, WINDOW_COLUMNS)
        if "bad_fits_shape" in quality_split(audit_row.get("quality_flags", "")) or not mode_records:
            status = "done_bad_fits_shape" if "bad_fits_shape" in quality_split(audit_row.get("quality_flags", "")) else "done_no_modes_for_l"
            flag = "bad_fits_shape" if status == "done_bad_fits_shape" else "no_modes_in_bin"
            save_df_csv(_empty_df(MODE_DIAG_COLUMNS), mode_diag_csv, MODE_DIAG_COLUMNS); save_df_csv(_empty_df(STACK_DIAG_COLUMNS), stack_diag_csv, STACK_DIAG_COLUMNS); save_df_csv(_empty_df(EXAMPLE_STACK_COLUMNS), example_csv, EXAMPLE_STACK_COLUMNS); save_df_csv(_empty_df(TASK_ERROR_COLUMNS), task_errors_csv, TASK_ERROR_COLUMNS)
            checkpoint_done(checkpoint_dir, gm_id, ell, {"status": status, "output_paths": output_paths, "quality_flags": [flag]})
            return {"status": status, "paths": output_paths}
        m_values = np.arange(0, ell + 1, dtype=np.float64)
        example_written = False
        for _, wrow in window_df.iterrows():
            window_flags = quality_split(wrow.get("quality_flags", ""))
            if not bool(wrow.get("accepted", False)):
                continue
            start_sample = int(wrow["start_sample"]); end_sample = int(wrow["end_sample"])
            try:
                ts = extract_windowed_m_timeseries(data, ell, start_sample, end_sample, window_mask)
                freq_uhz, power_by_m = compute_fft_power(ts, cadence_s)
            except Exception as exc:
                tb = traceback.format_exc()
                task_error_rows.append(make_task_error_row(gm_id, gmonth_label, ell, path, "fft", "nonfatal_window_failed", exc=exc, tb=tb, error_file=str(error_txt), quality_flags=window_flags + ["fft_failed", "task_exception"]))
                fft_flags = window_flags + ["fft_failed"]
                stack_rows.append({"gm_id": gm_id, "gmonth": gmonth, "gmonth_label": gmonth_label, "l": ell, "window_id": wrow["window_id"], "t_mid": wrow["t_mid"], "n_modes_for_sign": 0, "shift_sign_policy": args_dict["shift_sign"], "chosen_shift_sign": "none", "sharpness_before": math.nan, "sharpness_after_plus": math.nan, "sharpness_after_minus": math.nan, "sharpness_chosen": math.nan, "splitting_file": splitting.path, "splitting_missing": True, "quality_flags": quality_join(fft_flags)})
                continue
            sign_info = choose_shift_sign_for_window(freq_uhz, power_by_m, m_values, mode_records, splitting, ell, str(args_dict["shift_sign"]))
            stack_rows.append({"gm_id": gm_id, "gmonth": gmonth, "gmonth_label": gmonth_label, "l": ell, "window_id": wrow["window_id"], "t_mid": wrow["t_mid"], "n_modes_for_sign": sign_info["n_modes_for_sign"], "shift_sign_policy": args_dict["shift_sign"], "chosen_shift_sign": sign_info["chosen_shift_sign"], "sharpness_before": sign_info["sharpness_before"], "sharpness_after_plus": sign_info["sharpness_after_plus"], "sharpness_after_minus": sign_info["sharpness_after_minus"], "sharpness_chosen": sign_info["sharpness_chosen"], "splitting_file": splitting.path, "splitting_missing": sign_info["splitting_missing"], "quality_flags": quality_join(sign_info["quality_flags"])})
            for mode in mode_records:
                n = int(mode["n"])
                nu0 = safe_float(mode.get("nu_ref"), math.nan)
                gamma = safe_float(mode.get("gamma_ref"), math.nan)
                chosen_sign = str(sign_info["chosen_shift_sign"])
                coeffs, k_used, split_flags = get_splitting_coeffs(splitting, ell, n)
                try:
                    shifts = compute_splitting_shift(ell, m_values, coeffs, k_used) if k_used and chosen_sign != "none" else None
                    stacked = shift_and_stack_spectra(freq_uhz, power_by_m, m_values, shifts, chosen_sign)
                    stack_sharp = compute_stack_sharpness(freq_uhz, stacked, nu0, gamma) if math.isfinite(nu0) and math.isfinite(gamma) and gamma > 0 else math.nan
                    score_result = score_fisher_for_mode(freq_uhz, stacked, mode)
                    q = np.asarray(score_result["q"], dtype=np.float64); J = np.asarray(score_result["J"], dtype=np.float64)
                    flags = window_flags + split_flags + score_result["flags"]
                    mode_rows.append({"gm_id": gm_id, "gmonth": gmonth, "gmonth_label": gmonth_label, "l": ell, "n": n, "mode_id": f"n{n}_l{ell}", "window_id": wrow["window_id"], "t_start": wrow["t_start"], "t_mid": wrow["t_mid"], "t_end": wrow["t_end"], "fill": float(wrow["fill"]), "m_count": ell + 1, "m_min": 0, "m_max": ell, "m_weighting": "uniform", "splitting_file": splitting.path, "splitting_k_used": ",".join(str(k) for k in k_used), "shift_sign_policy": args_dict["shift_sign"], "chosen_shift_sign": chosen_sign, "sharpness_before": sign_info["sharpness_before"], "sharpness_after_plus": sign_info["sharpness_after_plus"], "sharpness_after_minus": sign_info["sharpness_after_minus"], "sharpness_chosen": sign_info["sharpness_chosen"], "stack_sharpness": stack_sharp, "nu_ref": nu0, "gamma_ref": gamma, "H_ref_used": score_result["H_ref_used"], "B_ref_used": score_result["B_ref_used"], "H_source": score_result["H_source"], "B_source": score_result["B_source"], "q_nu": float(q[0]) if q.size >= 1 else math.nan, "q_lngamma": float(q[1]) if q.size >= 2 else math.nan, "J_nu_nu": float(J[0, 0]) if J.shape == (2, 2) else math.nan, "J_nu_lngamma": float(J[0, 1]) if J.shape == (2, 2) else math.nan, "J_lngamma_lngamma": float(J[1, 1]) if J.shape == (2, 2) else math.nan, "fisher_cond_mode": score_result["fisher_cond_mode"], "n_freq_used": score_result["n_freq_used"], "mode_used": bool(score_result["mode_used"]), "quality_flags": quality_join(flags)})
                    if not example_written and math.isfinite(nu0) and math.isfinite(gamma) and 2500.0 <= nu0 <= 4000.0:
                        half = max(6.0 * gamma, 30.0); keep = (freq_uhz >= nu0 - half) & (freq_uhz <= nu0 + half)
                        for fval, pval in zip(freq_uhz[keep], stacked[keep]):
                            if math.isfinite(float(pval)):
                                example_rows.append({"gm_id": gm_id, "gmonth": gmonth, "gmonth_label": gmonth_label, "l": ell, "n": n, "window_id": wrow["window_id"], "t_mid": wrow["t_mid"], "chosen_shift_sign": chosen_sign, "freq_uhz": float(fval), "power": float(pval)})
                        example_written = True
                except Exception as exc:
                    tb = traceback.format_exc()
                    task_error_rows.append(make_task_error_row(gm_id, gmonth_label, ell, path, "mode_score", "nonfatal_mode_failed", exc=exc, tb=tb, error_file=str(error_txt), quality_flags=window_flags + split_flags + ["parsing_failed", "task_exception"]))
                    mode_rows.append(_failed_mode_row(gm_id, gmonth, gmonth_label, ell, n, mode, wrow, window_flags, split_flags, splitting, args_dict["shift_sign"], chosen_sign, sign_info))
                    continue
        audit_df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS); mode_df = pd.DataFrame(mode_rows, columns=MODE_DIAG_COLUMNS); stack_df = pd.DataFrame(stack_rows, columns=STACK_DIAG_COLUMNS); example_df = pd.DataFrame(example_rows, columns=EXAMPLE_STACK_COLUMNS); task_error_df = pd.DataFrame(task_error_rows, columns=TASK_ERROR_COLUMNS)
        save_df_csv(audit_df, audit_csv, AUDIT_COLUMNS); save_df_csv(window_df, windows_csv, WINDOW_COLUMNS); save_df_csv(mode_df, mode_diag_csv, MODE_DIAG_COLUMNS); save_df_csv(stack_df, stack_diag_csv, STACK_DIAG_COLUMNS); save_df_csv(example_df, example_csv, EXAMPLE_STACK_COLUMNS); save_df_csv(task_error_df, task_errors_csv, TASK_ERROR_COLUMNS)
        quality_all = []
        for df, col in [(audit_df, "quality_flags"), (window_df, "quality_flags"), (mode_df, "quality_flags"), (task_error_df, "quality_flags")]:
            if col in df.columns:
                for item in df[col].dropna().tolist():
                    quality_all.extend(quality_split(item))
        status = "done_with_nonfatal_errors" if len(task_error_df) else "done"
        ckpt = checkpoint_done(checkpoint_dir, gm_id, ell, {"status": status, "output_paths": output_paths, "quality_flags": sorted(set(quality_all)), "n_windows": int(len(window_df)), "n_modes_rows": int(len(mode_df)), "n_task_errors": int(len(task_error_df))})
        return {"status": status, "paths": output_paths, "checkpoint": ckpt, "n_task_errors": int(len(task_error_df))}
    except Exception as exc:
        tb = traceback.format_exc(); error_txt.write_text(tb, encoding="utf-8")
        task_error_rows.append(make_task_error_row(gm_id, gmonth_label, ell, path, "task", "failed", exc=exc, tb=tb, error_file=str(error_txt), quality_flags=["parsing_failed", "task_exception"]))
        audit_df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS) if audit_rows else pd.DataFrame([{"gm_id": gm_id, "gmonth_label": gmonth_label, "l_requested": ell, "path": path, "exists": Path(path).exists(), "quality_flags": "parsing_failed", "rejection_reason": "task_exception"}], columns=AUDIT_COLUMNS)
        task_error_df = pd.DataFrame(task_error_rows, columns=TASK_ERROR_COLUMNS)
        save_df_csv(audit_df, audit_csv, AUDIT_COLUMNS); save_df_csv(window_df, windows_csv, WINDOW_COLUMNS); save_df_csv(pd.DataFrame(mode_rows, columns=MODE_DIAG_COLUMNS), mode_diag_csv, MODE_DIAG_COLUMNS); save_df_csv(pd.DataFrame(stack_rows, columns=STACK_DIAG_COLUMNS), stack_diag_csv, STACK_DIAG_COLUMNS); save_df_csv(pd.DataFrame(example_rows, columns=EXAMPLE_STACK_COLUMNS), example_csv, EXAMPLE_STACK_COLUMNS); save_df_csv(task_error_df, task_errors_csv, TASK_ERROR_COLUMNS)
        ckpt = checkpoint_done(checkpoint_dir, gm_id, ell, {"status": "failed", "output_paths": output_paths, "error_file": str(error_txt), "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": tb, "quality_flags": ["parsing_failed", "task_exception"], "n_task_errors": int(len(task_error_df))})
        if task_error_rows:
            task_error_rows[-1]["checkpoint"] = ckpt
            save_df_csv(pd.DataFrame(task_error_rows, columns=TASK_ERROR_COLUMNS), task_errors_csv, TASK_ERROR_COLUMNS)
        return {"status": "failed", "paths": output_paths, "checkpoint": ckpt, "error_file": str(error_txt), "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": tb, "task_error": task_error_rows[-1] if task_error_rows else {}}


def make_frequency_bins(bin_widths: Sequence[float], freq_min: float, freq_max: float) -> List[Dict[str, float]]:
    bins: List[Dict[str, float]] = []
    for width in bin_widths:
        start = max(2000.0, freq_min) if abs(width - 500.0) < 1e-9 else freq_min
        edge = start
        while edge + width <= freq_max + 1e-9:
            bins.append({"freq_bin_low": float(edge), "freq_bin_high": float(edge + width), "freq_bin_center": float(edge + 0.5 * width), "bin_width": float(width)})
            edge += width
    return bins


def _read_csv_if_exists(path: str, columns: Sequence[str]) -> "pd.DataFrame":
    require_pandas(); p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns))
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=list(columns))
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[list(columns)]


def accumulate_frequency_bins(mode_diag_df: "pd.DataFrame", window_df: "pd.DataFrame", mode_table: "pd.DataFrame", args: argparse.Namespace, logger: logging.Logger) -> "pd.DataFrame":
    require_pandas()
    bins = make_frequency_bins(parse_bin_widths(args.bin_widths), args.freq_min, args.freq_max)
    rows: List[Dict[str, Any]] = []
    if window_df is None or len(window_df) == 0:
        logger.warning("No window rows available; bin time series will be empty with no_valid_windows.")
        return pd.DataFrame(rows, columns=BIN_COLUMNS)
    if "accepted" not in window_df.columns:
        window_df["accepted"] = False
    accepted_windows = window_df[window_df["accepted"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if len(accepted_windows) == 0:
        logger.warning("No valid windows passed fill threshold.")
        return pd.DataFrame(rows, columns=BIN_COLUMNS)
    if mode_diag_df is None or len(mode_diag_df) == 0:
        mode_diag_df = pd.DataFrame(columns=MODE_DIAG_COLUMNS)
    mode_diag_df = mode_diag_df.copy()
    mode_diag_df["mode_used"] = mode_diag_df.get("mode_used", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["nu_ref", "q_nu", "q_lngamma", "J_nu_nu", "J_nu_lngamma", "J_lngamma_lngamma", "fill"]:
        if col in mode_diag_df.columns:
            mode_diag_df[col] = pd.to_numeric(mode_diag_df[col], errors="coerce")
    for (gm_id, window_id), wgrp in accepted_windows.groupby(["gm_id", "window_id"], dropna=False):
        wfirst = wgrp.iloc[0]
        present_l = sorted(set(int(x) for x in pd.to_numeric(wgrp["l"], errors="coerce").dropna().tolist()))
        fill_values = pd.to_numeric(wgrp["fill"], errors="coerce").dropna().to_numpy(dtype=float)
        fill_median = float(np.nanmedian(fill_values)) if fill_values.size else math.nan
        fill_min_value = float(np.nanmin(fill_values)) if fill_values.size else math.nan
        mgrp = mode_diag_df[(mode_diag_df.get("gm_id", "") == gm_id) & (mode_diag_df.get("window_id", "") == window_id)].copy()
        for b in bins:
            low = b["freq_bin_low"]; high = b["freq_bin_high"]
            ref_modes = mode_table[mode_table["l"].isin(present_l) & (mode_table["nu_ref"] >= low) & (mode_table["nu_ref"] < high)]
            n_modes_total = int(len(ref_modes))
            bin_all = mgrp[(mgrp["nu_ref"] >= low) & (mgrp["nu_ref"] < high)] if len(mgrp) else pd.DataFrame(columns=MODE_DIAG_COLUMNS)
            bin_rows = bin_all[bin_all["mode_used"].astype(bool)].copy() if len(bin_all) else pd.DataFrame(columns=MODE_DIAG_COLUMNS)
            flags: List[str] = []
            for item in bin_all.get("quality_flags", pd.Series(dtype=str)).dropna().tolist():
                flags.extend(quality_split(item))
            n_modes_used = int(len(bin_rows))
            base = {"t_start": wfirst.get("t_start", ""), "t_mid": wfirst.get("t_mid", ""), "t_end": wfirst.get("t_end", ""), "gm_id": gm_id, "gmonth": wfirst.get("gmonth", ""), "freq_bin_low": low, "freq_bin_high": high, "freq_bin_center": b["freq_bin_center"], "bin_width": b["bin_width"], "n_modes_total": n_modes_total, "n_modes_used": n_modes_used, "n_modes_rejected": max(n_modes_total - n_modes_used, 0), "fill_median": fill_median, "fill_min": fill_min_value}
            if n_modes_total == 0 or n_modes_used == 0:
                flags.append("no_modes_in_bin")
                rows.append({**base, "l_list": ",".join(str(x) for x in present_l), "delta_nu": math.nan, "delta_lngamma": math.nan, "cov_nu_nu": math.nan, "cov_lngamma_lngamma": math.nan, "cov_nu_lngamma": math.nan, "sigma_nu": math.nan, "sigma_lngamma": math.nan, "corr_nu_lngamma": math.nan, "fisher_trace": math.nan, "fisher_det": math.nan, "fisher_cond": math.nan, "quality_flags": quality_join(flags)})
                continue
            q = np.array([float(np.nansum(bin_rows["q_nu"].to_numpy(dtype=float))), float(np.nansum(bin_rows["q_lngamma"].to_numpy(dtype=float)))], dtype=np.float64)
            J00 = float(np.nansum(bin_rows["J_nu_nu"].to_numpy(dtype=float))); J01 = float(np.nansum(bin_rows["J_nu_lngamma"].to_numpy(dtype=float))); J11 = float(np.nansum(bin_rows["J_lngamma_lngamma"].to_numpy(dtype=float)))
            J = np.array([[J00, J01], [J01, J11]], dtype=np.float64)
            fisher_trace = float(np.trace(J)) if np.all(np.isfinite(J)) else math.nan
            fisher_det = float(np.linalg.det(J)) if np.all(np.isfinite(J)) else math.nan
            try:
                fisher_cond = float(np.linalg.cond(J))
            except Exception:
                fisher_cond = math.inf
            try:
                if not np.all(np.isfinite(J)) or abs(float(np.linalg.det(J))) < 1e-30:
                    raise np.linalg.LinAlgError("bin Fisher singular")
                cov = np.linalg.inv(J)
            except Exception:
                flags.append("fisher_singular"); cov = np.linalg.pinv(J, rcond=1e-12)
            if not math.isfinite(fisher_cond) or fisher_cond > 1.0e12:
                flags.append("fisher_ill_conditioned")
            if not (math.isfinite(J00) and J00 > 0):
                flags.append("low_fisher_nu")
            if not (math.isfinite(J11) and J11 > 0):
                flags.append("low_fisher_gamma")
            delta = cov @ q if np.all(np.isfinite(cov)) and np.all(np.isfinite(q)) else np.array([math.nan, math.nan])
            cov00 = float(cov[0, 0]); cov11 = float(cov[1, 1]); cov01 = float(cov[0, 1])
            sigma_nu = math.sqrt(cov00) if math.isfinite(cov00) and cov00 >= 0 else math.nan
            sigma_lngamma = math.sqrt(cov11) if math.isfinite(cov11) and cov11 >= 0 else math.nan
            denom = sigma_nu * sigma_lngamma
            corr = cov01 / denom if math.isfinite(cov01) and math.isfinite(denom) and denom > 0 else math.nan
            used_l = sorted(set(int(x) for x in pd.to_numeric(bin_rows["l"], errors="coerce").dropna().tolist()))
            rows.append({**base, "l_list": ",".join(str(x) for x in used_l), "delta_nu": float(delta[0]), "delta_lngamma": float(delta[1]), "cov_nu_nu": cov00, "cov_lngamma_lngamma": cov11, "cov_nu_lngamma": cov01, "sigma_nu": sigma_nu, "sigma_lngamma": sigma_lngamma, "corr_nu_lngamma": corr, "fisher_trace": fisher_trace, "fisher_det": fisher_det, "fisher_cond": fisher_cond, "quality_flags": quality_join(flags)})
    return pd.DataFrame(rows, columns=BIN_COLUMNS)


def write_tables(outdir: str, tables: Mapping[str, "pd.DataFrame"], logger: logging.Logger) -> Dict[str, Dict[str, str]]:
    require_pandas(); table_dir = Path(outdir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Dict[str, str]] = {}
    versions = package_versions(); parquet_available = versions.get("pyarrow") != "not_available" or versions.get("fastparquet") != "not_available"
    for name, df in tables.items():
        if df is None:
            df = pd.DataFrame()
        csv_path = table_dir / f"{name}.csv"; df.to_csv(csv_path, index=False); outputs[name] = {"csv": str(csv_path)}
        logger.info("Wrote CSV: %s rows=%d", csv_path, len(df))
        parquet_path = table_dir / f"{name}.parquet"
        if parquet_available:
            try:
                df.to_parquet(parquet_path, index=False); outputs[name]["parquet"] = str(parquet_path); logger.info("Wrote parquet: %s", parquet_path)
            except Exception as exc:
                logger.warning("Parquet write failed for %s: %s; CSV retained.", name, exc)
        else:
            logger.warning("No pyarrow/fastparquet available; skipping parquet for %s.", name)
    return outputs


def make_diagnostic_plots(outdir: str, audit_df: "pd.DataFrame", window_df: "pd.DataFrame", mode_diag_df: "pd.DataFrame", bin_df: "pd.DataFrame", stack_diag_df: "pd.DataFrame", example_stack_df: "pd.DataFrame", logger: logging.Logger) -> None:
    plot_dir = Path(outdir) / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable; diagnostic plots skipped: %s", exc); return
    def _time_x(series):
        return pd.to_datetime(series, utc=True, errors="coerce") if pd is not None else series
    try:
        if window_df is not None and len(window_df) > 0:
            df = window_df.copy(); df["t_mid_dt"] = _time_x(df["t_mid"]); df["fill"] = pd.to_numeric(df["fill"], errors="coerce")
            plt.figure(figsize=(10, 5)); plt.scatter(df["t_mid_dt"], df["fill"], s=10); plt.xlabel("t_mid"); plt.ylabel("fill"); plt.title("MRVMT window fill vs time"); plt.tight_layout(); plt.savefig(plot_dir / "fill_vs_time.png", dpi=150); plt.close()
    except Exception as exc:
        logger.warning("fill_vs_time plot failed: %s", exc)
    try:
        if bin_df is not None and len(bin_df) > 0:
            df = bin_df.copy(); df["freq_bin_center"] = pd.to_numeric(df["freq_bin_center"], errors="coerce"); df["n_modes_used"] = pd.to_numeric(df["n_modes_used"], errors="coerce")
            grp = df.groupby(["bin_width", "freq_bin_center"], dropna=False)["n_modes_used"].mean().reset_index()
            plt.figure(figsize=(9, 5))
            for width, sub in grp.groupby("bin_width"):
                plt.plot(sub["freq_bin_center"], sub["n_modes_used"], marker="o", label=f"{width:g} uHz")
            plt.xlabel("frequency bin center [uHz]"); plt.ylabel("mean number of modes used"); plt.title("Number of modes used per bin"); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "modes_used_per_bin.png", dpi=150); plt.close()
    except Exception as exc:
        logger.warning("modes_used_per_bin plot failed: %s", exc)
    try:
        if bin_df is not None and len(bin_df) > 0:
            df = bin_df.copy(); df["freq_bin_center"] = pd.to_numeric(df["freq_bin_center"], errors="coerce"); df["fisher_trace"] = pd.to_numeric(df["fisher_trace"], errors="coerce")
            grp = df.groupby(["bin_width", "freq_bin_center"], dropna=False)["fisher_trace"].median().reset_index()
            plt.figure(figsize=(9, 5))
            for width, sub in grp.groupby("bin_width"):
                plt.plot(sub["freq_bin_center"], sub["fisher_trace"], marker="o", label=f"{width:g} uHz")
            plt.xlabel("frequency bin center [uHz]"); plt.ylabel("median Fisher trace"); plt.title("Fisher information vs frequency bin"); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "fisher_trace_vs_bin.png", dpi=150); plt.close()
    except Exception as exc:
        logger.warning("fisher_trace_vs_bin plot failed: %s", exc)
    try:
        if stack_diag_df is not None and len(stack_diag_df) > 0:
            df = stack_diag_df.copy()
            for col in ["sharpness_before", "sharpness_after_plus", "sharpness_after_minus", "sharpness_chosen"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            grp = df.groupby("l", dropna=False)[["sharpness_before", "sharpness_after_plus", "sharpness_after_minus", "sharpness_chosen"]].median().reset_index()
            plt.figure(figsize=(10, 5)); plt.plot(grp["l"], grp["sharpness_before"], marker="o", label="before/none"); plt.plot(grp["l"], grp["sharpness_after_plus"], marker="o", label="plus"); plt.plot(grp["l"], grp["sharpness_after_minus"], marker="o", label="minus"); plt.plot(grp["l"], grp["sharpness_chosen"], marker="o", label="chosen"); plt.xlabel("l"); plt.ylabel("median stack sharpness"); plt.title("Stack sharpness before/after splitting correction"); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "stack_sharpness_before_after.png", dpi=150); plt.close()
    except Exception as exc:
        logger.warning("stack_sharpness plot failed: %s", exc)
    try:
        if example_stack_df is not None and len(example_stack_df) > 0:
            df = example_stack_df.copy(); df["freq_uhz"] = pd.to_numeric(df["freq_uhz"], errors="coerce"); df["power"] = pd.to_numeric(df["power"], errors="coerce")
            for idx, (key, sub) in enumerate(list(df.groupby(["gm_id", "l", "n", "window_id"], dropna=False))[:5]):
                gm_id, ell, n, window_id = key
                plt.figure(figsize=(8, 5)); plt.plot(sub["freq_uhz"], sub["power"], marker=".", linewidth=1); plt.xlabel("frequency [uHz]"); plt.ylabel("stacked power"); plt.title(f"Example stacked spectrum gm={gm_id} l={ell} n={n} {window_id}"); plt.tight_layout(); plt.savefig(plot_dir / f"example_stacked_spectrum_{idx:02d}.png", dpi=150); plt.close()
    except Exception as exc:
        logger.warning("example stacked spectra plot failed: %s", exc)
    try:
        if bin_df is not None and len(bin_df) > 0:
            df = bin_df.copy(); df["t_mid_dt"] = _time_x(df["t_mid"])
            for col in ["freq_bin_low", "freq_bin_high", "bin_width", "delta_nu", "delta_lngamma"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            preferred = df[(np.isclose(df["freq_bin_low"], 3000.0)) & (np.isclose(df["freq_bin_high"], 3500.0)) & (np.isclose(df["bin_width"], 500.0))]
            if len(preferred) == 0:
                preferred = df.dropna(subset=["delta_nu", "delta_lngamma"]).head(200)
            plt.figure(figsize=(10, 5)); plt.plot(preferred["t_mid_dt"], preferred["delta_nu"], marker="o"); plt.xlabel("t_mid"); plt.ylabel("delta_nu [uHz]"); plt.title("delta_nu bin time series quick-look"); plt.tight_layout(); plt.savefig(plot_dir / "delta_nu_quicklook.png", dpi=150); plt.close()
            plt.figure(figsize=(10, 5)); plt.plot(preferred["t_mid_dt"], preferred["delta_lngamma"], marker="o"); plt.xlabel("t_mid"); plt.ylabel("delta_lngamma"); plt.title("delta_lngamma bin time series quick-look"); plt.tight_layout(); plt.savefig(plot_dir / "delta_lngamma_quicklook.png", dpi=150); plt.close()
    except Exception as exc:
        logger.warning("delta quicklook plot failed: %s", exc)


def audit_only_worker(task: Mapping[str, Any], args_dict: Mapping[str, Any]) -> Dict[str, Any]:
    require_pandas()
    gm_id = str(task["gm_id"]); gmonth_label = str(task["gmonth_label"]); ell = int(task["l"]); path = str(task["path"])
    audit_row, header_info, window_mask = audit_fits_file(path, ell, gm_id=gm_id, gmonth_label=gmonth_label, read_window=True, return_window=True)
    window_df = build_window_index(gm_id, header_info.get("gmonth", ""), gmonth_label, ell, path, header_info.get("ts_start", ""), int(header_info.get("n_time", 51840) or 51840), float(header_info.get("cadence_s", 60.0) or 60.0), window_mask, float(args_dict["window_days"]), float(args_dict["stride_days"]), float(args_dict["fill_min"]))
    return {"audit": audit_row, "windows": window_df.to_dict("records")}


def run_dry_run(args: argparse.Namespace, logger: logging.Logger, log_path: str) -> None:
    target_l_list = parse_l_list(args.target_l_list)
    tasks = discover_mrvmt_files(args.mrvmt_root, target_l_list, args.max_gms, logger)
    mode_table = load_mode_table(args.mode_table, logger)
    mode_table = filter_mode_table_for_targets(mode_table, target_l_list, args.freq_min, args.freq_max, logger)
    splitting = load_splitting_file(args.splitting_file, logger)
    n_internal_windows = max(int(math.floor((36.0 - args.window_days) / args.stride_days)) + 1, 0)
    planned = {"n_discovered_files": len(tasks), "n_target_l": len(target_l_list), "n_mode_table_rows": int(len(mode_table)), "n_planned_internal_windows_per_file": n_internal_windows, "n_planned_gm_l_window_jobs": int(len(tasks) * n_internal_windows), "splitting_parsed": bool(splitting.parsed), "splitting_metadata": splitting.metadata, "splitting_quality_flags": splitting.quality_flags, "dry_run_note": "No FITS arrays were read and no FFT/spectra were computed."}
    metadata_path = write_metadata(args, args.outdir, log_path, extra={"dry_run_plan": planned}, filename="metadata_preview.json")
    logger.info("Dry-run planned files: %d", len(tasks)); logger.info("Dry-run planned GM/l/window work units: %d", planned["n_planned_gm_l_window_jobs"]); logger.info("Metadata preview written: %s", metadata_path)
    print(json.dumps(planned, indent=2, sort_keys=True, default=json_default))


def run_audit_only(args: argparse.Namespace, logger: logging.Logger, log_path: str) -> None:
    target_l_list = parse_l_list(args.target_l_list)
    tasks = discover_mrvmt_files(args.mrvmt_root, target_l_list, args.max_gms, logger)
    args_dict = vars(args).copy()
    audit_rows: List[Dict[str, Any]] = []; window_rows: List[Dict[str, Any]] = []
    if args.nproc == 1:
        for task in tasks:
            try:
                result = audit_only_worker(task, args_dict); audit_rows.append(result["audit"]); window_rows.extend(result["windows"])
            except Exception as exc:
                logger.error("Audit failed for %s: %s", task, exc)
                audit_rows.append({"gm_id": task.get("gm_id", ""), "gmonth_label": task.get("gmonth_label", ""), "l_requested": task.get("l", ""), "path": task.get("path", ""), "exists": Path(str(task.get("path", ""))).exists(), "quality_flags": "parsing_failed", "rejection_reason": f"audit_exception:{exc}"})
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as executor:
            futures = [executor.submit(audit_only_worker, task, args_dict) for task in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result(); audit_rows.append(result["audit"]); window_rows.extend(result["windows"])
                except Exception as exc:
                    logger.error("Audit worker failed: %s", exc)
    audit_df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS); window_df = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    outputs = write_tables(args.outdir, {"audit": audit_df, "window_index": window_df}, logger)
    metadata_path = write_metadata(args, args.outdir, log_path, extra={"audit_only_summary": {"n_tasks": len(tasks), "n_audit_rows": len(audit_df), "n_window_rows": len(window_df), "outputs": outputs}})
    logger.info("Audit-only metadata written: %s", metadata_path)


def _collect_results_from_task_paths(results: Sequence[Mapping[str, Any]]) -> Dict[str, "pd.DataFrame"]:
    require_pandas()
    audit_parts = []; window_parts = []; mode_parts = []; stack_parts = []; example_parts = []; task_error_parts = []
    for result in results:
        paths = result.get("paths") or result.get("output_paths") or {}
        if not paths:
            continue
        audit_parts.append(_read_csv_if_exists(paths.get("audit", ""), AUDIT_COLUMNS))
        window_parts.append(_read_csv_if_exists(paths.get("window_index", ""), WINDOW_COLUMNS))
        mode_parts.append(_read_csv_if_exists(paths.get("mode_diagnostics", ""), MODE_DIAG_COLUMNS))
        stack_parts.append(_read_csv_if_exists(paths.get("stack_diagnostics", ""), STACK_DIAG_COLUMNS))
        example_parts.append(_read_csv_if_exists(paths.get("example_stacked_spectra", ""), EXAMPLE_STACK_COLUMNS))
        task_error_parts.append(_read_csv_if_exists(paths.get("task_errors", ""), TASK_ERROR_COLUMNS))
    def cat(parts: List["pd.DataFrame"], cols: Sequence[str]) -> "pd.DataFrame":
        return pd.concat(parts, ignore_index=True).reindex(columns=list(cols)) if parts else pd.DataFrame(columns=list(cols))
    return {"audit": cat(audit_parts, AUDIT_COLUMNS), "window_index": cat(window_parts, WINDOW_COLUMNS), "mode_diagnostics": cat(mode_parts, MODE_DIAG_COLUMNS), "stack_diagnostics": cat(stack_parts, STACK_DIAG_COLUMNS), "example_stacked_spectra": cat(example_parts, EXAMPLE_STACK_COLUMNS), "task_errors": cat(task_error_parts, TASK_ERROR_COLUMNS)}


def _log_task_result(logger: logging.Logger, task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    ell = int(task.get("l", result.get("l", 0)))
    status = str(result.get("status", "unknown"))
    if status == "failed":
        logger.error("Task %s l=%03d status=failed exception=%s: %s\n%s", task.get("gm_id"), ell, result.get("exception_type", ""), result.get("exception_message", ""), result.get("traceback", ""))
    elif "error" in status:
        logger.warning("Task %s l=%03d status=%s n_task_errors=%s", task.get("gm_id"), ell, status, result.get("n_task_errors", ""))
    else:
        logger.info("Task %s l=%03d status=%s", task.get("gm_id"), ell, status)


def write_minimal_failed_task_artifacts(task: Mapping[str, Any], args_dict: Mapping[str, Any], stage: str, exc: BaseException, tb: str) -> Dict[str, Any]:
    require_pandas()
    gm_id = str(task.get("gm_id", "")); gmonth_label = str(task.get("gmonth_label", "")); ell = int(task.get("l", 0)); path = str(task.get("path", ""))
    outdir = Path(str(args_dict["outdir"])); task_dir = outdir / "task_outputs"; checkpoint_dir = outdir / "checkpoints"; task_dir.mkdir(parents=True, exist_ok=True)
    task_key = make_task_key(gm_id, ell)
    audit_csv = task_dir / f"{task_key}__audit.csv"; windows_csv = task_dir / f"{task_key}__windows.csv"; mode_diag_csv = task_dir / f"{task_key}__mode_diagnostics.csv"; stack_diag_csv = task_dir / f"{task_key}__stack_diagnostics.csv"; example_csv = task_dir / f"{task_key}__example_stacked_spectra.csv"; task_errors_csv = task_dir / f"{task_key}__task_errors.csv"; error_txt = task_dir / f"{task_key}__error.txt"
    output_paths = {"audit": str(audit_csv), "window_index": str(windows_csv), "mode_diagnostics": str(mode_diag_csv), "stack_diagnostics": str(stack_diag_csv), "example_stacked_spectra": str(example_csv), "task_errors": str(task_errors_csv)}
    error_txt.write_text(tb, encoding="utf-8")
    audit_df = pd.DataFrame([{"gm_id": gm_id, "gmonth_label": gmonth_label, "l_requested": ell, "path": path, "exists": Path(path).exists(), "quality_flags": "parsing_failed;task_exception", "rejection_reason": f"{stage}_exception"}], columns=AUDIT_COLUMNS)
    err_row = make_task_error_row(gm_id, gmonth_label, ell, path, stage, "failed", exc=exc, tb=tb, error_file=str(error_txt), quality_flags=["parsing_failed", "task_exception"])
    save_df_csv(audit_df, audit_csv, AUDIT_COLUMNS); save_df_csv(_empty_df(WINDOW_COLUMNS), windows_csv, WINDOW_COLUMNS); save_df_csv(_empty_df(MODE_DIAG_COLUMNS), mode_diag_csv, MODE_DIAG_COLUMNS); save_df_csv(_empty_df(STACK_DIAG_COLUMNS), stack_diag_csv, STACK_DIAG_COLUMNS); save_df_csv(_empty_df(EXAMPLE_STACK_COLUMNS), example_csv, EXAMPLE_STACK_COLUMNS)
    ckpt = checkpoint_done(checkpoint_dir, gm_id, ell, {"status": "failed", "output_paths": output_paths, "error_file": str(error_txt), "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": tb, "quality_flags": ["parsing_failed", "task_exception"], "n_task_errors": 1})
    err_row["checkpoint"] = ckpt
    save_df_csv(pd.DataFrame([err_row], columns=TASK_ERROR_COLUMNS), task_errors_csv, TASK_ERROR_COLUMNS)
    return {"status": "failed", "paths": output_paths, "checkpoint": ckpt, "error_file": str(error_txt), "exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": tb, "task_error": err_row}


def run_science(args: argparse.Namespace, logger: logging.Logger, log_path: str) -> None:
    target_l_list = parse_l_list(args.target_l_list)
    tasks = discover_mrvmt_files(args.mrvmt_root, target_l_list, args.max_gms, logger)
    mode_table = load_mode_table(args.mode_table, logger)
    mode_table = filter_mode_table_for_targets(mode_table, target_l_list, args.freq_min, args.freq_max, logger)
    splitting = load_splitting_file(args.splitting_file, logger)
    mode_records_by_l: Dict[int, List[Dict[str, Any]]] = {int(ell): sub.sort_values(["nu_ref", "n"]).to_dict("records") for ell, sub in mode_table.groupby("l")}
    checkpoint_dir = Path(args.outdir) / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []; pending: List[Dict[str, Any]] = []
    for task in tasks:
        ckpt_path = checkpoint_path(checkpoint_dir, task["gm_id"], int(task["l"]))
        if ckpt_path.exists() and not args.overwrite:
            ckpt = load_checkpoint(ckpt_path)
            if ckpt and ckpt.get("output_paths"):
                paths = ckpt["output_paths"]
                if all(Path(str(p)).exists() for p in paths.values()):
                    logger.info("Skipping completed GM/l via checkpoint: %s l=%03d", task["gm_id"], int(task["l"]))
                    results.append({"status": "skipped_checkpoint", "paths": paths, "checkpoint": str(ckpt_path)})
                    continue
        pending.append(task)
    args_dict = vars(args).copy()
    logger.info("Processing %d pending GM/l tasks with nproc=%d", len(pending), args.nproc)
    if args.nproc == 1:
        for task in pending:
            ell = int(task["l"])
            result = process_gm_l_task(task, args_dict, mode_records_by_l.get(ell, []), splitting)
            _log_task_result(logger, task, result)
            results.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as executor:
            future_to_task = {executor.submit(process_gm_l_task, task, args_dict, mode_records_by_l.get(int(task["l"]), []), splitting): task for task in pending}
            for fut in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[fut]
                try:
                    result = fut.result(); _log_task_result(logger, task, result); results.append(result)
                except Exception as exc:
                    tb = traceback.format_exc()
                    logger.error("Task %s l=%03d failed in executor: %s\n%s", task["gm_id"], int(task["l"]), exc, tb)
                    results.append(write_minimal_failed_task_artifacts(task, args_dict, "executor", exc, tb))
    tables = _collect_results_from_task_paths(results)
    tables["bin_time_series"] = accumulate_frequency_bins(tables["mode_diagnostics"], tables["window_index"], mode_table, args, logger)
    outputs = write_tables(args.outdir, {"audit": tables["audit"], "window_index": tables["window_index"], "mode_diagnostics": tables["mode_diagnostics"], "bin_time_series": tables["bin_time_series"], "stack_diagnostics": tables["stack_diagnostics"], "example_stacked_spectra": tables["example_stacked_spectra"], "task_errors": tables["task_errors"]}, logger)
    make_diagnostic_plots(args.outdir, tables["audit"], tables["window_index"], tables["mode_diagnostics"], tables["bin_time_series"], tables["stack_diagnostics"], tables["example_stacked_spectra"], logger)
    chosen_shift_signs: Dict[str, Any] = {}
    stack_df = tables["stack_diagnostics"]
    if stack_df is not None and len(stack_df) > 0:
        for (gm_id, ell), sub in stack_df.groupby(["gm_id", "l"], dropna=False):
            counts = Counter(str(x) for x in sub["chosen_shift_sign"].dropna().tolist())
            chosen_shift_signs[f"{gm_id}/l{int(ell):03d}"] = {"counts_by_window": dict(counts), "window_signs": {str(row["window_id"]): str(row["chosen_shift_sign"]) for _, row in sub.iterrows()}}
    quality_all: List[str] = []
    for _name, df in tables.items():
        if isinstance(df, pd.DataFrame) and "quality_flags" in df.columns:
            for item in df["quality_flags"].dropna().tolist():
                quality_all.extend(quality_split(item))
    if len(tables["window_index"]) == 0 or not tables["window_index"].get("accepted", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"]).any():
        quality_all.append("no_valid_windows")
    metadata_path = write_metadata(args, args.outdir, log_path, extra={"science_summary": {"n_discovered_tasks": len(tasks), "n_pending_processed": len(pending), "n_results_collected": len(results), "n_audit_rows": int(len(tables["audit"])), "n_window_rows": int(len(tables["window_index"])), "n_mode_diagnostic_rows": int(len(tables["mode_diagnostics"])), "n_bin_rows": int(len(tables["bin_time_series"])), "n_task_error_rows": int(len(tables["task_errors"])), "quality_flags_seen": sorted(set(quality_all)), "outputs": outputs}, "splitting": {"splitting_file": splitting.path, "parsed": splitting.parsed, "metadata": splitting.metadata, "warnings": splitting.warnings, "quality_flags": splitting.quality_flags}, "chosen_shift_signs": chosen_shift_signs})
    logger.info("Science metadata written: %s", metadata_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logging(args.outdir, args.log_level)
    logger.info("Command line: %s", " ".join([sys.argv[0]] + sys.argv[1:]))
    logger.info("Hostname: %s", socket.gethostname())
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("Package versions: %s", package_versions())
    logger.info("Input MRVMT root: %s", args.mrvmt_root)
    logger.info("Mode table: %s", args.mode_table)
    logger.info("Splitting file: %s", args.splitting_file)
    logger.info("Output directory: %s", args.outdir)
    try:
        require_pandas()
        if args.dry_run:
            run_dry_run(args, logger, log_path)
        elif args.audit_only:
            run_audit_only(args, logger, log_path)
        else:
            run_science(args, logger, log_path)
        logger.info("Completed gong_lag_v4_mvp_ab.py")
        return 0
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        logger.error(traceback.format_exc())
        try:
            write_metadata(args, args.outdir, log_path, extra={"fatal_error": str(exc), "traceback": traceback.format_exc(), "quality_flags_seen": ["parsing_failed"]}, filename="metadata_failed.json")
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
