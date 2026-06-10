#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gong_lag_v4_mvp_c_hmi_degradation_injection.py

MVP-C: HMI-to-GONG degradation / known-lag injection validation.

Primary supported HMI formats:
  1. canonical_long:
       t_mid, freq_bin_low, freq_bin_high, delta_nu, delta_lngamma or delta_gamma
  2. extended_bins_12:
       wide-format 12-bin table with mid_date/start_date and
       dnu_bin{k}, dnu_err_bin{k}, dw_bin{k}, dw_err_bin{k}, nmodes_bin{k},
       for k=1..12 and fixed edges:
       1500,2000,2500,3000,3500,3750,4000,4500,5000,5500,6000,6500,7000

Scientific scope:
  validation-only; no final GONG lag claim; no full WMLTP; no leakage/window convolution.

Lag convention:
  Positive lag means the linewidth variable leads delta_nu.

Operational estimator:
  C(tau) = corr[gamma_residual(t), nu_residual(t + tau)]

Injection:
  tau_inj = +6 hr is implemented as gamma_injected(t) = gamma_original(t + 6 hr),
  so the validation estimator should peak near +6 hr if unbiased.
"""

from __future__ import annotations

import os

SLURM_CPUS_PER_TASK_RAW = os.environ.get("SLURM_CPUS_PER_TASK", "1")
try:
    SLURM_CPUS_PER_TASK = max(1, int(SLURM_CPUS_PER_TASK_RAW))
except Exception:
    SLURM_CPUS_PER_TASK = 1

# Numerical threading must be set before numpy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import datetime as dt
import importlib
import json
import logging
import math
import platform
import socket
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


GONG_RUN_DEFAULT = (
    "/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/"
    "runs/q1_prod_hmi2010_split_auto_all"
)

PRIMARY_HMI_DEFAULT = "/scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv"

EXTENDED_BINS_12_EDGES = [
    1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 3750.0, 4000.0,
    4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0,
]

HMI_ALIASES = {
    "t_mid": ["t_mid", "time", "date", "datetime", "timestamp", "mid_time", "window_mid", "mid_date"],
    "t_start": ["t_start", "start_time", "window_start", "start_date"],
    "t_end": ["t_end", "end_time", "window_end", "end_date"],
    "freq_bin_low": ["freq_bin_low", "nu_low", "frequency_low", "freq_low", "bin_low", "low_uhz", "nu_min"],
    "freq_bin_high": ["freq_bin_high", "nu_high", "frequency_high", "freq_high", "bin_high", "high_uhz", "nu_max"],
    "freq_bin_center": ["freq_bin_center", "nu_center", "frequency_center", "freq_center", "center_uhz", "nu", "frequency"],
    "bin_width": ["bin_width", "width", "freq_bin_width", "dnu_bin"],
    "delta_nu": ["delta_nu", "dnu", "nu_residual", "frequency_residual", "delta_frequency", "nu_shift", "dnu_resid"],
    "delta_lngamma": ["delta_lngamma", "delta_ln_gamma", "delta_lnGamma", "dlnGamma", "dln_gamma", "lngamma_residual", "ln_gamma_residual", "delta_ln_width", "dlnwidth"],
    "delta_gamma": ["delta_gamma", "dgamma", "delta_width", "width_residual", "gamma_residual", "fwhm_residual", "delta_fwhm", "dw"],
    "sigma_nu": ["sigma_nu", "err_nu", "nu_error", "dnu_error"],
    "sigma_lngamma": ["sigma_lngamma", "err_lngamma", "lngamma_error", "dlnGamma_error", "dln_gamma_error"],
    "sigma_gamma": ["sigma_gamma", "err_gamma", "gamma_error", "width_error", "dw_error"],
    "quality_flags": ["quality_flags", "flags", "flag", "quality", "bad_flags"],
    "nmodes": ["nmodes", "n_modes", "mode_count"],
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def time_gmtime(*args: Any) -> Any:
    import time
    return time.gmtime(*args)


def package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ["numpy", "pandas", "pyarrow", "fastparquet", "matplotlib"]:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "not_available"
    return out


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def normalize_name(name: Any) -> str:
    text = str(name).strip()
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("ν", "nu").replace("Γ", "gamma").replace("γ", "gamma")
    keep = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            keep.append(ch.lower())
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def quality_split(text: Any) -> List[str]:
    if text is None:
        return []
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return []
    return [x for x in s.split(";") if x]


def quality_join(flags: Iterable[Any]) -> str:
    clean = []
    for flag in flags:
        if flag is None:
            continue
        s = str(flag).strip()
        if s and s.lower() != "nan":
            clean.append(s)
    return ";".join(sorted(set(clean)))


def parse_csv_floats(spec: str, name: str) -> List[float]:
    out: List[float] = []
    for token in str(spec).replace(" ", "").split(","):
        if not token:
            continue
        try:
            out.append(float(token))
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"{name} contains non-float token: {token}") from exc
    if not out:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return out


def parse_pair(spec: str, name: str) -> Tuple[float, float]:
    vals = parse_csv_floats(spec, name)
    if len(vals) != 2:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly two comma-separated values")
    if vals[1] <= vals[0]:
        raise argparse.ArgumentTypeError(f"{name}: upper bound must exceed lower bound")
    return vals[0], vals[1]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GONG-Lag v4 MVP-C HMI-to-GONG degradation / injection validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--hmi-bin-timeseries", required=True, help="Explicit HMI bin time-series path; CSV or parquet.")
    parser.add_argument("--hmi-format", choices=["auto", "canonical_long", "extended_bins_12"], default="auto")
    parser.add_argument("--gong-run", default=GONG_RUN_DEFAULT, help="Accepted GONG/Q1 split-auto run directory.")
    parser.add_argument("--outdir", required=True, help="Output directory under /scratch1/dongpoba/gong_lag_v4/...")

    parser.add_argument("--freq-min", type=float, default=2000.0)
    parser.add_argument("--freq-max", type=float, default=4500.0)
    parser.add_argument("--bin-widths", default="250,500", help="Comma-separated bin widths in microHz.")
    parser.add_argument("--focus-bin-low", type=float, default=3000.0)
    parser.add_argument("--focus-bin-high", type=float, default=3500.0)

    parser.add_argument("--degradation-kernel-days", type=float, default=9.0)
    parser.add_argument("--target-stride-days", type=float, default=3.0)
    parser.add_argument("--detrend-days", type=float, default=81.0)

    parser.add_argument("--inject-lags-hours", default="-24,-12,0,6,12,24")
    parser.add_argument(
        "--injection-target",
        choices=["auto", "delta_lngamma", "delta_gamma"],
        default="auto",
        help="Linewidth-like column to shift relative to delta_nu.",
    )
    parser.add_argument("--n-injection-realizations", type=int, default=10)

    parser.add_argument(
        "--noise-model",
        choices=["none", "gaussian_covariance", "bootstrap_residual"],
        default="gaussian_covariance",
    )
    parser.add_argument("--covariance-scale", type=float, default=1.0)

    parser.add_argument("--lag-search-hours", default="-72,72")
    parser.add_argument("--lag-step-hours", type=float, default=72.0)
    parser.add_argument("--lag-weighting", choices=["unweighted", "inverse_variance"], default="unweighted")

    parser.add_argument("--bootstrap-n", type=int, default=200)
    parser.add_argument("--bootstrap-block-days", type=float, default=108.0)
    parser.add_argument("--random-seed", type=int, default=20260610)

    parser.add_argument("--apply-gong-missing-mask", dest="apply_gong_missing_mask", action="store_true", default=True)
    parser.add_argument("--no-gong-missing-mask", dest="apply_gong_missing_mask", action="store_false")
    parser.add_argument(
        "--finite-quality-mask",
        choices=["off", "basic", "strict"],
        default="basic",
        help="basic masks non-finite target covariance/time series; strict also masks non-empty GONG quality_flags.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")

    args = parser.parse_args(argv)

    if args.freq_max <= args.freq_min:
        parser.error("--freq-max must exceed --freq-min")
    if args.focus_bin_high <= args.focus_bin_low:
        parser.error("--focus-bin-high must exceed --focus-bin-low")
    if args.degradation_kernel_days <= 0:
        parser.error("--degradation-kernel-days must be positive")
    if args.target_stride_days <= 0:
        parser.error("--target-stride-days must be positive")
    if args.detrend_days <= 0:
        parser.error("--detrend-days must be positive")
    if args.lag_step_hours <= 0:
        parser.error("--lag-step-hours must be positive")
    if args.bootstrap_n < 0:
        parser.error("--bootstrap-n must be non-negative")
    if args.bootstrap_block_days <= 0:
        parser.error("--bootstrap-block-days must be positive")
    if args.n_injection_realizations < 1:
        parser.error("--n-injection-realizations must be >= 1")
    if args.covariance_scale < 0:
        parser.error("--covariance-scale must be non-negative")
    parse_csv_floats(args.bin_widths, "--bin-widths")
    parse_csv_floats(args.inject_lags_hours, "--inject-lags-hours")
    parse_pair(args.lag_search_hours, "--lag-search-hours")
    return args


def setup_outdir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is non-empty; use --overwrite to reuse: {outdir}")
    for sub in ["logs", "tables", "plots"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)


def setup_logging(outdir: Path, log_level: str) -> logging.Logger:
    log_path = outdir / "logs" / f"mvp_c_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logger = logging.getLogger("gong_lag_v4_mvp_c")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S")
    fmt.converter = time_gmtime

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, log_level.upper()))
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.info("Logging to %s", log_path)
    return logger


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suffix in [".csv", ".txt", ".dat", ""]:
        return pd.read_csv(path, comment="#")
    raise ValueError(f"Unsupported table extension for {path}; expected CSV/parquet")


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rename_by_aliases(df: pd.DataFrame, aliases: Mapping[str, Sequence[str]]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    norm_to_original = {normalize_name(c): c for c in df.columns}
    rename: Dict[str, str] = {}
    used: Dict[str, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in df.columns:
            used[canonical] = canonical
            continue
        for cand in candidates:
            key = normalize_name(cand)
            if key in norm_to_original:
                original = norm_to_original[key]
                rename[original] = canonical
                used[canonical] = original
                break
    return df.rename(columns=rename).copy(), used


def add_time_hours(df: pd.DataFrame, column: str = "t_mid") -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[column], utc=True, errors="coerce")
    out[column] = t
    hours = np.full(len(out), np.nan, dtype=float)
    good = t.notna().to_numpy()
    if np.any(good):
        hours[good] = t[good].astype("int64").astype(float).to_numpy() / 3.6e12
    out[f"{column}_hours"] = hours
    return out


def extended_col(raw: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    norm = {normalize_name(c): c for c in raw.columns}
    for c in candidates:
        if normalize_name(c) in norm:
            return norm[normalize_name(c)]
    return None


def normalize_hmi_extended_bins_12(raw: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    mid_col = extended_col(raw, ["mid_date", "t_mid", "mid_time"])
    start_col = extended_col(raw, ["start_date", "t_start", "start_time"])
    if mid_col is None:
        raise ValueError("extended_bins_12 requires mid_date")
    if start_col is None:
        raise ValueError("extended_bins_12 requires start_date")

    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    for k in range(1, 13):
        dnu_col = extended_col(raw, [f"dnu_bin{k}", f"dnu_bin_{k}", f"dnu{k}"])
        dnu_err_col = extended_col(raw, [f"dnu_err_bin{k}", f"dnu_err_bin_{k}", f"dnu_error_bin{k}", f"dnuerr_bin{k}"])
        dw_col = extended_col(raw, [f"dw_bin{k}", f"dw_bin_{k}", f"delta_width_bin{k}", f"dgamma_bin{k}"])
        dw_err_col = extended_col(raw, [f"dw_err_bin{k}", f"dw_err_bin_{k}", f"dw_error_bin{k}", f"dwerr_bin{k}"])
        nmodes_col = extended_col(raw, [f"nmodes_bin{k}", f"nmodes_bin_{k}", f"n_modes_bin{k}", f"nmode_bin{k}"])

        for label, col in [(f"dnu_bin{k}", dnu_col), (f"dw_bin{k}", dw_col)]:
            if col is None:
                missing_required.append(label)
        if dnu_col is None or dw_col is None:
            continue

        low = EXTENDED_BINS_12_EDGES[k - 1]
        high = EXTENDED_BINS_12_EDGES[k]
        for idx, r in raw.iterrows():
            t_start = pd.to_datetime(r[start_col], utc=True, errors="coerce")
            t_mid = pd.to_datetime(r[mid_col], utc=True, errors="coerce")
            t_end = t_start + pd.Timedelta(days=3) if pd.notna(t_start) else pd.NaT
            rows.append({
                "source_row": int(idx),
                "hmi_native_bin_index": int(k),
                "t_start": t_start,
                "t_mid": t_mid,
                "t_end": t_end,
                "freq_bin_low": low,
                "freq_bin_high": high,
                "freq_bin_center": 0.5 * (low + high),
                "bin_width": high - low,
                "delta_nu": pd.to_numeric(pd.Series([r[dnu_col]]), errors="coerce").iloc[0],
                "sigma_nu": pd.to_numeric(pd.Series([r[dnu_err_col]]), errors="coerce").iloc[0] if dnu_err_col else np.nan,
                "delta_gamma": pd.to_numeric(pd.Series([r[dw_col]]), errors="coerce").iloc[0],
                "sigma_gamma": pd.to_numeric(pd.Series([r[dw_err_col]]), errors="coerce").iloc[0] if dw_err_col else np.nan,
                "nmodes": pd.to_numeric(pd.Series([r[nmodes_col]]), errors="coerce").iloc[0] if nmodes_col else np.nan,
                "quality_flags": "",
            })

    if missing_required:
        raise ValueError("extended_bins_12 missing required columns: " + ",".join(missing_required[:20]))

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("extended_bins_12 normalization produced zero rows")

    # Do not fabricate delta_lngamma.  Primary linewidth variable is delta_gamma.
    gamma_col = "delta_gamma"
    df = add_time_hours(df, "t_mid")
    df = add_time_hours(df, "t_start")
    df = add_time_hours(df, "t_end")

    for c in ["freq_bin_low", "freq_bin_high", "bin_width", "delta_nu", "sigma_nu", "delta_gamma", "sigma_gamma", "nmodes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = (
        np.isfinite(df["t_mid_hours"])
        & np.isfinite(df["freq_bin_low"])
        & np.isfinite(df["freq_bin_high"])
        & (df["freq_bin_high"] > df["freq_bin_low"])
    )
    df = df[keep].copy()

    audit = {
        "hmi_format": "extended_bins_12",
        "original_columns": list(raw.columns),
        "canonical_columns": list(df.columns),
        "fixed_bin_edges_uhz": EXTENDED_BINS_12_EDGES,
        "gamma_column_used": gamma_col,
        "delta_lngamma_fabricated": False,
        "primary_linewidth_variable": "delta_gamma",
        "t_mid_source_column": mid_col,
        "t_start_source_column": start_col,
        "t_end_rule": "start_date + 3 days",
        "n_rows_after_basic_time_freq_filter": int(len(df)),
        "time_min": df["t_mid"].min().isoformat() if len(df) else "",
        "time_max": df["t_mid"].max().isoformat() if len(df) else "",
        "freq_min_seen": float(df["freq_bin_low"].min()) if len(df) else None,
        "freq_max_seen": float(df["freq_bin_high"].max()) if len(df) else None,
        "raw_shift_reference": "raw shift relative to solar-min reference window; 81-day detrending applied downstream",
    }
    return df, gamma_col, audit


def normalize_hmi_canonical_long(raw: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    df, used = rename_by_aliases(raw, HMI_ALIASES)

    if "t_mid" not in df.columns:
        raise ValueError("HMI table missing t_mid/time/date column")
    df = add_time_hours(df, "t_mid")

    if "t_start" in df.columns:
        df = add_time_hours(df, "t_start")
    if "t_end" in df.columns:
        df = add_time_hours(df, "t_end")

    if "freq_bin_low" not in df.columns or "freq_bin_high" not in df.columns:
        if "freq_bin_center" in df.columns and "bin_width" in df.columns:
            center = pd.to_numeric(df["freq_bin_center"], errors="coerce")
            width = pd.to_numeric(df["bin_width"], errors="coerce")
            df["freq_bin_low"] = center - 0.5 * width
            df["freq_bin_high"] = center + 0.5 * width
        else:
            raise ValueError("HMI table must contain freq_bin_low/freq_bin_high or freq_bin_center/bin_width")

    df["freq_bin_low"] = pd.to_numeric(df["freq_bin_low"], errors="coerce")
    df["freq_bin_high"] = pd.to_numeric(df["freq_bin_high"], errors="coerce")
    df["bin_width"] = df["freq_bin_high"] - df["freq_bin_low"]

    if "delta_nu" not in df.columns:
        raise ValueError("HMI table missing delta_nu/dnu column")
    df["delta_nu"] = pd.to_numeric(df["delta_nu"], errors="coerce")

    if args.injection_target == "auto":
        if "delta_lngamma" in df.columns:
            gamma_col = "delta_lngamma"
        elif "delta_gamma" in df.columns:
            gamma_col = "delta_gamma"
        else:
            raise ValueError("HMI table missing linewidth residual column: delta_lngamma or delta_gamma")
    else:
        gamma_col = args.injection_target
        if gamma_col not in df.columns:
            raise ValueError(f"HMI table missing requested --injection-target column: {gamma_col}")

    df[gamma_col] = pd.to_numeric(df[gamma_col], errors="coerce")

    for col in ["sigma_nu", "sigma_lngamma", "sigma_gamma", "nmodes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = (
        np.isfinite(df["t_mid_hours"])
        & np.isfinite(df["freq_bin_low"])
        & np.isfinite(df["freq_bin_high"])
        & (df["freq_bin_high"] > df["freq_bin_low"])
    )
    df = df[keep].copy()

    audit = {
        "hmi_format": "canonical_long",
        "original_columns": list(raw.columns),
        "canonical_columns": list(df.columns),
        "alias_map": used,
        "gamma_column_used": gamma_col,
        "delta_lngamma_fabricated": False,
        "primary_linewidth_variable": gamma_col,
        "n_rows_after_basic_time_freq_filter": int(len(df)),
        "time_min": df["t_mid"].min().isoformat() if len(df) else "",
        "time_max": df["t_mid"].max().isoformat() if len(df) else "",
        "freq_min_seen": float(df["freq_bin_low"].min()) if len(df) else None,
        "freq_max_seen": float(df["freq_bin_high"].max()) if len(df) else None,
    }
    return df, gamma_col, audit


def normalize_hmi_table(raw: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    fmt = args.hmi_format
    if fmt == "extended_bins_12":
        return normalize_hmi_extended_bins_12(raw, args)
    if fmt == "canonical_long":
        return normalize_hmi_canonical_long(raw, args)

    # auto: detect explicit wide-format columns first.
    norm_cols = {normalize_name(c) for c in raw.columns}
    extended_markers = {"mid_date", "start_date", "dnu_bin1", "dw_bin1", "dnu_bin12", "dw_bin12"}
    if extended_markers.issubset(norm_cols):
        return normalize_hmi_extended_bins_12(raw, args)
    return normalize_hmi_canonical_long(raw, args)


def normalize_gong_bin_table(raw: pd.DataFrame) -> pd.DataFrame:
    # MVP-A/B accepted bin_time_series.csv uses fill_median/fill_min, not always a
    # single column named fill.  MVP-C needs one effective fill column for masking.
    required_core = [
        "t_start", "t_mid", "t_end", "freq_bin_low", "freq_bin_high", "bin_width",
        "cov_nu_nu", "cov_lngamma_lngamma", "cov_nu_lngamma",
    ]
    missing = [c for c in required_core if c not in raw.columns]
    if missing:
        raise ValueError(f"GONG bin_time_series.csv missing core columns: {missing}")

    df = raw.copy()

    if "quality_flags" not in df.columns:
        df["quality_flags"] = ""

    # Backward/forward compatible fill mapping:
    #   preferred: fill
    #   accepted MVP-A/B output: fill_median, fill_min
    #   fallback only if neither is present: 1.0 with a diagnostic-compatible value
    if "fill" not in df.columns:
        if "fill_median" in df.columns:
            df["fill"] = df["fill_median"]
        elif "fill_min" in df.columns:
            df["fill"] = df["fill_min"]
        else:
            df["fill"] = 1.0

    # Derive sigma columns from covariance if absent.
    if "sigma_nu" not in df.columns:
        df["sigma_nu"] = np.sqrt(pd.to_numeric(df["cov_nu_nu"], errors="coerce").clip(lower=0))
    if "sigma_lngamma" not in df.columns:
        df["sigma_lngamma"] = np.sqrt(pd.to_numeric(df["cov_lngamma_lngamma"], errors="coerce").clip(lower=0))

    for col in ["t_start", "t_mid", "t_end"]:
        df = add_time_hours(df, col)

    numeric = [
        "freq_bin_low", "freq_bin_high", "bin_width",
        "cov_nu_nu", "cov_lngamma_lngamma", "cov_nu_lngamma",
        "sigma_nu", "sigma_lngamma", "fill", "fill_median", "fill_min",
        "delta_nu", "delta_lngamma",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "gm_id" not in df.columns:
        df["gm_id"] = ""
    if "gmonth" not in df.columns:
        df["gmonth"] = ""
    return df


def load_gong_run(gong_run: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    table_dir = gong_run / "tables"
    bin_path = table_dir / "bin_time_series.csv"
    window_path = table_dir / "window_index.csv"
    audit_path = table_dir / "audit.csv"
    meta_path = gong_run / "metadata.json"

    missing = [p for p in [bin_path, window_path, audit_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"GONG run is missing required table(s): {missing}")

    gong_bin = normalize_gong_bin_table(pd.read_csv(bin_path))
    window_index = pd.read_csv(window_path)
    audit = pd.read_csv(audit_path)
    meta = read_json_if_exists(meta_path)
    return gong_bin, window_index, audit, meta


def requested_bins(args: argparse.Namespace) -> pd.DataFrame:
    widths = parse_csv_floats(args.bin_widths, "--bin-widths")
    rows = []
    for width in widths:
        edge = float(args.freq_min)
        while edge + width <= float(args.freq_max) + 1e-9:
            rows.append({
                "freq_bin_low": edge,
                "freq_bin_high": edge + width,
                "freq_bin_center": edge + 0.5 * width,
                "bin_width": width,
            })
            edge += width
    if not rows:
        raise ValueError("No requested frequency bins were generated")
    return pd.DataFrame(rows)


def valid_cov(c00: float, c11: float, c01: float) -> bool:
    if not all(math.isfinite(x) for x in [c00, c11, c01]):
        return False
    if c00 < 0 or c11 < 0:
        return False
    return c00 * c11 - c01 * c01 >= -1e-18


def nearest_psd_cov(c00: float, c11: float, c01: float) -> Optional[np.ndarray]:
    if not all(math.isfinite(x) for x in [c00, c11, c01]):
        return None
    C = np.array([[c00, c01], [c01, c11]], dtype=float)
    C = 0.5 * (C + C.T)
    try:
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0.0, None)
        C2 = (vecs * vals) @ vecs.T
        if not np.all(np.isfinite(C2)):
            return None
        return 0.5 * (C2 + C2.T)
    except Exception:
        return None


def combine_gong_components(components: pd.DataFrame, low: float, high: float) -> Dict[str, Any]:
    flags: List[str] = []
    if len(components) == 0:
        return {
            "delta_nu_gong": np.nan,
            "delta_lngamma_gong": np.nan,
            "cov_nu_nu": np.nan,
            "cov_lngamma_lngamma": np.nan,
            "cov_nu_lngamma": np.nan,
            "sigma_nu": np.nan,
            "sigma_lngamma": np.nan,
            "fill": np.nan,
            "quality_flags": "missing_gong_covariance",
            "grid_mapping": "missing",
            "n_gong_components": 0,
        }

    exact = components[
        np.isclose(components["freq_bin_low"], low, atol=1e-6)
        & np.isclose(components["freq_bin_high"], high, atol=1e-6)
    ]
    if len(exact) >= 1:
        row = exact.iloc[0]
        flags = quality_split(row.get("quality_flags", ""))
        if not valid_cov(float(row["cov_nu_nu"]), float(row["cov_lngamma_lngamma"]), float(row["cov_nu_lngamma"])):
            flags.append("invalid_gong_covariance")
        return {
            "delta_nu_gong": float(row.get("delta_nu", np.nan)),
            "delta_lngamma_gong": float(row.get("delta_lngamma", np.nan)),
            "cov_nu_nu": float(row["cov_nu_nu"]),
            "cov_lngamma_lngamma": float(row["cov_lngamma_lngamma"]),
            "cov_nu_lngamma": float(row["cov_nu_lngamma"]),
            "sigma_nu": float(row["sigma_nu"]),
            "sigma_lngamma": float(row["sigma_lngamma"]),
            "fill": float(row["fill"]),
            "quality_flags": quality_join(flags),
            "grid_mapping": "exact",
            "n_gong_components": 1,
        }

    sub = components[
        (components["freq_bin_low"] >= low - 1e-6)
        & (components["freq_bin_high"] <= high + 1e-6)
        & (components["freq_bin_high"] > components["freq_bin_low"])
    ].copy()
    if len(sub) == 0:
        flags.append("missing_gong_covariance")
        return {
            "delta_nu_gong": np.nan,
            "delta_lngamma_gong": np.nan,
            "cov_nu_nu": np.nan,
            "cov_lngamma_lngamma": np.nan,
            "cov_nu_lngamma": np.nan,
            "sigma_nu": np.nan,
            "sigma_lngamma": np.nan,
            "fill": np.nan,
            "quality_flags": quality_join(flags),
            "grid_mapping": "missing",
            "n_gong_components": 0,
        }

    for q in sub.get("quality_flags", pd.Series(dtype=str)).fillna("").tolist():
        flags.extend(quality_split(q))

    info = np.zeros((2, 2), dtype=float)
    info_y = np.zeros(2, dtype=float)
    n_info = 0
    for _, r in sub.iterrows():
        C = nearest_psd_cov(float(r["cov_nu_nu"]), float(r["cov_lngamma_lngamma"]), float(r["cov_nu_lngamma"]))
        y = np.array([float(r.get("delta_nu", np.nan)), float(r.get("delta_lngamma", np.nan))], dtype=float)
        if C is None or not np.all(np.isfinite(y)):
            continue
        try:
            I = np.linalg.pinv(C, rcond=1e-14)
            info += I
            info_y += I @ y
            n_info += 1
        except Exception:
            continue

    if n_info > 0:
        Cc = np.linalg.pinv(info, rcond=1e-14)
        yc = Cc @ info_y
        cov00, cov11, cov01 = float(Cc[0, 0]), float(Cc[1, 1]), float(Cc[0, 1])
        dnu, dlng = float(yc[0]), float(yc[1])
        mapping = "composite_inverse_covariance"
    else:
        flags.append("invalid_gong_covariance")
        cov00 = float(np.nanmean(sub["cov_nu_nu"]))
        cov11 = float(np.nanmean(sub["cov_lngamma_lngamma"]))
        cov01 = float(np.nanmean(sub["cov_nu_lngamma"]))
        dnu = float(np.nanmean(sub.get("delta_nu", np.nan)))
        dlng = float(np.nanmean(sub.get("delta_lngamma", np.nan)))
        mapping = "composite_mean_covariance_fallback"

    return {
        "delta_nu_gong": dnu,
        "delta_lngamma_gong": dlng,
        "cov_nu_nu": cov00,
        "cov_lngamma_lngamma": cov11,
        "cov_nu_lngamma": cov01,
        "sigma_nu": math.sqrt(cov00) if math.isfinite(cov00) and cov00 >= 0 else np.nan,
        "sigma_lngamma": math.sqrt(cov11) if math.isfinite(cov11) and cov11 >= 0 else np.nan,
        "fill": float(np.nanmedian(sub["fill"])),
        "quality_flags": quality_join(flags),
        "grid_mapping": mapping,
        "n_gong_components": int(len(sub)),
    }


def build_target_grid(gong_bin: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    time_cols = ["t_start", "t_mid", "t_end", "t_start_hours", "t_mid_hours", "t_end_hours", "gm_id", "gmonth"]
    time_grid = gong_bin[time_cols].drop_duplicates(subset=["t_mid_hours"]).sort_values("t_mid_hours").reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for _, tr in time_grid.iterrows():
        gsub = gong_bin[np.isclose(gong_bin["t_mid_hours"], float(tr["t_mid_hours"]), atol=1e-9)]
        for _, br in bins.iterrows():
            low = float(br["freq_bin_low"])
            high = float(br["freq_bin_high"])
            comp = combine_gong_components(gsub, low, high)
            rows.append({
                "t_start": tr["t_start"],
                "t_mid": tr["t_mid"],
                "t_end": tr["t_end"],
                "t_start_hours": float(tr["t_start_hours"]),
                "t_mid_hours": float(tr["t_mid_hours"]),
                "t_end_hours": float(tr["t_end_hours"]),
                "gm_id": tr.get("gm_id", ""),
                "gmonth": tr.get("gmonth", ""),
                "freq_bin_low": low,
                "freq_bin_high": high,
                "freq_bin_center": float(br["freq_bin_center"]),
                "bin_width": float(br["bin_width"]),
                **comp,
            })
    return pd.DataFrame(rows)


@dataclass
class NativeBinSeries:
    freq_bin_low: float
    freq_bin_high: float
    bin_width: float
    times_hours: np.ndarray
    delta_nu: np.ndarray
    gamma: np.ndarray
    n_native_rows: int
    n_unique_times: int
    hmi_overlap_mapping: str
    quality_flags: str


def aggregate_hmi_to_native_bin(hmi: pd.DataFrame, low: float, high: float, gamma_col: str) -> NativeBinSeries:
    overlap = np.minimum(hmi["freq_bin_high"].to_numpy(float), high) - np.maximum(hmi["freq_bin_low"].to_numpy(float), low)
    sel = overlap > 1e-6
    if not np.any(sel):
        return NativeBinSeries(low, high, high - low, np.array([]), np.array([]), np.array([]), 0, 0, "no_overlap", "hmi_no_frequency_overlap")

    sub = hmi.loc[sel].copy()
    sub["_overlap_weight"] = overlap[sel]
    sub = sub[
        np.isfinite(sub["t_mid_hours"])
        & np.isfinite(sub["delta_nu"])
        & np.isfinite(sub[gamma_col])
        & np.isfinite(sub["_overlap_weight"])
        & (sub["_overlap_weight"] > 0)
    ].copy()

    if len(sub) == 0:
        return NativeBinSeries(low, high, high - low, np.array([]), np.array([]), np.array([]), 0, 0, "overlap_no_finite_values", "hmi_no_finite_values")

    sub["_w_nu"] = sub["_overlap_weight"] * sub["delta_nu"]
    sub["_w_gamma"] = sub["_overlap_weight"] * sub[gamma_col]

    grouped = sub.groupby("t_mid_hours", as_index=False).agg(
        weight_sum=("_overlap_weight", "sum"),
        w_nu=("_w_nu", "sum"),
        w_gamma=("_w_gamma", "sum"),
        t_mid=("t_mid", "first"),
    )
    grouped = grouped[grouped["weight_sum"] > 0].copy()
    grouped["delta_nu"] = grouped["w_nu"] / grouped["weight_sum"]
    grouped["gamma"] = grouped["w_gamma"] / grouped["weight_sum"]
    grouped = grouped.sort_values("t_mid_hours")

    flags = []
    if not np.any(
        np.isclose(sub["freq_bin_low"], low, atol=1e-6)
        & np.isclose(sub["freq_bin_high"], high, atol=1e-6)
    ):
        flags.append("hmi_frequency_bin_overlap_mapping")
    if sub["freq_bin_low"].min() > low + 1e-6 or sub["freq_bin_high"].max() < high - 1e-6:
        flags.append("hmi_partial_frequency_coverage")

    return NativeBinSeries(
        freq_bin_low=low,
        freq_bin_high=high,
        bin_width=high - low,
        times_hours=grouped["t_mid_hours"].to_numpy(float),
        delta_nu=grouped["delta_nu"].to_numpy(float),
        gamma=grouped["gamma"].to_numpy(float),
        n_native_rows=int(len(sub)),
        n_unique_times=int(len(grouped)),
        hmi_overlap_mapping="exact_or_overlap_weighted_frequency_average",
        quality_flags=quality_join(flags),
    )


def interp_nan(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    xs = x[keep]
    ys = y[keep]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    out = np.interp(x_new, xs, ys, left=np.nan, right=np.nan)
    out[(x_new < xs[0]) | (x_new > xs[-1])] = np.nan
    return out


def windowed_mean_sampled(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray, window_hours: float) -> Tuple[np.ndarray, np.ndarray]:
    source_t = np.asarray(source_t, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    target_t = np.asarray(target_t, dtype=float)
    keep = np.isfinite(source_t) & np.isfinite(source_y)
    out = np.full(target_t.shape, np.nan, dtype=float)
    counts = np.zeros(target_t.shape, dtype=int)
    if keep.sum() == 0:
        return out, counts
    t = source_t[keep]
    y = source_y[keep]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    cs = np.concatenate([[0.0], np.cumsum(y)])
    half = 0.5 * window_hours
    left = np.searchsorted(t, target_t - half, side="left")
    right = np.searchsorted(t, target_t + half, side="right")
    n = right - left
    valid = n > 0
    out[valid] = (cs[right[valid]] - cs[left[valid]]) / n[valid]
    counts[valid] = n[valid]
    return out, counts


def degrade_native_bin(native: NativeBinSeries, target_t: np.ndarray, lag_hours: float, kernel_days: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if native.n_unique_times < 2:
        n = len(target_t)
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=int), "hmi_insufficient_native_samples"

    # Positive lag means gamma leads nu:
    # gamma_injected(t) = gamma_original(t + lag_hours)
    gamma_injected_native = interp_nan(native.times_hours + lag_hours, native.times_hours, native.gamma)

    nu_degraded, n_nu = windowed_mean_sampled(
        native.times_hours,
        native.delta_nu,
        target_t,
        kernel_days * 24.0,
    )
    gamma_degraded, n_gamma = windowed_mean_sampled(
        native.times_hours,
        gamma_injected_native,
        target_t,
        kernel_days * 24.0,
    )
    counts = np.minimum(n_nu, n_gamma)
    flags = []
    if np.nanmean(counts > 0) < 0.5:
        flags.append("hmi_low_kernel_coverage")
    return nu_degraded, gamma_degraded, counts, quality_join(flags)


def running_mean_time_aware(times_hours: np.ndarray, values: np.ndarray, window_days: float) -> np.ndarray:
    return windowed_mean_sampled(times_hours, values, times_hours, window_days * 24.0)[0]


def detrend_81d(times_hours: np.ndarray, values: np.ndarray, detrend_days: float) -> np.ndarray:
    mean = running_mean_time_aware(times_hours, values, detrend_days)
    return values - mean


def weighted_corr(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        keep &= np.isfinite(w) & (w > 0)
    if keep.sum() < 4:
        return np.nan
    xv = x[keep]
    yv = y[keep]
    if w is None:
        xs = xv - np.nanmean(xv)
        ys = yv - np.nanmean(yv)
        denom = math.sqrt(float(np.nansum(xs * xs) * np.nansum(ys * ys)))
        return float(np.nansum(xs * ys) / denom) if denom > 0 else np.nan
    wv = w[keep].astype(float)
    wsum = float(np.sum(wv))
    if wsum <= 0:
        return np.nan
    mx = float(np.sum(wv * xv) / wsum)
    my = float(np.sum(wv * yv) / wsum)
    xs = xv - mx
    ys = yv - my
    denom = math.sqrt(float(np.sum(wv * xs * xs) * np.sum(wv * ys * ys)))
    return float(np.sum(wv * xs * ys) / denom) if denom > 0 else np.nan


def estimate_lag_cc(
    times_hours: np.ndarray,
    gamma_resid: np.ndarray,
    nu_resid: np.ndarray,
    lag_min: float,
    lag_max: float,
    lag_step: float,
    weighting: str = "unweighted",
    sigma_gamma: Optional[np.ndarray] = None,
    sigma_nu: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    lags = np.arange(lag_min, lag_max + 0.5 * lag_step, lag_step, dtype=float)
    corr = np.full(lags.shape, np.nan, dtype=float)
    n_pairs = np.zeros(lags.shape, dtype=int)

    for i, tau in enumerate(lags):
        nu_shift = interp_nan(times_hours + tau, times_hours, nu_resid)
        if weighting == "inverse_variance" and sigma_gamma is not None and sigma_nu is not None:
            sig_nu_shift = interp_nan(times_hours + tau, times_hours, sigma_nu)
            var = sigma_gamma * sigma_gamma + sig_nu_shift * sig_nu_shift
            w = np.where(np.isfinite(var) & (var > 0), 1.0 / var, np.nan)
        else:
            w = None
        keep = np.isfinite(gamma_resid) & np.isfinite(nu_shift)
        if w is not None:
            keep &= np.isfinite(w) & (w > 0)
        n_pairs[i] = int(keep.sum())
        corr[i] = weighted_corr(gamma_resid, nu_shift, w=w)

    if not np.any(np.isfinite(corr)):
        return {
            "tau_hat_hours": np.nan,
            "tau_discrete_hours": np.nan,
            "corr_max": np.nan,
            "corr_at_zero": np.nan,
            "n_pairs_at_max": 0,
            "lag_grid_hours": lags,
            "corr_grid": corr,
            "quality_flags": "lag_estimator_failed",
        }

    imax = int(np.nanargmax(corr))
    tau_disc = float(lags[imax])
    tau_hat = tau_disc
    flags: List[str] = []
    if 0 < imax < len(lags) - 1 and np.all(np.isfinite(corr[imax - 1:imax + 2])):
        y0, y1, y2 = corr[imax - 1], corr[imax], corr[imax + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-15:
            offset_steps = 0.5 * (y0 - y2) / denom
            if math.isfinite(offset_steps):
                offset_steps = float(np.clip(offset_steps, -1.0, 1.0))
                tau_hat = tau_disc + offset_steps * lag_step
        else:
            flags.append("parabolic_flat_peak")
    else:
        flags.append("lag_peak_at_search_edge")

    zero_idx = int(np.argmin(np.abs(lags)))
    return {
        "tau_hat_hours": float(tau_hat),
        "tau_discrete_hours": tau_disc,
        "corr_max": float(corr[imax]),
        "corr_at_zero": float(corr[zero_idx]) if len(corr) else np.nan,
        "n_pairs_at_max": int(n_pairs[imax]),
        "lag_grid_hours": lags,
        "corr_grid": corr,
        "quality_flags": quality_join(flags),
    }


def moving_block_bootstrap_lags(
    times_hours: np.ndarray,
    gamma_resid: np.ndarray,
    nu_resid: np.ndarray,
    lag_min: float,
    lag_max: float,
    lag_step: float,
    block_days: float,
    stride_days: float,
    bootstrap_n: int,
    rng: np.random.Generator,
    weighting: str,
) -> np.ndarray:
    if bootstrap_n <= 0:
        return np.array([], dtype=float)

    keep = np.isfinite(times_hours) & np.isfinite(gamma_resid) & np.isfinite(nu_resid)
    gamma = np.asarray(gamma_resid[keep], dtype=float)
    nu = np.asarray(nu_resid[keep], dtype=float)
    n = len(gamma)
    if n < 8:
        return np.full(bootstrap_n, np.nan, dtype=float)

    block_len = max(1, int(math.ceil(block_days / max(stride_days, 1e-9))))
    block_len = min(block_len, n)
    boot = np.full(bootstrap_n, np.nan, dtype=float)
    boot_times = np.arange(n, dtype=float) * stride_days * 24.0

    for b in range(bootstrap_n):
        idx: List[int] = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            idx.extend(((start + np.arange(block_len)) % n).astype(int).tolist())
        idx_arr = np.asarray(idx[:n], dtype=int)
        est = estimate_lag_cc(
            boot_times,
            gamma[idx_arr],
            nu[idx_arr],
            lag_min,
            lag_max,
            lag_step,
            weighting=weighting,
        )
        boot[b] = float(est["tau_hat_hours"]) if math.isfinite(float(est["tau_hat_hours"])) else np.nan
    return boot


def add_noise_for_bin(
    base: pd.DataFrame,
    noise_model: str,
    covariance_scale: float,
    rng: np.random.Generator,
    detrend_days_for_bootstrap_residual: float,
    gamma_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    flags: List[str] = []
    n = len(base)
    noise_nu = np.zeros(n, dtype=float)
    noise_gamma = np.zeros(n, dtype=float)

    if noise_model == "none" or covariance_scale == 0:
        return noise_nu, noise_gamma, flags

    if gamma_col == "delta_gamma":
        flags.append("linewidth_covariance_proxy_cov_lngamma_used_for_delta_gamma_noise")

    if noise_model == "gaussian_covariance":
        for i, row in enumerate(base.itertuples(index=False)):
            c00 = float(getattr(row, "cov_nu_nu"))
            c11 = float(getattr(row, "cov_lngamma_lngamma"))
            c01 = float(getattr(row, "cov_nu_lngamma"))
            C = nearest_psd_cov(c00, c11, c01)
            if C is None:
                flags.append("invalid_gong_covariance")
                noise_nu[i] = np.nan
                noise_gamma[i] = np.nan
                continue
            C = C * covariance_scale
            try:
                draw = rng.multivariate_normal(mean=np.zeros(2), cov=C, check_valid="ignore")
                noise_nu[i] = draw[0]
                noise_gamma[i] = draw[1]
            except Exception:
                flags.append("noise_draw_failed")
                noise_nu[i] = np.nan
                noise_gamma[i] = np.nan
        return noise_nu, noise_gamma, flags

    if noise_model == "bootstrap_residual":
        if "delta_nu_gong" not in base.columns or "delta_lngamma_gong" not in base.columns:
            flags.append("bootstrap_residual_missing_gong_columns")
            return np.full(n, np.nan), np.full(n, np.nan), flags
        t = base["t_mid_hours"].to_numpy(float)
        gnu = base["delta_nu_gong"].to_numpy(float)
        glng = base["delta_lngamma_gong"].to_numpy(float)
        rnu = detrend_81d(t, gnu, detrend_days_for_bootstrap_residual)
        rlng = detrend_81d(t, glng, detrend_days_for_bootstrap_residual)
        keep = np.isfinite(rnu) & np.isfinite(rlng)
        if keep.sum() < 4:
            flags.append("bootstrap_residual_insufficient_pool")
            return np.full(n, np.nan), np.full(n, np.nan), flags
        pool = np.column_stack([rnu[keep], rlng[keep]])
        draws = pool[rng.integers(0, len(pool), size=n)]
        scale = math.sqrt(covariance_scale)
        noise_nu = draws[:, 0] * scale
        noise_gamma = draws[:, 1] * scale
        return noise_nu, noise_gamma, flags

    raise ValueError(f"Unknown noise_model: {noise_model}")


def make_hmi_input_audit(hmi: pd.DataFrame, hmi_path: Path, gamma_col: str, audit_meta: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = hmi.groupby(["freq_bin_low", "freq_bin_high", "bin_width"], dropna=False)
    for (low, high, width), sub in grouped:
        rows.append({
            "hmi_bin_timeseries": str(hmi_path),
            "file_type": hmi_path.suffix.lower().lstrip("."),
            "hmi_format": audit_meta.get("hmi_format", ""),
            "gamma_column_used": gamma_col,
            "delta_lngamma_fabricated": bool(audit_meta.get("delta_lngamma_fabricated", False)),
            "freq_bin_low": low,
            "freq_bin_high": high,
            "bin_width": width,
            "n_rows": int(len(sub)),
            "time_min": sub["t_mid"].min().isoformat() if len(sub) else "",
            "time_max": sub["t_mid"].max().isoformat() if len(sub) else "",
            "n_finite_delta_nu": int(np.isfinite(sub["delta_nu"]).sum()),
            "n_finite_gamma": int(np.isfinite(sub[gamma_col]).sum()),
            "quality_flags": "",
        })
    if not rows:
        rows.append({
            "hmi_bin_timeseries": str(hmi_path),
            "file_type": hmi_path.suffix.lower().lstrip("."),
            "hmi_format": audit_meta.get("hmi_format", ""),
            "gamma_column_used": gamma_col,
            "delta_lngamma_fabricated": False,
            "freq_bin_low": np.nan,
            "freq_bin_high": np.nan,
            "bin_width": np.nan,
            "n_rows": 0,
            "time_min": "",
            "time_max": "",
            "n_finite_delta_nu": 0,
            "n_finite_gamma": 0,
            "quality_flags": "hmi_no_rows",
        })
    out = pd.DataFrame(rows)
    out.attrs["audit_meta"] = dict(audit_meta)
    return out


def finite_mask_for_target_grid(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    if args.apply_gong_missing_mask:
        if "fill" in df.columns:
            mask &= np.isfinite(df["fill"].to_numpy(float)) & (df["fill"].to_numpy(float) > 0)
    if args.finite_quality_mask in ["basic", "strict"]:
        for c in ["cov_nu_nu", "cov_lngamma_lngamma", "cov_nu_lngamma"]:
            mask &= np.isfinite(df[c].to_numpy(float))
    if args.finite_quality_mask == "strict":
        q = df.get("quality_flags", pd.Series([""] * len(df))).fillna("").astype(str)
        mask &= q.str.len().to_numpy() == 0
    return mask


def build_degraded_hmi(
    hmi: pd.DataFrame,
    gamma_col: str,
    target_grid: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    native_by_bin: Dict[Tuple[float, float, float], NativeBinSeries] = {}
    native_audit_rows: List[Dict[str, Any]] = []

    for (low, high, width), _ in target_grid.groupby(["freq_bin_low", "freq_bin_high", "bin_width"], dropna=False):
        native = aggregate_hmi_to_native_bin(hmi, float(low), float(high), gamma_col)
        native_by_bin[(float(low), float(high), float(width))] = native
        native_audit_rows.append({
            "freq_bin_low": float(low),
            "freq_bin_high": float(high),
            "bin_width": float(width),
            "n_hmi_native_rows": native.n_native_rows,
            "n_hmi_unique_times": native.n_unique_times,
            "hmi_overlap_mapping": native.hmi_overlap_mapping,
            "quality_flags": native.quality_flags,
        })

    inject_lags = parse_csv_floats(args.inject_lags_hours, "--inject-lags-hours")
    all_frames: List[pd.DataFrame] = []

    for lag in inject_lags:
        logger.info("Degrading HMI for injection lag %.3f hr", lag)
        for key, tg_bin in target_grid.groupby(["freq_bin_low", "freq_bin_high", "bin_width"], dropna=False):
            low, high, width = map(float, key)
            native = native_by_bin[(low, high, width)]
            target_t = tg_bin["t_mid_hours"].to_numpy(float)
            nu_deg, gamma_deg, hmi_count, degrade_flags = degrade_native_bin(
                native,
                target_t,
                lag,
                float(args.degradation_kernel_days),
            )

            base = tg_bin.copy().reset_index(drop=True)
            base["inject_lag_hours"] = float(lag)
            base["hmi_linewidth_variable"] = gamma_col
            base["hmi_delta_nu_degraded_noiseless"] = nu_deg
            base["hmi_gamma_degraded_noiseless"] = gamma_deg
            base["hmi_kernel_sample_count"] = hmi_count
            base["hmi_degradation_quality_flags"] = degrade_flags

            for realization in range(int(args.n_injection_realizations)):
                realization_base = base.copy()
                noise_nu, noise_gamma, noise_flags = add_noise_for_bin(
                    realization_base,
                    str(args.noise_model),
                    float(args.covariance_scale),
                    rng,
                    float(args.detrend_days),
                    gamma_col,
                )
                realization_base["realization"] = int(realization)
                realization_base["noise_nu"] = noise_nu
                realization_base["noise_gamma"] = noise_gamma
                realization_base["hmi_delta_nu_degraded"] = realization_base["hmi_delta_nu_degraded_noiseless"] + noise_nu
                realization_base["hmi_gamma_degraded"] = realization_base["hmi_gamma_degraded_noiseless"] + noise_gamma

                target_mask = finite_mask_for_target_grid(realization_base, args)
                data_mask = (
                    target_mask
                    & np.isfinite(realization_base["hmi_delta_nu_degraded"].to_numpy(float))
                    & np.isfinite(realization_base["hmi_gamma_degraded"].to_numpy(float))
                )
                realization_base["mask_accepted"] = data_mask
                combined_flags = []
                for q1, q2 in zip(
                    realization_base.get("quality_flags", pd.Series([""] * len(realization_base))).fillna(""),
                    realization_base.get("hmi_degradation_quality_flags", pd.Series([""] * len(realization_base))).fillna(""),
                ):
                    combined_flags.append(quality_join(quality_split(q1) + quality_split(q2) + noise_flags))
                realization_base["combined_quality_flags"] = combined_flags
                all_frames.append(realization_base)

    degraded = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    native_audit = pd.DataFrame(native_audit_rows)
    return degraded, native_audit


def recover_lags(
    degraded: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lag_min, lag_max = parse_pair(args.lag_search_hours, "--lag-search-hours")
    lag_step = float(args.lag_step_hours)

    rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []

    group_cols = ["inject_lag_hours", "realization", "freq_bin_low", "freq_bin_high", "bin_width"]
    for key, sub in degraded.groupby(group_cols, dropna=False):
        inject_lag, realization, low, high, width = key
        sub = sub.sort_values("t_mid_hours").copy()

        mask = sub["mask_accepted"].astype(bool).to_numpy()
        times = sub["t_mid_hours"].to_numpy(float)
        nu = sub["hmi_delta_nu_degraded"].to_numpy(float)
        gamma = sub["hmi_gamma_degraded"].to_numpy(float)
        sigma_nu = sub["sigma_nu"].to_numpy(float) if "sigma_nu" in sub.columns else None
        sigma_gamma = sub["sigma_lngamma"].to_numpy(float) if "sigma_lngamma" in sub.columns else None

        nu[~mask] = np.nan
        gamma[~mask] = np.nan
        if sigma_nu is not None:
            sigma_nu[~mask] = np.nan
        if sigma_gamma is not None:
            sigma_gamma[~mask] = np.nan

        nu_resid = detrend_81d(times, nu, float(args.detrend_days))
        gamma_resid = detrend_81d(times, gamma, float(args.detrend_days))

        est = estimate_lag_cc(
            times,
            gamma_resid,
            nu_resid,
            lag_min,
            lag_max,
            lag_step,
            weighting=str(args.lag_weighting),
            sigma_gamma=sigma_gamma,
            sigma_nu=sigma_nu,
        )

        boot = moving_block_bootstrap_lags(
            times,
            gamma_resid,
            nu_resid,
            lag_min,
            lag_max,
            lag_step,
            float(args.bootstrap_block_days),
            float(args.target_stride_days),
            int(args.bootstrap_n),
            rng,
            weighting=str(args.lag_weighting),
        )
        boot_finite = boot[np.isfinite(boot)]

        if len(boot_finite):
            ci16, ci50, ci84 = np.nanpercentile(boot_finite, [16, 50, 84])
            ci025, ci975 = np.nanpercentile(boot_finite, [2.5, 97.5])
            boot_std = float(np.nanstd(boot_finite, ddof=1)) if len(boot_finite) > 1 else np.nan
            boot_mean = float(np.nanmean(boot_finite))
        else:
            ci16 = ci50 = ci84 = ci025 = ci975 = boot_std = boot_mean = np.nan

        tau_hat = float(est["tau_hat_hours"])
        bias = tau_hat - float(inject_lag) if math.isfinite(tau_hat) else np.nan
        finite_fraction = float(np.nanmean(mask)) if len(mask) else 0.0
        ci68_contains = bool(ci16 <= float(inject_lag) <= ci84) if all(math.isfinite(x) for x in [ci16, ci84]) else False
        ci95_contains = bool(ci025 <= float(inject_lag) <= ci975) if all(math.isfinite(x) for x in [ci025, ci975]) else False

        quality = quality_split(est.get("quality_flags", ""))
        if finite_fraction < 0.5:
            quality.append("low_finite_fraction")
        if not math.isfinite(tau_hat):
            quality.append("lag_estimator_failed")

        common = {
            "inject_lag_hours": float(inject_lag),
            "realization": int(realization),
            "freq_bin_low": float(low),
            "freq_bin_high": float(high),
            "freq_bin_center": 0.5 * (float(low) + float(high)),
            "bin_width": float(width),
            "tau_hat_hours": tau_hat,
            "tau_discrete_hours": float(est["tau_discrete_hours"]) if math.isfinite(float(est["tau_discrete_hours"])) else np.nan,
            "bias_hours": bias,
            "corr_max": float(est["corr_max"]) if math.isfinite(float(est["corr_max"])) else np.nan,
            "corr_at_zero": float(est["corr_at_zero"]) if math.isfinite(float(est["corr_at_zero"])) else np.nan,
            "n_pairs_at_max": int(est["n_pairs_at_max"]),
            "n_target_points": int(len(sub)),
            "finite_fraction": finite_fraction,
            "bootstrap_n_requested": int(args.bootstrap_n),
            "bootstrap_n_finite": int(len(boot_finite)),
            "bootstrap_mean_hours": boot_mean,
            "bootstrap_std_hours": boot_std,
            "ci68_low_hours": float(ci16) if math.isfinite(ci16) else np.nan,
            "ci68_high_hours": float(ci84) if math.isfinite(ci84) else np.nan,
            "ci95_low_hours": float(ci025) if math.isfinite(ci025) else np.nan,
            "ci95_high_hours": float(ci975) if math.isfinite(ci975) else np.nan,
            "ci68_contains_injected": ci68_contains,
            "ci95_contains_injected": ci95_contains,
            "quality_flags": quality_join(quality),
        }
        rows.append(common)

        boot_rows.append({
            **{k: common[k] for k in [
                "inject_lag_hours", "realization", "freq_bin_low", "freq_bin_high",
                "freq_bin_center", "bin_width",
            ]},
            "bootstrap_n_requested": int(args.bootstrap_n),
            "bootstrap_n_finite": int(len(boot_finite)),
            "bootstrap_mean_hours": boot_mean,
            "bootstrap_std_hours": boot_std,
            "bootstrap_q02_5_hours": float(ci025) if math.isfinite(ci025) else np.nan,
            "bootstrap_q16_hours": float(ci16) if math.isfinite(ci16) else np.nan,
            "bootstrap_q50_hours": float(ci50) if math.isfinite(ci50) else np.nan,
            "bootstrap_q84_hours": float(ci84) if math.isfinite(ci84) else np.nan,
            "bootstrap_q97_5_hours": float(ci975) if math.isfinite(ci975) else np.nan,
            "ci68_contains_injected": ci68_contains,
            "ci95_contains_injected": ci95_contains,
            "quality_flags": quality_join(quality),
        })

    by_bin = pd.DataFrame(rows)
    boot_summary = pd.DataFrame(boot_rows)
    logger.info("Recovered lags for %d bin/lag/realization cases", len(by_bin))
    return by_bin, boot_summary


def summarize_recovery(by_bin: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if len(by_bin) == 0:
        return pd.DataFrame()

    def summarize_group(region: str, group_keys: List[str], frame: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for key, sub in frame.groupby(group_keys, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            d = dict(zip(group_keys, key))
            finite_tau = np.isfinite(sub["tau_hat_hours"].to_numpy(float))
            rows.append({
                "summary_region": region,
                **d,
                "n_cases": int(len(sub)),
                "n_finite_tau": int(finite_tau.sum()),
                "finite_tau_fraction": float(finite_tau.mean()) if len(finite_tau) else 0.0,
                "tau_hat_mean_hours": float(np.nanmean(sub["tau_hat_hours"])) if finite_tau.any() else np.nan,
                "tau_hat_median_hours": float(np.nanmedian(sub["tau_hat_hours"])) if finite_tau.any() else np.nan,
                "tau_hat_std_hours": float(np.nanstd(sub["tau_hat_hours"], ddof=1)) if finite_tau.sum() > 1 else np.nan,
                "bias_mean_hours": float(np.nanmean(sub["bias_hours"])) if finite_tau.any() else np.nan,
                "bias_median_hours": float(np.nanmedian(sub["bias_hours"])) if finite_tau.any() else np.nan,
                "abs_bias_median_hours": float(np.nanmedian(np.abs(sub["bias_hours"]))) if finite_tau.any() else np.nan,
                "bootstrap_std_median_hours": float(np.nanmedian(sub["bootstrap_std_hours"])),
                "coverage68_rate": float(np.nanmean(sub["ci68_contains_injected"].astype(float))),
                "coverage95_rate": float(np.nanmean(sub["ci95_contains_injected"].astype(float))),
                "finite_fraction_median": float(np.nanmedian(sub["finite_fraction"])),
            })
        return pd.DataFrame(rows)

    pieces = []
    pieces.append(
        summarize_group(
            "per_bin",
            ["inject_lag_hours", "freq_bin_low", "freq_bin_high", "freq_bin_center", "bin_width"],
            by_bin,
        )
    )

    main = by_bin[
        (by_bin["freq_bin_low"] >= float(args.freq_min) - 1e-6)
        & (by_bin["freq_bin_high"] <= float(args.freq_max) + 1e-6)
    ]
    if len(main):
        pieces.append(summarize_group(f"main_{args.freq_min:g}_{args.freq_max:g}", ["inject_lag_hours", "bin_width"], main))

    focus = by_bin[
        (by_bin["freq_bin_low"] >= float(args.focus_bin_low) - 1e-6)
        & (by_bin["freq_bin_high"] <= float(args.focus_bin_high) + 1e-6)
    ]
    if len(focus):
        pieces.append(summarize_group(f"focus_{args.focus_bin_low:g}_{args.focus_bin_high:g}", ["inject_lag_hours", "bin_width"], focus))

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def evaluate_acceptance(summary: pd.DataFrame, by_bin: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    criteria: List[Dict[str, Any]] = []

    def add(name: str, status: str, value: Any, threshold: str, detail: str) -> None:
        criteria.append({
            "criterion": name,
            "status": status,
            "value": value,
            "threshold_or_rule": threshold,
            "detail": detail,
        })

    focus_region = f"focus_{args.focus_bin_low:g}_{args.focus_bin_high:g}"
    main_region = f"main_{args.freq_min:g}_{args.freq_max:g}"

    if len(summary) == 0:
        add("nonempty_recovery", "FAIL", "empty", "must have finite recovery table", "No recovery rows were produced.")
    else:
        add("nonempty_recovery", "PASS", int(len(summary)), "summary table nonempty", "Recovery summary exists.")

    focus = summary[summary.get("summary_region", "") == focus_region] if len(summary) else pd.DataFrame()
    main = summary[summary.get("summary_region", "") == main_region] if len(summary) else pd.DataFrame()

    zero = focus[np.isclose(focus.get("inject_lag_hours", np.nan), 0.0)] if len(focus) else pd.DataFrame()
    if len(zero) == 0:
        add("tau0_no_systematic_positive_lag_focus", "WARN", "missing", "focus bin tau_inj=0 should be present", "No focus-bin tau_inj=0 summary.")
    else:
        positive_bias = float(np.nanmedian(zero["tau_hat_median_hours"]))
        if positive_bias > 18:
            status = "FAIL"
        elif positive_bias > 6:
            status = "WARN"
        else:
            status = "PASS"
        add(
            "tau0_no_systematic_positive_lag_focus",
            status,
            positive_bias,
            "PASS if median recovered tau for tau_inj=0 is <= +6 hr; FAIL if > +18 hr",
            "Checks whether degradation/noise produces artificial positive lag at tau_inj=0.",
        )

    recover_lags = focus[focus.get("inject_lag_hours", np.nan).isin([6.0, 12.0])] if len(focus) else pd.DataFrame()
    if len(recover_lags) < 1:
        add("plus6_plus12_recoverable_focus", "WARN", "missing", "+6 and +12 hr focus summaries should exist", "No +6/+12 focus rows.")
    else:
        abs_bias = float(np.nanmedian(np.abs(recover_lags["bias_median_hours"])))
        sign_reversed = bool(np.any(recover_lags["tau_hat_median_hours"].to_numpy(float) < -1e-6))
        if sign_reversed or abs_bias > 36:
            status = "FAIL"
        elif abs_bias > 18:
            status = "WARN"
        else:
            status = "PASS"
        add(
            "plus6_plus12_recoverable_focus",
            status,
            abs_bias,
            "PASS if median abs bias <=18 hr and no sign reversal; FAIL if >36 hr or sign-reversed",
            "Validation-only lag estimator should recover +6/+12 hr in 3000-3500 microHz focus region without severe washout.",
        )

    if len(main) == 0:
        add("main_range_bias_acceptable", "WARN", "missing", "main-region summaries should exist", "No main-region summary.")
    else:
        main_abs_bias = float(np.nanmedian(np.abs(main["bias_median_hours"])))
        if main_abs_bias > 36:
            status = "FAIL"
        elif main_abs_bias > 18:
            status = "WARN"
        else:
            status = "PASS"
        add(
            "main_range_bias_acceptable",
            status,
            main_abs_bias,
            "PASS if median abs bias <=18 hr; WARN <=36 hr; FAIL >36 hr",
            "Checks 2000-4500 microHz main range without collapsing per-bin output.",
        )

    finite_fraction = float(np.nanmedian(by_bin["finite_fraction"])) if len(by_bin) else 0.0
    if finite_fraction >= 0.80:
        status = "PASS"
    elif finite_fraction >= 0.50:
        status = "WARN"
    else:
        status = "FAIL"
    add(
        "finite_fraction_sufficient",
        status,
        finite_fraction,
        "PASS >=0.80; WARN >=0.50; FAIL <0.50",
        "Measures finite GONG-like target samples after degradation, masking, and noise.",
    )

    tau_finite_fraction = float(np.isfinite(by_bin["tau_hat_hours"]).mean()) if len(by_bin) else 0.0
    if tau_finite_fraction >= 0.80:
        status = "PASS"
    elif tau_finite_fraction >= 0.60:
        status = "WARN"
    else:
        status = "FAIL"
    add(
        "no_widespread_failed_bins",
        status,
        tau_finite_fraction,
        "PASS finite tau fraction >=0.80; WARN >=0.60; FAIL <0.60",
        "Checks widespread non-finite or failed lag estimates.",
    )

    enough_for_coverage = int(args.n_injection_realizations) >= 10 and int(args.bootstrap_n) >= 20
    if enough_for_coverage and len(by_bin):
        cov95 = float(np.nanmean(by_bin["ci95_contains_injected"].astype(float)))
        cov68 = float(np.nanmean(by_bin["ci68_contains_injected"].astype(float)))
        if cov95 >= 0.75 and cov68 >= 0.45:
            status = "PASS"
        elif cov95 >= 0.55:
            status = "WARN"
        else:
            status = "FAIL"
        add(
            "bootstrap_uncertainty_not_obviously_anti_conservative",
            status,
            {"coverage95": cov95, "coverage68": cov68},
            "PASS if 95% coverage >=0.75 and 68% coverage >=0.45 for sufficient reps; WARN if 95% >=0.55",
            "Coverage criterion is heuristic and should be interpreted with number of injection realizations.",
        )
    else:
        add(
            "bootstrap_uncertainty_not_obviously_anti_conservative",
            "WARN",
            {"n_injection_realizations": int(args.n_injection_realizations), "bootstrap_n": int(args.bootstrap_n)},
            "Need >=10 realizations and bootstrap_n>=20 for meaningful coverage screen",
            "Insufficient repetitions/bootstrap for strong coverage acceptance.",
        )

    statuses = [c["status"] for c in criteria]
    if any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif all(s == "PASS" for s in statuses):
        overall = "PASS"
    else:
        overall = "WARN"

    acceptance_df = pd.DataFrame(criteria)
    acceptance_json = {
        "overall_status": overall,
        "criteria": criteria,
        "scope": "MVP-C validation-only; not final GONG lag claim",
        "pass_conditions_interpreted": [
            "tau_inj=0 does not systematically become positive lag",
            "+6/+12 hr recoverable in 3000-3500 microHz without severe bias",
            "2000-4500 microHz main range has acceptable bias",
            "bootstrap uncertainty not obviously anti-conservative",
            "finite fraction high and no widespread failed bins",
        ],
        "notes": [
            "This acceptance is generated by conservative heuristics; review CSV tables and plots before promoting any scientific statement.",
            "Per-bin recovery table is the primary product; regional summaries are only supplemental.",
        ],
    }
    return acceptance_df, acceptance_json


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    hmi_audit_meta: Mapping[str, Any],
    gong_meta: Mapping[str, Any],
    acceptance_json: Mapping[str, Any],
    summary: pd.DataFrame,
) -> None:
    report = outdir / "MVP_C_REPORT.md"
    focus_region = f"focus_{args.focus_bin_low:g}_{args.focus_bin_high:g}"
    main_region = f"main_{args.freq_min:g}_{args.freq_max:g}"

    lines = []
    lines.append("# MVP-C HMI-to-GONG Degradation / Injection Validation Report")
    lines.append("")
    lines.append(f"- generated_utc: `{utc_now()}`")
    lines.append(f"- hmi_bin_timeseries: `{args.hmi_bin_timeseries}`")
    lines.append(f"- hmi_format: `{args.hmi_format}`")
    lines.append(f"- gong_run: `{args.gong_run}`")
    lines.append(f"- overall_status: `{acceptance_json.get('overall_status', 'UNKNOWN')}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is validation-only lag recovery for known injected lags. It is not a final GONG lag claim and not a full WMLTP implementation.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    cfg = {
        "degradation_kernel_days": args.degradation_kernel_days,
        "target_stride_days": args.target_stride_days,
        "target_grid_source": "GONG Q1 split-auto run",
        "detrend_days": args.detrend_days,
        "noise_model": args.noise_model,
        "covariance_scale": args.covariance_scale,
        "random_seed": args.random_seed,
        "inject_lags_hours": args.inject_lags_hours,
        "lag_search_hours": args.lag_search_hours,
        "lag_step_hours": args.lag_step_hours,
        "bootstrap_n": args.bootstrap_n,
        "bootstrap_block_days": args.bootstrap_block_days,
        "positive_lag_convention": "gamma residuals lead nu residuals",
        "operational_correlation": "C(tau)=corr[gamma'(t), nu'(t+tau)]",
        "detrend_method": "time_aware_centered_running_mean",
        "slurm_cpus_per_task": SLURM_CPUS_PER_TASK,
    }
    for k, v in cfg.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Input audit")
    lines.append("")
    lines.append(f"- HMI format: `{hmi_audit_meta.get('hmi_format', '')}`")
    lines.append(f"- HMI gamma column used: `{hmi_audit_meta.get('gamma_column_used', '')}`")
    lines.append(f"- delta_lngamma fabricated: `{hmi_audit_meta.get('delta_lngamma_fabricated', False)}`")
    lines.append(f"- HMI time range: `{hmi_audit_meta.get('time_min', '')}` to `{hmi_audit_meta.get('time_max', '')}`")
    lines.append(f"- HMI frequency range seen: `{hmi_audit_meta.get('freq_min_seen', '')}` to `{hmi_audit_meta.get('freq_max_seen', '')}` microHz")
    if hmi_audit_meta.get("fixed_bin_edges_uhz"):
        lines.append(f"- fixed HMI native bin edges [microHz]: `{hmi_audit_meta.get('fixed_bin_edges_uhz')}`")
    lines.append(f"- GONG metadata keys available: `{sorted(gong_meta.keys())[:20]}`")
    lines.append("")
    lines.append("## Acceptance summary")
    lines.append("")
    lines.append("| criterion | status | value | detail |")
    lines.append("|---|---:|---:|---|")
    for c in acceptance_json.get("criteria", []):
        value = c.get("value", "")
        lines.append(f"| {c.get('criterion')} | {c.get('status')} | `{value}` | {c.get('detail')} |")
    lines.append("")
    lines.append("## Key summary rows")
    lines.append("")
    if len(summary):
        subset = summary[summary["summary_region"].isin(["per_bin", main_region, focus_region])].head(40)
        lines.append("```csv")
        lines.append(subset.to_csv(index=False))
        lines.append("```")
    else:
        lines.append("No recovery summary rows.")
    lines.append("")
    lines.append("## Required review")
    lines.append("")
    lines.append("Review per-bin `tables/injection_recovery_by_bin.csv` before using regional PASS/WARN/FAIL labels.")
    lines.append("The focus region and main region summaries are supplemental; do not collapse all bins into a single scientific conclusion.")
    report.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(outdir: Path, args: argparse.Namespace, extra: Mapping[str, Any]) -> None:
    metadata = {
        "generated_utc": utc_now(),
        "command_line": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": package_versions(),
        "slurm": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK", ""),
            "script_interpreted_slurm_cpus_per_task": SLURM_CPUS_PER_TASK,
        },
        "thread_policy": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        },
        "inputs": {
            "hmi_bin_timeseries": args.hmi_bin_timeseries,
            "hmi_format": args.hmi_format,
            "gong_run": args.gong_run,
        },
        "configuration": vars(args),
        "degradation": {
            "degradation_kernel_days": args.degradation_kernel_days,
            "target_stride_days": args.target_stride_days,
            "target_grid_source": "GONG Q1 split-auto run",
            "noise_model": args.noise_model,
            "covariance_scale": args.covariance_scale,
            "random_seed": args.random_seed,
            "detrend_method": "time_aware_centered_running_mean",
        },
        "lag_convention": {
            "positive_lag_means": "delta_lngamma or delta_gamma leads delta_nu",
            "operational_definition": "C(tau)=corr[gamma'(t), nu'(t+tau)]",
            "injection_definition": "gamma_injected(t)=gamma_original(t+tau_inj)",
        },
        "extended_bins_12": {
            "fixed_bin_edges_uhz": EXTENDED_BINS_12_EDGES,
            "delta_lngamma_fabricated": False,
            "primary_linewidth_variable_for_that_format": "delta_gamma",
        },
        "scope": "MVP-C validation-only; not final GONG lag claim",
        **dict(extra),
    }
    with (outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=json_default)


def make_plots(
    outdir: Path,
    args: argparse.Namespace,
    degraded: pd.DataFrame,
    by_bin: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable; plots skipped without failing: %s", exc)
        return

    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        offsets = np.linspace(-2 * args.degradation_kernel_days, 2 * args.degradation_kernel_days, 801)
        kernel = (np.abs(offsets) <= 0.5 * args.degradation_kernel_days).astype(float)
        plt.figure(figsize=(8, 4))
        plt.plot(offsets, kernel)
        plt.xlabel("offset from target t_mid [days]")
        plt.ylabel("kernel weight")
        plt.title("9-day effective degradation kernel")
        plt.tight_layout()
        plt.savefig(plot_dir / "degradation_kernel_illustration.png", dpi=150)
        plt.close()
    except Exception as exc:
        logger.warning("kernel plot failed: %s", exc)

    try:
        focus = degraded[
            np.isclose(degraded["freq_bin_low"], args.focus_bin_low, atol=1e-6)
            & np.isclose(degraded["freq_bin_high"], args.focus_bin_high, atol=1e-6)
            & np.isclose(degraded["inject_lag_hours"], 0.0, atol=1e-6)
            & (degraded["realization"] == 0)
        ].copy()
        if len(focus) == 0:
            focus = degraded[(degraded["realization"] == 0)].head(1000).copy()
        if len(focus):
            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(focus["t_mid"], utc=True), focus["hmi_delta_nu_degraded"], marker="o", linewidth=1)
            plt.xlabel("GONG-like t_mid")
            plt.ylabel("degraded delta_nu")
            plt.title("HMI-to-GONG-degraded quicklook: delta_nu")
            plt.tight_layout()
            plt.savefig(plot_dir / "hmi_degraded_delta_nu_selected_bin.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(focus["t_mid"], utc=True), focus["hmi_gamma_degraded"], marker="o", linewidth=1)
            plt.xlabel("GONG-like t_mid")
            plt.ylabel("degraded gamma variable")
            plt.title("HMI-to-GONG-degraded quicklook: linewidth variable")
            plt.tight_layout()
            plt.savefig(plot_dir / "hmi_degraded_gamma_selected_bin.png", dpi=150)
            plt.close()
    except Exception as exc:
        logger.warning("HMI degraded quicklook plots failed: %s", exc)

    try:
        if len(by_bin):
            plt.figure(figsize=(7, 6))
            plt.scatter(by_bin["inject_lag_hours"], by_bin["tau_hat_hours"], s=12)
            mn = min(np.nanmin(by_bin["inject_lag_hours"]), np.nanmin(by_bin["tau_hat_hours"]))
            mx = max(np.nanmax(by_bin["inject_lag_hours"]), np.nanmax(by_bin["tau_hat_hours"]))
            plt.plot([mn, mx], [mn, mx], linestyle="--")
            plt.xlabel("injected lag [hr]")
            plt.ylabel("recovered lag [hr]")
            plt.title("Injected vs recovered lag")
            plt.tight_layout()
            plt.savefig(plot_dir / "injected_vs_recovered_lag.png", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            grouped = by_bin.groupby("inject_lag_hours", as_index=False)["bias_hours"].median()
            plt.plot(grouped["inject_lag_hours"], grouped["bias_hours"], marker="o")
            plt.axhline(0, linestyle="--")
            plt.xlabel("injected lag [hr]")
            plt.ylabel("median recovery bias [hr]")
            plt.title("Recovery bias vs injected lag")
            plt.tight_layout()
            plt.savefig(plot_dir / "recovery_bias_vs_tau_inj.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10, 5))
            g2 = by_bin.groupby(["freq_bin_center", "bin_width"], as_index=False)["bias_hours"].median()
            for width, sub in g2.groupby("bin_width"):
                plt.plot(sub["freq_bin_center"], sub["bias_hours"], marker="o", label=f"{width:g} microHz")
            plt.axhline(0, linestyle="--")
            plt.xlabel("frequency bin center [microHz]")
            plt.ylabel("median bias [hr]")
            plt.title("Recovery bias by frequency bin")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "recovery_bias_by_frequency_bin.png", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            cov = by_bin.groupby("inject_lag_hours", as_index=False).agg(
                coverage68=("ci68_contains_injected", "mean"),
                coverage95=("ci95_contains_injected", "mean"),
            )
            plt.plot(cov["inject_lag_hours"], cov["coverage68"], marker="o", label="68% CI")
            plt.plot(cov["inject_lag_hours"], cov["coverage95"], marker="o", label="95% CI")
            plt.xlabel("injected lag [hr]")
            plt.ylabel("coverage fraction")
            plt.title("Bootstrap CI coverage")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "bootstrap_ci_coverage.png", dpi=150)
            plt.close()

            null = by_bin[np.isclose(by_bin["inject_lag_hours"], 0.0, atol=1e-6)]
            if len(null):
                plt.figure(figsize=(8, 5))
                plt.hist(null["tau_hat_hours"].dropna(), bins=20)
                plt.xlabel("recovered lag for tau_inj=0 [hr]")
                plt.ylabel("count")
                plt.title("tau_inj=0 null recovered lag distribution")
                plt.tight_layout()
                plt.savefig(plot_dir / "tau_inj_zero_null_distribution.png", dpi=150)
                plt.close()

            detail = by_bin[
                (by_bin["freq_bin_low"] >= args.focus_bin_low - 1e-6)
                & (by_bin["freq_bin_high"] <= args.focus_bin_high + 1e-6)
            ]
            if len(detail):
                plt.figure(figsize=(9, 5))
                gd = detail.groupby("inject_lag_hours", as_index=False).agg(
                    tau_hat=("tau_hat_hours", "median"),
                    ci_low=("ci68_low_hours", "median"),
                    ci_high=("ci68_high_hours", "median"),
                )
                yerr = np.vstack([gd["tau_hat"] - gd["ci_low"], gd["ci_high"] - gd["tau_hat"]])
                plt.errorbar(gd["inject_lag_hours"], gd["tau_hat"], yerr=yerr, marker="o", capsize=3)
                plt.plot(gd["inject_lag_hours"], gd["inject_lag_hours"], linestyle="--")
                plt.xlabel("injected lag [hr]")
                plt.ylabel("median recovered lag [hr]")
                plt.title(f"{args.focus_bin_low:g}-{args.focus_bin_high:g} microHz detailed recovery")
                plt.tight_layout()
                plt.savefig(plot_dir / "focus_3000_3500_detailed_recovery.png", dpi=150)
                plt.close()
    except Exception as exc:
        logger.warning("recovery plots failed: %s", exc)


def write_outputs(
    outdir: Path,
    hmi_input_audit: pd.DataFrame,
    gong_grid_audit: pd.DataFrame,
    degraded: pd.DataFrame,
    recovery_by_bin: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    acceptance_summary: pd.DataFrame,
    acceptance_json: Mapping[str, Any],
) -> None:
    tables = {
        "hmi_input_audit": hmi_input_audit,
        "gong_grid_audit": gong_grid_audit,
        "degraded_hmi_timeseries": degraded,
        "injection_recovery_by_bin": recovery_by_bin,
        "injection_recovery_summary": recovery_summary,
        "bootstrap_summary": bootstrap_summary,
        "acceptance_summary": acceptance_summary,
    }
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(table_dir / f"{name}.csv", index=False)
    with (outdir / "mvp_c_acceptance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(acceptance_json, f, indent=2, sort_keys=True, default=json_default)


def run_audit_only_or_dry(
    args: argparse.Namespace,
    outdir: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    hmi_path = Path(args.hmi_bin_timeseries)
    gong_run = Path(args.gong_run)

    audit_rows = [{
        "hmi_bin_timeseries": str(hmi_path),
        "exists": hmi_path.exists(),
        "file_type": hmi_path.suffix.lower().lstrip("."),
        "hmi_format_requested": args.hmi_format,
        "dry_run": dry_run,
        "quality_flags": "" if hmi_path.exists() else "hmi_input_missing",
    }]
    hmi_audit = pd.DataFrame(audit_rows)

    gong_rows = [{
        "gong_run": str(gong_run),
        "exists": gong_run.exists(),
        "bin_time_series_exists": (gong_run / "tables" / "bin_time_series.csv").exists(),
        "window_index_exists": (gong_run / "tables" / "window_index.csv").exists(),
        "audit_exists": (gong_run / "tables" / "audit.csv").exists(),
        "metadata_exists": (gong_run / "metadata.json").exists(),
        "quality_flags": "",
    }]
    gong_audit = pd.DataFrame(gong_rows)

    hmi_meta: Dict[str, Any] = {}
    if not dry_run:
        raw_hmi = read_table(hmi_path)
        hmi, gamma_col, hmi_meta = normalize_hmi_table(raw_hmi, args)
        hmi_audit = make_hmi_input_audit(hmi, hmi_path, gamma_col, hmi_meta)

        gong_bin, _window, _audit, _meta = load_gong_run(gong_run)
        bins = requested_bins(args)
        target_grid = build_target_grid(gong_bin, bins)
        gong_audit = target_grid

    empty = pd.DataFrame()
    acceptance_json = {
        "overall_status": "AUDIT_ONLY" if not dry_run else "DRY_RUN",
        "criteria": [],
        "scope": "No injection recovery was executed.",
    }
    acceptance_summary = pd.DataFrame(acceptance_json["criteria"])
    write_outputs(outdir, hmi_audit, gong_audit, empty, empty, empty, empty, acceptance_summary, acceptance_json)
    write_metadata(outdir, args, {"audit_only": bool(args.audit_only), "dry_run": bool(args.dry_run), "hmi_input_audit_meta": hmi_meta})
    write_report(outdir, args, hmi_meta, {}, acceptance_json, pd.DataFrame())
    logger.info("Completed %s", "dry-run" if dry_run else "audit-only")


def run_pipeline(args: argparse.Namespace, outdir: Path, logger: logging.Logger) -> None:
    if not os.environ.get("SLURM_JOB_ID") and not (args.audit_only or args.dry_run):
        raise RuntimeError(
            "Refusing non-audit MVP-C science run outside Slurm. "
            "Submit through sbatch/salloc on a compute allocation."
        )

    hmi_path = Path(args.hmi_bin_timeseries)
    gong_run = Path(args.gong_run)

    if not hmi_path.exists():
        raise FileNotFoundError(f"HMI input path does not exist: {hmi_path}")
    if not gong_run.exists():
        raise FileNotFoundError(f"GONG run directory does not exist: {gong_run}")

    if args.dry_run:
        run_audit_only_or_dry(args, outdir, logger, dry_run=True)
        return
    if args.audit_only:
        run_audit_only_or_dry(args, outdir, logger, dry_run=False)
        return

    logger.info("Reading HMI bin time series: %s", hmi_path)
    raw_hmi = read_table(hmi_path)
    hmi, gamma_col, hmi_meta = normalize_hmi_table(raw_hmi, args)
    hmi_input_audit = make_hmi_input_audit(hmi, hmi_path, gamma_col, hmi_meta)
    logger.info("HMI rows after normalization: %d; gamma column: %s; format: %s", len(hmi), gamma_col, hmi_meta.get("hmi_format"))

    logger.info("Reading GONG/Q1 run: %s", gong_run)
    gong_bin, gong_window, gong_audit_raw, gong_meta = load_gong_run(gong_run)
    logger.info("GONG bin rows: %d; window_index rows: %d; audit rows: %d", len(gong_bin), len(gong_window), len(gong_audit_raw))

    bins = requested_bins(args)
    target_grid = build_target_grid(gong_bin, bins)
    gong_grid_audit = target_grid.copy()
    logger.info("Target GONG-like grid rows: %d", len(target_grid))

    rng = np.random.default_rng(int(args.random_seed))

    degraded, native_audit = build_degraded_hmi(hmi, gamma_col, target_grid, args, rng, logger)
    if len(native_audit):
        hmi_input_audit = hmi_input_audit.merge(
            native_audit,
            on=["freq_bin_low", "freq_bin_high", "bin_width"],
            how="left",
            suffixes=("", "_target"),
        )
    logger.info("Degraded HMI rows: %d", len(degraded))

    recovery_by_bin, bootstrap_summary = recover_lags(degraded, args, rng, logger)
    recovery_summary = summarize_recovery(recovery_by_bin, args)
    acceptance_summary, acceptance_json = evaluate_acceptance(recovery_summary, recovery_by_bin, args)

    write_outputs(
        outdir,
        hmi_input_audit,
        gong_grid_audit,
        degraded,
        recovery_by_bin,
        recovery_summary,
        bootstrap_summary,
        acceptance_summary,
        acceptance_json,
    )

    write_metadata(
        outdir,
        args,
        {
            "hmi_input_audit_meta": hmi_meta,
            "gong_metadata": gong_meta,
            "gamma_column_used": gamma_col,
            "n_hmi_rows_normalized": int(len(hmi)),
            "n_target_grid_rows": int(len(target_grid)),
            "n_degraded_rows": int(len(degraded)),
            "n_recovery_rows": int(len(recovery_by_bin)),
            "acceptance": acceptance_json,
        },
    )
    write_report(outdir, args, hmi_meta, gong_meta, acceptance_json, recovery_summary)
    make_plots(outdir, args, degraded, recovery_by_bin, logger)

    logger.info("MVP-C overall status: %s", acceptance_json.get("overall_status"))
    logger.info("Wrote outputs under %s", outdir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)

    try:
        setup_outdir(outdir, bool(args.overwrite))
        logger = setup_logging(outdir, args.log_level)
        logger.info("Command line: %s", " ".join(sys.argv))
        logger.info("SLURM_JOB_ID=%s SLURM_CPUS_PER_TASK=%s interpreted=%s", os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_CPUS_PER_TASK", ""), SLURM_CPUS_PER_TASK)
        logger.info("Thread policy: OPENBLAS=%s OMP=%s MKL=%s", os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"))
        run_pipeline(args, outdir, logger)
        return 0
    except Exception as exc:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "logs").mkdir(parents=True, exist_ok=True)
        err = {
            "generated_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "scope": "MVP-C validation-only",
        }
        with (outdir / "metadata_failed.json").open("w", encoding="utf-8") as f:
            json.dump(err, f, indent=2, sort_keys=True)
        sys.stderr.write(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
