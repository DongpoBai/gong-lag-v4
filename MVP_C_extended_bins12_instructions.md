# GONG-Lag v4 MVP-C — extended_bins_12 primary HMI input instructions

Static generation date: 2026-06-10  
Scope: MVP-C HMI-to-GONG degradation / known-lag injection validation only.

This document replaces the earlier pasted Markdown instructions. It incorporates the total-control decision that the MVP-C primary HMI input is the staged 12-bin extended HMI table:

```text
/scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv
```

The source staging path is:

```text
/home1/dongpoba/3day_extended_bins.csv
```

Candidate 2 is retained only for later robustness reconstruction:

```text
/home1/dongpoba/hmi_mdi_all_modes_3day.parquet
```

Do not use Candidate 2 as MVP-C primary input.

---

## 1. MVP-C scientific target

MVP-C asks whether a known HMI bin time series, after GONG-like degradation, still recovers injected hour-scale lag without unacceptable bias. The validation pieces are:

```text
9-day effective averaging kernel
3-day GONG-like target stride/grid
GONG-like t_mid grid
GONG-like covariance/noise
81-day running-mean detrending
validation-only CC lag estimator with parabolic sub-step interpolation
moving-block bootstrap
```

The science question is whether the degradation process:

```text
1. turns tau_inj = 0 hr into a systematic positive lag
2. washes out +6 hr / +12 hr
3. creates unacceptable bias or under-covered uncertainty
```

This remains validation-only. It is not a final GONG lag claim, not full WMLTP, and not independent GONG verification.

Lag convention:

```text
positive lag means delta_gamma or delta_lngamma leads delta_nu
C(tau) = corr[gamma'(t), nu'(t + tau)]
tau_inj = +6 hr means linewidth residuals lead frequency residuals by 6 hr
```

---

## 2. Primary HMI input format: `extended_bins_12`

Use the explicit CLI setting:

```bash
--hmi-format=extended_bins_12
```

The loader must not guess bin edges. It must use exactly:

```text
1500,2000,2500,3000,3500,3750,4000,4500,5000,5500,6000,6500,7000
```

For each `k = 1..12`:

```text
t_mid       = mid_date
t_start     = start_date
t_end       = start_date + 3 days
delta_nu    = dnu_bin{k}
sigma_nu    = dnu_err_bin{k}
delta_gamma = dw_bin{k}
sigma_gamma = dw_err_bin{k}
nmodes      = nmodes_bin{k}
```

The file has no `delta_lngamma`. Do not fabricate `delta_lngamma`.

For MVP-C primary runs:

```text
primary linewidth variable = delta_gamma
```

The HMI table is a raw shift table relative to a solar-minimum reference window. It is not already an 81-day detrended residual table. The MVP-C script applies downstream 81-day detrending:

```text
x'(t) = x(t) - running_mean_81d(x)
```

Implementation note for noise: the accepted GONG/Q1 covariance is in the `delta_nu` / `delta_lngamma` coordinates. When `--hmi-format=extended_bins_12` uses `delta_gamma`, the script records that `cov_lngamma_lngamma` is used only as a linewidth-noise proxy for the `delta_gamma` validation injection. It does not relabel or fabricate `delta_lngamma`.

---

## 3. Files in this downloadable set

The downloadable set contains:

```text
MVP_C_extended_bins12_instructions.md
gong_lag_v4_mvp_c_hmi_degradation_injection.py
slurm/mvp_c_smoke_debug.job
slurm/mvp_c_small_main.job
slurm/mvp_c_production_oneweek.job
README_MVP_C_SLURM.md
mvp_c_extended_bins12_static_bundle.zip
```

The Python entrypoint adds:

```text
--hmi-format auto|canonical_long|extended_bins_12
```

and implements explicit `extended_bins_12` loader support.

---

## 4. Stage and inspect only the explicit HMI candidate

Do this on Discovery login node only for staging and exact-path inspection. Do not run the science pipeline on the login node.

```bash
mkdir -p /scratch1/dongpoba/gong_lag_v4/inputs/hmi

cp -p /home1/dongpoba/3day_extended_bins.csv \
  /scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv

ls -lh /scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv
stat /scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv
head -2 /scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv
```

Allowed here: exact-path `ls`, `stat`, `head`.

Not allowed:

```text
find /project2/erhodes_44 ...
grep -R /project2/erhodes_44 ...
du -sh /project2/erhodes_44
```

---

## 5. Install the generated files on Discovery

Assuming the downloaded bundle is unpacked in a temporary working directory:

```bash
cd /path/to/unpacked/mvp_c_extended_bins12_static_bundle

mkdir -p /scratch1/dongpoba/gong_lag_v4_git_backup
mkdir -p /scratch1/dongpoba/gong_lag_v4_git_backup/slurm
mkdir -p /scratch1/dongpoba/gong_lag_v4/mvp_c_hmi_degradation/logs/slurm

install -m 750 gong_lag_v4_mvp_c_hmi_degradation_injection.py \
  /scratch1/dongpoba/gong_lag_v4_git_backup/gong_lag_v4_mvp_c_hmi_degradation_injection.py

install -m 750 slurm/mvp_c_smoke_debug.job \
  /scratch1/dongpoba/gong_lag_v4_git_backup/slurm/mvp_c_smoke_debug.job

install -m 750 slurm/mvp_c_small_main.job \
  /scratch1/dongpoba/gong_lag_v4_git_backup/slurm/mvp_c_small_main.job

install -m 750 slurm/mvp_c_production_oneweek.job \
  /scratch1/dongpoba/gong_lag_v4_git_backup/slurm/mvp_c_production_oneweek.job

install -m 640 README_MVP_C_SLURM.md \
  /scratch1/dongpoba/gong_lag_v4_git_backup/README_MVP_C_SLURM.md
```

---

## 6. Required default environment for Slurm jobs

The Slurm templates default to:

```bash
HMI_BIN_TIMESERIES=/scratch1/dongpoba/gong_lag_v4/inputs/hmi/3day_extended_bins.csv
HMI_FORMAT=extended_bins_12
GONG_RUN=/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/runs/q1_prod_hmi2010_split_auto_all
```

Every job includes:

```bash
#SBATCH --account=erhodes_44
```

Every job sets:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

No GPU partition is used. No `gluscevi_339` account is used.

---

## 7. Submit smoke/debug audit

The debug job is audit-only. It reads and normalizes the HMI input, reads the GONG target grid, writes audit outputs, and does not perform the full injection recovery.

```bash
cd /scratch1/dongpoba/gong_lag_v4_git_backup
sbatch slurm/mvp_c_smoke_debug.job
```

Expected run directory pattern:

```text
/scratch1/dongpoba/gong_lag_v4/mvp_c_hmi_degradation/smoke_debug_<JOBID>
```

---

## 8. Submit small main validation

Run this only after smoke/debug audit produced the expected audit outputs.

```bash
cd /scratch1/dongpoba/gong_lag_v4_git_backup
sbatch slurm/mvp_c_small_main.job
```

Small main uses:

```text
inject_lags_hours = 0,6,12
n_injection_realizations = 3
noise_model = gaussian_covariance
bootstrap_n = 50
```

Expected run directory pattern:

```text
/scratch1/dongpoba/gong_lag_v4/mvp_c_hmi_degradation/small_main_<JOBID>
```

---

## 9. Submit production oneweek validation

Use this only after small main outputs are acceptable.

```bash
cd /scratch1/dongpoba/gong_lag_v4_git_backup
sbatch slurm/mvp_c_production_oneweek.job
```

Production uses:

```text
inject_lags_hours = -24,-12,0,6,12,24
n_injection_realizations = 50
noise_model = gaussian_covariance
bootstrap_n = 500
```

Expected run directory pattern:

```text
/scratch1/dongpoba/gong_lag_v4/mvp_c_hmi_degradation/production_oneweek_<JOBID>
```

---

## 10. Output checks

Set `OUTDIR` to the actual run directory.

```bash
OUTDIR=/scratch1/dongpoba/gong_lag_v4/mvp_c_hmi_degradation/small_main_<JOBID>

ls -lh "$OUTDIR"/metadata.json
ls -lh "$OUTDIR"/mvp_c_acceptance_summary.json
ls -lh "$OUTDIR"/MVP_C_REPORT.md
ls -lh "$OUTDIR"/tables/hmi_input_audit.csv
ls -lh "$OUTDIR"/tables/gong_grid_audit.csv
ls -lh "$OUTDIR"/tables/degraded_hmi_timeseries.csv
ls -lh "$OUTDIR"/tables/injection_recovery_by_bin.csv
ls -lh "$OUTDIR"/tables/injection_recovery_summary.csv
ls -lh "$OUTDIR"/tables/bootstrap_summary.csv
ls -lh "$OUTDIR"/tables/acceptance_summary.csv

head -5 "$OUTDIR"/tables/hmi_input_audit.csv
head -5 "$OUTDIR"/tables/injection_recovery_by_bin.csv
head -20 "$OUTDIR"/tables/injection_recovery_summary.csv
head -40 "$OUTDIR"/MVP_C_REPORT.md
grep -n "overall_status" "$OUTDIR"/mvp_c_acceptance_summary.json
grep -n "hmi_format" "$OUTDIR"/metadata.json
grep -n "delta_lngamma_fabricated" "$OUTDIR"/metadata.json
```

The checks are exact-file checks and do not recursively scan project trees.

---

## 11. Expected MVP-C outputs

Each run directory should contain:

```text
metadata.json
logs/
tables/hmi_input_audit.csv
tables/gong_grid_audit.csv
tables/degraded_hmi_timeseries.csv
tables/injection_recovery_by_bin.csv
tables/injection_recovery_summary.csv
tables/bootstrap_summary.csv
tables/acceptance_summary.csv
mvp_c_acceptance_summary.json
MVP_C_REPORT.md
plots/
```

If `matplotlib` is unavailable, CSV/JSON/MD outputs should still be written and the script logs a warning.

---

## 12. Monitoring

```bash
squeue --me
jobinfo <id>
seff <id>
sacct
scancel <id>
```

After the first submitted job, use:

```bash
myaccount
```

Confirm `erhodes_44` usage increased and `gluscevi_339` did not move.

---

## 13. Short interactive smoke allocation only

Use `salloc` only for short smoke checks. Do not use it for production.

```bash
salloc --partition=debug --cpus-per-task=4 --mem=16G --time=1:00:00 --account=erhodes_44
```

or:

```bash
salloc --partition=main --cpus-per-task=8 --mem=16G --time=2:00:00 --account=erhodes_44
```

Inside the allocation:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

## 14. SU cost awareness

```text
SU is charged by reservation, not actual usage.
Per minute: 1 CPU = 1 SU, 4GB memory = 1 SU.
Example: 16 CPU + 32GB for 24h ≈ 34,560 SU.
Do not over-request cpus-per-task or memory.
```

---

## 15. Acceptance interpretation

The script writes `mvp_c_acceptance_summary.json` and `tables/acceptance_summary.csv`, but these are conservative screening labels. Review the per-bin CSV before promoting any statement.

PASS intent:

```text
tau_inj = 0 hr does not produce systematic positive lag
+6 hr and +12 hr recover in 3000-3500 microHz without severe bias
2000-4500 microHz main range bias is acceptable
bootstrap uncertainty is not obviously anti-conservative
finite fraction is high
no widespread non-finite / failed bins
```

WARN intent:

```text
partial bin recovery
mixed bias or coverage
focus region usable but other bins unstable
tau_inj = 0 hr has mild positive tendency but within uncertainty
```

FAIL intent:

```text
tau_inj = 0 hr systematically becomes positive lag
+6/+12 hr washed out or sign-reversed
many bins non-finite
bootstrap fails
GONG-like grid mapping is wrong
```

---

## 16. Static-generation caveat

The files in this bundle are static-generation products. They have not been used here to run a Discovery science pipeline. The intended first runtime action is the Slurm smoke/debug audit job.
