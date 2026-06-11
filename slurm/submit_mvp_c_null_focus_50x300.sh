#!/bin/bash
set -euo pipefail

RUN_ROOT=/scratch1/dongpoba/gong_lag_v4/mvp_c_hmi_degradation/runs/null_focus_50x300
LOG_ROOT=${RUN_ROOT}/logs/slurm

mkdir -p "${LOG_ROOT}"

cd /scratch1/dongpoba/gong_lag_v4_git_backup

sbatch slurm/mvp_c_null_focus_50x300.job
