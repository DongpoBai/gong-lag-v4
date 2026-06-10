# WMLTP-Q1 splitting validation report

Generated: `2026-06-10T18:01:14.588457+00:00`

Acceptance category: **PASS**

## Run directories

- baseline: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/runs/q1_prod_hmi2010_nosplit_all`
- split: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/runs/q1_prod_hmi2010_split_auto_all`

## Scope

This report compares an existing no-splitting AB run with an existing splitting-corrected AB run.
It does not run AB, does not submit jobs, and does not perform lag inference.

## Acceptance reasoning

No fail or warning reasons were triggered by the scripted criteria.

## Key metrics

- `l_gt_0_median_ratio_chosen_to_before`: `7.1738058698114004`
- `l_gt_0_median_ratio_split_to_baseline_chosen`: `7.1738058698114004`
- `l_gt_0_fraction_ratio_lt_0p95`: `0.045840032154340835`
- `range_2000_4500_median_fisher_trace_ratio`: `1.5855004645439172`
- `range_2000_4500_median_sigma_nu_ratio`: `0.8121824690045547`
- `range_2000_4500_median_sigma_lngamma_ratio`: `0.76827413119733`
- `range_2000_4500_split_fraction_finite_delta_nu`: `1.0`
- `range_2000_4500_split_fraction_finite_delta_lngamma`: `1.0`
- `range_3000_3500_split_fraction_finite_delta_nu`: `1.0`
- `range_3000_3500_split_fraction_finite_delta_lngamma`: `1.0`
- `missing_splitting_mode_diagnostics_baseline`: `1184910`
- `missing_splitting_mode_diagnostics_split`: `202150`

## Output tables

- `q1_run_health_summary.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_run_health_summary.csv`
- `q1_grid_consistency.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_grid_consistency.csv`
- `q1_stack_sharpness_by_l.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_stack_sharpness_by_l.csv`
- `q1_stack_sharpness_overall.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_stack_sharpness_overall.csv`
- `q1_bin_fisher_comparison.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_bin_fisher_comparison.csv`
- `q1_mode_fisher_comparison.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_mode_fisher_comparison.csv`
- `q1_quality_flag_comparison.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_quality_flag_comparison.csv`
- `q1_time_series_difference_summary.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_time_series_difference_summary.csv`
- `q1_shift_sign_summary.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_shift_sign_summary.csv`
- `q1_missing_splitting_by_l_n.csv`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/tables/q1_missing_splitting_by_l_n.csv`
- `q1_acceptance_summary.json`: `/scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_prod_hmi2010_split_auto_vs_nosplit/q1_acceptance_summary.json`

## Plot status

- matplotlib_available: `False`
- warning: matplotlib not available; plots skipped: No module named 'matplotlib'

## Interpretation guardrail

This is a Q1 shift-and-stack / score-Fisher diagnostics validation only.
Do not interpret `delta_nu` / `delta_lngamma` differences here as a lag result.
Do not claim independent GONG verification from this report.