# Q1-low-l full HMI2010 reference splitting validation PASS

Status: PASS.

Scope:
- WMLTP-Q1 only
- rotational Legendre splitting correction
- stored-nonnegative-m shift-and-stack
- downstream score/Fisher diagnostics

Not included:
- complete WMLTP
- leakage model
- temporal-window convolution
- Nigam-Kosovichev asymmetry
- negative-m reconstruction
- lag analysis
- independent GONG verification

Inputs:
- baseline: /scratch1/dongpoba/gong_lag_v4/runs/mvp_ab_fixed_science_nosplit_all
- split-auto: /scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/runs/q1_low_l_hmi2010_split_auto_all
- comparison: /scratch1/dongpoba/gong_lag_v4/q1_splitting_validation/comparisons/q1_low_l_hmi2010_split_auto_vs_accepted_nosplit
- AB script: /scratch1/dongpoba/gong_lag_v4/scripts/gong_lag_v4_mvp_ab_q1_flat_spl.py
- splitting reference: /project2/erhodes_44/rcf-04/astro10/wmltp/theor1.0/HMI.100430to0728.mpts.v1.spl.mod.dat

Acceptance summary:
- acceptance_category: PASS
- fail_reasons: []
- warn_reasons: []
- baseline checkpoints: 1555
- split checkpoints: 1555
- task_errors: 0
- l>0 median sharpness_chosen/sharpness_before: 1.1771274021189453
- l>0 fraction ratio > 1: 0.747427652733119
- l>0 fraction ratio > 1.05: 0.6770900321543408
- l>0 fraction ratio < 0.95: 0.1832797427652733
- 2000-4500 uHz median fisher_trace ratio: 0.9973094195347576
- 2000-4500 uHz median sigma_nu ratio: 1.0017327006796042
- 2000-4500 uHz median sigma_lngamma ratio: 1.00440212544019
- 3000-3500 uHz finite delta_nu fraction: 1.0
- 3000-3500 uHz finite delta_lngamma fraction: 1.0
- mode_diagnostics missing_splitting baseline -> split: 410520 -> 118180

Interpretation:
HMI-2010 reference splitting is active and improves full low-l stack sharpness while preserving usable score/Fisher diagnostics. This validates the Q1 splitting-corrected machinery on the accepted low-l grid, but it is not a lag result and not complete WMLTP.
