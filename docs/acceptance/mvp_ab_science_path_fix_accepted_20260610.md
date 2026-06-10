# GONG-Lag v4 MVP-A/B science-path fix acceptance

Accepted run on CARC:

`/scratch1/dongpoba/gong_lag_v4/runs/mvp_ab_fixed_science_nosplit_all`

Acceptance evidence:

- audit.csv: 1555 rows
- window_index.csv: 15550 rows
- task_errors.csv: 0 rows
- mode_diagnostics.csv: 410520 rows
- bin_time_series.csv: 59090 rows
- stack_diagnostics.csv: 15550 rows
- checkpoints: 1555
- all checkpoints status=done
- all checkpoints n_task_errors=0
- audit.shape_ok all true
- audit.window_binary all true
- audit/window_index quality_flags all empty
- assertion result: FULL-RUN SCIENCE PATH CHECK PASSED

Scope:

- Fixed science path runtime failure only.
- No HMI degradation implemented.
- No lag analysis implemented.
- No leakage model, Nigam-Kosovichev estimator, or temporal-window convolution implemented.
- Run used --splitting-file none and --shift-sign none, so missing_splitting is expected.
