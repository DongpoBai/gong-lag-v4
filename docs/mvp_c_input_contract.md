# MVP-C input contract

MVP-C goal:

HMI-to-GONG degradation and lag injection test.

MVP-C does not make a formal GONG lag claim.

## GONG input

Accepted MVP-A/B table:

`/scratch1/dongpoba/gong_lag_v4/runs/mvp_ab_fixed_science_nosplit_all/tables/bin_time_series.csv`

Required GONG columns:

- t_start
- t_mid
- t_end
- freq_bin_low
- freq_bin_high
- freq_bin_center
- delta_nu
- delta_lngamma
- sigma_nu
- sigma_lngamma
- quality_flags

## HMI input

Preferred HMI input is already reduced to bin-level or mode-level time series.

Minimum bin-level columns:

- t_mid or timestamp
- freq_bin_low
- freq_bin_high or freq_bin_center
- delta_nu
- delta_lngamma or delta_gamma or gamma_residual
- uncertainty columns if available

If only mode-level HMI tables are available, required columns are:

- timestamp or epoch
- l
- n
- nu or delta_nu
- gamma or delta_lngamma
- frequency or nu_ref
- quality flag / bad flag if available

## MVP-C outputs

- tables/degradation_map.csv
- tables/injection_cases.csv
- tables/injection_recovery.csv
- tables/injection_summary.csv
- metadata.json
- logs

## Science rule

Positive lag means δΓ or δlnΓ leads δν.

No independent GONG verification claim is allowed until HMI-to-GONG degradation and injection tests pass.
