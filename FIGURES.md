# Figure Guide

## Keep

- `images/traffic_b3_threshold_sensitivity.png` — **Data validity**: Shows threshold choice does not materially damage corridor coverage.
- `images/traffic_b4_example_speed_heatmap.png` — **Breakdown definitions**: Visually anchors the distinction between congestion state and abrupt onset.
- `images/traffic_b4_part2_state_vs_onset_counts.png` — **Breakdown definitions**: Directly communicates that the operational definitions measure different phenomena.
- `images/traffic_b5p2b_classification_robustness.png` — **Propagation**: Shows which episode classifications survive alternative lag windows.
- `images/traffic_b5p3_recovery_survival.png` — **Recovery**: Shows the long tail of sustained congestion while respecting right censoring.
- `images/traffic_b6_leave_one_out_duration_robustness.png` — **Robustness**: Demonstrates that headline duration metrics are not dominated by one station or date.
- `images/traffic_b6_duration_by_time_period.png` — **Duration**: Directly communicates the strong midday-duration result.

## Keep Supporting

- `images/traffic_b3_weakest_station_threshold.png` — **Data validity**: Shows weakest-link coverage, important for spatial analysis.
- `images/traffic_b4_part2_onset_timing_normalized.png` — **Breakdown definitions**: Compares timing without allowing the high-count state method to flatten rare onset methods.
- `images/traffic_b5p1_event_centered_profiles.png` — **Precursors**: Shows the abrupt speed collapse and surrounding flow/occupancy behavior.
- `images/traffic_b5p1_breakdown_speed_heatmap.png` — **Precursors**: Shows event-to-event heterogeneity in the ten abrupt onsets.
- `images/traffic_b5p1_preconditions_vs_controls.png` — **Precursors**: Shows matched-control differences are modest rather than dramatic.
- `images/traffic_b5p2b_station_participation_by_window.png` — **Propagation**: Makes sensitivity of episodes 3 and 5 easy to see.
- `images/traffic_b5p3_duration_by_station.png` — **Duration**: Shows persistent duration differences across corridor stations.
- `images/traffic_b5p3_recovery_order.png` — **Recovery**: Shows recovery order differs across corridor episodes.
- `images/traffic_b6_abrupt_onset_concentration.png` — **Limitations**: Makes the small and date-concentrated abrupt-onset sample transparent.

## Audit Only

- `images/traffic_b3_measurements_by_coverage.png` — **Audit**: Useful diagnostic but not a headline result.
- `images/traffic_b3_exact_repeat_diagnostic.png` — **Audit**: Diagnostic only; exact repeats do not establish imputation.
- `images/traffic_b3_lane_consistency.png` — **Audit**: Validates metadata-defined physical lanes rather than a substantive result.

## Exclude

- `images/traffic_b5p4_breakdown_probability_plm.png` — **Probability audit**: Maximum probability is substantively tiny and the uncensored sample is too sparse for a useful portfolio chart.

## How to read the leave-one-out robustness plot

The full-sample dot is the statistic using all data. The station and date ranges show the minimum and maximum value obtained after removing one station or one date at a time. They are sensitivity ranges, not confidence intervals.
