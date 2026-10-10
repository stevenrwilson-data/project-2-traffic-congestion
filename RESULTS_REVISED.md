# Results — Revised After Final Challenge Review

## Abrupt-onset data continuity

The LA temporal rule uses a five-minute speed drop. This audit checks the filtered 100%-observed data at t-10, t-5, t, t+5 and t+10 for every abrupt onset.

| onset_time | event_date | station_id | present_minus_10 | present_minus_5 | present_at_onset | present_plus_5 | present_plus_10 | temporal_drop_pair_contiguous | nearby_5min_context_complete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-01 11:10:00 | 2026-02-01 | 716152 | True | True | True | True | True | True | True |
| 2026-02-27 08:00:00 | 2026-02-27 | 717185 | True | True | True | True | True | True | True |
| 2026-02-27 08:00:00 | 2026-02-27 | 717190 | True | True | True | True | True | True | True |
| 2026-03-02 05:10:00 | 2026-03-02 | 717198 | True | True | True | True | True | True | True |
| 2026-03-02 05:15:00 | 2026-03-02 | 717190 | True | True | True | True | True | True | True |
| 2026-03-02 05:20:00 | 2026-03-02 | 717185 | True | True | True | True | True | True | True |
| 2026-03-02 05:20:00 | 2026-03-02 | 717195 | True | True | True | True | True | True | True |
| 2026-03-19 10:40:00 | 2026-03-19 | 716152 | True | True | True | True | True | True | True |
| 2026-04-30 10:20:00 | 2026-04-30 | 716152 | True | True | True | True | True | True | True |
| 2026-05-31 14:10:00 | 2026-05-31 | 717185 | True | True | True | True | True | True | True |

March 2 t-5/t pairs are contiguous at all four onset stations: **True**.

March 2 t-5/t/t+5 context is complete at all four onset stations: **True**.

## Clustering claim audit

| event_date | onsets | first_onset | last_onset | full_span_minutes | maximum_consecutive_gap_minutes | would_15_minute_gap_split_date |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-01 | 1 | 2026-02-01 11:10:00 | 2026-02-01 11:10:00 | 0.0 |  | False |
| 2026-02-27 | 2 | 2026-02-27 08:00:00 | 2026-02-27 08:00:00 | 0.0 | 0.0 | False |
| 2026-03-02 | 4 | 2026-03-02 05:10:00 | 2026-03-02 05:20:00 | 10.0 | 5.0 | False |
| 2026-03-19 | 1 | 2026-03-19 10:40:00 | 2026-03-19 10:40:00 | 0.0 |  | False |
| 2026-04-30 | 1 | 2026-04-30 10:20:00 | 2026-04-30 10:20:00 | 0.0 |  | False |
| 2026-05-31 | 1 | 2026-05-31 14:10:00 | 2026-05-31 14:10:00 | 0.0 |  | False |

15-minute clustering provides a real split challenge: **False**.

## Original versus full-context duration medians

| batch_7a_time_period | original_05_20_median_minutes | full_day_context_median_minutes | change_minutes |
| --- | --- | --- | --- |
| AM | 50.0 | 50.0 | 0.0 |
| Midday | 170.0 | 170.0 | 0.0 |
| PM | 45.0 | 50.0 | 5.0 |

## Short-gap merging sensitivity

| merge_gap_minutes | events_total | am_events | midday_events | pm_events | am_median_duration | midday_median_duration | pm_median_duration | midday_minus_max_am_pm | midday_episodes_crossing_15_00_pct | merged_components_gt1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1462 | 401 | 698 | 363 | 50.0 | 170.0 | 50.0 | 120.0 | 60.89 | 0 |
| 5 | 1331 | 381 | 658 | 292 | 50.0 | 222.5 | 60.0 | 162.5 | 64.89 | 119 |
| 10 | 1241 | 370 | 629 | 242 | 50.0 | 255.0 | 65.0 | 190.0 | 67.89 | 174 |
| 15 | 1149 | 353 | 596 | 200 | 55.0 | 275.0 | 70.0 | 205.0 | 71.48 | 222 |

Midday-start episodes remain longer than both AM and PM under every tested merge rule: **True**.

## Adjacent-pair bottleneck ranking

| bottleneck_rank | upstream_travel_order | upstream_station_id | upstream_name | downstream_travel_order | downstream_station_id | downstream_name | paired_intervals | queue_signature_intervals | queue_signature_pct_all_intervals | queue_signature_pct_when_upstream_congested | reverse_signature_intervals | persistent_queue_signature_runs_15min | median_downstream_minus_upstream_speed_when_signature |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 716152 | LARK ELLEN | 4 | 717195 | AZUSA 1 | 19066 | 1314 | 6.892 | 18.012 | 492 | 172 | 9.3 |
| 2 | 2 | 717190 | VINCENT 2 | 3 | 716152 | LARK ELLEN | 20106 | 513 | 2.551 | 6.58 | 336 | 36 | 21.8 |
| 3 | 4 | 717195 | AZUSA 1 | 5 | 717198 | AZUSA 2 | 19788 | 391 | 1.976 | 5.894 | 198 | 31 | 3.3 |
| 4 | 1 | 717185 | VINCENT 1 | 2 | 717190 | VINCENT 2 | 21008 | 104 | 0.495 | 1.29 | 100 | 6 | 2.35 |

Top-ranked pair: **716152 LARK ELLEN → 717195 AZUSA 1**.

## Downstream discharge-flow audit

| top_ranked_upstream_station | top_ranked_downstream_station | abrupt_onsets | onsets_with_free_downstream_discharge_station | median_discharge_flow_vphpl_when_free | minimum_discharge_flow_vphpl_when_free | maximum_discharge_flow_vphpl_when_free | plm_revival_recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 716152 | 717195 | 10 | 9 | 1200.0 | 0.0 | 1698.0 | False |

## March 2 concentration

| date | abrupt_onsets | stations | share_of_all_abrupt_onsets_pct | cause_assigned | recommended_wording |
| --- | --- | --- | --- | --- | --- |
| 2026-03-02 | 4 | 4 | 40.0 | False | Four of ten abrupt onsets occurred on March 2, 2026. The current analysis does not assign a causal incident or weather explanation to that concentration. |

## Corridor-selection provenance

| final_corridor | final_freeway | final_direction | final_station_count | final_span_miles | earlier_210w_candidates_visible_in_final_comparison | selection_basis | recommended_writeup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fwy_10_E_physical_run_6 | 10 | E | 5 | 1.2040000000000006 | 1 | Corrected physical-adjacency construction followed by weakest-link-first traffic-content selection. | Earlier exploratory rankings included a 210 W candidate. The final analysis uses I-10 E after the corrected Batch 2 workflow rebuilt corridors using physical adjacency and selected the primary corridor using weakest-station coverage, simultaneous coverage, recurring low-speed evidence, and tie-breakers. The final corridor is short, only 1.20 miles, so propagation claims are explicitly limited to this local corridor scale. |

## Revised finding status

| finding | status | basis |
| --- | --- | --- |
| Sustained congestion and abrupt breakdown are distinct operational phenomena. | SUPPORTED | 1,457 sustained-state events versus 10 abrupt temporal onsets and 7 spatial bottleneck events. The temporal-onset sample is also checked for filtered-data continuity at t-5 and t. |
| The 10 abrupt onsets form six same-day corridor episodes. | DESCRIPTIVE_NOT_ROBUSTNESS | The 15-minute rule does not split any multi-onset date, so 15/30/45-minute invariance is not a strong robustness test. |
| Episodes beginning 10:00–15:00 last longer than AM or PM starts. | SUPPORTED_AFTER_FRAGMENTATION_TEST | Full-day recovery context plus 0/5/10/15-minute short-gap merging sensitivity. |
| A recurring local bottleneck signature can be ranked across adjacent station pairs. | SUPPORTED_DIAGNOSTIC | Top pair: 716152 → 717195; Lark Ellen → Azusa 1 = True. |
| The original product-limit probability chart remains excluded. | EXCLUDE | Upstream queued flows were not appropriate capacity measurements; the downstream discharge audit is more defensible, but ten abrupt events remain too sparse. |
