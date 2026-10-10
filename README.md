# Traffic Breakdown and Congestion Dynamics on an I-10 East Corridor

## Overview

This project analyzes congestion and traffic breakdown on a five-station eastbound I-10 corridor in Caltrans District 7 using Caltrans PeMS five-minute station data from February 1–May 31, 2026.

The project separates sustained congestion from abrupt breakdown onset, studies corridor propagation and recovery, and tests the main results for sensitivity to observation thresholds, station choice, date choice, clustering windows, and propagation windows.

## Data and corridor

- Five physically adjacent eastbound I-10 stations: 717185, 717190, 716152, 717195, 717198.
- Final 100%-observed analytical sample: **164,657 rows** and **2,540.1 simultaneous corridor-hours**.
- The 80% and 100% observation thresholds retained exactly the same selected-corridor rows.

## Headline findings

### 1. Sustained congestion and abrupt breakdown are empirically different traffic phenomena on this corridor.

**Evidence:** The sustained-state rule produced 1457 events, the LA abrupt-collapse rule 10, and the Caltrans spatial rule 7, with extremely low pairwise onset overlap.

**Caveat:** The definitions answer different operational questions rather than competing to identify one universal event set.

### 2. Corridor-level abrupt-breakdown episode construction is robust to the temporal clustering rule.

**Evidence:** 15-, 30-, and 45-minute clustering windows all produced 6 corridor episodes.

**Caveat:** Only six corridor episodes were identified, so the result is strong for this sample but not a universal traffic-flow claim.

### 3. Midday sustained-congestion episodes last substantially longer than AM or PM episodes.

**Evidence:** Median durations were 50 minutes AM, 170 midday, and 45 PM. The midday advantage remained positive after every leave-one-station and leave-one-date exclusion.

**Caveat:** Time-period comparisons are observational and may reflect demand, geometry, incidents, or recurring operational conditions.

### 4. Sustained congestion is often persistent and has a long duration tail.

**Evidence:** Median duration was 60 minutes and the 90th percentile was 355 minutes.

**Caveat:** The sustained-state rule is an operational definition, and some episodes are right-censored by the analysis window.

## Selected quantitative results

- Sustained-congestion events: **1,457**.
- LA-rule abrupt onsets: **10**.
- Caltrans spatial bottleneck events: **7**.
- Corridor-level abrupt-breakdown episodes: **6**.
- Median sustained-congestion duration: **60 minutes**; p90: **355 minutes**.
- AM / Midday / PM median durations: **50 / 170 / 45 minutes**.
- Five minutes before abrupt onset, event/control medians were speed **58.10/61.95 mph**, flow **419.0/443.5 veh/5 min**, occupancy **0.1087/0.1150**.
- Largest single-date share of LA-rule onsets: **40%** on 2026-03-02.

## Robustness

- 15-, 30-, and 45-minute clustering windows all produced **6 episodes**.
- **4/6** propagation classifications were unchanged across 30/60/120-minute windows.
- Midday remained longer than AM/PM after every leave-one-station and leave-one-date exclusion: **True**.

## Repository guide

- `prepare_traffic.py` — reproducible analysis pipeline.
- `METHODS.md` — definitions and analytical design.
- `RESULTS.md` — final quantitative findings.
- `LIMITATIONS.md` — scope and inferential limitations.
- `FIGURES.md` — figure-by-figure interpretation.
- `audit_reports/` — complete batch-by-batch paper trail.
- `portfolio_figures/` — shortlisted presentation figures.

## Status

The analytical phase is complete. Batch 7 packages the work for repository and portfolio presentation.
