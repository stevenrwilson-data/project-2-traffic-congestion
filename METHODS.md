# Methods

## Data

Caltrans PeMS five-minute station data were analyzed for February 1–May 31, 2026. The final analysis uses a five-station eastbound I-10 corridor.

## Observation quality

The selected corridor retained 164,657 rows at both the 80% and 100% observed thresholds, so the strict 100% threshold was used.

## Event definitions

- **Sustained congestion:** speed below 50 mph for at least three consecutive five-minute intervals.
- **LA temporal abrupt breakdown:** at least a 20 mph five-minute speed drop ending below 40 mph.
- **Caltrans spatial bottleneck:** upstream/downstream speed drop of at least 20 mph, downstream speed below 40 mph, pair gap below 3 miles, sustained in at least 5 of 7 consecutive intervals.

## Pre-breakdown comparison

Abrupt-onset events were aligned at onset and compared with controls matched on station, weekday/weekend status, and half-hour time bin on different dates. Controls were kept at least 60 minutes from any LA-rule onset.

## Corridor episodes and propagation

Same-day abrupt onsets were clustered into corridor episodes. Episode construction was tested at 15-, 30-, and 45-minute clustering gaps. Propagation classification was tested at 30-, 60-, and 120-minute windows; 60 minutes was retained as the primary operational window.

## Duration and recovery

Sustained-congestion duration is measured from first to last five-minute state interval. Right-censored events are retained in the recovery analysis rather than treated as complete.

## Robustness

Headline duration metrics were recomputed after removing each station and each date one at a time. These ranges measure influence/sensitivity; they are not confidence intervals.

## Final adjudication

Conclusions were labeled supported, supported with caveat, suggestive, supported methodologically, or unsupported for substantive claim.
