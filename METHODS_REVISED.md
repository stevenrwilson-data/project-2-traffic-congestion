# Methods — Revised After Final Challenge Review

## Time-of-day boundaries

- AM: 05:00–09:59
- Midday: 10:00–14:59
- PM: 15:00–19:59

## Duration correction

Sustained-congestion states were re-detected from full-day 100%-observed station data. Only episodes whose observed onset occurred from 05:00 through 19:59 were assigned to AM, Midday, or PM, so recovery after 20:00 is no longer artificially truncated.

To test fragmentation, same-station congestion-state episodes were also merged when separated by only 5, 10, or 15 minutes at or above 50 mph. The 0-minute case preserves the unmerged full-context result.

## Bottleneck localization

Each adjacent station pair was tested for a queue signature: upstream speed below 50 mph while the immediately downstream station remained at or above 50 mph. The ranking uses persistent 15-minute runs first, then conditional queue-signature frequency.

## Flow audit

The original capacity-probability attempt used flow at whichever station triggered an abrupt temporal onset. The corrective audit instead examines the downstream station of the top-ranked bottleneck pair five minutes before each abrupt onset, conditional on that downstream station remaining free-flowing.

## Episode clustering correction

The earlier claim that six episodes were robust to 15/30/45-minute clustering is no longer used as a headline robustness result. The corrective audit explicitly tests whether a 15-minute gap could have split any multi-onset date.
