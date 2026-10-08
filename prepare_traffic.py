import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Install once in your virtual environment: python -m pip install pandas pyarrow
# Run from your project folder: python prepare_traffic.py
# Or: python prepare_traffic.py --input /path/to/station_5min
# All imports stay at the absolute top. Original compressed files are untouched.
# Reference: PeMS Station 5-Minute field specification (12 station fields,
# then five fields per lane). Verify against one actual export before analysis.
BASE_COLUMNS = [
    'timestamp', 'station_id', 'district', 'freeway', 'direction', 'lane_type',
    'station_length_miles', 'samples', 'observed_pct', 'flow_veh_5min',
    'occupancy_fraction', 'speed_mph',
]
LANE_FIELDS = ['samples', 'flow_veh_5min', 'occupancy_fraction', 'speed_mph', 'observed']
STRING_COLUMNS = ['timestamp', 'direction', 'lane_type']


def inspect_format(path):
    """Scan widths first so different lane counts do not shift column meanings."""
    widths = set()
    rows = 0
    with gzip.open(path, 'rt', newline='') as source:
        for row in csv.reader(source):
            if not row:
                continue
            width = len(row)
            if width < 12 or (width - 12) % 5:
                raise ValueError(f'{path.name}: invalid field count {width} at row {rows + 1}')
            widths.add(width)
            rows += 1
    if not rows:
        raise ValueError(f'{path.name}: empty archive')
    names = BASE_COLUMNS.copy()
    for lane in range(1, (max(widths) - 12) // 5 + 1):
        names.extend(f'lane_{lane}_{field}' for field in LANE_FIELDS)
    return names, rows, sorted(widths)


def prepare_chunk(frame, source_name, date, min_observed, congestion_speed):
    """Preserve values, add flags; never silently discard or impute observations."""
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    # Keep PeMS local wall time. Do not silently assign UTC or resolve DST.
    for col in frame.columns:
        if col not in STRING_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').astype('float64')
    frame['invalid_timestamp'] = frame.timestamp.isna()
    frame['date_mismatch'] = frame.timestamp.dt.strftime('%Y-%m-%d').ne(date)
    frame['invalid_station_id'] = frame.station_id.isna() | frame.station_id.le(0) | frame.station_id.mod(1).ne(0)
    frame['invalid_speed'] = frame.speed_mph.isna() | ~frame.speed_mph.between(0, 120)
    frame['invalid_flow'] = frame.flow_veh_5min.isna() | frame.flow_veh_5min.lt(0)
    frame['invalid_occupancy'] = frame.occupancy_fraction.isna() | ~frame.occupancy_fraction.between(0, 1)
    frame['invalid_observed_pct'] = frame.observed_pct.isna() | ~frame.observed_pct.between(0, 100)
    frame['low_observed'] = frame.invalid_observed_pct | frame.observed_pct.lt(min_observed)
    frame['off_5min_grid'] = frame.timestamp.dt.minute.mod(5).ne(0) | frame.timestamp.dt.second.ne(0)
    flags = ['invalid_timestamp', 'date_mismatch', 'invalid_station_id', 'invalid_speed',
             'invalid_flow', 'invalid_occupancy', 'low_observed', 'off_5min_grid']
    frame['analysis_eligible'] = ~frame[flags].any(axis=1) & frame.lane_type.eq('ML')
    # Hourly equivalent rate, NOT an observed one-hour vehicle count.
    frame['flow_veh_hour_equivalent'] = frame.flow_veh_5min * 12
    frame['occupancy_pct'] = frame.occupancy_fraction * 100
    frame['hour'] = frame.timestamp.dt.hour.astype('Int64')
    frame['weekday'] = frame.timestamp.dt.dayofweek.astype('Int64')
    frame['is_weekend'] = frame.weekday.ge(5)
    # Exploratory threshold only; low speed alone does not establish breakdown/gridlock.
    frame['below_speed_threshold'] = frame.speed_mph.lt(congestion_speed).where(~frame.invalid_speed).astype('boolean')
    frame['source_file'] = source_name
    return frame


def main():
    parser = argparse.ArgumentParser(description='Prepare PeMS compressed station data, one day at a time.')
    parser.add_argument('--input', type=Path, default=Path('station_5min'))
    parser.add_argument('--output', type=Path, default=Path('traffic_prepared'))
    parser.add_argument('--start', default='2026-02-01')
    parser.add_argument('--end', default='2026-05-31')
    parser.add_argument('--chunksize', type=int, default=100000)
    parser.add_argument('--min-observed', type=float, default=80)
    parser.add_argument('--congestion-speed', type=float, default=45)
    args = parser.parse_args()
    if args.chunksize < 1 or not 0 <= args.min_observed <= 100 or not 0 < args.congestion_speed <= 120:
        parser.error('Check chunksize, min-observed, and congestion-speed ranges.')
    expected = pd.date_range(args.start, args.end)
    if len(expected) == 0:
        parser.error('End date must be on or after start date.')
    files = [(day.strftime('%Y-%m-%d'), args.input / f'd07_text_station_5min_{day:%Y_%m_%d}.txt.gz') for day in expected]
    missing = [date for date, path in files if not path.is_file()]
    available = [(date, path) for date, path in files if path.is_file()]
    if not available:
        parser.error(f'No matching archives found in {args.input.resolve()}')
    if args.output.exists():
        parser.error('Output already exists. Choose a new --output folder to prevent mixing runs.')
    args.output.mkdir(parents=True)
    manifest = {'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'missing_dates': missing, 'files': [], 'timestamp_basis': 'PeMS local wall time; timezone not assigned',
                'notes': ['All lane types and lane fields retained.', 'Eligibility selects ML with valid station measures and observation coverage.',
                          'Duplicates are flagged, retained, and excluded from eligibility.',
                          'No breakdown events, density estimates, or gridlock claims are produced.']}
    try:
        for number, (date, path) in enumerate(available, 1):
            print(f'[{number}/{len(available)}] {path.name}', flush=True)
            names, raw_rows, widths = inspect_format(path)
            audit = {'source_file': path.name, 'date': date, 'raw_rows': raw_rows, 'field_counts': widths, 'rows_written': 0}
            seen = set()
            staging = Path(tempfile.mkdtemp(prefix='.day-', dir=args.output))
            try:
                reader = pd.read_csv(path, header=None, names=names, dtype='string', compression='gzip', chunksize=args.chunksize)
                for part, frame in enumerate(reader):
                    frame = prepare_chunk(frame, path.name, date, args.min_observed, args.congestion_speed)
                    keys = list(zip(frame.timestamp, frame.station_id))
                    duplicated = []
                    for key in keys:
                        duplicated.append(key in seen)
                        seen.add(key)
                    frame['duplicate_station_timestamp'] = duplicated
                    frame['analysis_eligible'] &= ~frame.duplicate_station_timestamp
                    for col in frame.select_dtypes(include=['bool', 'boolean']).columns:
                        audit[col] = audit.get(col, 0) + int(frame[col].sum())
                    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), staging / f'part-{part:05d}.parquet', compression='snappy')
                    audit['rows_written'] += len(frame)
                staging.rename(args.output / f'date={date}')
            except Exception:
                shutil.rmtree(staging)
                raise
            manifest['files'].append(audit)
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            pd.DataFrame(manifest['files']).to_csv(args.output / 'quality_audit.csv', index=False)
    except Exception as error:
        manifest['error'] = str(error)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        raise
    print(f'Done: {len(available)} days; {len(missing)} missing dates. Output: {args.output.resolve()}')
    print('Inspect quality_audit.csv before selecting analysis_eligible rows.')


# Later analysis: inspect speed-flow and speed-occupancy relationships;
# identify sustained congestion and contiguous pre/post-breakdown windows;
# compare weekdays and rush hours. Station metadata is needed to place
# detectors along corridors. Do not bridge missing time intervals as events.
# Read a day for inspection: pd.read_parquet('traffic_prepared/date=2026-02-01')
if __name__ == '__main__':
    main()
