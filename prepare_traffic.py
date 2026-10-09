import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ===================== SWITCHES: EDIT HERE =====================
# All OFF: running this file reads no data and writes no outputs.
# Change only the sections you want to run to True; save; then run:
# python prepare_traffic.py
RUN_PREPARATION = False
RUN_FILTER_AUDIT = False
RUN_THRESHOLD_COMPARISON = False
RUN_STATION_COVERAGE_AUDIT = False
RUN_FILTERED_SELECTION = False
RUN_CONTINUITY_AUDIT = False
RUN_STATION_RANKING = True

# ===================== SETTINGS: EDIT HERE =====================
RAW_DATA_FOLDER = Path('station_5min')
PREPARED_DATA_FOLDER = Path('traffic_prepared')
FILTER_REPORT_FOLDER = Path('audit_reports/filter_review')
STATION_REPORT_FOLDER = Path('audit_reports/station_coverage')
CONTINUITY_REPORT_FOLDER = Path('audit_reports/continuity')
RANKING_REPORT_FOLDER = Path('audit_reports/station_ranking')
SELECTION_FOLDER = Path('traffic_essential_40')
START_DATE = '2026-02-01'
END_DATE = '2026-05-31'
SELECTED_LANE_TYPES = ['ML']
MIN_OBSERVED_PCT = 40

# Existing output folders are protected. To rerun a section, change its
# output folder above to a new name. Ignore large selection folders in Git.
# The original preparation's 80% eligibility flag stays unchanged.
# MIN_OBSERVED_PCT controls the revised selection, not original preparation.
# Threshold audits always compare 0, 5, 20, 40, 60, 80, 100% for ML.
# Switching an audit on does NOT switch dataset filtering on.
# FF can be audited by lane type, but a speed-required selection excludes
# FF records without speed. Study their suitability before changing rules.
# No scientific justification for a cutoff is implied by these switches.

# ===================== 1. ORIGINAL PREPARATION =====================


# Install once in your virtual environment: python -m pip install pandas pyarrow
# Enable RUN_PREPARATION above to prepare raw files.
# Input and output paths are configured at the top.
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


# Validate widths before parsing so lane fields cannot shift silently.
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


# Stream raw archives to daily parts; retain measurements and quality flags.
def run_preparation():
    parser = argparse.ArgumentParser(description='Prepare PeMS compressed station data, one day at a time.')
    parser.add_argument('--input', type=Path, default=RAW_DATA_FOLDER)
    parser.add_argument('--output', type=Path, default=PREPARED_DATA_FOLDER)
    parser.add_argument('--start', default=START_DATE)
    parser.add_argument('--end', default=END_DATE)
    parser.add_argument('--chunksize', type=int, default=100000)
    parser.add_argument('--min-observed', type=float, default=80)
    parser.add_argument('--congestion-speed', type=float, default=45)
    args = parser.parse_args([])
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

# ===================== 2. FILTER AND THRESHOLD AUDITS =====================
FLAGS = [
    'invalid_timestamp', 'date_mismatch', 'invalid_station_id',
    'invalid_speed', 'invalid_flow', 'invalid_occupancy',
    'invalid_observed_pct', 'low_observed', 'off_5min_grid',
    'duplicate_station_timestamp',
]
COLUMNS = ['lane_type', 'observed_pct', 'analysis_eligible'] + FLAGS
THRESHOLDS = [0, 5, 20, 40, 60, 80, 100]


# One shared read pass calculates sequential exclusions, overlapping flags,
# threshold sensitivity, and optional selection without overwriting source rows.
def run_filter_review(args):
    files = sorted(args.input.glob('date=*/part-*.parquet'))
    if not files:
        raise ValueError(f'No prepared Parquet files in {args.input}')
    if args.reports.exists():
        raise ValueError('Report directory exists; choose a new --reports path.')
    if args.write_selection and args.selection_output.exists():
        raise ValueError('Selection directory exists; choose a new --selection-output path.')
    args.reports.mkdir(parents=True)
    if args.write_selection:
        args.selection_output.mkdir(parents=True)
    labels = [
        'All saved rows', 'Selected lane types', 'Valid timestamp and date',
        'Valid station ID', 'Valid speed', 'No duplicate station/timestamp',
        f'Valid observation percentage and coverage >={args.min_observed:g}%',
    ]
    totals = dict.fromkeys(labels, 0)
    lane_summary = {}
    threshold_counts = dict.fromkeys(THRESHOLDS, 0)
    essential_ml = 0
    valid_coverage_ml = 0
    zero_coverage_ml = 0
    old_eligible = 0
    intersections = dict.fromkeys(['both', 'new_only', 'old_only', 'neither'], 0)
    overlap_counts = {}
    day_reports = {}
    last_day = None
    settings = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': str(args.input), 'lane_types': args.lane_types,
        'minimum_observed_pct': args.min_observed,
        'required': ['valid timestamp/date', 'valid station ID', 'valid speed',
                     'no duplicate station/timestamp', 'valid coverage >= cutoff'],
        'optional': ['valid flow', 'valid occupancy', 'five-minute grid'],
        'threshold_comparison_scope': 'ML with essential checks',
        'thresholds': THRESHOLDS, 'write_selection': args.write_selection,
        'file_count': len(files), 'status': 'running',
        'notes': ['Sequential exclusions depend on order.',
                  'Flag totals overlap; do not add them.',
                  '0% threshold includes zero-observed records.',
                  'Original analysis_eligible retains original rules.',
                  'Selected rows receive eligible_essential_selected=True.'],
    }
    settings_path = args.reports / 'run_settings.json'
    settings_path.write_text(json.dumps(settings, indent=2))
    try:
        for path in files:
            day = path.parent.name
            if day != last_day:
                print(f'Checking {day}', flush=True)
                last_day = day
            df = pd.read_parquet(path, columns=None if args.write_selection else COLUMNS)
            old = df.analysis_eligible.fillna(False)
            old_eligible += int(old.sum())
            for lane, group in df.groupby('lane_type', dropna=False):
                key = 'MISSING' if pd.isna(lane) else str(lane)
                counts = lane_summary.setdefault(key, dict.fromkeys(['rows', 'original_eligible', 'zero_observed', 'observed_ge40', 'observed_ge80'] + FLAGS, 0))
                counts['rows'] += len(group)
                counts['original_eligible'] += int(group.analysis_eligible.sum())
                counts['zero_observed'] += int(group.observed_pct.eq(0).sum())
                counts['observed_ge40'] += int(group.observed_pct.ge(40).sum())
                counts['observed_ge80'] += int(group.observed_pct.ge(80).sum())
                for flag in FLAGS:
                    counts[flag] += int(group[flag].sum())
            scope = df.lane_type.isin(args.lane_types)
            essential = (~df.invalid_timestamp & ~df.date_mismatch
                         & ~df.invalid_station_id & ~df.invalid_speed
                         & ~df.duplicate_station_timestamp).fillna(False)
            conditions = [scope, ~df.invalid_timestamp & ~df.date_mismatch,
                          ~df.invalid_station_id, ~df.invalid_speed,
                          ~df.duplicate_station_timestamp,
                          ~df.invalid_observed_pct & df.observed_pct.ge(args.min_observed)]
            keep = pd.Series(True, index=df.index)
            totals[labels[0]] += len(df)
            for label, condition in zip(labels[1:], conditions):
                keep &= condition.fillna(False)
                totals[label] += int(keep.sum())
            ml_essential = df.lane_type.eq('ML').fillna(False) & essential
            essential_ml += int(ml_essential.sum())
            coverage = df.loc[ml_essential & ~df.invalid_observed_pct, 'observed_pct']
            valid_coverage_ml += len(coverage)
            zero_coverage_ml += int(coverage.eq(0).sum())
            for cutoff in THRESHOLDS:
                threshold_counts[cutoff] += int(coverage.ge(cutoff).sum())
            intersections['both'] += int((old & keep).sum())
            intersections['new_only'] += int((~old & keep).sum())
            intersections['old_only'] += int((old & ~keep).sum())
            intersections['neither'] += int((~old & ~keep).sum())
            # Exact combinations of existing quality flags within selected scope.
            patterns = pd.Series(0, index=df.index, dtype='int64')
            for bit, flag in enumerate(FLAGS):
                patterns += df[flag].fillna(True).astype('int64') * (1 << bit)
            for pattern, count in patterns[scope].value_counts().items():
                overlap_counts[int(pattern)] = overlap_counts.get(int(pattern), 0) + int(count)
            daily = day_reports.setdefault(day, {'rows': 0, 'original_eligible': 0, 'new_eligible': 0})
            daily['rows'] += len(df)
            daily['original_eligible'] += int(old.sum())
            daily['new_eligible'] += int(keep.sum())
            if args.write_selection and keep.any():
                selected = df.loc[keep].copy()
                selected['eligible_essential_selected'] = True
                destination = args.selection_output / day
                destination.mkdir(exist_ok=True)
                selected.to_parquet(destination / path.name, index=False)
        audit = pd.DataFrame({'condition': labels, 'rows_remaining': [totals[x] for x in labels]})
        audit['removed_at_step'] = (audit.rows_remaining.shift(1) - audit.rows_remaining).fillna(0).astype('int64')
        if RUN_FILTER_AUDIT:
            audit.to_csv(args.reports / 'sequential_filter_audit.csv', index=False)
        pd.DataFrame.from_dict(lane_summary, orient='index').rename_axis('lane_type').to_csv(args.reports / 'lane_type_quality.csv')
        thresholds = pd.DataFrame({'minimum_observed_pct': THRESHOLDS, 'rows_retained': [threshold_counts[t] for t in THRESHOLDS]})
        thresholds['pct_of_essential_mainline'] = 100 * thresholds.rows_retained / essential_ml if essential_ml else float('nan')
        if RUN_THRESHOLD_COMPARISON:
            thresholds.to_csv(args.reports / 'observation_threshold_audit.csv', index=False)
        pd.DataFrame([{'flag_combination': ';'.join(flag for bit, flag in enumerate(FLAGS) if pattern & (1 << bit)) or 'none', 'rows': count}
                      for pattern, count in sorted(overlap_counts.items())]).to_csv(args.reports / 'overlapping_flags_in_scope.csv', index=False)
        pd.DataFrame.from_dict(day_reports, orient='index').rename_axis('date').to_csv(args.reports / 'daily_selection_counts.csv')
        pd.Series(intersections, name='rows').rename_axis('category').to_csv(args.reports / 'old_vs_new_selection.csv')
        settings.update(status='complete', total_rows=totals[labels[0]], original_eligible=old_eligible,
                        new_eligible=totals[labels[-1]], essential_mainline=essential_ml,
                        valid_coverage_mainline=valid_coverage_ml, zero_coverage_mainline=zero_coverage_ml)
        print('\nSequential audit:\n' + audit.to_string(index=False))
        print('\nMainline threshold comparison:\n' + thresholds.round(2).to_string(index=False))
        print(f'\nExactly 0% observed among essential mainline: {zero_coverage_ml:,}')
        print(f'Reports saved: {args.reports.resolve()}')
    except BaseException as error:
        settings.update(status='incomplete', error=str(error))
        raise
    finally:
        settings_path.write_text(json.dumps(settings, indent=2))


# ===================== 3. STATION COVERAGE AUDIT =====================
# Aggregate by station, route, date, and hour to distinguish persistent zero
# coverage from intermittent observations. No physical failure is inferred.
def run_station_coverage(args):
    files = sorted(args.input.glob('date=*/part-*.parquet'))
    if not files:
        raise SystemExit('No prepared files found.')
    if args.output.exists():
        raise SystemExit('Output exists. Use a new --output directory to preserve prior results.')
    args.output.mkdir(parents=True)
    settings = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'input': str(args.input), 'scope': 'All ML rows, before coverage filtering',
                'notes': ['No stations or measurements are removed.',
                          'Never observed means never positive observed_pct in the supplied period.',
                          'Daily rows do not certify complete coverage; missing intervals require a later continuity audit.']}
    metadata = args.output / 'run_settings.json'
    metadata.write_text(json.dumps(settings, indent=2))
    daily_parts = []
    hour_parts = []
    columns = ['station_id', 'freeway', 'direction', 'lane_type', 'timestamp',
               'observed_pct', 'invalid_observed_pct', 'invalid_station_id']
    last_day = None
    try:
        for path in files:
            day = path.parent.name.removeprefix('date=')
            if day != last_day:
                print(f'Checking {day}', flush=True)
                last_day = day
            df = pd.read_parquet(path, columns=columns)
            df = df.loc[df.lane_type.eq('ML') & ~df.invalid_station_id].copy()
            if df.empty:
                continue
            valid = ~df.invalid_observed_pct
            df['rows'] = 1
            df['valid_coverage_rows'] = valid.astype('int64')
            df['zero_rows'] = (valid & df.observed_pct.eq(0)).astype('int64')
            df['positive_rows'] = (valid & df.observed_pct.gt(0)).astype('int64')
            df['ge40_rows'] = (valid & df.observed_pct.ge(40)).astype('int64')
            df['ge80_rows'] = (valid & df.observed_pct.ge(80)).astype('int64')
            df['observed_sum'] = df.observed_pct.where(valid, 0)
            df['date'] = day
            measures = ['rows', 'valid_coverage_rows', 'zero_rows', 'positive_rows',
                        'ge40_rows', 'ge80_rows', 'observed_sum']
            keys = ['station_id', 'freeway', 'direction']
            daily_parts.append(df.groupby(keys + ['date'], dropna=False)[measures].sum())
            df['hour'] = df.timestamp.dt.hour
            hour_parts.append(df.groupby(keys + ['hour'], dropna=False)[measures].sum())
        if not daily_parts:
            raise ValueError('No mainline rows with valid station IDs found.')
        daily = pd.concat(daily_parts).groupby(level=[0, 1, 2, 3], dropna=False).sum().reset_index()
        hourly = pd.concat(hour_parts).groupby(level=[0, 1, 2, 3], dropna=False).sum().reset_index()
        keys = ['station_id', 'freeway', 'direction']
        summary = daily.groupby(keys, dropna=False)[measures].sum()
        days = daily.assign(days_present=1, days_any_observed=daily.positive_rows.gt(0).astype('int64'),
                            days_any_ge40=daily.ge40_rows.gt(0).astype('int64'),
                            days_any_ge80=daily.ge80_rows.gt(0).astype('int64'))
        summary = summary.join(days.groupby(keys, dropna=False)[['days_present', 'days_any_observed', 'days_any_ge40', 'days_any_ge80']].sum())
        for table in [daily, hourly, summary]:
            denom = table.valid_coverage_rows.replace(0, float('nan'))
            table['zero_pct'] = 100 * table.zero_rows / denom
            table['positive_pct'] = 100 * table.positive_rows / denom
            table['ge40_pct'] = 100 * table.ge40_rows / denom
            table['ge80_pct'] = 100 * table.ge80_rows / denom
            table['mean_observed_pct'] = table.observed_sum / denom
        summary['coverage_group'] = 'mixed zero and positive'
        summary.loc[summary.zero_rows.eq(0) & summary.positive_rows.gt(0), 'coverage_group'] = 'positive throughout valid records'
        summary.loc[summary.positive_rows.eq(0) & summary.valid_coverage_rows.gt(0), 'coverage_group'] = 'never positive observed'
        summary.loc[summary.valid_coverage_rows.eq(0), 'coverage_group'] = 'no valid coverage values'
        summary.reset_index().to_csv(args.output / 'station_coverage.csv', index=False)
        daily.to_csv(args.output / 'station_daily_coverage.csv', index=False)
        hourly.to_csv(args.output / 'station_hour_coverage.csv', index=False)
        groups = summary.groupby('coverage_group').agg(station_route_groups=('rows', 'size'), rows=('rows', 'sum'), zero_rows=('zero_rows', 'sum'), positive_rows=('positive_rows', 'sum'))
        groups.to_csv(args.output / 'coverage_group_summary.csv')
        dates = daily.groupby('date')[measures].sum()
        dates.to_csv(args.output / 'daily_network_coverage.csv')
        print('\nCoverage groups:\n' + groups.to_string())
        print(f'\nUnique station IDs: {summary.reset_index().station_id.nunique():,}')
        print(f'Mainline rows audited: {int(summary.rows.sum()):,}')
        print(f'Zero-coverage rows: {int(summary.zero_rows.sum()):,}')
        print(f'Reports: {args.output.resolve()}')
        settings.update(status='complete', mainline_rows=int(summary.rows.sum()), zero_rows=int(summary.zero_rows.sum()),
                        unique_stations=int(summary.reset_index().station_id.nunique()), source_days=len(dates))
    except BaseException as error:
        settings.update(status='incomplete', error=str(error))
        raise
    finally:
        metadata.write_text(json.dumps(settings, indent=2))




# ===================== 4. DAILY CONTINUITY AUDIT =====================
# Reads ALL valid-identity/time ML rows before coverage filtering.
# Positive means observed_pct > 0, not a scientifically validated cutoff.
# Runs stop at missing intervals, duplicates, state changes, and midnight.
# Run length is number of consecutive five-minute records times five;
# it describes measurement windows, not exact traffic-event duration.
# Daily resets and local clock changes limit interpretation of these runs.
def run_continuity_audit():
    folders = sorted(PREPARED_DATA_FOLDER.glob('date=*'))
    if not folders:
        raise ValueError('No prepared daily folders found.')
    if CONTINUITY_REPORT_FOLDER.exists():
        raise ValueError('Choose a new CONTINUITY_REPORT_FOLDER before rerunning.')
    CONTINUITY_REPORT_FOLDER.mkdir(parents=True)
    settings = {'status': 'running', 'input': str(PREPARED_DATA_FOLDER),
                'scope': 'ML before coverage filtering; valid station ID and timestamp',
                'positive_definition': 'valid observed_pct > 0',
                'run_rules': 'Exact five-minute spacing, same coverage state, reset each day',
                'created_utc': datetime.now(timezone.utc).isoformat()}
    metadata = CONTINUITY_REPORT_FOLDER / 'run_settings.json'
    metadata.write_text(json.dumps(settings, indent=2))
    summaries = []
    columns = ['station_id', 'timestamp', 'lane_type', 'observed_pct',
               'invalid_observed_pct', 'invalid_timestamp', 'invalid_station_id']
    try:
        for folder in folders:
            files = sorted(folder.glob('part-*.parquet'))
            if not files:
                continue
            print(f'Checking continuity: {folder.name}', flush=True)
            chunks = []
            for path in files:
                chunk = pd.read_parquet(path, columns=columns)
                chunk = chunk.loc[chunk.lane_type.eq('ML') & ~chunk.invalid_timestamp & ~chunk.invalid_station_id]
                chunks.append(chunk)
            day = pd.concat(chunks, ignore_index=True)
            results = []
            for station, group in day.groupby('station_id'):
                group = group.sort_values('timestamp').copy()
                group['coverage_state'] = 'invalid'
                valid = ~group.invalid_observed_pct
                group.loc[valid & group.observed_pct.eq(0), 'coverage_state'] = 'zero'
                group.loc[valid & group.observed_pct.gt(0), 'coverage_state'] = 'positive'
                spacing = group.timestamp.diff()
                same_state = group.coverage_state.eq(group.coverage_state.shift())
                group['run_id'] = (~spacing.eq(pd.Timedelta(minutes=5)) | ~same_state).cumsum()
                runs = group.groupby('run_id').agg(coverage_state=('coverage_state', 'first'), intervals=('timestamp', 'size'))
                positive = runs.loc[runs.coverage_state.eq('positive'), 'intervals']
                zero = runs.loc[runs.coverage_state.eq('zero'), 'intervals']
                results.append({
                    'date': folder.name.removeprefix('date='), 'station_id': int(station),
                    'rows': len(group), 'positive_rows': int(group.coverage_state.eq('positive').sum()),
                    'zero_rows': int(group.coverage_state.eq('zero').sum()),
                    'invalid_coverage_rows': int(group.coverage_state.eq('invalid').sum()),
                    'longest_positive_run_minutes': int(positive.max()) * 5 if not positive.empty else 0,
                    'longest_zero_run_minutes': int(zero.max()) * 5 if not zero.empty else 0,
                    'gaps_over_5_minutes': int(spacing.gt(pd.Timedelta(minutes=5)).sum()),
                    'duplicate_timestamps': int(group.timestamp.duplicated().sum()),
                })
            if results:
                report = pd.DataFrame(results)
                output = CONTINUITY_REPORT_FOLDER / 'station_daily_continuity.csv'
                report.to_csv(output, mode='a', header=not output.exists(), index=False)
                summaries.append({'date': folder.name.removeprefix('date='), 'station_days': len(report),
                                  **{f'station_days_positive_run_ge{minutes}min': int(report.longest_positive_run_minutes.ge(minutes).sum())
                                     for minutes in [30, 60, 120]}})
        if summaries:
            summary = pd.DataFrame(summaries)
            summary.to_csv(CONTINUITY_REPORT_FOLDER / 'daily_continuity_summary.csv', index=False)
            print('\nContinuity totals:\n' + summary.drop(columns='date').sum().to_string())
        settings['status'] = 'complete'
        print(f'Saved: {CONTINUITY_REPORT_FOLDER.resolve()}')
    except BaseException as error:
        settings.update(status='incomplete', error=str(error))
        raise
    finally:
        metadata.write_text(json.dumps(settings, indent=2))


# ===================== 5. RANK STATIONS FROM SAVED REPORTS =====================
def run_station_ranking():
    """Rank continuity candidates; no raw observations are read or filtered.

    Primary ranking: number of days with a positive-coverage run >=120 min.
    Tie breakers: positive-row percentage, then station ID for reproducibility.
    Positive coverage means >0%, not full observation or validated reliability.
    A long run could occur overnight: this does not establish rush-hour coverage.
    Existing continuity runs reset daily and do not bridge missing intervals.
    """
    continuity_path = CONTINUITY_REPORT_FOLDER / 'station_daily_continuity.csv'
    coverage_path = STATION_REPORT_FOLDER / 'station_coverage.csv'
    for source in [continuity_path, coverage_path]:
        if not source.is_file():
            raise ValueError(f'Required completed audit report missing: {source}')
    if RANKING_REPORT_FOLDER.exists():
        raise ValueError('Choose a new RANKING_REPORT_FOLDER to preserve prior results.')

    # Validate source run status before combining reports. Incomplete CSVs can
    # exist after an interrupted audit and must not masquerade as full results.
    for folder in [CONTINUITY_REPORT_FOLDER, STATION_REPORT_FOLDER]:
        status = json.loads((folder / 'run_settings.json').read_text())
        if status.get('status') != 'complete':
            raise ValueError(f'Source audit is not complete: {folder}')

    daily = pd.read_csv(continuity_path)
    coverage = pd.read_csv(coverage_path)
    if daily.empty or coverage.empty:
        raise ValueError('Source reports must contain observations.')
    if daily.duplicated(['station_id', 'date']).any():
        raise ValueError('Duplicate station-days in continuity report; investigate before ranking.')

    # Each threshold records whether the station had at least ONE run that day;
    # days at different thresholds overlap and must never be added together.
    for minutes in [30, 60, 120]:
        daily[f'days_run_ge{minutes}min'] = daily.longest_positive_run_minutes.ge(minutes).astype('int64')
    daily['days_any_positive'] = daily.positive_rows.gt(0).astype('int64')
    ranking = daily.groupby('station_id').agg(
        days_present=('date', 'nunique'),
        days_any_positive=('days_any_positive', 'sum'),
        days_run_ge30min=('days_run_ge30min', 'sum'),
        days_run_ge60min=('days_run_ge60min', 'sum'),
        days_run_ge120min=('days_run_ge120min', 'sum'),
        rows=('rows', 'sum'), positive_rows=('positive_rows', 'sum'),
        zero_rows=('zero_rows', 'sum'),
        invalid_coverage_rows=('invalid_coverage_rows', 'sum'),
        longest_positive_run_minutes=('longest_positive_run_minutes', 'max'),
        median_daily_longest_positive_run_minutes=('longest_positive_run_minutes', 'median'),
        gaps_over_5_minutes=('gaps_over_5_minutes', 'sum'),
        duplicate_timestamps=('duplicate_timestamps', 'sum'),
    ).reset_index()

    # Keep both denominators: days actually represented and all dates in the
    # source report. Missing station-days must not inflate apparent coverage.
    study_days = daily.date.nunique()
    ranking['positive_rows_pct'] = 100 * ranking.positive_rows / ranking.rows
    ranking['days_ge120_pct_of_present'] = 100 * ranking.days_run_ge120min / ranking.days_present
    ranking['days_ge120_pct_of_study'] = 100 * ranking.days_run_ge120min / study_days

    # Preserve multiple route/direction assignments rather than silently choosing
    # the first. Multiple assignments are flagged for later metadata investigation.
    def join_assignments(values):
        return '; '.join(sorted(set(values.dropna().astype(str))))

    identity = coverage.groupby('station_id').agg(
        freeway=('freeway', join_assignments),
        direction=('direction', join_assignments),
        route_direction_groups=('station_id', 'size'),
        coverage_report_rows=('rows', 'sum'),
        coverage_report_positive_rows=('positive_rows', 'sum'),
    ).reset_index()
    ranking = ranking.merge(identity, on='station_id', how='left', validate='one_to_one', indicator=True)
    if not ranking['_merge'].eq('both').all():
        raise ValueError('Station IDs differ between reports; investigate their provenance.')
    ranking = ranking.drop(columns='_merge')
    # Reconciliation catches reports from different periods or inconsistent inputs.
    if not ranking.rows.eq(ranking.coverage_report_rows).all() or not ranking.positive_rows.eq(ranking.coverage_report_positive_rows).all():
        raise ValueError('Coverage and continuity counts disagree; do not rank mismatched reports.')
    ranking['multiple_route_assignments'] = ranking.route_direction_groups.gt(1)
    ranking = ranking.sort_values(['days_run_ge120min', 'positive_rows_pct', 'station_id'], ascending=[False, False, True]).reset_index(drop=True)
    ranking.insert(0, 'rank', range(1, len(ranking) + 1))

    # Save ALL stations, including zero-coverage stations, to document the full
    # comparison. This is a candidate ranking, not a final station selection.
    RANKING_REPORT_FOLDER.mkdir(parents=True)
    ranking.to_csv(RANKING_REPORT_FOLDER / 'station_ranking.csv', index=False)
    settings = {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': [str(continuity_path), str(coverage_path)],
        'study_days': int(study_days), 'stations_ranked': len(ranking),
        'ranking_order': ['days_run_ge120min descending', 'positive_rows_pct descending', 'station_id ascending'],
        'limitations': ['Positive is >0%, not validated reliability.',
                        'No rush-hour requirement.', 'Daily run resets.',
                        'No final station exclusion or traffic-event detection.'],
    }
    (RANKING_REPORT_FOLDER / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    display = ['rank', 'station_id', 'freeway', 'direction', 'days_present', 'days_run_ge120min', 'positive_rows_pct']
    print('\nTop 20 continuity candidates:\n' + ranking[display].head(20).round(2).to_string(index=False))
    print(f'\nStations ranked: {len(ranking):,}; study dates: {study_days}')
    print(f'Stations with at least one two-hour positive run: {int(ranking.days_run_ge120min.gt(0).sum()):,}')
    print(f'Saved: {RANKING_REPORT_FOLDER.resolve()}')

# ===================== 6. RUN ENABLED SECTIONS =====================
def main():
    switches = {
        'Preparation': RUN_PREPARATION,
        'Filter audit': RUN_FILTER_AUDIT,
        'Threshold comparison': RUN_THRESHOLD_COMPARISON,
        'Station coverage audit': RUN_STATION_COVERAGE_AUDIT,
        'Filtered selection': RUN_FILTERED_SELECTION,
        'Continuity audit': RUN_CONTINUITY_AUDIT,
        'Station ranking': RUN_STATION_RANKING,
    }
    for label, enabled in switches.items():
        print(f'{label}: {"ON" if enabled else "OFF"}')
    if not any(switches.values()):
        print('Nothing enabled. Change a RUN_ switch at the top to True and save.')
        return
    if not 0 <= MIN_OBSERVED_PCT <= 100:
        raise ValueError('MIN_OBSERVED_PCT must be between 0 and 100.')
    if RUN_PREPARATION:
        run_preparation()
    if RUN_FILTER_AUDIT or RUN_THRESHOLD_COMPARISON or RUN_FILTERED_SELECTION:
        run_filter_review(argparse.Namespace(
            input=PREPARED_DATA_FOLDER, reports=FILTER_REPORT_FOLDER,
            selection_output=SELECTION_FOLDER,
            write_selection=RUN_FILTERED_SELECTION,
            min_observed=MIN_OBSERVED_PCT, lane_types=SELECTED_LANE_TYPES,
        ))
    if RUN_STATION_COVERAGE_AUDIT:
        run_station_coverage(argparse.Namespace(
            input=PREPARED_DATA_FOLDER, output=STATION_REPORT_FOLDER,
        ))

    if RUN_CONTINUITY_AUDIT:
        run_continuity_audit()

    if RUN_STATION_RANKING:
        run_station_ranking()


if __name__ == '__main__':
    main()
