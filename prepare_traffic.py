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
RUN_STATION_RANKING = False
RUN_TOP_STATION_REVIEW = False
RUN_RANKING_NETWORK_CHECKS = False
RUN_BATCH_1 = False

# ===================== SETTINGS: EDIT HERE =====================
RAW_DATA_FOLDER = Path('station_5min')
PREPARED_DATA_FOLDER = Path('traffic_prepared')
FILTER_REPORT_FOLDER = Path('audit_reports/filter_review')
STATION_REPORT_FOLDER = Path('audit_reports/station_coverage')
CONTINUITY_REPORT_FOLDER = Path('audit_reports/continuity')
RANKING_REPORT_FOLDER = Path('audit_reports/station_ranking')
TOP_REVIEW_REPORT_FOLDER = Path('audit_reports/top_station_review')
TOP_STATIONS_TO_REVIEW = 20
REVIEW_CHECKS_REPORT_FOLDER = Path('audit_reports/ranking_network_checks')
DATES_TO_CHECK = ['2026-02-28', '2026-05-27']
SELECTION_FOLDER = Path('traffic_essential_40')
START_DATE = '2026-02-01'
END_DATE = '2026-05-31'
SELECTED_LANE_TYPES = ['ML']
MIN_OBSERVED_PCT = 40
# Separate folder preserves the incomplete first attempt for the audit trail.
BATCH_1_REPORT_FOLDER = Path('audit_reports/batch_1_observation_quality_v2')
BATCH_1_THRESHOLDS = [0, 5, 20, 40, 60, 80, 100]
# These are diagnostic flags, NOT scientific exclusions or final selection rules.
BATCH_1_SHARED_ZERO_PCT = 95
BATCH_1_RUN_MINUTES = 120

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
                # Arrow Boolean arrays have no Boolean cumulative-sum kernel.
                group['run_id'] = (~spacing.eq(pd.Timedelta(minutes=5)) | ~same_state).fillna(True).astype('int64').cumsum()
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


# ===================== 6. TOP-STATION COVERAGE REVIEW =====================
def run_top_station_review():
    """Review saved coverage reports; do not scan or filter traffic measurements.

    Top N is a review convenience, not a final geographic station selection.
    Percentages use all listed records, including invalid coverage values.
    No-positive days are not automatically hardware outages. Shared calendar
    dates identify patterns worth investigating, not a proven common cause.
    This checks daily totals; it cannot locate within-day or rush-hour gaps.
    """
    if TOP_STATIONS_TO_REVIEW < 1:
        raise ValueError('TOP_STATIONS_TO_REVIEW must be positive.')
    if TOP_REVIEW_REPORT_FOLDER.exists():
        raise ValueError('Choose a new TOP_REVIEW_REPORT_FOLDER before rerunning.')
    # Require completed sources to avoid combining interrupted report files.
    for folder in [RANKING_REPORT_FOLDER, STATION_REPORT_FOLDER]:
        settings = json.loads((folder / 'run_settings.json').read_text())
        if settings.get('status') != 'complete':
            raise ValueError(f'Source report incomplete: {folder}')
    ranking = pd.read_csv(RANKING_REPORT_FOLDER / 'station_ranking.csv')
    daily = pd.read_csv(STATION_REPORT_FOLDER / 'station_daily_coverage.csv')
    if ranking.station_id.duplicated().any():
        raise ValueError('Ranking contains duplicate station IDs.')
    # Use the existing rank order, not an unrecorded new selection rule.
    candidates = ranking.sort_values('rank').head(TOP_STATIONS_TO_REVIEW).copy()
    if candidates.empty:
        raise ValueError('No ranked stations found.')
    measures = ['rows', 'valid_coverage_rows', 'zero_rows', 'positive_rows',
                'ge40_rows', 'ge80_rows', 'observed_sum']
    # Sum route assignments within station-day; do not double-count days when
    # a station has multiple documented route/direction assignments.
    daily = daily.loc[daily.station_id.isin(candidates.station_id)]
    daily = daily.groupby(['station_id', 'date'])[measures].sum().reset_index()
    totals = daily.groupby('station_id')[measures].sum().reset_index()
    review = candidates[['rank', 'station_id', 'freeway', 'direction', 'days_present',
                         'days_run_ge120min', 'rows', 'positive_rows']].merge(
        totals, on='station_id', suffixes=('_ranking', ''), validate='one_to_one')
    # Ensure these sources describe the same underlying row counts.
    if len(review) != len(candidates) or not review.rows.eq(review.rows_ranking).all() or not review.positive_rows.eq(review.positive_rows_ranking).all():
        raise ValueError('Ranking and daily coverage counts disagree.')
    for name in ['positive', 'zero', 'ge40', 'ge80']:
        review[f'{name}_pct_of_all_rows'] = 100 * review[f'{name}_rows'] / review.rows
    review['invalid_coverage_rows'] = review.rows - review.valid_coverage_rows
    review['mean_observed_pct_valid_rows'] = review.observed_sum / review.valid_coverage_rows.replace(0, float('nan'))
    # Keep no-positive days separate from missing days and invalid values.
    no_positive = daily.loc[daily.positive_rows.eq(0)].copy()
    no_positive['all_records_valid_zero'] = no_positive.zero_rows.eq(no_positive.rows)
    date_sets = no_positive.groupby('station_id').date.apply(lambda values: set(values)).to_dict()
    review['days_no_positive'] = review.station_id.map(lambda station: len(date_sets.get(station, set())))
    review['no_positive_dates'] = review.station_id.map(lambda station: '; '.join(sorted(date_sets.get(station, set()))))
    dates_present = daily.groupby('station_id').date.nunique()
    if not review.station_id.map(dates_present).eq(review.days_present).all():
        raise ValueError('Source reports disagree on days represented.')
    # Compare station pairs on dates represented for BOTH stations. A missing
    # station-day is not silently treated as a zero-observed day.
    represented = daily.groupby('station_id').date.apply(lambda values: set(values)).to_dict()
    pairs = []
    ids = candidates.station_id.tolist()
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            common = represented[first] & represented[second]
            a = date_sets.get(first, set()) & common
            b = date_sets.get(second, set()) & common
            pairs.append({'station_a': first, 'station_b': second,
                          'common_dates_present': len(common),
                          'shared_no_positive_days': len(a & b),
                          'either_no_positive_days': len(a | b),
                          'same_nonempty_no_positive_dates': bool(a) and a == b,
                          'shared_dates': '; '.join(sorted(a & b))})
    calendar = no_positive.groupby('date').agg(stations_no_positive=('station_id', 'nunique'),
                                             stations_all_records_zero=('all_records_valid_zero', 'sum'))
    # Save the report and methodology; no traffic observations are removed.
    TOP_REVIEW_REPORT_FOLDER.mkdir(parents=True)
    review.to_csv(TOP_REVIEW_REPORT_FOLDER / 'top_station_coverage.csv', index=False)
    no_positive.to_csv(TOP_REVIEW_REPORT_FOLDER / 'no_positive_station_days.csv', index=False)
    calendar.to_csv(TOP_REVIEW_REPORT_FOLDER / 'shared_no_positive_dates.csv')
    pd.DataFrame(pairs).to_csv(TOP_REVIEW_REPORT_FOLDER / 'station_pair_date_comparison.csv', index=False)
    settings = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'stations_reviewed': len(review), 'top_n_setting': TOP_STATIONS_TO_REVIEW,
                'inputs': [str(RANKING_REPORT_FOLDER), str(STATION_REPORT_FOLDER)],
                'percentage_denominator': 'All represented rows',
                'no_positive_definition': 'No valid observed_pct >0 that day',
                'limitations': ['Daily totals do not locate within-day gaps.',
                               'Shared dates do not establish outage causes.',
                               'No final analysis cutoff or geographic selection.']}
    (TOP_REVIEW_REPORT_FOLDER / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    display = ['rank', 'station_id', 'freeway', 'direction', 'positive_pct_of_all_rows',
               'ge40_pct_of_all_rows', 'ge80_pct_of_all_rows', 'days_no_positive']
    print('\nTop-station coverage review:\n' + review[display].round(2).to_string(index=False))
    print('\nDates with no positive coverage among candidates:\n' + (calendar.to_string() if not calendar.empty else 'None'))
    print(f'\nSaved: {TOP_REVIEW_REPORT_FOLDER.resolve()}')


# ===================== 7. CHECK RANKING TIES AND NETWORK COVERAGE =====================
def run_ranking_network_checks():
    """Test review concerns using saved reports, without choosing stations.

    Ranking ties are computed using the saved full-precision numerical values.
    Rounded console output is not used to identify ties. Equal scores do not
    establish equal reliability. No-positive station-days do not establish
    hardware failure; missing station-days are reported separately.
    """
    if REVIEW_CHECKS_REPORT_FOLDER.exists():
        raise ValueError('Choose a new REVIEW_CHECKS_REPORT_FOLDER before rerunning.')
    # An interrupted audit can leave CSVs behind: require completed sources.
    for folder in [RANKING_REPORT_FOLDER, STATION_REPORT_FOLDER]:
        status = json.loads((folder / 'run_settings.json').read_text())
        if status.get('status') != 'complete':
            raise ValueError(f'Source audit incomplete: {folder}')
    ranking = pd.read_csv(RANKING_REPORT_FOLDER / 'station_ranking.csv')
    daily = pd.read_csv(STATION_REPORT_FOLDER / 'station_daily_coverage.csv')
    if ranking.empty or ranking.station_id.duplicated().any():
        raise ValueError('Ranking must have one nonempty row per station.')

    # Reconstruct the actual sort fields, including a deterministic station-ID
    # tie breaker. Group before that tie breaker to reveal equivalent scores.
    ranking = ranking.sort_values(['days_run_ge120min', 'positive_rows_pct', 'station_id'], ascending=[False, False, True])
    ties = ranking.groupby(['days_run_ge120min', 'positive_rows_pct'], dropna=False).agg(
        stations=('station_id', 'size'),
        station_ids=('station_id', lambda values: '; '.join(str(int(x)) for x in sorted(values))),
    ).reset_index().sort_values(['days_run_ge120min', 'positive_rows_pct'], ascending=[False, False])
    best = ranking.iloc[0]
    best_primary_count = int(ranking.days_run_ge120min.eq(best.days_run_ge120min).sum())
    best_exact_count = int((ranking.days_run_ge120min.eq(best.days_run_ge120min) & ranking.positive_rows_pct.eq(best.positive_rows_pct)).sum())
    # Tie at the displayed top-N boundary determines whether an arbitrary
    # station-ID ordering cuts through a larger group with equivalent scores.
    boundary = ranking.iloc[min(TOP_STATIONS_TO_REVIEW, len(ranking)) - 1]
    boundary_group = ranking.loc[ranking.days_run_ge120min.eq(boundary.days_run_ge120min) & ranking.positive_rows_pct.eq(boundary.positive_rows_pct)]
    boundary_included = int(ranking.head(TOP_STATIONS_TO_REVIEW).station_id.isin(boundary_group.station_id).sum())

    # Aggregate multiple route assignments within each station-day. Keep invalid
    # coverage distinguishable from valid zero rather than calling both outages.
    measures = ['rows', 'valid_coverage_rows', 'zero_rows', 'positive_rows', 'ge40_rows', 'ge80_rows']
    daily = daily.groupby(['station_id', 'date'])[measures].sum().reset_index()
    station_ids = set(daily.station_id)
    if station_ids != set(ranking.station_id):
        raise ValueError('Station identities differ between ranking and coverage reports.')
    totals = daily.groupby('station_id')[['rows', 'positive_rows']].sum()
    expected = ranking.set_index('station_id')[['rows', 'positive_rows']].sort_index()
    if not totals.sort_index().eq(expected).all().all():
        raise ValueError('Source report row counts disagree.')
    daily['no_positive'] = daily.positive_rows.eq(0)
    daily['all_valid_zero'] = daily.zero_rows.eq(daily.rows)
    # Full calendar summary lets the two requested dates be compared with other
    # dates; percentages are descriptive and imply no failure cause.
    network = daily.groupby('date').agg(
        stations_present=('station_id', 'nunique'),
        stations_no_positive=('no_positive', 'sum'),
        stations_all_valid_zero=('all_valid_zero', 'sum'),
        rows=('rows', 'sum'), zero_rows=('zero_rows', 'sum'),
        positive_rows=('positive_rows', 'sum'), ge40_rows=('ge40_rows', 'sum'),
        ge80_rows=('ge80_rows', 'sum'), valid_coverage_rows=('valid_coverage_rows', 'sum'),
    ).reset_index()
    network['stations_absent_from_date'] = len(station_ids) - network.stations_present
    network['zero_pct_all_rows'] = 100 * network.zero_rows / network.rows
    network['positive_pct_all_rows'] = 100 * network.positive_rows / network.rows
    network['ge80_pct_all_rows'] = 100 * network.ge80_rows / network.rows
    network['invalid_coverage_rows'] = network.rows - network.valid_coverage_rows
    # Left join requested dates so an absent date is shown as absent, not zero.
    selected_dates = pd.DataFrame({'date': DATES_TO_CHECK}).merge(network, on='date', how='left', validate='one_to_one')
    selected_dates['date_available'] = selected_dates.stations_present.notna()
    REVIEW_CHECKS_REPORT_FOLDER.mkdir(parents=True)
    ties.to_csv(REVIEW_CHECKS_REPORT_FOLDER / 'exact_ranking_score_groups.csv', index=False)
    network.to_csv(REVIEW_CHECKS_REPORT_FOLDER / 'network_daily_observation_check.csv', index=False)
    selected_dates.to_csv(REVIEW_CHECKS_REPORT_FOLDER / 'requested_dates_observation_check.csv', index=False)
    settings = {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': [str(RANKING_REPORT_FOLDER), str(STATION_REPORT_FOLDER)],
        'best_primary_days': int(best.days_run_ge120min),
        'stations_with_best_primary_days': best_primary_count,
        'stations_with_best_exact_score': best_exact_count,
        'top_n': TOP_STATIONS_TO_REVIEW,
        'boundary_tie_group_size': len(boundary_group),
        'boundary_tie_stations_included': boundary_included,
        'dates_checked': DATES_TO_CHECK,
        'limitations': ['Observed percentages are not independently verified by this check.',
                       'No-positive days do not identify outage causes.',
                       'No cutoff, station selection, or date exclusion is adopted.'],
    }
    (REVIEW_CHECKS_REPORT_FOLDER / 'ranking_network_check_settings.json').write_text(json.dumps(settings, indent=2))
    print(f'\nStations with the best two-hour-day count ({int(best.days_run_ge120min)}): {best_primary_count}')
    print(f'Stations tied on BOTH actual ranking score fields at the top: {best_exact_count}')
    print(f'Tie group at top-{TOP_STATIONS_TO_REVIEW} boundary: {len(boundary_group)} stations; {boundary_included} included')
    display = ['date', 'date_available', 'stations_present', 'stations_absent_from_date', 'stations_no_positive', 'stations_all_valid_zero', 'zero_pct_all_rows', 'positive_pct_all_rows', 'ge80_pct_all_rows']
    print('\nRequested-date network check:\n' + selected_dates[display].round(3).to_string(index=False))
    print(f'\nSaved: {REVIEW_CHECKS_REPORT_FOLDER.resolve()}')

# ===================== BATCH 1. COMBINED OBSERVATION QUALITY =====================
# Questions 1-6 are calculated together in one pass over prepared daily parts.
# Legacy audits remain reproducible above; do not enable them with this batch
# unless you deliberately want their separate outputs and additional scans.


def batch_1_day_metrics(frame, day):
    """Return daily station metrics, exact coverage counts, and hourly summaries.

    Retention uses ALL saved mainline records as its denominator. Continuity
    requires a valid timestamp/date/station/speed, exact five-minute spacing,
    and no duplicate timestamp copies. Flow and occupancy are NOT required:
    this is a preliminary speed/coverage audit, not the final traffic dataset.
    """
    frame = frame.copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    # Missing/invalid identifiers cannot be attributed to a physical station.
    # Preserve their count in the manifest instead of silently hiding it.
    invalid_ids = frame.invalid_station_id.fillna(True)
    unattributed = int(invalid_ids.sum())
    frame = frame.loc[~invalid_ids].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), unattributed
    frame['station_id'] = frame.station_id.astype('int64')
    coverage_valid = (~frame.invalid_observed_pct.fillna(True)
                      & frame.observed_pct.between(0, 100))
    frame['zero_rows'] = coverage_valid & frame.observed_pct.eq(0)
    frame['positive_rows'] = coverage_valid & frame.observed_pct.gt(0)
    frame['invalid_coverage_rows'] = ~coverage_valid
    # Exclude ALL copies of ambiguous station/timestamp duplicates from runs.
    # Retention counts still describe saved records; neither source nor old
    # analysis_eligible flags are rewritten by this audit.
    duplicate_copies = frame.duplicated(['station_id', 'timestamp'], keep=False)
    required = ['invalid_timestamp', 'date_mismatch', 'invalid_station_id',
                'invalid_speed', 'off_5min_grid', 'duplicate_station_timestamp']
    frame['essential_rows'] = (~frame[required].fillna(True).any(axis=1)
                               & ~duplicate_copies
                               & frame.timestamp.notna()
                               & frame.timestamp.dt.strftime('%Y-%m-%d').eq(day))
    frame['duplicate_copies'] = duplicate_copies
    frame['invalid_time_rows'] = (frame.timestamp.isna()
                                  | frame.invalid_timestamp.fillna(True)
                                  | frame.date_mismatch.fillna(True))
    for threshold in BATCH_1_THRESHOLDS:
        frame[f'ge{threshold}_rows'] = coverage_valid & frame.observed_pct.ge(threshold)
    summed = ['zero_rows', 'positive_rows', 'invalid_coverage_rows', 'essential_rows',
              'duplicate_copies', 'invalid_time_rows'] + [f'ge{t}_rows' for t in BATCH_1_THRESHOLDS]
    groups = frame.groupby('station_id', sort=True)
    daily = groups[summed].sum()
    daily['rows'] = groups.size()
    # Route changes are reported, not interpreted as geographic adjacency.
    frame['route_label'] = (frame.freeway.astype('string').fillna('UNKNOWN') + '/'
                            + frame.direction.astype('string').fillna('UNKNOWN'))
    daily['routes_in_day'] = frame.groupby('station_id').route_label.nunique()
    routes = frame.groupby('station_id').route_label.agg(lambda s: '|'.join(sorted(set(s))))
    daily['route_labels'] = routes

    # Sort once per day, then use vectorized runs for all thresholds. Parts may
    # split station series; concatenating the daily parts first avoids breaks
    # introduced merely by Parquet chunk boundaries.
    ordered = frame.sort_values(['station_id', 'timestamp'], kind='stable')
    same_station = ordered.station_id.eq(ordered.station_id.shift())
    consecutive = (same_station & ordered.timestamp.diff().eq(pd.Timedelta(minutes=5))
                   & ordered.route_label.eq(ordered.route_label.shift()))
    for label, threshold in [('positive', None)] + [(f'ge{t}', t) for t in BATCH_1_THRESHOLDS]:
        meets = (ordered.positive_rows if threshold is None else ordered[f'ge{threshold}_rows'])
        valid = ordered.essential_rows & meets
        # Invalid or below-threshold intervals break a run. Missing timestamps
        # also break it; never bridge gaps or connect separate stations.
        starts = valid & ~(consecutive & valid.shift(fill_value=False))
        # Parquet may restore Boolean flags as bool[pyarrow]. Count integer
        # run starts explicitly; Arrow does not implement cumsum on Booleans.
        # The conversion also gives identical run IDs with NumPy-backed flags.
        run_id = starts.fillna(False).astype('int64').cumsum()
        lengths = ordered.loc[valid, ['station_id']].copy()
        lengths['run_id'] = run_id.loc[valid]
        run_lengths = lengths.groupby(['station_id', 'run_id']).size()
        longest = run_lengths.groupby(level=0).max() if not run_lengths.empty else pd.Series(dtype='int64')
        daily[f'{label}_speed_valid_rows'] = valid.groupby(ordered.station_id).sum()
        # N five-minute interval records represent N*5 minutes of interval
        # exposure, not (N-1)*5 minutes between timestamp endpoints.
        daily[f'{label}_longest_run_minutes'] = longest.reindex(daily.index, fill_value=0) * 5

    daily['date'] = day
    # Clock-aware expected day length: March 8 has 276, not 288, intervals.
    local_day = pd.Timestamp(day, tz='America/Los_Angeles')
    next_day = pd.Timestamp(pd.Timestamp(day) + pd.Timedelta(days=1), tz='America/Los_Angeles')
    expected = int((next_day - local_day) / pd.Timedelta(minutes=5))
    daily['expected_intervals_if_active_all_day'] = expected
    valid_times = frame.loc[frame.essential_rows, ['station_id', 'timestamp']]
    unique_times = valid_times.groupby('station_id').timestamp.nunique()
    daily['unique_speed_valid_timestamps'] = unique_times.reindex(daily.index, fill_value=0)
    # Wall-clock gaps across DST still break runs. They must not be labeled a
    # detector outage; the report explicitly marks the DST transition date.
    daily['dst_transition_day'] = expected != 288

    distribution = (frame.loc[coverage_valid].groupby(['station_id', 'observed_pct'])
                    .size().rename('rows').reset_index())
    hour_frame = frame.loc[~frame.invalid_time_rows].copy()
    hour_frame['hour'] = hour_frame.timestamp.dt.hour
    # Network/hour percentages describe records, not traffic volumes. A
    # station with positive records elsewhere in the hour can still have gaps.
    hourly = hour_frame.groupby('hour')[['zero_rows', 'positive_rows', 'invalid_coverage_rows']].sum()
    hourly['rows'] = hour_frame.groupby('hour').size()
    hourly['stations_present'] = hour_frame.groupby('hour').station_id.nunique()
    zero_station_hours = (hour_frame.groupby(['hour', 'station_id']).zero_rows.all()
                          .groupby(level=0).sum())
    hourly['stations_all_zero'] = zero_station_hours
    hourly['date'] = day
    return daily.reset_index(), distribution, hourly.reset_index(), unattributed


def run_batch_1():
    """Produce reusable quality tables; do not filter or select source records."""
    output = BATCH_1_REPORT_FOLDER
    if output.exists():
        raise ValueError(f'Batch 1 output exists: {output}. Choose a new folder; no files overwritten.')
    if (not BATCH_1_THRESHOLDS or len(set(BATCH_1_THRESHOLDS)) != len(BATCH_1_THRESHOLDS)
            or any(t < 0 or t > 100 or int(t) != t for t in BATCH_1_THRESHOLDS)):
        raise ValueError('Batch 1 thresholds must be distinct whole percentages between 0 and 100.')
    if not 0 <= BATCH_1_SHARED_ZERO_PCT <= 100 or BATCH_1_RUN_MINUTES <= 0:
        raise ValueError('Check the shared-zero diagnostic percentage and run duration.')
    dates = pd.date_range(START_DATE, END_DATE)
    if dates.empty:
        raise ValueError('Batch 1 end date precedes start date.')
    inputs = [(d.strftime('%Y-%m-%d'), sorted((PREPARED_DATA_FOLDER / f'date={d:%Y-%m-%d}').glob('part-*.parquet'))) for d in dates]
    missing = [day for day, paths in inputs if not paths]
    if len(missing) == len(inputs):
        raise ValueError(f'No prepared daily parts found in {PREPARED_DATA_FOLDER}.')
    output.mkdir(parents=True)
    settings_path = output / 'batch_1_run_settings.json'
    settings = {'status': 'incomplete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'input': str(PREPARED_DATA_FOLDER), 'start_date': START_DATE, 'end_date': END_DATE,
                'scope': 'ML', 'thresholds': BATCH_1_THRESHOLDS,
                'run_minutes_diagnostic': BATCH_1_RUN_MINUTES,
                'shared_zero_pct_diagnostic': BATCH_1_SHARED_ZERO_PCT,
                'missing_dates': missing, 'selection_adopted': False,
                'continuity_requirements': ['valid time/date/station/speed', 'five-minute grid',
                                           'exclude every duplicate copy', 'same route',
                                           'do not bridge gaps or midnight'],
                'coverage_count_denominator': 'All saved ML rows attributed to a valid station ID',
                'threshold_zero_warning': '>=0 includes zero coverage; baseline only, not observed usability.'}
    settings_path.write_text(json.dumps(settings, indent=2))
    columns = ['timestamp', 'station_id', 'freeway', 'direction', 'lane_type', 'observed_pct',
               'invalid_observed_pct', 'invalid_timestamp', 'date_mismatch', 'invalid_station_id',
               'invalid_speed', 'off_5min_grid', 'duplicate_station_timestamp']
    daily_parts, distributions, hour_parts = [], [], []
    all_saved_rows = mainline_rows = unattributed_rows = 0
    try:
        for day, paths in inputs:
            if not paths:
                continue
            print(f'Batch 1: {day}', flush=True)
            # Read only needed columns, discard non-mainline records immediately,
            # and hold at most one day's raw rows. Source Parquet is untouched.
            frames = []
            for path in paths:
                part = pd.read_parquet(path, columns=columns)
                all_saved_rows += len(part)
                part = part.loc[part.lane_type.eq('ML')].copy()
                mainline_rows += len(part)
                frames.append(part)
            day_frame = pd.concat(frames, ignore_index=True)
            daily, distribution, hourly, unattributed = batch_1_day_metrics(day_frame, day)
            unattributed_rows += unattributed
            daily_parts.append(daily)
            distributions.append(distribution)
            hour_parts.append(hourly)
            del day_frame, frames
        daily = pd.concat(daily_parts, ignore_index=True)
        if daily.empty:
            raise ValueError('No attributable mainline station records found.')
        if int(daily.rows.sum()) + unattributed_rows != mainline_rows:
            raise ValueError('Batch 1 row reconciliation failed.')
        distribution = pd.concat(distributions, ignore_index=True)
        distribution = distribution.groupby(['station_id', 'observed_pct'], as_index=False).rows.sum()
        hourly = pd.concat(hour_parts, ignore_index=True)
        grouped = daily.groupby('station_id')
        counts = ['rows', 'zero_rows', 'positive_rows', 'invalid_coverage_rows', 'essential_rows',
                  'duplicate_copies', 'invalid_time_rows'] + [f'ge{t}_rows' for t in BATCH_1_THRESHOLDS]
        station = grouped[counts].sum()
        station['days_present'] = grouped.date.nunique()
        station['days_absent_from_requested_window'] = len(dates) - station.days_present
        station['days_with_positive_coverage'] = daily.positive_rows.gt(0).groupby(daily.station_id).sum()
        station['route_labels'] = grouped.route_labels.agg(lambda s: '|'.join(sorted(set(s))))
        station['positive_pct_all_rows'] = 100 * station.positive_rows / station.rows
        labels = ['positive'] + [f'ge{t}' for t in BATCH_1_THRESHOLDS]
        for label in labels:
            station[f'{label}_speed_valid_rows'] = grouped[f'{label}_speed_valid_rows'].sum()
            station[f'{label}_longest_run_minutes'] = grouped[f'{label}_longest_run_minutes'].max()
            station[f'{label}_days_run_ge{BATCH_1_RUN_MINUTES}min'] = (
                daily[f'{label}_longest_run_minutes'].ge(BATCH_1_RUN_MINUTES)
                .groupby(daily.station_id).sum())
            if label != 'positive':
                station[f'{label}_pct_all_rows'] = 100 * station[f'{label}_rows'] / station.rows
        daily['station_all_zero'] = daily.zero_rows.eq(daily.rows)
        daily['station_no_positive'] = daily.positive_rows.eq(0)
        network = daily.groupby('date')[counts].sum()
        network['stations_present'] = daily.groupby('date').station_id.nunique()
        network['stations_absent_from_study_pool'] = len(station) - network.stations_present
        network['stations_all_zero'] = daily.groupby('date').station_all_zero.sum()
        network['stations_no_positive'] = daily.groupby('date').station_no_positive.sum()
        for table in [network, hourly]:
            table['zero_pct_all_rows'] = 100 * table.zero_rows / table.rows
            table['positive_pct_all_rows'] = 100 * table.positive_rows / table.rows
            table['shared_zero_diagnostic'] = table.zero_pct_all_rows.ge(BATCH_1_SHARED_ZERO_PCT)
        shared_dates = network.index[network.shared_zero_diagnostic].tolist()
        # A high network-zero percentage may reflect persistently unobserved
        # stations. Also summarize only stations EVER positive in this window;
        # neither denominator proves a feed outage or hardware failure.
        positive_ids = station.index[station.positive_rows.gt(0)]
        active_daily = daily.loc[daily.station_id.isin(positive_ids)]
        active_network = active_daily.groupby('date')[['rows', 'zero_rows', 'positive_rows']].sum()
        network['ever_positive_pool_rows'] = active_network.rows.reindex(network.index, fill_value=0)
        network['zero_pct_ever_positive_pool'] = (100 * active_network.zero_rows / active_network.rows).reindex(network.index)
        network['shared_zero_ever_positive_pool'] = network.zero_pct_ever_positive_pool.ge(BATCH_1_SHARED_ZERO_PCT)
        shared_dates = network.index[network.shared_zero_diagnostic | network.shared_zero_ever_positive_pool].tolist()
        shared_station_dates = daily.loc[daily.date.isin(shared_dates) & daily.station_all_zero]
        station['shared_zero_dates_with_station_all_zero'] = (
            shared_station_dates.groupby('station_id').date.agg(lambda s: '|'.join(sorted(s)))
            .reindex(station.index, fill_value=''))
        # This is a descriptive candidate POOL, not a quality ranking or cutoff.
        # Keep every station with any positive observation for later comparison.
        station['candidate_pool_any_positive'] = station.positive_rows.gt(0)
        station.reset_index().to_csv(output / 'batch_1_station_quality.csv', index=False)
        # This wide, reusable daily intermediate is compressed Parquet rather
        # than a huge CSV. Keep it local/ignored in Git; commit compact reports.
        daily.to_parquet(output / 'batch_1_station_daily_quality.parquet', index=False)
        distribution.to_csv(output / 'batch_1_station_observation_distribution.csv', index=False)
        distribution.groupby('observed_pct', as_index=False).rows.sum().to_csv(output / 'batch_1_network_observation_distribution.csv', index=False)
        network.reset_index().to_csv(output / 'batch_1_network_daily_quality.csv', index=False)
        hourly.to_csv(output / 'batch_1_network_hourly_quality.csv', index=False)
        network.loc[shared_dates].reset_index().to_csv(output / 'batch_1_shared_zero_dates.csv', index=False)
        station.loc[station.candidate_pool_any_positive].reset_index().to_csv(output / 'batch_1_candidate_pool.csv', index=False)
        totals = {f'>={t}%': int(station[f'ge{t}_rows'].sum()) for t in BATCH_1_THRESHOLDS}
        report = '\n'.join([
            'BATCH 1 — OBSERVATION QUALITY AND MISSINGNESS',
            f'Saved rows scanned: {all_saved_rows:,}; mainline rows: {mainline_rows:,}',
            f'Unattributed invalid-station rows: {unattributed_rows:,}',
            f'Stations: {len(station):,}; ever-positive candidate pool: {len(positive_ids):,}',
            f'Zero-observed rows: {int(station.zero_rows.sum()):,}',
            f'Invalid coverage rows: {int(station.invalid_coverage_rows.sum()):,}',
            'Threshold retention (all attributed ML records): ' + json.dumps(totals),
            'Shared zero diagnostic dates: ' + ', '.join(shared_dates),
            'Missing prepared dates: ' + ', '.join(missing),
            '', 'INTERPRETATION AND DECISION GATE',
            'Candidate pool includes every ever-positive station; no final threshold or corridor chosen.',
            'Threshold 0 includes zero-observed rows. Positive coverage is not fully observed coverage.',
            'Speed continuity additionally requires valid identity/time/date/speed/grid and no duplicate copies.',
            'Runs stop at midnight, route changes, invalid rows, and missing timestamps.',
            'March 8 local day is 23 hours; its clock jump is not evidence of detector failure.',
            'Observed coverage describes recorded flags, not verified hardware health or causal outage diagnosis.',
            'Station absence differs from a present record with zero coverage.',
            'Run lengths are interval counts times five minutes; daily longest runs do not measure complete-day coverage.',
            'Reusable CSV summaries and a daily Parquet intermediate answer these questions without redundant charts.',
            'Next: obtain historical station metadata and compare geographically coherent corridors.',
        ])
        (output / 'batch_1_audit_summary.txt').write_text(report + '\n')
        settings.update(status='complete', saved_rows_scanned=all_saved_rows,
                        mainline_rows=mainline_rows, unattributed_rows=unattributed_rows,
                        stations=len(station), candidate_pool_stations=len(positive_ids))
        print('\n' + report)
        print(f'\nSaved Batch 1 reports: {output.resolve()}')
    except Exception as error:
        settings.update(status='incomplete', error=str(error))
        raise
    finally:
        settings_path.write_text(json.dumps(settings, indent=2))


# ===================== RUN ENABLED SECTIONS =====================
def main():
    switches = {
        'Preparation': RUN_PREPARATION,
        'Filter audit': RUN_FILTER_AUDIT,
        'Threshold comparison': RUN_THRESHOLD_COMPARISON,
        'Station coverage audit': RUN_STATION_COVERAGE_AUDIT,
        'Filtered selection': RUN_FILTERED_SELECTION,
        'Continuity audit': RUN_CONTINUITY_AUDIT,
        'Station ranking': RUN_STATION_RANKING,
        'Top-station review': RUN_TOP_STATION_REVIEW,
        'Ranking/network checks': RUN_RANKING_NETWORK_CHECKS,
        'Batch 1 observation quality': RUN_BATCH_1,
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

    if RUN_TOP_STATION_REVIEW:
        run_top_station_review()

    if RUN_RANKING_NETWORK_CHECKS:
        run_ranking_network_checks()

    if RUN_BATCH_1:
        run_batch_1()


if __name__ == '__main__':
    main()
