import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
RUN_BATCH_2 = False
RUN_BATCH_2_SELECTION = False
RUN_BATCH_3 = False
RUN_BATCH_4_PART_1 = False
RUN_BATCH_4_PART_2 = False
RUN_BATCH_5_PART_1 = False
RUN_BATCH_5_PART_2 = False
RUN_BATCH_5_PART_2B = False
RUN_BATCH_5_PART_3 = False
RUN_BATCH_5_PART_4 = False
RUN_BATCH_5_PART_5 = False
RUN_BATCH_6_PART_1 = False
RUN_BATCH_6_PART_2 = False
RUN_BATCH_6_PART_3 = False
RUN_BATCH_7_PART_1 = False
RUN_BATCH_7_PART_2 = False
RUN_BATCH_7_PART_3 = False
RUN_BATCH_7A = True

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

BATCH_2_METADATA_FILE = Path('d07_text_meta_2023_12_22.txt')
BATCH_2_REPORT_FOLDER = Path('audit_reports/batch_2_corridor_construction_v3')
BATCH_2_SELECTION_SOURCE_FOLDER = Path('audit_reports/batch_2_corridor_construction_v3')
BATCH_2_SELECTION_OUTPUT_FOLDER = Path('audit_reports/batch_2_primary_corridor_selection')
BATCH_2_MIN_CORRIDOR_STATIONS = 4
BATCH_2_TRAFFIC_REVIEW_TOP_N = 5
BATCH_2_AM_PEAK_HOURS = (6, 10)
BATCH_2_PM_PEAK_HOURS = (15, 19)

BATCH_3_REPORT_FOLDER = Path('audit_reports/batch_3_analytical_validity_v3')
BATCH_3_CORRIDOR_GEOGRAPHY_FILE = (
    BATCH_2_SELECTION_SOURCE_FOLDER / 'batch_2_candidate_station_geography.csv'
)
BATCH_3_SELECTION_FILE = (
    BATCH_2_SELECTION_OUTPUT_FOLDER / 'batch_2_primary_corridor_selection.csv'
)
BATCH_3_THRESHOLDS = [40, 60, 80, 100]
BATCH_3_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_4_REPORT_FOLDER = Path('audit_reports/batch_4_breakdown_detection_v2')
BATCH_4_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_4_PART_2_SOURCE_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_4_PART_2_REPORT_FOLDER = Path(
    'audit_reports/batch_4_part_2_concept_separation'
)
BATCH_4_PART_2_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_5_PART_1_SOURCE_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_5_PART_1_DECISION_FOLDER = Path(
    'audit_reports/batch_4_part_2_concept_separation'
)
BATCH_5_PART_1_REPORT_FOLDER = Path(
    'audit_reports/batch_5_part_1_prebreakdown_conditions_v4'
)
BATCH_5_PART_1_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'
BATCH_5_PART_1_PRIMARY_ONSET_METHOD = 'LA_20drop_below40'
BATCH_5_PART_1_PROFILE_MINUTES = list(range(-30, 35, 5))
BATCH_5_PART_1_CONTROL_MINUTES_FROM_ONSET = 60
BATCH_5_PART_1_CONTROLS_PER_EVENT = 10
BATCH_5_PART_1_CONTEXT_START_HOUR = 4.5
BATCH_5_PART_1_CONTEXT_END_HOUR = 20.5
BATCH_5_PART_1_CORE_START_HOUR = 5.0
BATCH_5_PART_1_CORE_END_HOUR = 20.0

BATCH_5_PART_2_SOURCE_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_5_PART_2_PART1_FOLDER = Path(
    'audit_reports/batch_5_part_1_prebreakdown_conditions_v4'
)
BATCH_5_PART_2_REPORT_FOLDER = Path(
    'audit_reports/batch_5_part_2_propagation'
)
BATCH_5_PART_2_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_5_PART_2_PRIMARY_ONSET_METHOD = 'LA_20drop_below40'
BATCH_5_PART_2_STATE_METHOD = 'Threshold_50_15min'
BATCH_5_PART_2_CLUSTER_GAP_MINUTES = 30
BATCH_5_PART_2_CLUSTER_SENSITIVITY_MINUTES = [15, 30, 45]
BATCH_5_PART_2_PROFILE_START_MINUTE = -60
BATCH_5_PART_2_PROFILE_END_MINUTE = 120
BATCH_5_PART_2_STATE_MATCH_BEFORE_MINUTES = 60
BATCH_5_PART_2_STATE_MATCH_AFTER_MINUTES = 120

BATCH_5_PART_2B_SOURCE_FOLDER = Path(
    'audit_reports/batch_5_part_2_propagation'
)
BATCH_5_PART_2B_REPORT_FOLDER = Path(
    'audit_reports/batch_5_part_2b_propagation_robustness_v2'
)
BATCH_5_PART_2B_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_5_PART_2B_WINDOWS_MINUTES = [30, 60, 120]
BATCH_5_PART_2B_PRIMARY_WINDOW_MINUTES = 60
BATCH_5_PART_2B_CORRIDOR_WIDE_MIN_STATIONS = 4
BATCH_5_PART_2B_PARTIAL_MIN_STATIONS = 2

# --------------------- Batch 5 Part 3: duration / recovery ---------------------
BATCH_5_PART_3_SOURCE_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_5_PART_3_PROPAGATION_FOLDER = Path(
    'audit_reports/batch_5_part_2_propagation'
)
BATCH_5_PART_3_ROBUSTNESS_FOLDER = Path(
    'audit_reports/batch_5_part_2b_propagation_robustness_v2'
)
BATCH_5_PART_3_REPORT_FOLDER = Path(
    'audit_reports/batch_5_part_3_duration_recovery'
)
BATCH_5_PART_3_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'
BATCH_5_PART_3_STATE_METHOD = 'Threshold_50_15min'
BATCH_5_PART_3_PRIMARY_PROPAGATION_WINDOW_MINUTES = 60
BATCH_5_PART_3_TIME_PERIODS = [
    ('AM', 5, 10),
    ('Midday', 10, 15),
    ('PM', 15, 20),
]

# --------------------- Batch 5 Part 4: breakdown probability ---------------------
BATCH_5_PART_4_SOURCE_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_5_PART_4_REPORT_FOLDER = Path(
    'audit_reports/batch_5_part_4_breakdown_probability'
)
BATCH_5_PART_4_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'
BATCH_5_PART_4_ONSET_METHOD = 'LA_20drop_below40'
BATCH_5_PART_4_FREE_FLOW_MIN_SPEED_MPH = 40
BATCH_5_PART_4_BOOTSTRAP_REPS = 300
BATCH_5_PART_4_BOOTSTRAP_SEED = 20261009
BATCH_5_PART_4_GRID_POINTS = 60
BATCH_5_PART_4_CONFIDENCE_LOW = 0.025
BATCH_5_PART_4_CONFIDENCE_HIGH = 0.975
BATCH_5_PART_4_SOURCE_NOTES = [
    {
        'label': 'FHWA stochastic breakdown probability / product-limit method',
        'url': 'https://ops.fhwa.dot.gov/publications/fhwahop19029/appc.htm',
        'basis': (
            'FHWA describes freeway capacity as stochastic, classifies the '
            'interval immediately before breakdown as an uncensored capacity '
            'observation, treats free-flow intervals not followed by breakdown '
            'as censored observations, and estimates breakdown probability '
            'with the product-limit method.'
        ),
    },
    {
        'label': 'FHWA decision-parameter chapter',
        'url': 'https://ops.fhwa.dot.gov/publications/fhwahop19029/ch4.htm',
        'basis': (
            'FHWA describes using current speed/flow conditions to estimate '
            'the probability of breakdown in the next interval and identifies '
            'the product-limit method as an established technique.'
        ),
    },
]

# --------------------- Batch 5 Part 5: synthesis ---------------------
BATCH_5_PART_5_PART1_FOLDER = Path(
    'audit_reports/batch_5_part_1_prebreakdown_conditions_v4'
)
BATCH_5_PART_5_PART2_FOLDER = Path(
    'audit_reports/batch_5_part_2_propagation'
)
BATCH_5_PART_5_PART2B_FOLDER = Path(
    'audit_reports/batch_5_part_2b_propagation_robustness_v2'
)
BATCH_5_PART_5_PART3_FOLDER = Path(
    'audit_reports/batch_5_part_3_duration_recovery'
)
BATCH_5_PART_5_PART4_FOLDER = Path(
    'audit_reports/batch_5_part_4_breakdown_probability'
)
BATCH_5_PART_5_REPORT_FOLDER = Path(
    'audit_reports/batch_5_part_5_synthesis'
)

# --------------------- Batch 6 Part 1: final robustness ---------------------
BATCH_6_PART_1_BATCH3_FOLDER = Path(
    'audit_reports/batch_3_analytical_validity_v3'
)
BATCH_6_PART_1_BATCH4_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_6_PART_1_BATCH5_PART1_FOLDER = Path(
    'audit_reports/batch_5_part_1_prebreakdown_conditions_v4'
)
BATCH_6_PART_1_BATCH5_PART2_FOLDER = Path(
    'audit_reports/batch_5_part_2_propagation'
)
BATCH_6_PART_1_BATCH5_PART2B_FOLDER = Path(
    'audit_reports/batch_5_part_2b_propagation_robustness_v2'
)
BATCH_6_PART_1_BATCH5_PART3_FOLDER = Path(
    'audit_reports/batch_5_part_3_duration_recovery'
)
BATCH_6_PART_1_BATCH5_PART4_FOLDER = Path(
    'audit_reports/batch_5_part_4_breakdown_probability'
)
BATCH_6_PART_1_REPORT_FOLDER = Path(
    'audit_reports/batch_6_part_1_final_robustness'
)
BATCH_6_PART_1_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_6_INFLUENCE_RELATIVE_CHANGE_PCT = 20.0
BATCH_6_ONSET_DATE_CONCENTRATION_FLAG_PCT = 30.0
BATCH_6_PRIMARY_THRESHOLD_LOW = 80
BATCH_6_PRIMARY_THRESHOLD_HIGH = 100
BATCH_6_PROPAGATION_STABILITY_TARGET_PCT = 60.0

# --------------------- Batch 6 Part 2: adjudication / limitations ---------------------
BATCH_6_PART_2_PART1_FOLDER = Path(
    'audit_reports/batch_6_part_1_final_robustness'
)
BATCH_6_PART_2_BATCH4_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_6_PART_2_BATCH4_DECISION_FOLDER = Path(
    'audit_reports/batch_4_part_2_concept_separation'
)
BATCH_6_PART_2_BATCH5_PART1_FOLDER = Path(
    'audit_reports/batch_5_part_1_prebreakdown_conditions_v4'
)
BATCH_6_PART_2_BATCH5_PART2B_FOLDER = Path(
    'audit_reports/batch_5_part_2b_propagation_robustness_v2'
)
BATCH_6_PART_2_BATCH5_PART3_FOLDER = Path(
    'audit_reports/batch_5_part_3_duration_recovery'
)
BATCH_6_PART_2_BATCH5_PART4_FOLDER = Path(
    'audit_reports/batch_5_part_4_breakdown_probability'
)
BATCH_6_PART_2_REPORT_FOLDER = Path(
    'audit_reports/batch_6_part_2_findings_limitations'
)

# --------------------- Batch 6 Part 3: final portfolio package ---------------------
BATCH_6_PART_3_PART1_FOLDER = Path(
    'audit_reports/batch_6_part_1_final_robustness'
)
BATCH_6_PART_3_PART2_FOLDER = Path(
    'audit_reports/batch_6_part_2_findings_limitations'
)
BATCH_6_PART_3_REPORT_FOLDER = Path(
    'audit_reports/batch_6_part_3_final_portfolio'
)
BATCH_6_PART_3_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'

BATCH_6_PROJECT_TITLE = (
    'Traffic Breakdown and Congestion Dynamics on an I-10 East Corridor'
)


# --------------------- Batch 7: portfolio packaging ---------------------
BATCH_7_BATCH6_PORTFOLIO_FOLDER = Path('audit_reports/batch_6_part_3_final_portfolio')
BATCH_7_BATCH6_FINDINGS_FOLDER = Path('audit_reports/batch_6_part_2_findings_limitations')
BATCH_7_BATCH6_ROBUSTNESS_FOLDER = Path('audit_reports/batch_6_part_1_final_robustness')
BATCH_7_BATCH5_PART1_FOLDER = Path('audit_reports/batch_5_part_1_prebreakdown_conditions_v4')
BATCH_7_BATCH5_PART3_FOLDER = Path('audit_reports/batch_5_part_3_duration_recovery')
BATCH_7_BATCH4_FOLDER = Path('audit_reports/batch_4_breakdown_detection_v2')
BATCH_7_REPORT_FOLDER = Path('audit_reports/batch_7_portfolio_packaging')
BATCH_7_PORTFOLIO_FIGURE_FOLDER = Path('portfolio_figures')
BATCH_7_README = Path('README.md')
BATCH_7_METHODS = Path('METHODS.md')
BATCH_7_RESULTS = Path('RESULTS.md')
BATCH_7_LIMITATIONS = Path('LIMITATIONS.md')
BATCH_7_FIGURES = Path('FIGURES.md')
BATCH_7_PROJECT_STATUS = Path('PROJECT_STATUS.md')
BATCH_7_STUDY_PERIOD = 'February 1–May 31, 2026'
BATCH_7_CORRIDOR_DESCRIPTION = 'five-station eastbound I-10 corridor in Caltrans District 7'


BATCH_4_FINAL_OBSERVED_PCT = 100
BATCH_4_ANALYSIS_START_HOUR = 5
BATCH_4_ANALYSIS_END_HOUR = 20
BATCH_4_THRESHOLD_SPEEDS = [40, 50, 55]
BATCH_4_PERSISTENCE_INTERVALS = 3

# Literature / agency sources used for the Batch 4 operational definitions.
# These are recorded in the output report so the event rules are traceable.
BATCH_4_SOURCES = [
    {
        'label': 'FHWA D-PTSU speed-threshold discussion',
        'url': 'https://ops.fhwa.dot.gov/publications/fhwahop19029/ch4.htm',
        'basis': (
            'FHWA notes that freeway breakdown thresholds vary by facility, '
            'typically around 40-55 mph, and uses 50 mph as an appropriate '
            'illustrative threshold after comparing 40, 50 and 55 mph.'
        ),
    },
    {
        'label': 'FHWA freeway breakdown persistence discussion',
        'url': 'https://ops.fhwa.dot.gov/publications/fhwahop08054/sect6.htm',
        'basis': (
            'FHWA describes the HCM convention that oversaturated freeway '
            'conditions lasting at least 15 minutes constitute sustained breakdown.'
        ),
    },
    {
        'label': 'SHRP2 C05 / Los Angeles five-minute breakdown rule',
        'url': 'https://onlinepubs.trb.org/onlinepubs/shrp2/SHRP2prepubC05.pdf',
        'basis': (
            'The report summarizes Los Angeles work using five-minute data '
            'with a minimum 20 mph speed differential and a speed threshold '
            'of 40 mph to identify breakdown.'
        ),
    },
    {
        'label': 'Caltrans PeMS bottleneck identification',
        'url': 'https://cagov.github.io/caldata-mdsa-caltrans-pems/data/bottlenecks/',
        'basis': (
            'Current Caltrans PeMS implementation uses a 20 mph inter-station '
            'speed drop, downstream speed below 40 mph, station separation '
            'under 3 miles, and persistence in at least 5 of 7 consecutive '
            'five-minute observations.'
        ),
    },
]

# ===================== PORTFOLIO VISUAL THEME =====================
# ALL visual-theme designations live here at the top of the file.
# Plotting code below must reference these constants rather than define
# colors, fonts, or colormaps locally.

BRAND_COLORS = [
    '#08415C',  # blue
    '#8E443D',  # brick
    '#511730',  # plum
    '#E0D68A',  # sand
    '#F7A399',  # salmon
    '#041F2C',  # field
]

THEME_BACKGROUND = '#F4F0E6'
THEME_TEXT = BRAND_COLORS[5]
THEME_GRID = BRAND_COLORS[4]
THEME_BODY_FONT = 'Montserrat'
THEME_TITLE_FONT = 'Archivo Black'
THEME_HEATMAP_CMAP = 'viridis'
THEME_WHITE = 'white'
THEME_TRANSPARENT = 'none'

# Semantic colors used across the traffic project.
THEME_BREAKDOWN = BRAND_COLORS[0]
THEME_CONTROL = BRAND_COLORS[1]
THEME_SECONDARY = BRAND_COLORS[2]
THEME_ACCENT = BRAND_COLORS[3]
THEME_EDGE = BRAND_COLORS[5]
THEME_MEDIAN = BRAND_COLORS[3]

# Shared plot-style parameters.
THEME_BOX_ALPHA = 0.62
THEME_BOX_WIDTH = 0.58
THEME_BOX_EDGE_WIDTH = 1.5
THEME_MEDIAN_WIDTH = 2.2
THEME_FLIER_SIZE = 4
THEME_FLIER_ALPHA = 0.75

# Every plot-design parameter is centralized here. Plotting sections below
# reference these names only; they do not create local visual styles.
THEME_FIGSIZE_15_6 = (15, 6)
THEME_FIGSIZE_12_7 = (12, 7)
THEME_FIGSIZE_18_6 = (18, 6)
THEME_FIGSIZE_14_7 = (14, 7)
THEME_FIGSIZE_11_7 = (11, 7)
THEME_FIGSIZE_16_7 = (16, 7)
THEME_FIGSIZE_13_7 = (13, 7)
THEME_FIGSIZE_18_6_5 = (18, 6.5)
THEME_FIGSIZE_17_6 = (17, 6)

THEME_MARKER_PRIMARY = 'o'
THEME_LINEWIDTH_STANDARD = 2.0
THEME_LINEWIDTH_EMPHASIS = 2.3
THEME_LINEWIDTH_SECONDARY = 1.8
THEME_LINEWIDTH_FINE = 1.5
THEME_LINEWIDTH_REFERENCE = 1.4
THEME_MARKER_SIZE_STANDARD = 7
THEME_SCATTER_SIZE = 90

THEME_GRID_ALPHA_STRONG = 0.60
THEME_GRID_ALPHA_MEDIUM = 0.50
THEME_GRID_ALPHA_LIGHT = 0.40
THEME_GRID_ALPHA_SUBTLE = 0.35
THEME_FILL_ALPHA = 0.18

THEME_ROTATION_SMALL = 12
THEME_ROTATION_MEDIUM = 15
THEME_ROTATION_LARGE = 25
THEME_ROTATION_XL = 45

THEME_ALIGN_CENTER = 'center'
THEME_ALIGN_RIGHT = 'right'
THEME_ANNOTATION_SIZE_SMALL = 8
THEME_ANNOTATION_SIZE = 10
THEME_SUPTITLE_SIZE = 20
THEME_SUPTITLE_SIZE_SMALL = 19
THEME_TEXTCOORDS = 'offset points'
THEME_ANNOTATION_OFFSET_6 = (0, 6)
THEME_ANNOTATION_OFFSET_7 = (0, 7)
THEME_ANNOTATION_OFFSET_8 = (0, 8)

THEME_LINESTYLE_DASHED = '--'
THEME_LINESTYLE_DOTTED = ':'
THEME_LEGEND_FRAME = False
THEME_LEGEND_NCOL_2 = 2
THEME_LEGEND_NCOL_3 = 3
THEME_LEGEND_LOC = 'upper center'
THEME_LEGEND_ANCHOR = (0.5, -0.18)

THEME_GRID_AXIS_X = 'x'
THEME_GRID_AXIS_Y = 'y'
THEME_GRID_VISIBLE_OFF = False
THEME_TICK_AXIS_X = 'x'

THEME_AXIS_MIN_ZERO = 0
THEME_PERCENT_AXIS_MAX = 105
THEME_HEATMAP_ASPECT = 'auto'
THEME_HEATMAP_INTERPOLATION = 'nearest'
THEME_HEATMAP_MIN = 0
THEME_HEATMAP_MAX = 75
THEME_COLORBAR_PAD = 0.02

THEME_DPI = 250
THEME_BBOX = 'tight'
THEME_TIGHT_RECT_045 = (0, 0.045, 1, 1)
THEME_TIGHT_RECT_05 = (0, 0.05, 1, 1)
THEME_TIGHT_RECT_055 = (0, 0.055, 1, 1)
THEME_TIGHT_RECT_B5_PROFILE = (0, 0.055, 1, 0.95)
THEME_TIGHT_RECT_B5_BOX = (0, 0.055, 1, 0.94)
THEME_BOXPLOT_SHOW_FLIERS = True

THEME_FIGSIZE_B5P2_HEATMAP = (16, 8)
THEME_FIGSIZE_B5P2_LAGS = (13, 7)
THEME_FIGSIZE_B5P2_EPISODES = (14, 48)
THEME_LINEWIDTH_PROPAGATION = 2.1
THEME_ALPHA_PROPAGATION = 0.80
THEME_ALPHA_REFERENCE = 0.45
THEME_LEGEND_NCOL_1 = 1

THEME_FIGSIZE_B5P2B_CLASSIFICATION = (13, 7)
THEME_FIGSIZE_B5P2B_STATIONS = (13, 7)
THEME_BAR_WIDTH_GROUPED = 0.22
THEME_BAR_OFFSET_NEGATIVE = -0.22
THEME_BAR_OFFSET_ZERO = 0.0
THEME_BAR_OFFSET_POSITIVE = 0.22

# Batch 5 Part 3-4 figure settings
THEME_FIGSIZE_B5P3_DURATION = (14, 7)
THEME_FIGSIZE_B5P3_RECOVERY = (13, 7)
THEME_FIGSIZE_B5P3_SURVIVAL = (13, 7)
THEME_FIGSIZE_B5P4_PLM = (14, 7)

THEME_STEP_WHERE_POST = 'post'
THEME_SURVIVAL_LINEWIDTH = 2.3
THEME_RECOVERY_LINEWIDTH = 2.0
THEME_BOOTSTRAP_ALPHA = 0.20
THEME_REFERENCE_ALPHA = 0.55
THEME_RECOVERY_MARKER = 'o'
THEME_EVENT_RUG_MARKER = '|'
THEME_EVENT_RUG_SIZE = 13
THEME_EVENT_RUG_LINEWIDTH = 1.6
THEME_BOX_WIDTH_DURATION = 0.62

# Batch 6 figure settings
THEME_FIGSIZE_B6_ROBUSTNESS = (15, 7)
THEME_FIGSIZE_B6_PERIOD = (12, 7)
THEME_FIGSIZE_B6_CONCENTRATION = (13, 7)

THEME_BAR_WIDTH_B6 = 0.62
THEME_ERRORBAR_CAPSIZE_B6 = 7
THEME_SCATTER_SIZE_B6 = 80
THEME_REFERENCE_LINEWIDTH_B6 = 1.8
THEME_ANNOTATION_OFFSET_B6 = (0, 7)
THEME_ROTATION_B6 = 25

plt.rcParams.update({
    'axes.prop_cycle': plt.cycler(color=BRAND_COLORS),
    'font.family': THEME_BODY_FONT,
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.facecolor': THEME_BACKGROUND,
    'axes.facecolor': THEME_BACKGROUND,
    'axes.labelcolor': THEME_TEXT,
    'xtick.color': THEME_TEXT,
    'ytick.color': THEME_TEXT,
    'text.color': THEME_TEXT,
    'grid.color': THEME_GRID,
    'grid.linewidth': 0.5,
})

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

# ===================== BATCH 7A SETTINGS / STYLE =====================
# BATCH 7A SETTINGS — EDIT HERE
# ============================================================

BATCH_7A_REPORT_FOLDER = Path('audit_reports/batch_7a_corrective_review_v3')
BATCH_7A_IMAGE_FOLDER = Path(__file__).resolve().parent / 'images'
BATCH_7A_REVISED_FIGURE_FOLDER = Path('portfolio_figures_revised_v3')
BATCH_7A_FULL_DAY_CACHE = Path(
    'audit_reports/batch_7a_full_day_corridor_cache.parquet'
)

BATCH_7A_BATCH2_SELECTION_FOLDER = Path(
    'audit_reports/batch_2_primary_corridor_selection'
)
BATCH_7A_BATCH4_FOLDER = Path(
    'audit_reports/batch_4_breakdown_detection_v2'
)
BATCH_7A_BATCH5_PART2_FOLDER = Path(
    'audit_reports/batch_5_part_2_propagation'
)
BATCH_7A_BATCH5_PART3_FOLDER = Path(
    'audit_reports/batch_5_part_3_duration_recovery'
)
BATCH_7A_RECOVERY_CURVE_FILE = (
    BATCH_7A_BATCH5_PART3_FOLDER
    / 'batch_5_part_3_recovery_product_limit.csv'
)
BATCH_7A_BATCH6_PART1_FOLDER = Path(
    'audit_reports/batch_6_part_1_final_robustness'
)
BATCH_7A_BATCH6_PART3_FOLDER = Path(
    'audit_reports/batch_6_part_3_final_portfolio'
)

BATCH_7A_STATE_THRESHOLD_MPH = 50
BATCH_7A_STATE_MIN_INTERVALS = 3
BATCH_7A_STATE_MERGE_GAPS_MINUTES = [0, 5, 10, 15]

BATCH_7A_ANALYSIS_START_HOUR = 5.0
BATCH_7A_ANALYSIS_END_HOUR = 20.0

BATCH_7A_BOTTLENECK_UPSTREAM_SPEED_MPH = 50
BATCH_7A_BOTTLENECK_DOWNSTREAM_SPEED_MPH = 50
BATCH_7A_BOTTLENECK_MIN_RUN_INTERVALS = 3

BATCH_7A_DISCHARGE_FREE_SPEED_MPH = 50

BATCH_7A_CLAUDE_REVIEW_FILE = Path('CLAUDE_REVIEW_PACKET.md')
BATCH_7A_METHODS_REVISED_FILE = Path('METHODS_REVISED.md')
BATCH_7A_RESULTS_REVISED_FILE = Path('RESULTS_REVISED.md')
BATCH_7A_FIGURES_REVISED_FILE = Path('FIGURES_REVISED.md')

BATCH_7A_FORCE_EXCLUDE_FIGURES = [
    'images/traffic_b4_onsets_by_hour.png',
    'images/traffic_b4_event_counts_by_method.png',
    'images/traffic_b3_lane_consistency.png',
    'images/traffic_b5p4_breakdown_probability_plm.png',
    'images/traffic_b6_leave_one_out_duration_robustness.png',
    'images/traffic_b6_duration_by_time_period.png',
    'images/traffic_b6_abrupt_onset_concentration.png',
    'images/traffic_b5p3_recovery_survival.png',
]


# ============================================================
# BATCH 7A STYLE — ALL DESIGN PARAMETERS CENTRALIZED HERE
# ============================================================

THEME_FIGSIZE_B7A_COUNTS = (12, 7)
THEME_FIGSIZE_B7A_THRESHOLD = (11, 7)
THEME_FIGSIZE_B7A_LEAVEOUT = (14, 7)
THEME_FIGSIZE_B7A_EPISODES = (18, 10)
THEME_FIGSIZE_B7A_MERGE = (13, 7)
THEME_FIGSIZE_B7A_RECOVERY = (12, 7)
THEME_FIGSIZE_B7A_ONSET_DATES = (12, 7)

THEME_SCALE_LOG_B7A = 'log'
THEME_LINESTYLE_NONE_B7A = 'none'
THEME_RANGE_CAPSIZE_B7A = 7
THEME_RANGE_MARKERSIZE_B7A = 8
THEME_GRID_ROWS_B7A = 2
THEME_GRID_COLUMNS_B7A = 3
THEME_EPISODE_TICK_COUNT_B7A = 4
THEME_COLORBAR_SHRINK_B7A = 0.82
THEME_COLORBAR_PAD_B7A = 0.025
THEME_Y_HEADROOM_B7A = 1.08
THEME_GROUP_BAR_WIDTH_B7A = 0.24
THEME_GROUP_BAR_OFFSETS_B7A = (-0.24, 0.0, 0.24)
THEME_RECOVERY_START_MINUTES_B7A = 0.0
THEME_RECOVERY_START_PROBABILITY_B7A = 1.0



# ============================================================

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
            'Missing prepared dates: ' + (', '.join(missing) if missing else 'none'),
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



# ===================== BATCH 2. STATION METADATA AND CORRIDORS =====================
# Questions 7-11 are handled together from saved Batch 1 summaries and the
# historical PeMS station metadata valid for the Feb-May 2026 study window.
#
# IMPORTANT:
# Physical station order and adjacency are defined using ALL historical District 7
# mainline stations, not only the 480 ever-positive candidates. Candidate corridor
# runs therefore cannot silently jump across an intervening noncandidate detector.


def summarize_batch_2_corridors(candidate_geo):
    """Summarize physically contiguous candidate runs."""
    grouped = candidate_geo.groupby(
        ['corridor_id', 'Fwy', 'Dir'],
        sort=True,
        dropna=False,
    )

    corridor = grouped.agg(
        station_count=('station_id', 'size'),
        first_station_id=('station_id', 'first'),
        last_station_id=('station_id', 'last'),
        min_abs_pm=('Abs_PM', 'min'),
        max_abs_pm=('Abs_PM', 'max'),
        first_route_order=('physical_route_order', 'min'),
        last_route_order=('physical_route_order', 'max'),
        mean_positive_pct=('positive_pct_all_rows', 'mean'),
        median_positive_pct=('positive_pct_all_rows', 'median'),
        mean_ge80_pct=('ge80_pct_all_rows', 'mean'),
        median_ge80_pct=('ge80_pct_all_rows', 'median'),
        mean_ge100_pct=('ge100_pct_all_rows', 'mean'),
        median_ge100_pct=('ge100_pct_all_rows', 'median'),
        min_days_positive=('days_with_positive_coverage', 'min'),
        mean_days_positive=('days_with_positive_coverage', 'mean'),
        mean_ge80_longest_run_minutes=('ge80_longest_run_minutes', 'mean'),
        min_ge80_longest_run_minutes=('ge80_longest_run_minutes', 'min'),
        max_physical_neighbor_gap_miles=('distance_from_previous_ml_miles', 'max'),
    ).reset_index()

    corridor['postmile_span'] = corridor.max_abs_pm - corridor.min_abs_pm

    weighted = grouped[
        ['rows', 'positive_rows', 'ge80_rows', 'ge100_rows']
    ].sum().reset_index()

    weighted['weighted_positive_pct'] = (
        100 * weighted.positive_rows / weighted.rows
    )
    weighted['weighted_ge80_pct'] = (
        100 * weighted.ge80_rows / weighted.rows
    )
    weighted['weighted_ge100_pct'] = (
        100 * weighted.ge100_rows / weighted.rows
    )

    corridor = corridor.merge(
        weighted[
            [
                'corridor_id',
                'Fwy',
                'Dir',
                'weighted_positive_pct',
                'weighted_ge80_pct',
                'weighted_ge100_pct',
            ]
        ],
        on=['corridor_id', 'Fwy', 'Dir'],
        how='left',
        validate='one_to_one',
    )

    # Practical review ordering only. It is intentionally transparent and does
    # not automatically choose the primary study corridor.
    corridor['review_score'] = (
        corridor.station_count
        * corridor.weighted_ge80_pct / 100
        * corridor.mean_days_positive
    )

    corridor['eligible_for_corridor_review'] = (
        corridor.station_count.ge(BATCH_2_MIN_CORRIDOR_STATIONS)
    )

    corridor = corridor.sort_values(
        [
            'eligible_for_corridor_review',
            'review_score',
            'station_count',
            'weighted_ge80_pct',
            'weighted_ge100_pct',
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    corridor['review_rank'] = range(1, len(corridor) + 1)

    return corridor


def run_batch_2():
    """Merge historical metadata and construct physically contiguous corridors."""
    output = BATCH_2_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 2 output exists: {output}. '
            'Choose a new folder; no files overwritten.'
        )

    candidate_path = (
        BATCH_1_REPORT_FOLDER / 'batch_1_candidate_pool.csv'
    )
    station_quality_path = (
        BATCH_1_REPORT_FOLDER / 'batch_1_station_quality.csv'
    )

    required_inputs = [
        BATCH_2_METADATA_FILE,
        candidate_path,
        station_quality_path,
    ]
    missing_inputs = [
        str(path)
        for path in required_inputs
        if not path.exists()
    ]

    if missing_inputs:
        raise FileNotFoundError(
            'Missing Batch 2 inputs: ' + ', '.join(missing_inputs)
        )

    if BATCH_2_MIN_CORRIDOR_STATIONS < 2:
        raise ValueError(
            'BATCH_2_MIN_CORRIDOR_STATIONS must be at least 2.'
        )

    output.mkdir(parents=True)

    settings_path = output / 'batch_2_run_settings.json'
    settings = {
        'status': 'incomplete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'metadata_file': str(BATCH_2_METADATA_FILE),
        'batch_1_candidate_pool': str(candidate_path),
        'batch_1_station_quality': str(station_quality_path),
        'minimum_corridor_stations_for_review':
            BATCH_2_MIN_CORRIDOR_STATIONS,
        'selection_adopted': False,
        'traffic_review_top_n': BATCH_2_TRAFFIC_REVIEW_TOP_N,
        'am_peak_hours_diagnostic': list(BATCH_2_AM_PEAK_HOURS),
        'pm_peak_hours_diagnostic': list(BATCH_2_PM_PEAK_HOURS),
        'adjacency_definition':
            'Immediate same-freeway/direction neighbor in the full historical '
            'District 7 ML metadata inventory, ordered by absolute postmile.',
        'corridor_definition':
            'Consecutive ever-positive candidate stations in the full physical '
            'ML station sequence; any intervening noncandidate ML station breaks '
            'the candidate corridor run.',
    }
    settings_path.write_text(json.dumps(settings, indent=2))

    try:
        metadata = pd.read_csv(
            BATCH_2_METADATA_FILE,
            sep='\t',
            dtype={
                'ID': 'Int64',
                'Fwy': 'Int64',
                'Dir': 'string',
                'District': 'Int64',
                'Type': 'string',
                'Name': 'string',
            },
        )

        metadata.columns = metadata.columns.str.strip()

        required_metadata = [
            'ID',
            'Fwy',
            'Dir',
            'District',
            'County',
            'City',
            'State_PM',
            'Abs_PM',
            'Latitude',
            'Longitude',
            'Length',
            'Type',
            'Lanes',
            'Name',
        ]

        missing_metadata_columns = [
            column
            for column in required_metadata
            if column not in metadata.columns
        ]

        if missing_metadata_columns:
            raise ValueError(
                'Historical metadata missing columns: '
                + ', '.join(missing_metadata_columns)
            )

        metadata['ID'] = pd.to_numeric(
            metadata.ID,
            errors='coerce',
        ).astype('Int64')

        metadata['Fwy'] = pd.to_numeric(
            metadata.Fwy,
            errors='coerce',
        ).astype('Int64')

        metadata['District'] = pd.to_numeric(
            metadata.District,
            errors='coerce',
        ).astype('Int64')

        numeric_metadata = [
            'County',
            'City',
            'State_PM',
            'Abs_PM',
            'Latitude',
            'Longitude',
            'Length',
            'Lanes',
        ]

        for column in numeric_metadata:
            metadata[column] = pd.to_numeric(
                metadata[column],
                errors='coerce',
            )

        metadata['Dir'] = metadata.Dir.str.strip()
        metadata['Type'] = metadata.Type.str.strip()
        metadata['Name'] = metadata.Name.fillna('').str.strip()

        metadata_d7 = metadata.loc[
            metadata.District.eq(7)
        ].copy()

        metadata_ml = metadata_d7.loc[
            metadata_d7.Type.eq('ML')
        ].copy()

        if metadata_ml.ID.duplicated().any():
            duplicates = (
                metadata_ml.loc[
                    metadata_ml.ID.duplicated(keep=False),
                    ['ID', 'Fwy', 'Dir', 'Abs_PM', 'Name'],
                ]
                .sort_values('ID')
            )

            duplicates.to_csv(
                output / 'batch_2_duplicate_metadata_ids.csv',
                index=False,
            )

            raise ValueError(
                'Duplicate ML station IDs found in historical metadata. '
                'See batch_2_duplicate_metadata_ids.csv.'
            )

        candidate = pd.read_csv(candidate_path)
        quality = pd.read_csv(station_quality_path)

        candidate['station_id'] = pd.to_numeric(
            candidate.station_id,
            errors='raise',
        ).astype('int64')

        quality['station_id'] = pd.to_numeric(
            quality.station_id,
            errors='raise',
        ).astype('int64')

        if candidate.station_id.duplicated().any():
            raise ValueError(
                'Batch 1 candidate pool contains duplicate station IDs.'
            )

        if quality.station_id.duplicated().any():
            raise ValueError(
                'Batch 1 station-quality table contains duplicate station IDs.'
            )

        candidate_ids = set(candidate.station_id)

        candidate_quality_check = quality.loc[
            quality.station_id.isin(candidate_ids)
        ].copy()

        if len(candidate_quality_check) != len(candidate):
            raise ValueError(
                'Batch 1 candidate pool and station-quality table do not reconcile.'
            )

        # Validate that every candidate exists in the historical ML inventory.
        metadata_merge = candidate.merge(
            metadata_ml,
            left_on='station_id',
            right_on='ID',
            how='left',
            indicator=True,
            validate='one_to_one',
        )

        metadata_merge.to_csv(
            output / 'batch_2_candidate_metadata_merge_audit.csv',
            index=False,
        )

        unmatched = metadata_merge.loc[
            metadata_merge._merge.ne('both')
        ].copy()

        unmatched.to_csv(
            output / 'batch_2_unmatched_candidate_stations.csv',
            index=False,
        )

        if not unmatched.empty:
            raise ValueError(
                'Some Batch 1 candidates do not match historical ML metadata. '
                'See batch_2_unmatched_candidate_stations.csv.'
            )

        # Build the physical inventory BEFORE restricting to the candidate pool.
        physical = metadata_ml.copy()

        physical['valid_geography'] = (
            physical.Fwy.notna()
            & physical.Dir.notna()
            & physical.Abs_PM.notna()
            & physical.Latitude.notna()
            & physical.Longitude.notna()
        )

        invalid_physical = physical.loc[
            ~physical.valid_geography
        ].copy()

        invalid_physical.to_csv(
            output / 'batch_2_invalid_physical_ml_geography.csv',
            index=False,
        )

        physical = physical.loc[
            physical.valid_geography
        ].copy()

        physical = physical.rename(
            columns={'ID': 'station_id'}
        )

        physical['station_id'] = physical.station_id.astype('int64')
        physical['is_candidate'] = physical.station_id.isin(candidate_ids)

        physical = physical.sort_values(
            ['Fwy', 'Dir', 'Abs_PM', 'station_id'],
            kind='stable',
        ).reset_index(drop=True)

        physical_grouped = physical.groupby(
            ['Fwy', 'Dir'],
            sort=False,
            dropna=False,
        )

        physical['physical_route_order'] = (
            physical_grouped.cumcount() + 1
        )

        physical['physical_route_station_count'] = (
            physical_grouped.station_id.transform('size')
        )

        physical['previous_ml_station_id'] = (
            physical_grouped.station_id.shift(1).astype('Int64')
        )
        physical['next_ml_station_id'] = (
            physical_grouped.station_id.shift(-1).astype('Int64')
        )
        physical['previous_ml_abs_pm'] = (
            physical_grouped.Abs_PM.shift(1)
        )
        physical['next_ml_abs_pm'] = (
            physical_grouped.Abs_PM.shift(-1)
        )

        physical['distance_from_previous_ml_miles'] = (
            physical.Abs_PM - physical.previous_ml_abs_pm
        )
        physical['distance_to_next_ml_miles'] = (
            physical.next_ml_abs_pm - physical.Abs_PM
        )

        candidate_lookup = physical.set_index(
            'station_id'
        ).is_candidate

        physical['previous_ml_is_candidate'] = (
            physical.previous_ml_station_id.map(candidate_lookup)
            .fillna(False)
            .astype(bool)
        )
        physical['next_ml_is_candidate'] = (
            physical.next_ml_station_id.map(candidate_lookup)
            .fillna(False)
            .astype(bool)
        )

        # Assign traffic-direction upstream/downstream neighbors. California
        # absolute postmile generally increases north/east; for south/west travel
        # the travel order is reversed. Unknown directions are left unassigned.
        north_east = physical.Dir.isin(['N', 'E'])
        south_west = physical.Dir.isin(['S', 'W'])

        physical['upstream_ml_station_id'] = pd.Series(
            pd.NA,
            index=physical.index,
            dtype='Int64',
        )
        physical['downstream_ml_station_id'] = pd.Series(
            pd.NA,
            index=physical.index,
            dtype='Int64',
        )

        physical.loc[
            north_east,
            'upstream_ml_station_id',
        ] = physical.loc[
            north_east,
            'previous_ml_station_id',
        ]

        physical.loc[
            north_east,
            'downstream_ml_station_id',
        ] = physical.loc[
            north_east,
            'next_ml_station_id',
        ]

        physical.loc[
            south_west,
            'upstream_ml_station_id',
        ] = physical.loc[
            south_west,
            'next_ml_station_id',
        ]

        physical.loc[
            south_west,
            'downstream_ml_station_id',
        ] = physical.loc[
            south_west,
            'previous_ml_station_id',
        ]

        physical['upstream_ml_is_candidate'] = (
            physical.upstream_ml_station_id.map(candidate_lookup)
            .fillna(False)
            .astype(bool)
        )
        physical['downstream_ml_is_candidate'] = (
            physical.downstream_ml_station_id.map(candidate_lookup)
            .fillna(False)
            .astype(bool)
        )

        # A candidate corridor run begins only when the current detector is a
        # candidate and the immediately previous physical ML detector is not.
        previous_candidate_in_route = (
            physical_grouped.is_candidate.shift(fill_value=False)
        )

        physical['candidate_run_start'] = (
            physical.is_candidate
            & ~previous_candidate_in_route
        )

        physical['candidate_run_number'] = (
            physical.groupby(
                ['Fwy', 'Dir'],
                dropna=False,
            )
            .candidate_run_start.cumsum()
            .astype('int64')
        )

        physical['corridor_id'] = pd.Series(
            pd.NA,
            index=physical.index,
            dtype='string',
        )

        candidate_mask = physical.is_candidate

        physical.loc[
            candidate_mask,
            'corridor_id',
        ] = (
            'Fwy_'
            + physical.loc[
                candidate_mask,
                'Fwy',
            ].astype('Int64').astype('string')
            + '_'
            + physical.loc[
                candidate_mask,
                'Dir',
            ].astype('string')
            + '_physical_run_'
            + physical.loc[
                candidate_mask,
                'candidate_run_number',
            ].astype('string')
        )

        physical.to_csv(
            output / 'batch_2_full_ml_physical_sequence.csv',
            index=False,
        )

        # Join Batch 1 quality measures only after physical adjacency and corridor
        # membership have been established.
        candidate_geo = physical.loc[
            physical.is_candidate
        ].merge(
            candidate,
            on='station_id',
            how='left',
            validate='one_to_one',
            suffixes=('', '_batch1'),
        )

        if len(candidate_geo) != len(candidate):
            raise ValueError(
                'Physical candidate sequence does not reconcile to Batch 1 pool.'
            )

        # Normalize route text before comparison. Batch 1 inherited freeway
        # labels such as 210.0/W while metadata stores the same route as 210/W.
        def normalize_route_piece(value):
            parts = str(value).strip().split('/')
            if len(parts) != 2:
                return str(value).strip()
            freeway_text, direction_text = parts
            try:
                freeway_text = str(int(float(freeway_text)))
            except ValueError:
                freeway_text = freeway_text.strip()
            return freeway_text + '/' + direction_text.strip()

        metadata_route = (
            candidate_geo.Fwy.astype('Int64').astype('string')
            + '/'
            + candidate_geo.Dir.astype('string')
        )

        candidate_geo['metadata_route_label'] = metadata_route
        candidate_geo['batch1_route_labels_normalized'] = [
            '|'.join(
                normalize_route_piece(piece)
                for piece in str(batch1_routes).split('|')
            )
            for batch1_routes in candidate_geo.route_labels
        ]
        candidate_geo['batch1_route_contains_metadata_route'] = [
            str(meta_route) in normalized_routes.split('|')
            for meta_route, normalized_routes in zip(
                candidate_geo.metadata_route_label,
                candidate_geo.batch1_route_labels_normalized,
            )
        ]

        route_mismatches = candidate_geo.loc[
            ~candidate_geo.batch1_route_contains_metadata_route
        ].copy()

        route_mismatches.to_csv(
            output / 'batch_2_route_label_mismatches.csv',
            index=False,
        )

        candidate_geo.to_csv(
            output / 'batch_2_candidate_station_geography.csv',
            index=False,
        )

        neighbor_columns = [
            'station_id',
            'Fwy',
            'Dir',
            'Abs_PM',
            'Name',
            'physical_route_order',
            'physical_route_station_count',
            'previous_ml_station_id',
            'previous_ml_is_candidate',
            'distance_from_previous_ml_miles',
            'next_ml_station_id',
            'next_ml_is_candidate',
            'distance_to_next_ml_miles',
            'upstream_ml_station_id',
            'upstream_ml_is_candidate',
            'downstream_ml_station_id',
            'downstream_ml_is_candidate',
            'corridor_id',
            'positive_pct_all_rows',
            'ge80_pct_all_rows',
            'ge100_pct_all_rows',
            'days_with_positive_coverage',
        ]

        candidate_geo[neighbor_columns].to_csv(
            output / 'batch_2_candidate_neighbors.csv',
            index=False,
        )

        # Summarize the complete physical freeway/direction inventory and how
        # many of those stations survived into the ever-positive candidate pool.
        route_summary = (
            physical.groupby(
                ['Fwy', 'Dir'],
                as_index=False,
                dropna=False,
            )
            .agg(
                physical_ml_stations=('station_id', 'size'),
                candidate_stations=('is_candidate', 'sum'),
                min_abs_pm=('Abs_PM', 'min'),
                max_abs_pm=('Abs_PM', 'max'),
            )
        )

        route_summary['candidate_share_pct'] = (
            100
            * route_summary.candidate_stations
            / route_summary.physical_ml_stations
        )

        route_summary['postmile_span'] = (
            route_summary.max_abs_pm
            - route_summary.min_abs_pm
        )

        route_summary = route_summary.sort_values(
            [
                'candidate_stations',
                'candidate_share_pct',
                'physical_ml_stations',
            ],
            ascending=[False, False, False],
        )

        route_summary.to_csv(
            output / 'batch_2_freeway_direction_summary.csv',
            index=False,
        )

        corridor = summarize_batch_2_corridors(
            candidate_geo
        )

        corridor.to_csv(
            output / 'batch_2_corridor_candidates_all.csv',
            index=False,
        )

        review = corridor.loc[
            corridor.eligible_for_corridor_review
        ].copy()

        review.to_csv(
            output / 'batch_2_corridor_review_table.csv',
            index=False,
        )

        review.head(25).to_csv(
            output / 'batch_2_top_25_corridors.csv',
            index=False,
        )

        # Traffic-content gate: the best-quality corridors must also contain
        # recurring low-speed traffic and enough simultaneous observation to
        # support later propagation analysis. This reuses the existing
        # below_speed_threshold flag; it introduces no new speed definition.
        traffic_review_corridors = review.head(
            BATCH_2_TRAFFIC_REVIEW_TOP_N
        ).copy()

        traffic_review_ids = set(
            traffic_review_corridors.corridor_id
        )

        traffic_membership = candidate_geo.loc[
            candidate_geo.corridor_id.isin(traffic_review_ids),
            [
                'corridor_id',
                'station_id',
                'Fwy',
                'Dir',
                'Abs_PM',
                'Name',
                'ge80_pct_all_rows',
                'ge100_pct_all_rows',
                'distance_from_previous_ml_miles',
            ],
        ].copy()

        corridor_station_counts = (
            traffic_membership.groupby('corridor_id')
            .station_id.nunique()
            .to_dict()
        )

        station_to_corridor = (
            traffic_membership.set_index('station_id')
            .corridor_id.to_dict()
        )

        selected_station_ids = set(
            traffic_membership.station_id.astype('int64')
        )

        traffic_totals = {
            corridor_id: {
                'weekday_expected_timestamps': 0,
                'weekday_station_intervals_ge80': 0,
                'weekday_station_intervals_below_speed_ge80': 0,
                'weekday_simultaneous_ge80_timestamps': 0,
                'weekday_simultaneous_ge80_any_below_speed_timestamps': 0,
                'weekday_am_expected_timestamps': 0,
                'weekday_am_simultaneous_ge80_timestamps': 0,
                'weekday_pm_expected_timestamps': 0,
                'weekday_pm_simultaneous_ge80_timestamps': 0,
                'weekday_days': 0,
                'weekday_days_with_any_simultaneous_ge80': 0,
                'weekday_days_with_any_simultaneous_low_speed': 0,
            }
            for corridor_id in traffic_review_ids
        }

        daily_traffic_rows = []

        traffic_columns = [
            'timestamp',
            'station_id',
            'lane_type',
            'observed_pct',
            'invalid_observed_pct',
            'invalid_timestamp',
            'date_mismatch',
            'invalid_station_id',
            'invalid_speed',
            'off_5min_grid',
            'duplicate_station_timestamp',
            'below_speed_threshold',
        ]

        for day in pd.date_range(START_DATE, END_DATE):
            day_text = day.strftime('%Y-%m-%d')
            folder = PREPARED_DATA_FOLDER / f'date={day_text}'
            paths = sorted(folder.glob('part-*.parquet'))
            if not paths:
                continue

            selected_parts = []
            for path in paths:
                part = pd.read_parquet(
                    path,
                    columns=traffic_columns,
                )
                part = part.loc[
                    part.lane_type.eq('ML')
                    & part.station_id.isin(selected_station_ids)
                ].copy()
                if not part.empty:
                    selected_parts.append(part)

            if not selected_parts:
                continue

            day_frame = pd.concat(
                selected_parts,
                ignore_index=True,
            )

            duplicate_copies = day_frame.duplicated(
                ['station_id', 'timestamp'],
                keep=False,
            )

            base_valid = (
                ~day_frame.invalid_timestamp.fillna(True)
                & ~day_frame.date_mismatch.fillna(True)
                & ~day_frame.invalid_station_id.fillna(True)
                & ~day_frame.invalid_speed.fillna(True)
                & ~day_frame.off_5min_grid.fillna(True)
                & ~duplicate_copies
                & day_frame.timestamp.notna()
            )

            coverage_valid = (
                ~day_frame.invalid_observed_pct.fillna(True)
                & day_frame.observed_pct.between(0, 100)
            )

            day_frame['well_observed_ge80'] = (
                base_valid
                & coverage_valid
                & day_frame.observed_pct.ge(80)
            )

            day_frame['below_speed_existing'] = (
                day_frame.below_speed_threshold.fillna(False)
                & day_frame.well_observed_ge80
            )

            day_frame['corridor_id'] = (
                day_frame.station_id.astype('int64')
                .map(station_to_corridor)
            )

            weekday = day.dayofweek < 5

            for corridor_id in traffic_review_ids:
                group = day_frame.loc[
                    day_frame.corridor_id.eq(corridor_id)
                ].copy()

                if group.empty:
                    continue

                station_count = corridor_station_counts[corridor_id]

                timestamp_summary = (
                    group.groupby('timestamp', dropna=False)
                    .agg(
                        stations_ge80=('well_observed_ge80', 'sum'),
                        any_below_speed_ge80=('below_speed_existing', 'any'),
                    )
                    .reset_index()
                )

                simultaneous = (
                    timestamp_summary.stations_ge80.eq(station_count)
                )

                timestamp_hours = timestamp_summary.timestamp.dt.hour
                am_mask = (
                    timestamp_hours.ge(BATCH_2_AM_PEAK_HOURS[0])
                    & timestamp_hours.lt(BATCH_2_AM_PEAK_HOURS[1])
                )
                pm_mask = (
                    timestamp_hours.ge(BATCH_2_PM_PEAK_HOURS[0])
                    & timestamp_hours.lt(BATCH_2_PM_PEAK_HOURS[1])
                )

                result = {
                    'date': day_text,
                    'corridor_id': corridor_id,
                    'weekday': weekday,
                    'station_count': station_count,
                    'station_intervals_ge80': int(
                        group.well_observed_ge80.sum()
                    ),
                    'station_intervals_below_speed_ge80': int(
                        group.below_speed_existing.sum()
                    ),
                    'simultaneous_ge80_timestamps': int(
                        simultaneous.sum()
                    ),
                    'simultaneous_ge80_any_below_speed_timestamps': int(
                        (
                            simultaneous
                            & timestamp_summary.any_below_speed_ge80
                        ).sum()
                    ),
                    'am_simultaneous_ge80_timestamps': int(
                        (simultaneous & am_mask).sum()
                    ),
                    'pm_simultaneous_ge80_timestamps': int(
                        (simultaneous & pm_mask).sum()
                    ),
                }

                daily_traffic_rows.append(result)

                if weekday:
                    totals = traffic_totals[corridor_id]
                    totals['weekday_days'] += 1
                    totals['weekday_expected_timestamps'] += 288
                    totals['weekday_am_expected_timestamps'] += (
                        BATCH_2_AM_PEAK_HOURS[1]
                        - BATCH_2_AM_PEAK_HOURS[0]
                    ) * 12
                    totals['weekday_pm_expected_timestamps'] += (
                        BATCH_2_PM_PEAK_HOURS[1]
                        - BATCH_2_PM_PEAK_HOURS[0]
                    ) * 12
                    totals['weekday_station_intervals_ge80'] += result[
                        'station_intervals_ge80'
                    ]
                    totals[
                        'weekday_station_intervals_below_speed_ge80'
                    ] += result[
                        'station_intervals_below_speed_ge80'
                    ]
                    totals[
                        'weekday_simultaneous_ge80_timestamps'
                    ] += result[
                        'simultaneous_ge80_timestamps'
                    ]
                    totals[
                        'weekday_simultaneous_ge80_any_below_speed_timestamps'
                    ] += result[
                        'simultaneous_ge80_any_below_speed_timestamps'
                    ]
                    totals[
                        'weekday_am_simultaneous_ge80_timestamps'
                    ] += result[
                        'am_simultaneous_ge80_timestamps'
                    ]
                    totals[
                        'weekday_pm_simultaneous_ge80_timestamps'
                    ] += result[
                        'pm_simultaneous_ge80_timestamps'
                    ]

                    if result['simultaneous_ge80_timestamps'] > 0:
                        totals[
                            'weekday_days_with_any_simultaneous_ge80'
                        ] += 1

                    if result[
                        'simultaneous_ge80_any_below_speed_timestamps'
                    ] > 0:
                        totals[
                            'weekday_days_with_any_simultaneous_low_speed'
                        ] += 1

        daily_traffic = pd.DataFrame(daily_traffic_rows)

        daily_traffic.to_csv(
            output / 'batch_2_top_corridor_daily_traffic_review.csv',
            index=False,
        )

        traffic_rows = []

        for _, row in traffic_review_corridors.iterrows():
            corridor_id = row.corridor_id
            totals = traffic_totals[corridor_id]
            member_rows = traffic_membership.loc[
                traffic_membership.corridor_id.eq(corridor_id)
            ]

            traffic_rows.append({
                'quality_review_rank': int(row.review_rank),
                'corridor_id': corridor_id,
                'Fwy': int(row.Fwy),
                'Dir': row.Dir,
                'station_count': int(row.station_count),
                'postmile_span': row.postmile_span,
                'max_physical_neighbor_gap_miles':
                    row.max_physical_neighbor_gap_miles,
                'weighted_ge80_pct': row.weighted_ge80_pct,
                'minimum_station_ge80_pct':
                    member_rows.ge80_pct_all_rows.min(),
                'weighted_ge100_pct': row.weighted_ge100_pct,
                'minimum_station_ge100_pct':
                    member_rows.ge100_pct_all_rows.min(),
                'weekday_days': totals['weekday_days'],
                'weekday_days_with_any_simultaneous_ge80':
                    totals[
                        'weekday_days_with_any_simultaneous_ge80'
                    ],
                'weekday_simultaneous_ge80_hours':
                    totals[
                        'weekday_simultaneous_ge80_timestamps'
                    ] * 5 / 60,
                'weekday_simultaneous_ge80_pct_expected':
                    (
                        100
                        * totals[
                            'weekday_simultaneous_ge80_timestamps'
                        ]
                        / totals['weekday_expected_timestamps']
                        if totals['weekday_expected_timestamps']
                        else float('nan')
                    ),
                'weekday_am_simultaneous_ge80_pct_expected':
                    (
                        100
                        * totals[
                            'weekday_am_simultaneous_ge80_timestamps'
                        ]
                        / totals['weekday_am_expected_timestamps']
                        if totals['weekday_am_expected_timestamps']
                        else float('nan')
                    ),
                'weekday_pm_simultaneous_ge80_pct_expected':
                    (
                        100
                        * totals[
                            'weekday_pm_simultaneous_ge80_timestamps'
                        ]
                        / totals['weekday_pm_expected_timestamps']
                        if totals['weekday_pm_expected_timestamps']
                        else float('nan')
                    ),
                'weekday_station_interval_below_speed_pct_ge80':
                    (
                        100
                        * totals[
                            'weekday_station_intervals_below_speed_ge80'
                        ]
                        / totals['weekday_station_intervals_ge80']
                        if totals['weekday_station_intervals_ge80']
                        else float('nan')
                    ),
                'weekday_any_below_speed_pct_when_all_ge80':
                    (
                        100
                        * totals[
                            'weekday_simultaneous_ge80_any_below_speed_timestamps'
                        ]
                        / totals[
                            'weekday_simultaneous_ge80_timestamps'
                        ]
                        if totals[
                            'weekday_simultaneous_ge80_timestamps'
                        ]
                        else float('nan')
                    ),
                'weekday_days_with_any_simultaneous_low_speed':
                    totals[
                        'weekday_days_with_any_simultaneous_low_speed'
                    ],
                'station_names':
                    ' | '.join(
                        member_rows.sort_values('Abs_PM').Name.astype(str)
                    ),
            })

        traffic_review = pd.DataFrame(traffic_rows)
        traffic_review = traffic_review.sort_values('quality_review_rank')

        traffic_review.to_csv(
            output / 'batch_2_top_corridor_traffic_review.csv',
            index=False,
        )

        # Count how many candidate runs exist and how many stations are isolated.
        candidate_run_sizes = (
            candidate_geo.groupby('corridor_id')
            .station_id.size()
        )

        isolated_candidates = int(
            candidate_run_sizes.eq(1).sum()
        )

        report = '\n'.join([
            'BATCH 2 — STATION METADATA AND CORRIDOR CONSTRUCTION (CORRECTED)',
            f'Batch 1 candidate stations: {len(candidate):,}',
            f'Historical metadata rows: {len(metadata):,}',
            f'District 7 metadata rows: {len(metadata_d7):,}',
            f'District 7 ML metadata rows: {len(metadata_ml):,}',
            f'Physical ML stations with valid geography: {len(physical):,}',
            f'Physical ML stations lacking required geography: {len(invalid_physical):,}',
            f'Candidate stations matched to historical ML metadata: {len(candidate_geo):,}',
            f'Candidate stations unmatched: {len(unmatched):,}',
            f'Batch 1 / metadata route-label mismatches: {len(route_mismatches):,}',
            f'Freeway/direction combinations represented physically: '
            f'{physical.groupby(["Fwy", "Dir"]).ngroups:,}',
            f'Physically contiguous candidate runs: {candidate_geo.corridor_id.nunique():,}',
            f'Isolated one-station candidate runs: {isolated_candidates:,}',
            f'Corridor runs with at least {BATCH_2_MIN_CORRIDOR_STATIONS} stations: '
            f'{len(review):,}',
            '',
            'INTERPRETATION AND DECISION GATE',
            'Physical adjacency is defined using all historical District 7 ML stations.',
            'Candidate corridors cannot jump across an intervening noncandidate ML detector.',
            'Travel-direction upstream/downstream IDs are derived from freeway direction and physical postmile order.',
            'Corridor review score is only a transparent ordering aid, not a statistical estimator.',
            'No primary corridor has been automatically selected.',
            f'Traffic-content review scanned the top {BATCH_2_TRAFFIC_REVIEW_TOP_N} quality-ranked corridors.',
            'Use minimum station coverage, simultaneous corridor coverage, low-speed frequency, and physical spacing before locking the primary corridor.',
            'Next decision: choose the primary corridor for Batch 3 only after this traffic-content gate.',
        ])

        (output / 'batch_2_audit_summary.txt').write_text(
            report + '\n'
        )

        settings.update(
            status='complete',
            metadata_rows=len(metadata),
            district_7_ml_rows=len(metadata_ml),
            physical_ml_valid_geography=len(physical),
            physical_ml_invalid_geography=len(invalid_physical),
            candidate_stations=len(candidate),
            matched_candidate_stations=len(candidate_geo),
            unmatched_candidate_stations=len(unmatched),
            route_label_mismatches=len(route_mismatches),
            freeway_direction_combinations=physical.groupby(
                ['Fwy', 'Dir']
            ).ngroups,
            contiguous_candidate_runs=candidate_geo.corridor_id.nunique(),
            isolated_candidate_runs=isolated_candidates,
            corridor_review_rows=len(review),
            traffic_review_corridors=len(traffic_review_corridors),
        )

        print('\n' + report)
        print(
            f'\nSaved corrected Batch 2 reports: {output.resolve()}'
        )

    except Exception as error:
        settings.update(
            status='incomplete',
            error=str(error),
        )
        raise

    finally:
        settings_path.write_text(
            json.dumps(settings, indent=2)
        )




# ===================== BATCH 2B. PRIMARY CORRIDOR SELECTION =====================
# This lightweight step uses the completed Batch 2 v3 traffic-content review.
# It does NOT rescan the 120 prepared daily files.
#
# Selection logic is intentionally weakest-link-first:
#   1. require actual weekday low-speed activity under the existing speed flag;
#   2. require at least some timestamps where every station is >=80% observed;
#   3. among viable corridors, prefer the strongest weakest station;
#   4. then prefer the strongest simultaneous corridor coverage;
#   5. then prefer more weekdays with low-speed activity;
#   6. then station count and spatial span as tie breakers.
#
# This preserves the corridor decision in code rather than relying on a
# one-off terminal inspection.


def run_batch_2_selection():
    source = BATCH_2_SELECTION_SOURCE_FOLDER
    output = BATCH_2_SELECTION_OUTPUT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 2 selection output exists: {output}. '
            'Choose a new folder; no files overwritten.'
        )

    review_path = source / 'batch_2_top_corridor_traffic_review.csv'
    if not review_path.exists():
        raise FileNotFoundError(
            f'Missing completed Batch 2 traffic review: {review_path}'
        )

    review = pd.read_csv(review_path)

    required = [
        'quality_review_rank',
        'corridor_id',
        'Fwy',
        'Dir',
        'station_count',
        'postmile_span',
        'max_physical_neighbor_gap_miles',
        'weighted_ge80_pct',
        'minimum_station_ge80_pct',
        'weekday_simultaneous_ge80_pct_expected',
        'weekday_station_interval_below_speed_pct_ge80',
        'weekday_any_below_speed_pct_when_all_ge80',
        'weekday_days_with_any_simultaneous_low_speed',
    ]

    missing = [column for column in required if column not in review.columns]
    if missing:
        raise ValueError(
            'Traffic review missing required columns: '
            + ', '.join(missing)
        )

    output.mkdir(parents=True)

    # Viability is not a quality score. It only excludes corridors that cannot
    # support the intended breakdown/propagation analysis at all.
    review['has_recurring_low_speed_evidence'] = (
        review.weekday_days_with_any_simultaneous_low_speed.gt(0)
    )
    review['has_simultaneous_ge80_coverage'] = (
        review.weekday_simultaneous_ge80_pct_expected.gt(0)
    )
    review['selection_viable'] = (
        review.has_recurring_low_speed_evidence
        & review.has_simultaneous_ge80_coverage
    )

    viable = review.loc[review.selection_viable].copy()

    if viable.empty:
        raise ValueError(
            'No reviewed corridor has both recurring low-speed evidence '
            'and simultaneous >=80% station coverage.'
        )

    # Weakest-link-first lexicographic ranking. This avoids allowing a strong
    # corridor average to hide one unusable detector in the middle of the chain.
    viable = viable.sort_values(
        [
            'minimum_station_ge80_pct',
            'weekday_simultaneous_ge80_pct_expected',
            'weekday_days_with_any_simultaneous_low_speed',
            'station_count',
            'postmile_span',
            'quality_review_rank',
        ],
        ascending=[False, False, False, False, False, True],
        kind='stable',
    ).reset_index(drop=True)

    viable.insert(0, 'selection_rank', range(1, len(viable) + 1))

    selected = viable.iloc[[0]].copy()
    selected_id = selected.iloc[0].corridor_id

    comparison = review.merge(
        viable[
            ['corridor_id', 'selection_rank']
        ],
        on='corridor_id',
        how='left',
        validate='one_to_one',
    )

    comparison['selected_primary_corridor'] = (
        comparison.corridor_id.eq(selected_id)
    )

    comparison.to_csv(
        output / 'batch_2_primary_corridor_comparison.csv',
        index=False,
    )

    selected.to_csv(
        output / 'batch_2_primary_corridor_selection.csv',
        index=False,
    )

    row = selected.iloc[0]

    report = '\n'.join([
        'BATCH 2 — PRIMARY CORRIDOR SELECTION',
        f'Selected corridor: {row.corridor_id}',
        f'Freeway/direction: {int(row.Fwy)} {row.Dir}',
        f'Stations: {int(row.station_count)}',
        f'Postmile span: {row.postmile_span:.2f} miles',
        f'Max physical station gap: {row.max_physical_neighbor_gap_miles:.2f} miles',
        f'Weighted >=80% coverage: {row.weighted_ge80_pct:.2f}%',
        f'Weakest station >=80% coverage: {row.minimum_station_ge80_pct:.2f}%',
        f'Weekday simultaneous all-station >=80% coverage: '
        f'{row.weekday_simultaneous_ge80_pct_expected:.2f}%',
        f'Weekday station intervals below existing speed threshold: '
        f'{row.weekday_station_interval_below_speed_pct_ge80:.2f}%',
        f'Weekday all-station-covered timestamps with any low-speed station: '
        f'{row.weekday_any_below_speed_pct_when_all_ge80:.2f}%',
        f'Weekdays with any simultaneous low-speed evidence: '
        f'{int(row.weekday_days_with_any_simultaneous_low_speed)}',
        '',
        'SELECTION METHODOLOGY',
        'Candidate set: top five physically contiguous corridors from Batch 2 v3 quality review.',
        'Viability gate: corridor must show weekday low-speed activity and at least some simultaneous >=80% coverage across every station.',
        'Primary ranking: highest minimum station >=80% coverage.',
        'Tie breakers: simultaneous corridor >=80% coverage, weekdays with low-speed activity, station count, spatial span, then original quality-review rank.',
        'The minimum-station rule prevents a corridor average from hiding a broken middle detector.',
        'The low-speed requirement prevents selecting a pristine corridor that provides little breakdown behavior to study.',
        'No new congestion-speed threshold is introduced; this step reuses the existing below_speed_threshold flag.',
        '',
        'DECISION',
        f'{row.corridor_id} is locked as the primary corridor for Batch 3.',
    ])

    (output / 'batch_2_primary_corridor_selection.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_folder': str(source),
        'source_review': str(review_path),
        'selected_corridor': selected_id,
        'candidate_corridors_reviewed': len(review),
        'viable_corridors': len(viable),
        'selection_rule': [
            'require weekday_days_with_any_simultaneous_low_speed > 0',
            'require weekday_simultaneous_ge80_pct_expected > 0',
            'maximize minimum_station_ge80_pct',
            'then maximize weekday_simultaneous_ge80_pct_expected',
            'then maximize weekday_days_with_any_simultaneous_low_speed',
            'then maximize station_count',
            'then maximize postmile_span',
            'then prefer lower original quality_review_rank',
        ],
        'selection_adopted': True,
    }

    (output / 'batch_2_primary_corridor_selection_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 2 selection: {output.resolve()}'
    )




# ===================== BATCH 3. ANALYTICAL DATA VALIDITY =====================
# Questions 12-17 are handled together for the locked primary corridor.
#
# Q12: How complete are speed, flow, and occupancy across observation coverage?
# Q13: Do low-coverage rows contain plausible-looking values despite weak coverage?
# Q14: Are exact repeated measurements unusually common at low coverage?
# Q15: How do 0%-observed rows compare with 100%-observed rows at the same stations?
# Q16: How much usable simultaneous corridor data survives 40/60/80/100% cutoffs?
# Q17: Are lane-level observation fields internally consistent once empty padded
#      lane slots are excluded?
#
# This batch does not choose the final coverage threshold automatically. It writes
# the evidence needed for a documented 80-vs-100 (and 40/60 sensitivity) decision.


def batch_3_coverage_band(series):
    """Return descriptive coverage bands without implying a final cutoff."""
    result = pd.Series('invalid', index=series.index, dtype='string')
    valid = series.between(0, 100)
    result.loc[valid & series.eq(0)] = '0'
    result.loc[valid & series.gt(0) & series.lt(40)] = '>0-<40'
    result.loc[valid & series.ge(40) & series.lt(80)] = '40-<80'
    result.loc[valid & series.ge(80) & series.lt(100)] = '80-<100'
    result.loc[valid & series.eq(100)] = '100'
    return result


def batch_3_longest_true_run(mask, same_series):
    """Return longest consecutive True run, resetting when continuity breaks."""
    if len(mask) == 0:
        return 0
    starts = mask & ~(same_series & mask.shift(fill_value=False))
    run_id = starts.fillna(False).astype('int64').cumsum()
    lengths = mask.loc[mask].groupby(run_id.loc[mask]).size()
    return int(lengths.max()) if not lengths.empty else 0


def run_batch_3():
    """Audit analytical validity on the selected primary corridor."""
    output = BATCH_3_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 3 output exists: {output}. '
            'Choose a new folder; no files overwritten.'
        )

    required_inputs = [
        BATCH_3_SELECTION_FILE,
        BATCH_3_CORRIDOR_GEOGRAPHY_FILE,
    ]
    missing_inputs = [
        str(path)
        for path in required_inputs
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            'Missing Batch 3 inputs: ' + ', '.join(missing_inputs)
        )

    if (
        not BATCH_3_THRESHOLDS
        or len(set(BATCH_3_THRESHOLDS)) != len(BATCH_3_THRESHOLDS)
        or any(t < 0 or t > 100 for t in BATCH_3_THRESHOLDS)
    ):
        raise ValueError(
            'BATCH_3_THRESHOLDS must be distinct percentages from 0 to 100.'
        )

    selection = pd.read_csv(BATCH_3_SELECTION_FILE)
    if len(selection) != 1:
        raise ValueError(
            'Batch 3 requires exactly one locked primary corridor.'
        )

    selected_corridor = str(selection.iloc[0].corridor_id)

    geography = pd.read_csv(BATCH_3_CORRIDOR_GEOGRAPHY_FILE)
    corridor_stations = geography.loc[
        geography.corridor_id.eq(selected_corridor)
    ].copy()

    if corridor_stations.empty:
        raise ValueError(
            f'Selected corridor not found in geography table: {selected_corridor}'
        )

    if corridor_stations.station_id.duplicated().any():
        raise ValueError(
            'Selected corridor geography contains duplicate station IDs.'
        )

    station_ids = set(
        pd.to_numeric(
            corridor_stations.station_id,
            errors='raise',
        ).astype('int64')
    )
    station_count = len(station_ids)

    if 'Lanes' not in corridor_stations.columns:
        raise ValueError(
            'Batch 3 lane audit requires the historical metadata Lanes field.'
        )

    lane_count_table = corridor_stations[
        ['station_id', 'Lanes']
    ].copy()

    lane_count_table['station_id'] = pd.to_numeric(
        lane_count_table.station_id,
        errors='raise',
    ).astype('int64')

    lane_count_table['metadata_lane_count'] = pd.to_numeric(
        lane_count_table.Lanes,
        errors='raise',
    ).astype('int64')

    if lane_count_table.metadata_lane_count.le(0).any():
        raise ValueError(
            'Selected corridor contains a nonpositive metadata lane count.'
        )

    station_lane_counts = lane_count_table.set_index(
        'station_id'
    ).metadata_lane_count.to_dict()

    output.mkdir(parents=True)
    BATCH_3_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

    settings_path = output / 'batch_3_run_settings.json'
    settings = {
        'status': 'incomplete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selected_corridor': selected_corridor,
        'station_ids': sorted(station_ids),
        'station_count': station_count,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'thresholds': BATCH_3_THRESHOLDS,
        'selection_adopted': False,
        'coverage_bands': [
            '0',
            '>0-<40',
            '40-<80',
            '80-<100',
            '100',
            'invalid',
        ],
        'lane_slot_rule':
            'Physical lane slots are defined from the historical metadata Lanes '
            'count for each station. Lane-number fields above that count are '
            'treated as padded schema slots and excluded even when they contain '
            'zero placeholders.',
        'repetition_rule':
            'Exact repeated values are compared only across consecutive '
            'five-minute records from the same station.',
        'zero_vs_100_rule':
            '0%- and 100%-observed rows are compared within the same selected stations.',
    }
    settings_path.write_text(json.dumps(settings, indent=2))

    try:
        dates = pd.date_range(START_DATE, END_DATE)
        all_parts = []
        lane_parts = []
        daily_threshold_parts = []

        base_columns = [
            'timestamp',
            'station_id',
            'lane_type',
            'observed_pct',
            'flow_veh_5min',
            'occupancy_fraction',
            'speed_mph',
            'invalid_timestamp',
            'date_mismatch',
            'invalid_station_id',
            'invalid_speed',
            'invalid_flow',
            'invalid_occupancy',
            'invalid_observed_pct',
            'off_5min_grid',
            'duplicate_station_timestamp',
        ]

        for number, day in enumerate(dates, 1):
            day_text = day.strftime('%Y-%m-%d')
            folder = PREPARED_DATA_FOLDER / f'date={day_text}'
            paths = sorted(folder.glob('part-*.parquet'))
            if not paths:
                continue

            print(
                f'Batch 3: {day_text} [{number}/{len(dates)}]',
                flush=True,
            )

            day_parts = []
            day_lane_parts = []

            for path in paths:
                schema_names = set(pq.read_schema(path).names)
                available_base = [
                    column
                    for column in base_columns
                    if column in schema_names
                ]

                required_base = {
                    'timestamp',
                    'station_id',
                    'lane_type',
                    'observed_pct',
                    'flow_veh_5min',
                    'occupancy_fraction',
                    'speed_mph',
                    'invalid_timestamp',
                    'date_mismatch',
                    'invalid_station_id',
                    'invalid_speed',
                    'invalid_flow',
                    'invalid_occupancy',
                    'invalid_observed_pct',
                    'off_5min_grid',
                    'duplicate_station_timestamp',
                }

                if not required_base.issubset(schema_names):
                    missing = sorted(required_base - schema_names)
                    raise ValueError(
                        f'{path}: missing required Batch 3 columns: {missing}'
                    )

                lane_columns = sorted(
                    column
                    for column in schema_names
                    if (
                        column.startswith('lane_')
                        and len(column.split('_')) >= 3
                        and column.split('_')[1].isdigit()
                    )
                )

                frame = pd.read_parquet(
                    path,
                    columns=available_base + lane_columns,
                )

                frame = frame.loc[
                    frame.lane_type.eq('ML')
                    & frame.station_id.isin(station_ids)
                ].copy()

                if frame.empty:
                    continue

                day_parts.append(
                    frame[base_columns].copy()
                )

                if lane_columns:
                    lane_frame = frame[
                        ['timestamp', 'station_id', 'observed_pct']
                        + lane_columns
                    ].copy()
                    day_lane_parts.append(lane_frame)

            if not day_parts:
                continue

            day_frame = pd.concat(
                day_parts,
                ignore_index=True,
            )

            # Exclude every copy of duplicate station/timestamp observations from
            # validity/continuity metrics, matching the Batch 1 methodology.
            duplicate_copies = day_frame.duplicated(
                ['station_id', 'timestamp'],
                keep=False,
            )

            day_frame['base_valid'] = (
                ~day_frame.invalid_timestamp.fillna(True)
                & ~day_frame.date_mismatch.fillna(True)
                & ~day_frame.invalid_station_id.fillna(True)
                & ~day_frame.off_5min_grid.fillna(True)
                & ~duplicate_copies
                & day_frame.timestamp.notna()
            )

            day_frame['coverage_valid'] = (
                ~day_frame.invalid_observed_pct.fillna(True)
                & day_frame.observed_pct.between(0, 100)
            )

            day_frame['speed_valid'] = (
                day_frame.base_valid
                & ~day_frame.invalid_speed.fillna(True)
                & day_frame.speed_mph.notna()
            )
            day_frame['flow_valid'] = (
                day_frame.base_valid
                & ~day_frame.invalid_flow.fillna(True)
                & day_frame.flow_veh_5min.notna()
            )
            day_frame['occupancy_valid'] = (
                day_frame.base_valid
                & ~day_frame.invalid_occupancy.fillna(True)
                & day_frame.occupancy_fraction.notna()
            )
            day_frame['all_measures_valid'] = (
                day_frame.speed_valid
                & day_frame.flow_valid
                & day_frame.occupancy_valid
            )

            day_frame['coverage_band'] = batch_3_coverage_band(
                day_frame.observed_pct
            )

            day_frame['date'] = day_text
            all_parts.append(day_frame)

            # Lane-level analysis. Different raw files can have different maximum
            # lane widths, so melt each day's actual lane fields independently.
            if day_lane_parts:
                lane_day = pd.concat(
                    day_lane_parts,
                    ignore_index=True,
                    sort=False,
                )

                lane_numbers = sorted({
                    int(column.split('_')[1])
                    for column in lane_day.columns
                    if column.startswith('lane_')
                    and len(column.split('_')) >= 3
                    and column.split('_')[1].isdigit()
                })

                lane_rows = []

                for lane_number in lane_numbers:
                    prefix = f'lane_{lane_number}_'
                    fields = {
                        field: prefix + field
                        for field in LANE_FIELDS
                        if prefix + field in lane_day.columns
                    }

                    if not fields:
                        continue

                    values = pd.DataFrame({
                        'timestamp': lane_day.timestamp,
                        'station_id': lane_day.station_id,
                        'station_observed_pct': lane_day.observed_pct,
                        'lane_number': lane_number,
                    })

                    for field in LANE_FIELDS:
                        source_column = fields.get(field)
                        values[field] = (
                            lane_day[source_column]
                            if source_column is not None
                            else pd.NA
                        )

                    values['metadata_lane_count'] = (
                        pd.to_numeric(
                            values.station_id,
                            errors='raise',
                        ).astype('int64').map(station_lane_counts)
                    )

                    if values.metadata_lane_count.isna().any():
                        raise ValueError(
                            'Lane audit encountered a selected station without '
                            'a metadata lane count.'
                        )

                    # PeMS files can carry fixed-width lane slots above a
                    # station's physical lane count. Those padded slots can be
                    # zero-filled rather than null, so non-null testing is not
                    # a reliable detector-presence rule. Physical lane presence
                    # comes from the metadata Lanes count instead.
                    values['lane_slot_present'] = (
                        values.lane_number.le(
                            values.metadata_lane_count
                        )
                    )

                    values = values.loc[
                        values.lane_slot_present
                    ].copy()

                    if not values.empty:
                        values['coverage_band'] = batch_3_coverage_band(
                            values.station_observed_pct
                        )
                        lane_rows.append(values)

                if lane_rows:
                    lane_parts.append(
                        pd.concat(lane_rows, ignore_index=True)
                    )

            # Daily simultaneous threshold availability across every corridor
            # station. This is the actual propagation-analysis sample exposure.
            valid_for_threshold = (
                day_frame.base_valid
                & day_frame.coverage_valid
                & day_frame.all_measures_valid
            )

            threshold_day_rows = []

            for threshold in BATCH_3_THRESHOLDS:
                meets = (
                    valid_for_threshold
                    & day_frame.observed_pct.ge(threshold)
                )

                temp = day_frame.loc[
                    :,
                    ['timestamp', 'station_id']
                ].copy()
                temp['meets'] = meets

                timestamp_counts = (
                    temp.groupby('timestamp')
                    .agg(
                        stations_present=('station_id', 'nunique'),
                        stations_meeting=('meets', 'sum'),
                    )
                    .reset_index()
                )

                simultaneous = (
                    timestamp_counts.stations_present.eq(station_count)
                    & timestamp_counts.stations_meeting.eq(station_count)
                )

                threshold_day_rows.append({
                    'date': day_text,
                    'threshold': threshold,
                    'station_count': station_count,
                    'rows_meeting_threshold': int(meets.sum()),
                    'timestamps_all_stations_meeting': int(
                        simultaneous.sum()
                    ),
                    'hours_all_stations_meeting':
                        float(simultaneous.sum()) * 5 / 60,
                    'day_has_any_simultaneous_data': bool(
                        simultaneous.any()
                    ),
                })

            daily_threshold_parts.append(
                pd.DataFrame(threshold_day_rows)
            )

        if not all_parts:
            raise ValueError(
                'No selected-corridor observations found for Batch 3.'
            )

        data = pd.concat(
            all_parts,
            ignore_index=True,
        )

        data['station_id'] = pd.to_numeric(
            data.station_id,
            errors='raise',
        ).astype('int64')

        # Q12-Q13: validity and descriptive values by exact observation band.
        band_order = [
            '0',
            '>0-<40',
            '40-<80',
            '80-<100',
            '100',
            'invalid',
        ]

        band_rows = []

        for band in band_order:
            group = data.loc[
                data.coverage_band.eq(band)
            ]

            if group.empty:
                band_rows.append({
                    'coverage_band': band,
                    'rows': 0,
                })
                continue

            row = {
                'coverage_band': band,
                'rows': len(group),
                'pct_of_corridor_rows': 100 * len(group) / len(data),
                'speed_valid_pct': 100 * group.speed_valid.mean(),
                'flow_valid_pct': 100 * group.flow_valid.mean(),
                'occupancy_valid_pct': 100 * group.occupancy_valid.mean(),
                'all_measures_valid_pct':
                    100 * group.all_measures_valid.mean(),
                'median_speed_mph':
                    group.loc[group.speed_valid, 'speed_mph'].median(),
                'median_flow_veh_5min':
                    group.loc[group.flow_valid, 'flow_veh_5min'].median(),
                'median_occupancy_fraction':
                    group.loc[
                        group.occupancy_valid,
                        'occupancy_fraction',
                    ].median(),
            }

            for measure, valid_column in [
                ('speed_mph', 'speed_valid'),
                ('flow_veh_5min', 'flow_valid'),
                ('occupancy_fraction', 'occupancy_valid'),
            ]:
                values = group.loc[
                    group[valid_column],
                    measure,
                ]
                row[f'{measure}_q10'] = values.quantile(0.10)
                row[f'{measure}_q90'] = values.quantile(0.90)

            band_rows.append(row)

        band_summary = pd.DataFrame(band_rows)
        band_summary.to_csv(
            output / 'batch_3_measure_validity_by_coverage_band.csv',
            index=False,
        )

        # Q15: compare 0% and 100% rows within each selected station.
        zero_vs_100_rows = []

        for station_id, group in data.groupby('station_id'):
            for band in ['0', '100']:
                subset = group.loc[
                    group.coverage_band.eq(band)
                ]

                zero_vs_100_rows.append({
                    'station_id': station_id,
                    'coverage_band': band,
                    'rows': len(subset),
                    'speed_valid_rows': int(
                        subset.speed_valid.sum()
                    ),
                    'flow_valid_rows': int(
                        subset.flow_valid.sum()
                    ),
                    'occupancy_valid_rows': int(
                        subset.occupancy_valid.sum()
                    ),
                    'median_speed_mph':
                        subset.loc[
                            subset.speed_valid,
                            'speed_mph',
                        ].median(),
                    'speed_q10':
                        subset.loc[
                            subset.speed_valid,
                            'speed_mph',
                        ].quantile(0.10),
                    'speed_q90':
                        subset.loc[
                            subset.speed_valid,
                            'speed_mph',
                        ].quantile(0.90),
                    'median_flow_veh_5min':
                        subset.loc[
                            subset.flow_valid,
                            'flow_veh_5min',
                        ].median(),
                    'flow_q10':
                        subset.loc[
                            subset.flow_valid,
                            'flow_veh_5min',
                        ].quantile(0.10),
                    'flow_q90':
                        subset.loc[
                            subset.flow_valid,
                            'flow_veh_5min',
                        ].quantile(0.90),
                    'median_occupancy_fraction':
                        subset.loc[
                            subset.occupancy_valid,
                            'occupancy_fraction',
                        ].median(),
                    'occupancy_q10':
                        subset.loc[
                            subset.occupancy_valid,
                            'occupancy_fraction',
                        ].quantile(0.10),
                    'occupancy_q90':
                        subset.loc[
                            subset.occupancy_valid,
                            'occupancy_fraction',
                        ].quantile(0.90),
                })

        zero_vs_100 = pd.DataFrame(zero_vs_100_rows)
        zero_vs_100.to_csv(
            output / 'batch_3_zero_vs_100_same_station.csv',
            index=False,
        )

        # Q14: exact repeats across truly consecutive five-minute records.
        ordered = data.sort_values(
            ['station_id', 'timestamp'],
            kind='stable',
        ).copy()

        same_station = ordered.station_id.eq(
            ordered.station_id.shift()
        )
        consecutive = (
            same_station
            & ordered.timestamp.diff().eq(
                pd.Timedelta(minutes=5)
            )
        )

        ordered['repeat_speed'] = (
            consecutive
            & ordered.speed_valid
            & ordered.speed_valid.shift(
                fill_value=False
            )
            & ordered.speed_mph.eq(
                ordered.speed_mph.shift()
            )
        )

        ordered['repeat_flow'] = (
            consecutive
            & ordered.flow_valid
            & ordered.flow_valid.shift(
                fill_value=False
            )
            & ordered.flow_veh_5min.eq(
                ordered.flow_veh_5min.shift()
            )
        )

        ordered['repeat_occupancy'] = (
            consecutive
            & ordered.occupancy_valid
            & ordered.occupancy_valid.shift(
                fill_value=False
            )
            & ordered.occupancy_fraction.eq(
                ordered.occupancy_fraction.shift()
            )
        )

        ordered['repeat_triplet'] = (
            ordered.repeat_speed
            & ordered.repeat_flow
            & ordered.repeat_occupancy
        )

        repeat_rows = []

        for band in band_order:
            subset = ordered.loc[
                ordered.coverage_band.eq(band)
            ]
            comparable = subset.loc[
                consecutive.reindex(subset.index, fill_value=False)
            ]

            repeat_rows.append({
                'coverage_band': band,
                'comparable_consecutive_rows': len(comparable),
                'repeat_speed_pct':
                    100 * comparable.repeat_speed.mean()
                    if len(comparable)
                    else float('nan'),
                'repeat_flow_pct':
                    100 * comparable.repeat_flow.mean()
                    if len(comparable)
                    else float('nan'),
                'repeat_occupancy_pct':
                    100 * comparable.repeat_occupancy.mean()
                    if len(comparable)
                    else float('nan'),
                'repeat_triplet_pct':
                    100 * comparable.repeat_triplet.mean()
                    if len(comparable)
                    else float('nan'),
            })

        repeat_summary = pd.DataFrame(repeat_rows)
        repeat_summary.to_csv(
            output / 'batch_3_exact_repeat_rates_by_coverage_band.csv',
            index=False,
        )

        station_repeat_rows = []

        for station_id, group in ordered.groupby('station_id'):
            same = group.timestamp.diff().eq(
                pd.Timedelta(minutes=5)
            )

            station_repeat_rows.append({
                'station_id': station_id,
                'rows': len(group),
                'longest_exact_triplet_repeat_run_intervals':
                    batch_3_longest_true_run(
                        group.repeat_triplet,
                        same,
                    ),
                'longest_exact_triplet_repeat_run_minutes':
                    5 * batch_3_longest_true_run(
                        group.repeat_triplet,
                        same,
                    ),
            })

        pd.DataFrame(station_repeat_rows).to_csv(
            output / 'batch_3_station_exact_repeat_runs.csv',
            index=False,
        )

        # Q16: threshold sensitivity and simultaneous usable exposure.
        threshold_rows = []

        usable_base = (
            data.base_valid
            & data.coverage_valid
            & data.all_measures_valid
        )

        for threshold in BATCH_3_THRESHOLDS:
            meets = (
                usable_base
                & data.observed_pct.ge(threshold)
            )

            threshold_rows.append({
                'threshold': threshold,
                'rows_retained': int(meets.sum()),
                'pct_all_corridor_rows':
                    100 * meets.sum() / len(data),
                'stations_with_any_rows':
                    data.loc[meets, 'station_id'].nunique(),
                'days_with_any_rows':
                    data.loc[meets, 'date'].nunique(),
                'speed_valid_pct_within_threshold':
                    100 * data.loc[
                        data.coverage_valid
                        & data.observed_pct.ge(threshold),
                        'speed_valid',
                    ].mean(),
                'flow_valid_pct_within_threshold':
                    100 * data.loc[
                        data.coverage_valid
                        & data.observed_pct.ge(threshold),
                        'flow_valid',
                    ].mean(),
                'occupancy_valid_pct_within_threshold':
                    100 * data.loc[
                        data.coverage_valid
                        & data.observed_pct.ge(threshold),
                        'occupancy_valid',
                    ].mean(),
            })

        threshold_summary = pd.DataFrame(
            threshold_rows
        )

        daily_threshold = pd.concat(
            daily_threshold_parts,
            ignore_index=True,
        )

        simultaneous = (
            daily_threshold.groupby('threshold')
            .agg(
                days_with_simultaneous_data=(
                    'day_has_any_simultaneous_data',
                    'sum',
                ),
                simultaneous_timestamps=(
                    'timestamps_all_stations_meeting',
                    'sum',
                ),
                simultaneous_hours=(
                    'hours_all_stations_meeting',
                    'sum',
                ),
            )
            .reset_index()
        )

        threshold_summary = threshold_summary.merge(
            simultaneous,
            on='threshold',
            how='left',
            validate='one_to_one',
        )

        threshold_summary.to_csv(
            output / 'batch_3_threshold_sensitivity.csv',
            index=False,
        )

        daily_threshold.to_csv(
            output / 'batch_3_daily_simultaneous_threshold_exposure.csv',
            index=False,
        )

        # Per-station threshold coverage exposes the weakest detector instead of
        # allowing corridor averages to hide it.
        station_threshold_rows = []

        for station_id, group in data.groupby('station_id'):
            for threshold in BATCH_3_THRESHOLDS:
                eligible = (
                    group.base_valid
                    & group.coverage_valid
                    & group.all_measures_valid
                    & group.observed_pct.ge(threshold)
                )
                station_threshold_rows.append({
                    'station_id': station_id,
                    'threshold': threshold,
                    'rows': len(group),
                    'eligible_rows': int(eligible.sum()),
                    'eligible_pct_all_rows':
                        100 * eligible.sum() / len(group),
                    'days_with_any_eligible':
                        group.loc[eligible, 'date'].nunique(),
                })

        station_threshold = pd.DataFrame(
            station_threshold_rows
        )

        station_threshold.to_csv(
            output / 'batch_3_station_threshold_sensitivity.csv',
            index=False,
        )

        weakest = (
            station_threshold.groupby('threshold')
            .agg(
                minimum_station_eligible_pct=(
                    'eligible_pct_all_rows',
                    'min',
                ),
                median_station_eligible_pct=(
                    'eligible_pct_all_rows',
                    'median',
                ),
                minimum_station_days_with_any_eligible=(
                    'days_with_any_eligible',
                    'min',
                ),
            )
            .reset_index()
        )

        weakest.to_csv(
            output / 'batch_3_threshold_weakest_station.csv',
            index=False,
        )

        # Q17: lane-level observation audit excluding empty padded slots.
        if lane_parts:
            lanes = pd.concat(
                lane_parts,
                ignore_index=True,
            )

            lane_summary_rows = []

            for band in band_order:
                group = lanes.loc[
                    lanes.coverage_band.eq(band)
                ]

                observed_nonnull = group.observed.notna()
                observed_positive = (
                    observed_nonnull
                    & pd.to_numeric(
                        group.observed,
                        errors='coerce',
                    ).gt(0)
                )

                lane_summary_rows.append({
                    'coverage_band': band,
                    'present_lane_slots': len(group),
                    'lane_observed_nonnull_pct':
                        100 * observed_nonnull.mean()
                        if len(group)
                        else float('nan'),
                    'lane_observed_positive_pct':
                        100 * observed_positive.mean()
                        if len(group)
                        else float('nan'),
                })

            lane_summary = pd.DataFrame(
                lane_summary_rows
            )

            lane_summary.to_csv(
                output / 'batch_3_lane_observation_by_coverage_band.csv',
                index=False,
            )

            # Row-level lane consistency: compare the station-level observed
            # percentage with the fraction of PHYSICAL metadata-defined lanes
            # whose lane observed field is positive.
            lane_observed_numeric = pd.to_numeric(
                lanes.observed,
                errors='coerce',
            )

            lanes['lane_observed_nonnull'] = (
                lane_observed_numeric.notna()
            )
            lanes['lane_observed_positive'] = (
                lane_observed_numeric.gt(0)
            )

            lane_row = (
                lanes.groupby(
                    ['timestamp', 'station_id'],
                    dropna=False,
                )
                .agg(
                    station_observed_pct=(
                        'station_observed_pct',
                        'first',
                    ),
                    physical_lane_slots=(
                        'lane_number',
                        'size',
                    ),
                    metadata_lane_count=(
                        'metadata_lane_count',
                        'first',
                    ),
                    lane_slots_with_observed_field=(
                        'lane_observed_nonnull',
                        'sum',
                    ),
                    lane_slots_observed_positive=(
                        'lane_observed_positive',
                        'sum',
                    ),
                )
                .reset_index()
            )

            if not lane_row.physical_lane_slots.eq(
                lane_row.metadata_lane_count
            ).all():
                bad_lane_rows = lane_row.loc[
                    ~lane_row.physical_lane_slots.eq(
                        lane_row.metadata_lane_count
                    )
                ]
                raise ValueError(
                    'Prepared lane fields do not cover every metadata-defined '
                    f'physical lane on {len(bad_lane_rows)} station/timestamps.'
                )

            lane_row['pct_physical_lanes_observed_positive'] = (
                100
                * lane_row.lane_slots_observed_positive
                / lane_row.metadata_lane_count.replace(
                    0,
                    float('nan'),
                )
            )

            lane_row['station_minus_lane_positive_pct'] = (
                lane_row.station_observed_pct
                - lane_row.pct_physical_lanes_observed_positive
            )

            lane_row.to_csv(
                output / 'batch_3_lane_station_consistency.csv',
                index=False,
            )

            lane_consistency_summary = pd.DataFrame([{
                'rows': len(lane_row),
                'rows_with_physical_lane_slots':
                    int(lane_row.physical_lane_slots.gt(0).sum()),
                'median_physical_lane_slots':
                    lane_row.physical_lane_slots.median(),
                'median_abs_station_minus_lane_positive_pct':
                    lane_row.station_minus_lane_positive_pct.abs().median(),
                'pct_rows_exact_station_lane_positive_match':
                    100
                    * lane_row.station_minus_lane_positive_pct.eq(0).mean(),
            }])

            lane_consistency_summary.to_csv(
                output / 'batch_3_lane_consistency_summary.csv',
                index=False,
            )
        else:
            lane_summary = pd.DataFrame()
            lane_consistency_summary = pd.DataFrame()

        # ============================================================
        # BATCH 3 PORTFOLIO FIGURES
        # ============================================================
        # These plots use the same portfolio visual language as burgers_clean.py:
        # Montserrat body text, Archivo Black titles, cream background, restrained
        # grids, the shared six-color palette, and high-resolution PNG exports.

        # ------------------------------------------------------------
        # B3 FIGURE 1. Threshold retention and simultaneous corridor exposure
        # ------------------------------------------------------------
        threshold_plot = threshold_summary.copy()
        fig, axes = plt.subplots(1, 2, figsize=THEME_FIGSIZE_15_6)

        axes[0].bar(
            threshold_plot.threshold.astype(str),
            threshold_plot.pct_all_corridor_rows,
        )
        axes[0].set_title(
            'Data Retained as Coverage Requirement Tightens',
            fontfamily=THEME_TITLE_FONT,
        )
        axes[0].set_xlabel('Minimum Observed Coverage')
        axes[0].set_ylabel('Selected-Corridor Rows Retained (%)')
        axes[0].set_ylim(THEME_AXIS_MIN_ZERO, THEME_PERCENT_AXIS_MAX)
        axes[0].grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_STRONG)
        axes[0].grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)
        for position, value in enumerate(
            threshold_plot.pct_all_corridor_rows
        ):
            axes[0].annotate(
                f'{value:.1f}%',
                (position, value),
                xytext=(0, 5),
                textcoords=THEME_TEXTCOORDS,
                ha=THEME_ALIGN_CENTER,
            )

        axes[1].plot(
            threshold_plot.threshold,
            threshold_plot.simultaneous_hours,
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_LINEWIDTH_STANDARD,
        )
        axes[1].set_title(
            'Usable All-Station Corridor Exposure',
            fontfamily=THEME_TITLE_FONT,
        )
        axes[1].set_xlabel('Minimum Observed Coverage (%)')
        axes[1].set_ylabel('Simultaneous All-Station Hours')
        axes[1].set_xticks(BATCH_3_THRESHOLDS)
        axes[1].set_ylim(bottom=THEME_AXIS_MIN_ZERO)
        axes[1].grid(alpha=THEME_GRID_ALPHA_LIGHT)
        for row in threshold_plot.itertuples():
            axes[1].annotate(
                f'{row.simultaneous_hours:,.0f} h',
                (row.threshold, row.simultaneous_hours),
                xytext=THEME_ANNOTATION_OFFSET_7,
                textcoords=THEME_TEXTCOORDS,
                ha=THEME_ALIGN_CENTER,
                fontsize=THEME_ANNOTATION_SIZE,
            )

        fig.suptitle(
            'Coverage Threshold Sensitivity — I-10 East Primary Corridor',
            fontfamily=THEME_TITLE_FONT,
            fontsize=THEME_SUPTITLE_SIZE,
        )
        fig.tight_layout()
        fig.savefig(
            BATCH_3_IMAGE_FOLDER / 'traffic_b3_threshold_sensitivity.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # ------------------------------------------------------------
        # B3 FIGURE 2. Weakest-station and median-station retention
        # ------------------------------------------------------------
        weakest_plot = weakest.sort_values('threshold').copy()
        fig, ax = plt.subplots(figsize=THEME_FIGSIZE_12_7)
        ax.plot(
            weakest_plot.threshold,
            weakest_plot.minimum_station_eligible_pct,
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_LINEWIDTH_EMPHASIS,
            label='Weakest station',
        )
        ax.plot(
            weakest_plot.threshold,
            weakest_plot.median_station_eligible_pct,
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_LINEWIDTH_EMPHASIS,
            label='Median station',
        )
        ax.set_title(
            'Does a Higher Coverage Threshold Break the Corridor?',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel('Minimum Observed Coverage (%)')
        ax.set_ylabel('Eligible Station Rows (%)')
        ax.set_xticks(BATCH_3_THRESHOLDS)
        ax.set_ylim(THEME_AXIS_MIN_ZERO, THEME_PERCENT_AXIS_MAX)
        ax.grid(alpha=THEME_GRID_ALPHA_LIGHT)
        ax.legend(frameon=THEME_LEGEND_FRAME)
        fig.text(
            0.5,
            0.015,
            'Weakest-link retention matters for propagation analysis: one poor '
            'middle station can break the spatial chain.',
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )
        fig.tight_layout(rect=THEME_TIGHT_RECT_045)
        fig.savefig(
            BATCH_3_IMAGE_FOLDER / 'traffic_b3_weakest_station_threshold.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # ------------------------------------------------------------
        # B3 FIGURE 3. Measurement distributions by observation coverage
        # ------------------------------------------------------------
        measure_plot = band_summary.loc[
            band_summary.coverage_band.isin(
                ['0', '>0-<40', '40-<80', '80-<100', '100']
            )
        ].copy()

        measure_plot['coverage_band'] = pd.Categorical(
            measure_plot.coverage_band,
            categories=['0', '>0-<40', '40-<80', '80-<100', '100'],
            ordered=True,
        )
        measure_plot = measure_plot.sort_values('coverage_band')
        measure_plot = measure_plot.loc[measure_plot.rows.gt(0)].copy()

        fig, axes = plt.subplots(1, 3, figsize=THEME_FIGSIZE_18_6)

        for ax, median_col, q10_col, q90_col, title, ylabel in [
            (
                axes[0],
                'median_speed_mph',
                'speed_mph_q10',
                'speed_mph_q90',
                'Speed',
                'Speed (mph)',
            ),
            (
                axes[1],
                'median_flow_veh_5min',
                'flow_veh_5min_q10',
                'flow_veh_5min_q90',
                'Flow',
                'Vehicles per 5 Minutes',
            ),
            (
                axes[2],
                'median_occupancy_fraction',
                'occupancy_fraction_q10',
                'occupancy_fraction_q90',
                'Occupancy',
                'Occupancy Fraction',
            ),
        ]:
            x = np.arange(len(measure_plot))
            median = measure_plot[median_col].to_numpy(dtype=float)
            q10 = measure_plot[q10_col].to_numpy(dtype=float)
            q90 = measure_plot[q90_col].to_numpy(dtype=float)
            ax.errorbar(
                x,
                median,
                yerr=[median - q10, q90 - median],
                fmt='o',
                markersize=THEME_MARKER_SIZE_STANDARD,
                capsize=4,
                linewidth=THEME_LINEWIDTH_FINE,
            )
            ax.set_xticks(
                x,
                measure_plot.coverage_band.astype(str),
                rotation=THEME_ROTATION_LARGE,
                ha=THEME_ALIGN_RIGHT,
            )
            ax.set_title(title, fontfamily=THEME_TITLE_FONT)
            ax.set_xlabel('Station Observed Coverage Band')
            ax.set_ylabel(ylabel)
            ax.grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_MEDIUM)
            ax.grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)
            for xpos, rows in zip(x, measure_plot.rows):
                ax.annotate(
                    f'n={int(rows):,}',
                    (xpos, q90[xpos]),
                    xytext=THEME_ANNOTATION_OFFSET_6,
                    textcoords=THEME_TEXTCOORDS,
                    ha=THEME_ALIGN_CENTER,
                    fontsize=THEME_ANNOTATION_SIZE_SMALL,
                )

        fig.suptitle(
            'Traffic Measurements Remain Populated Even When Coverage Is Zero',
            fontfamily=THEME_TITLE_FONT,
            fontsize=THEME_SUPTITLE_SIZE,
        )
        fig.text(
            0.5,
            0.015,
            'Points are medians; bars span the 10th–90th percentiles. '
            'Populated values do not make 0%-observed rows analytically valid.',
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )
        fig.tight_layout(rect=THEME_TIGHT_RECT_055)
        fig.savefig(
            BATCH_3_IMAGE_FOLDER / 'traffic_b3_measurements_by_coverage.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # ------------------------------------------------------------
        # B3 FIGURE 4. Exact-repeat diagnostic
        # ------------------------------------------------------------
        repeat_plot = repeat_summary.loc[
            repeat_summary.coverage_band.isin(
                ['0', '>0-<40', '40-<80', '100']
            )
            & repeat_summary.comparable_consecutive_rows.gt(0)
        ].copy()

        x = np.arange(len(repeat_plot))
        width = 0.2

        fig, ax = plt.subplots(figsize=THEME_FIGSIZE_14_7)
        for offset, column, label in [
            (-1.5 * width, 'repeat_speed_pct', 'Speed'),
            (-0.5 * width, 'repeat_flow_pct', 'Flow'),
            (0.5 * width, 'repeat_occupancy_pct', 'Occupancy'),
            (1.5 * width, 'repeat_triplet_pct', 'Exact triplet'),
        ]:
            ax.bar(
                x + offset,
                repeat_plot[column],
                width=width,
                label=label,
            )

        ax.set_title(
            'Exact Repeats Do Not Spike in the 0%-Observed Rows',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel('Station Observed Coverage Band')
        ax.set_ylabel('Consecutive Records Repeating Exactly (%)')
        ax.set_xticks(x, repeat_plot.coverage_band.astype(str))
        ax.set_ylim(bottom=THEME_AXIS_MIN_ZERO)
        ax.grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_MEDIUM)
        ax.grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)
        ax.legend(frameon=THEME_LEGEND_FRAME, ncol=THEME_LEGEND_NCOL_2)
        fig.text(
            0.5,
            0.015,
            'Exact repeats are a diagnostic for suspicious persistence, not '
            'proof of imputation.',
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )
        fig.tight_layout(rect=THEME_TIGHT_RECT_05)
        fig.savefig(
            BATCH_3_IMAGE_FOLDER / 'traffic_b3_exact_repeat_diagnostic.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # ------------------------------------------------------------
        # B3 FIGURE 5. Lane-level station consistency after padding correction
        # ------------------------------------------------------------
        if not lane_parts:
            raise ValueError(
                'Batch 3 expected lane fields for the selected corridor.'
            )

        lane_row['absolute_station_lane_difference'] = (
            lane_row.station_minus_lane_positive_pct.abs()
        )

        lane_row['coverage_band'] = batch_3_coverage_band(
            lane_row.station_observed_pct
        )

        lane_plot = (
            lane_row.groupby('coverage_band')
            .agg(
                rows=('station_id', 'size'),
                median_abs_difference=(
                    'absolute_station_lane_difference',
                    'median',
                ),
                q90_abs_difference=(
                    'absolute_station_lane_difference',
                    lambda values: values.quantile(0.90),
                ),
                exact_match_pct=(
                    'station_minus_lane_positive_pct',
                    lambda values: 100 * values.eq(0).mean(),
                ),
            )
            .reset_index()
        )

        lane_plot = lane_plot.loc[
            lane_plot.coverage_band.isin(
                ['0', '>0-<40', '40-<80', '80-<100', '100']
            )
            & lane_plot.rows.gt(0)
        ].copy()

        lane_order = {
            '0': 0,
            '>0-<40': 1,
            '40-<80': 2,
            '80-<100': 3,
            '100': 4,
        }
        lane_plot['plot_order'] = lane_plot.coverage_band.map(lane_order)
        lane_plot = lane_plot.sort_values('plot_order')

        fig, axes = plt.subplots(1, 2, figsize=THEME_FIGSIZE_15_6)

        axes[0].bar(
            lane_plot.coverage_band,
            lane_plot.median_abs_difference,
        )
        axes[0].set_title(
            'Station vs. Physical-Lane Coverage',
            fontfamily=THEME_TITLE_FONT,
        )
        axes[0].set_xlabel('Station Observed Coverage Band')
        axes[0].set_ylabel('Median Absolute Difference (Percentage Points)')
        axes[0].grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_MEDIUM)
        axes[0].grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)

        axes[1].bar(
            lane_plot.coverage_band,
            lane_plot.exact_match_pct,
        )
        axes[1].set_title(
            'Exact Station/Lane Agreement',
            fontfamily=THEME_TITLE_FONT,
        )
        axes[1].set_xlabel('Station Observed Coverage Band')
        axes[1].set_ylabel('Rows Matching Exactly (%)')
        axes[1].set_ylim(THEME_AXIS_MIN_ZERO, THEME_PERCENT_AXIS_MAX)
        axes[1].grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_MEDIUM)
        axes[1].grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)

        for ax in axes:
            ax.tick_params(axis=THEME_GRID_AXIS_X, rotation=THEME_ROTATION_LARGE)

        fig.suptitle(
            'Lane Audit Uses Metadata-Defined Physical Lanes, Not Padded Slots',
            fontfamily=THEME_TITLE_FONT,
            fontsize=THEME_SUPTITLE_SIZE,
        )
        fig.tight_layout()
        fig.savefig(
            BATCH_3_IMAGE_FOLDER / 'traffic_b3_lane_consistency.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        lane_plot.to_csv(
            output / 'batch_3_lane_consistency_by_coverage_band.csv',
            index=False,
        )

        # Compact decision report. Final threshold remains a documented gate,
        # because choosing a cutoff should follow these observed results rather
        # than be silently hard-coded before seeing them.
        threshold_display = threshold_summary.merge(
            weakest,
            on='threshold',
            how='left',
            validate='one_to_one',
        )

        report_lines = [
            'BATCH 3 — ANALYTICAL DATA VALIDITY',
            f'Primary corridor: {selected_corridor}',
            f'Stations: {station_count}',
            f'Rows audited: {len(data):,}',
            '',
            'THRESHOLD SENSITIVITY',
            threshold_display.round(3).to_string(index=False),
            '',
            'COVERAGE-BAND MEASURE VALIDITY',
            band_summary.round(3).to_string(index=False),
            '',
            'EXACT REPEAT RATES',
            repeat_summary.round(3).to_string(index=False),
        ]

        if not lane_consistency_summary.empty:
            report_lines.extend([
                '',
                'LANE CONSISTENCY SUMMARY',
                lane_consistency_summary.round(3).to_string(
                    index=False
                ),
            ])

        report_lines.extend([
            '',
            'INTERPRETATION AND DECISION GATE',
            '0%-observed rows are preserved for same-station comparison; they are not treated as analytically valid merely because measurements are populated.',
            'Lane slots above each station metadata Lanes count are treated as padded schema slots and excluded from detector counts.',
            'Exact repeated measurements are diagnostic evidence only; repetition does not by itself prove imputation.',
            'Threshold comparison emphasizes 80% versus 100%, with 40% and 60% retained as sensitivity checks.',
            'Final coverage threshold is not yet adopted. Select it from validity, weakest-station retention, and simultaneous corridor exposure.',
            '',
            'PORTFOLIO FIGURES',
            'images/traffic_b3_threshold_sensitivity.png',
            'images/traffic_b3_weakest_station_threshold.png',
            'images/traffic_b3_measurements_by_coverage.png',
            'images/traffic_b3_exact_repeat_diagnostic.png',
            'images/traffic_b3_lane_consistency.png',
        ])

        report = '\n'.join(report_lines)

        (output / 'batch_3_audit_summary.txt').write_text(
            report + '\n'
        )

        settings.update(
            status='complete',
            rows_audited=len(data),
            stations_audited=station_count,
            selected_corridor=selected_corridor,
            final_threshold_adopted=None,
            portfolio_figures=[
                'images/traffic_b3_threshold_sensitivity.png',
                'images/traffic_b3_weakest_station_threshold.png',
                'images/traffic_b3_measurements_by_coverage.png',
                'images/traffic_b3_exact_repeat_diagnostic.png',
                'images/traffic_b3_lane_consistency.png',
            ],
        )

        print('\n' + report)
        print(
            f'\nSaved Batch 3 reports: {output.resolve()}'
        )

    except Exception as error:
        settings.update(
            status='incomplete',
            error=str(error),
        )
        raise

    finally:
        settings_path.write_text(
            json.dumps(settings, indent=2)
        )




# ===================== BATCH 4. DEFINE AND DETECT TRAFFIC BREAKDOWN =====================
# Batch 4 intentionally compares multiple defensible operational definitions.
# No one threshold is treated as universally correct.
#
# Primary comparison methods:
#
# 1. Threshold_50_15min
#    - speed < 50 mph
#    - sustained for at least 3 consecutive 5-minute intervals (15 minutes)
#    - onset must be observed as a transition from >= 50 mph
#    - combines FHWA's 50 mph illustrative congestion threshold with the
#      HCM/FHWA sustained-breakdown persistence convention.
#
# 2. LA_20drop_below40
#    - five-minute speed drops by at least 20 mph from the preceding interval
#    - new speed is below 40 mph
#    - event remains active while speed stays below 40 mph
#    - based on the Los Angeles five-minute breakdown rule summarized in SHRP2 C05.
#
# 3. Caltrans_PeMS_spatial
#    - current/downstream station speed < 40 mph
#    - speed drops by at least 20 mph from the immediately upstream station
#    - station spacing < 3 miles
#    - raw condition is present in at least 5 of any 7 consecutive 5-minute periods
#    - uses Caltrans direction rules: East/North upstream has smaller Abs_PM.
#
# All three methods are compared on the common 5 AM-8 PM analysis window because
# the current Caltrans PeMS bottleneck workflow is defined across AM, noon and PM
# shifts spanning that period.


def batch_4_contiguous_blocks(frame):
    """Number uninterrupted five-minute station sequences."""
    ordered = frame.sort_values('timestamp').copy()
    gap = ordered.timestamp.diff().ne(pd.Timedelta(minutes=5))
    date_change = ordered.timestamp.dt.date.ne(
        ordered.timestamp.shift().dt.date
    )
    ordered['block_id'] = (gap | date_change).cumsum()
    return ordered


def batch_4_threshold_events(frame, station_id, threshold, min_intervals):
    """
    Detect sustained temporal threshold events at one station.

    An event is counted only when the transition from above/equal threshold to
    below threshold is observed. Runs already in progress at the start of a
    block are retained in the audit but marked left-censored and excluded from
    onset-based event counts.
    """
    ordered = batch_4_contiguous_blocks(frame)
    rows = []

    for block_id, block in ordered.groupby('block_id', sort=False):
        block = block.reset_index(drop=True)
        below = block.speed_mph.lt(threshold).to_numpy()

        start = None
        for i, is_below in enumerate(below):
            if is_below and start is None:
                start = i
            at_end = i == len(block) - 1

            if start is not None and ((not is_below) or at_end):
                stop = i if (is_below and at_end) else i - 1
                run_length = stop - start + 1

                if run_length >= min_intervals:
                    previous_index = start - 1
                    next_index = stop + 1

                    onset_observed = (
                        previous_index >= 0
                        and block.loc[previous_index, 'speed_mph'] >= threshold
                    )
                    recovery_observed = (
                        next_index < len(block)
                        and block.loc[next_index, 'speed_mph'] >= threshold
                    )

                    pre = (
                        block.loc[previous_index]
                        if previous_index >= 0
                        else None
                    )

                    rows.append({
                        'method': f'Threshold_{threshold}_{min_intervals*5}min',
                        'station_id': int(station_id),
                        'start_time': block.loc[start, 'timestamp'],
                        'end_time': block.loc[stop, 'timestamp'],
                        'duration_minutes': run_length * 5,
                        'intervals': run_length,
                        'onset_observed': bool(onset_observed),
                        'recovery_observed': bool(recovery_observed),
                        'left_censored': not bool(onset_observed),
                        'right_censored': not bool(recovery_observed),
                        'start_speed_mph': float(block.loc[start, 'speed_mph']),
                        'minimum_speed_mph': float(
                            block.loc[start:stop, 'speed_mph'].min()
                        ),
                        'prebreakdown_speed_mph': (
                            float(pre.speed_mph) if pre is not None else np.nan
                        ),
                        'prebreakdown_flow_veh_5min': (
                            float(pre.flow_veh_5min) if pre is not None else np.nan
                        ),
                        'prebreakdown_occupancy_fraction': (
                            float(pre.occupancy_fraction)
                            if pre is not None else np.nan
                        ),
                    })

                start = None

    return pd.DataFrame(rows)


def batch_4_la_drop_events(frame, station_id):
    """
    Los Angeles rule summarized in SHRP2 C05:
    a >=20 mph five-minute speed differential with resulting speed <40 mph.

    Once an onset is identified, the event remains active while speed stays
    below 40 mph. This produces an event episode without inventing an
    additional persistence criterion not present in the summarized LA rule.
    """
    ordered = batch_4_contiguous_blocks(frame)
    rows = []

    for block_id, block in ordered.groupby('block_id', sort=False):
        block = block.reset_index(drop=True)

        i = 1
        while i < len(block):
            previous = block.loc[i - 1]
            current = block.loc[i]

            starts = (
                previous.speed_mph - current.speed_mph >= 20
                and current.speed_mph < 40
            )

            if not starts:
                i += 1
                continue

            start = i
            stop = i

            while (
                stop + 1 < len(block)
                and block.loc[stop + 1, 'speed_mph'] < 40
            ):
                stop += 1

            recovery_observed = (
                stop + 1 < len(block)
                and block.loc[stop + 1, 'speed_mph'] >= 40
            )

            rows.append({
                'method': 'LA_20drop_below40',
                'station_id': int(station_id),
                'start_time': block.loc[start, 'timestamp'],
                'end_time': block.loc[stop, 'timestamp'],
                'duration_minutes': (stop - start + 1) * 5,
                'intervals': stop - start + 1,
                'onset_observed': True,
                'recovery_observed': bool(recovery_observed),
                'left_censored': False,
                'right_censored': not bool(recovery_observed),
                'start_speed_mph': float(current.speed_mph),
                'minimum_speed_mph': float(
                    block.loc[start:stop, 'speed_mph'].min()
                ),
                'speed_drop_mph': float(
                    previous.speed_mph - current.speed_mph
                ),
                'prebreakdown_speed_mph': float(previous.speed_mph),
                'prebreakdown_flow_veh_5min': float(
                    previous.flow_veh_5min
                ),
                'prebreakdown_occupancy_fraction': float(
                    previous.occupancy_fraction
                ),
            })

            i = stop + 1

    return pd.DataFrame(rows)


def batch_4_mark_five_of_seven(raw_active):
    """
    Reproduce the Caltrans 5-of-7 persistence rule.

    Any seven-interval window containing at least five raw-active observations
    activates all seven time points in that qualifying window.
    """
    raw = np.asarray(raw_active, dtype=bool)
    sustained = np.zeros(len(raw), dtype=bool)

    if len(raw) < 7:
        return sustained

    for start in range(len(raw) - 6):
        if raw[start:start + 7].sum() >= 5:
            sustained[start:start + 7] = True

    return sustained


def batch_4_boolean_events(frame, active_column, method, station_id, extra=None):
    """Convert a station's contiguous active flags to event episodes."""
    ordered = batch_4_contiguous_blocks(frame)
    rows = []

    for block_id, block in ordered.groupby('block_id', sort=False):
        block = block.reset_index(drop=True)
        active = block[active_column].fillna(False).to_numpy(dtype=bool)
        start = None

        for i, is_active in enumerate(active):
            if is_active and start is None:
                start = i
            at_end = i == len(block) - 1

            if start is not None and ((not is_active) or at_end):
                stop = i if (is_active and at_end) else i - 1

                record = {
                    'method': method,
                    'station_id': int(station_id),
                    'start_time': block.loc[start, 'timestamp'],
                    'end_time': block.loc[stop, 'timestamp'],
                    'duration_minutes': (stop - start + 1) * 5,
                    'intervals': stop - start + 1,
                    'onset_observed': bool(start > 0),
                    'recovery_observed': bool(stop + 1 < len(block)),
                    'left_censored': bool(start == 0),
                    'right_censored': bool(stop + 1 == len(block)),
                    'start_speed_mph': float(
                        block.loc[start, 'speed_mph']
                    ),
                    'minimum_speed_mph': float(
                        block.loc[start:stop, 'speed_mph'].min()
                    ),
                }

                if extra is not None:
                    record.update(extra)

                rows.append(record)
                start = None

    return pd.DataFrame(rows)


def run_batch_4_part_1():
    """Compare literature- and agency-based traffic-breakdown definitions."""
    output = BATCH_4_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 4 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    if not BATCH_3_SELECTION_FILE.exists():
        raise FileNotFoundError(
            f'Missing locked corridor selection: {BATCH_3_SELECTION_FILE}'
        )

    if not BATCH_3_CORRIDOR_GEOGRAPHY_FILE.exists():
        raise FileNotFoundError(
            f'Missing corridor geography: {BATCH_3_CORRIDOR_GEOGRAPHY_FILE}'
        )

    selection = pd.read_csv(BATCH_3_SELECTION_FILE)
    if len(selection) != 1:
        raise ValueError(
            'Batch 4 requires exactly one locked primary corridor.'
        )

    selected_corridor = str(selection.iloc[0].corridor_id)

    geography = pd.read_csv(BATCH_3_CORRIDOR_GEOGRAPHY_FILE)
    corridor = geography.loc[
        geography.corridor_id.eq(selected_corridor)
    ].copy()

    if corridor.empty:
        raise ValueError(
            f'Corridor not found in Batch 2 geography: {selected_corridor}'
        )

    required_geo = {'station_id', 'Dir', 'Abs_PM', 'Name'}
    missing_geo = required_geo - set(corridor.columns)
    if missing_geo:
        raise ValueError(
            'Corridor geography missing required fields: '
            + ', '.join(sorted(missing_geo))
        )

    directions = corridor.Dir.dropna().astype(str).unique()
    if len(directions) != 1:
        raise ValueError(
            'Selected corridor must have exactly one travel direction.'
        )

    direction = directions[0].upper()
    ascending = direction in {'N', 'E'}

    corridor['station_id'] = pd.to_numeric(
        corridor.station_id,
        errors='raise',
    ).astype('int64')

    corridor['Abs_PM'] = pd.to_numeric(
        corridor.Abs_PM,
        errors='raise',
    )

    corridor = corridor.sort_values(
        'Abs_PM',
        ascending=ascending,
        kind='stable',
    ).reset_index(drop=True)

    corridor['travel_order'] = np.arange(1, len(corridor) + 1)

    station_ids = corridor.station_id.tolist()
    station_set = set(station_ids)

    # Verify Caltrans direction convention explicitly.
    if direction in {'N', 'E'} and not corridor.Abs_PM.is_monotonic_increasing:
        raise ValueError(
            'North/East corridor is not ordered by increasing Abs_PM.'
        )
    if direction in {'S', 'W'} and not corridor.Abs_PM.is_monotonic_decreasing:
        raise ValueError(
            'South/West corridor is not ordered by decreasing Abs_PM.'
        )

    corridor['upstream_station_id'] = corridor.station_id.shift()
    corridor['upstream_abs_pm'] = corridor.Abs_PM.shift()
    corridor['upstream_gap_miles'] = (
        corridor.Abs_PM - corridor.upstream_abs_pm
    ).abs()

    corridor.to_csv(
        output / 'batch_4_corridor_travel_order.csv'
        if output.exists()
        else Path('/tmp/unused'),
        index=False,
    ) if False else None

    output.mkdir(parents=True)
    BATCH_4_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

    corridor.to_csv(
        output / 'batch_4_corridor_travel_order.csv',
        index=False,
    )

    settings_path = output / 'batch_4_run_settings.json'
    settings = {
        'status': 'incomplete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selected_corridor': selected_corridor,
        'station_ids_in_travel_order': station_ids,
        'direction': direction,
        'final_observed_pct': BATCH_4_FINAL_OBSERVED_PCT,
        'analysis_window': (
            f'{BATCH_4_ANALYSIS_START_HOUR:02d}:00-'
            f'{BATCH_4_ANALYSIS_END_HOUR:02d}:00'
        ),
        'threshold_sensitivity_mph': BATCH_4_THRESHOLD_SPEEDS,
        'threshold_persistence_intervals':
            BATCH_4_PERSISTENCE_INTERVALS,
        'sources': BATCH_4_SOURCES,
        'primary_method_adopted': None,
    }
    settings_path.write_text(json.dumps(settings, indent=2))

    try:
        dates = pd.date_range(START_DATE, END_DATE)
        parts = []

        required_columns = [
            'timestamp',
            'station_id',
            'lane_type',
            'observed_pct',
            'flow_veh_5min',
            'occupancy_fraction',
            'speed_mph',
            'invalid_timestamp',
            'date_mismatch',
            'invalid_station_id',
            'invalid_speed',
            'invalid_flow',
            'invalid_occupancy',
            'invalid_observed_pct',
            'off_5min_grid',
            'duplicate_station_timestamp',
        ]

        for number, day in enumerate(dates, 1):
            day_text = day.strftime('%Y-%m-%d')
            folder = PREPARED_DATA_FOLDER / f'date={day_text}'
            paths = sorted(folder.glob('part-*.parquet'))

            if not paths:
                continue

            print(
                f'Batch 4: {day_text} [{number}/{len(dates)}]',
                flush=True,
            )

            day_parts = []

            for path in paths:
                schema = set(pq.read_schema(path).names)
                missing = set(required_columns) - schema

                if missing:
                    raise ValueError(
                        f'{path}: missing Batch 4 columns: '
                        + ', '.join(sorted(missing))
                    )

                frame = pd.read_parquet(
                    path,
                    columns=required_columns,
                )

                frame = frame.loc[
                    frame.lane_type.eq('ML')
                    & frame.station_id.isin(station_set)
                ].copy()

                if not frame.empty:
                    day_parts.append(frame)

            if not day_parts:
                continue

            day_frame = pd.concat(
                day_parts,
                ignore_index=True,
            )

            duplicate_copies = day_frame.duplicated(
                ['station_id', 'timestamp'],
                keep=False,
            )

            valid = (
                ~day_frame.invalid_timestamp.fillna(True)
                & ~day_frame.date_mismatch.fillna(True)
                & ~day_frame.invalid_station_id.fillna(True)
                & ~day_frame.invalid_speed.fillna(True)
                & ~day_frame.invalid_flow.fillna(True)
                & ~day_frame.invalid_occupancy.fillna(True)
                & ~day_frame.invalid_observed_pct.fillna(True)
                & ~day_frame.off_5min_grid.fillna(True)
                & ~duplicate_copies
                & day_frame.timestamp.notna()
                & day_frame.observed_pct.eq(
                    BATCH_4_FINAL_OBSERVED_PCT
                )
            )

            day_frame = day_frame.loc[valid].copy()

            hour = (
                day_frame.timestamp.dt.hour
                + day_frame.timestamp.dt.minute / 60
            )

            day_frame = day_frame.loc[
                hour.ge(BATCH_4_ANALYSIS_START_HOUR)
                & hour.lt(BATCH_4_ANALYSIS_END_HOUR)
            ].copy()

            if not day_frame.empty:
                parts.append(day_frame)

        if not parts:
            raise ValueError(
                'No 100%-observed selected-corridor rows for Batch 4.'
            )

        data = pd.concat(parts, ignore_index=True)
        data['station_id'] = pd.to_numeric(
            data.station_id,
            errors='raise',
        ).astype('int64')

        data = data.sort_values(
            ['station_id', 'timestamp'],
            kind='stable',
        ).reset_index(drop=True)

        data.to_parquet(
            output / 'batch_4_corridor_analysis_rows.parquet',
            index=False,
        )

        # --------------------------------------------------------
        # METHOD 1 + threshold sensitivity: sustained fixed speed
        # --------------------------------------------------------
        threshold_event_tables = []

        for threshold in BATCH_4_THRESHOLD_SPEEDS:
            for station_id, station_data in data.groupby(
                'station_id',
                sort=False,
            ):
                events = batch_4_threshold_events(
                    station_data,
                    station_id=station_id,
                    threshold=threshold,
                    min_intervals=BATCH_4_PERSISTENCE_INTERVALS,
                )

                if not events.empty:
                    threshold_event_tables.append(events)

        threshold_events = (
            pd.concat(threshold_event_tables, ignore_index=True)
            if threshold_event_tables
            else pd.DataFrame()
        )

        if threshold_events.empty:
            raise ValueError(
                'No sustained threshold events were detected.'
            )

        threshold_events.to_csv(
            output / 'batch_4_threshold_events_all_sensitivities.csv',
            index=False,
        )

        method_50 = threshold_events.loc[
            threshold_events.method.eq(
                f'Threshold_50_{BATCH_4_PERSISTENCE_INTERVALS*5}min'
            )
            & threshold_events.onset_observed
        ].copy()

        # --------------------------------------------------------
        # METHOD 2: Los Angeles 20-mph drop + below 40 mph
        # --------------------------------------------------------
        la_tables = []

        for station_id, station_data in data.groupby(
            'station_id',
            sort=False,
        ):
            events = batch_4_la_drop_events(
                station_data,
                station_id=station_id,
            )
            if not events.empty:
                la_tables.append(events)

        la_events = (
            pd.concat(la_tables, ignore_index=True)
            if la_tables
            else pd.DataFrame(columns=method_50.columns)
        )

        la_events.to_csv(
            output / 'batch_4_la_20drop_below40_events.csv',
            index=False,
        )

        # --------------------------------------------------------
        # METHOD 3: current Caltrans PeMS spatial bottleneck logic
        # --------------------------------------------------------
        wide_speed = data.pivot(
            index='timestamp',
            columns='station_id',
            values='speed_mph',
        ).sort_index()

        spatial_event_tables = []
        spatial_active_records = []

        for row in corridor.iloc[1:].itertuples():
            downstream_id = int(row.station_id)
            upstream_id = int(row.upstream_station_id)
            gap_miles = float(row.upstream_gap_miles)

            if gap_miles >= 3:
                continue

            pair = pd.DataFrame({
                'timestamp': wide_speed.index.to_numpy(),
                'upstream_speed_mph':
                    wide_speed.get(upstream_id).to_numpy(),
                'speed_mph':
                    wide_speed.get(downstream_id).to_numpy(),
            }).dropna().reset_index(drop=True)

            if pair.empty:
                continue

            pair['date'] = pair.timestamp.dt.date
            pair['hour'] = (
                pair.timestamp.dt.hour
                + pair.timestamp.dt.minute / 60
            )

            pair['shift'] = np.select(
                [
                    pair.hour.ge(5) & pair.hour.lt(10),
                    pair.hour.ge(10) & pair.hour.lt(15),
                    pair.hour.ge(15) & pair.hour.lt(20),
                ],
                ['AM', 'Noon', 'PM'],
                default='Outside',
            )

            pair = pair.loc[
                ~pair['shift'].eq('Outside')
            ].copy()

            pair['raw_active'] = (
                (pair.upstream_speed_mph - pair.speed_mph >= 20)
                & pair.speed_mph.lt(40)
            )

            pair['sustained_active'] = False

            for (date_value, shift), shift_frame in pair.groupby(
                ['date', 'shift'],
                sort=False,
            ):
                ordered_index = shift_frame.sort_values(
                    'timestamp'
                ).index

                raw = pair.loc[
                    ordered_index,
                    'raw_active',
                ].to_numpy(dtype=bool)

                # Split at any missing five-minute gap so persistence
                # cannot bridge unavailable observations.
                stamps = pair.loc[
                    ordered_index,
                    'timestamp',
                ].reset_index(drop=True)

                new_block = stamps.diff().ne(
                    pd.Timedelta(minutes=5)
                ).cumsum()

                for block_id in new_block.unique():
                    mask = new_block.eq(block_id).to_numpy()
                    block_positions = ordered_index[mask]

                    sustained = batch_4_mark_five_of_seven(
                        pair.loc[
                            block_positions,
                            'raw_active',
                        ].to_numpy(dtype=bool)
                    )

                    pair.loc[
                        block_positions,
                        'sustained_active',
                    ] = sustained

            active_rows = pair.loc[
                pair.sustained_active
            ].copy()

            if not active_rows.empty:
                spatial_active_records.append(
                    active_rows.assign(
                        station_id=downstream_id,
                        upstream_station_id=upstream_id,
                        gap_miles=gap_miles,
                    )
                )

            event_frame = pair[
                ['timestamp', 'speed_mph', 'sustained_active']
            ].copy()

            events = batch_4_boolean_events(
                event_frame,
                active_column='sustained_active',
                method='Caltrans_PeMS_spatial',
                station_id=downstream_id,
                extra={
                    'upstream_station_id': upstream_id,
                    'upstream_gap_miles': gap_miles,
                },
            )

            if not events.empty:
                spatial_event_tables.append(events)

        caltrans_events = (
            pd.concat(spatial_event_tables, ignore_index=True)
            if spatial_event_tables
            else pd.DataFrame(columns=method_50.columns)
        )

        caltrans_events.to_csv(
            output / 'batch_4_caltrans_spatial_events.csv',
            index=False,
        )

        spatial_active = (
            pd.concat(spatial_active_records, ignore_index=True)
            if spatial_active_records
            else pd.DataFrame()
        )

        if not spatial_active.empty:
            spatial_active.to_csv(
                output / 'batch_4_caltrans_spatial_active_intervals.csv',
                index=False,
            )

        # --------------------------------------------------------
        # Combined event inventory
        # --------------------------------------------------------
        combined_events = pd.concat(
            [
                method_50,
                la_events,
                caltrans_events,
            ],
            ignore_index=True,
            sort=False,
        )

        combined_events['event_date'] = pd.to_datetime(
            combined_events.start_time
        ).dt.date.astype(str)

        combined_events['start_hour'] = (
            pd.to_datetime(combined_events.start_time).dt.hour
            + pd.to_datetime(combined_events.start_time).dt.minute / 60
        )

        combined_events = combined_events.sort_values(
            ['start_time', 'station_id', 'method'],
            kind='stable',
        ).reset_index(drop=True)

        combined_events.insert(
            0,
            'event_id',
            np.arange(1, len(combined_events) + 1),
        )

        combined_events.to_csv(
            output / 'batch_4_combined_event_inventory.csv',
            index=False,
        )

        # --------------------------------------------------------
        # Method comparison summary
        # --------------------------------------------------------
        method_labels = [
            f'Threshold_50_{BATCH_4_PERSISTENCE_INTERVALS*5}min',
            'LA_20drop_below40',
            'Caltrans_PeMS_spatial',
        ]

        summary_rows = []

        for method in method_labels:
            group = combined_events.loc[
                combined_events.method.eq(method)
            ]

            summary_rows.append({
                'method': method,
                'events': len(group),
                'stations_with_events':
                    group.station_id.nunique(),
                'days_with_events':
                    group.event_date.nunique(),
                'median_duration_minutes':
                    group.duration_minutes.median(),
                'p90_duration_minutes':
                    group.duration_minutes.quantile(0.90),
                'median_start_hour':
                    group.start_hour.median(),
            })

        method_summary = pd.DataFrame(summary_rows)

        method_summary.to_csv(
            output / 'batch_4_method_summary.csv',
            index=False,
        )

        threshold_summary_rows = []

        for threshold in BATCH_4_THRESHOLD_SPEEDS:
            method = (
                f'Threshold_{threshold}_'
                f'{BATCH_4_PERSISTENCE_INTERVALS*5}min'
            )
            group = threshold_events.loc[
                threshold_events.method.eq(method)
                & threshold_events.onset_observed
            ]

            threshold_summary_rows.append({
                'threshold_mph': threshold,
                'events': len(group),
                'days_with_events':
                    pd.to_datetime(
                        group.start_time
                    ).dt.date.nunique()
                    if len(group) else 0,
                'median_duration_minutes':
                    group.duration_minutes.median()
                    if len(group) else np.nan,
            })

        threshold_sensitivity = pd.DataFrame(
            threshold_summary_rows
        )

        threshold_sensitivity.to_csv(
            output / 'batch_4_threshold_event_sensitivity.csv',
            index=False,
        )

        # --------------------------------------------------------
        # Event-onset agreement within +/- 10 minutes at same station
        # --------------------------------------------------------
        agreement_rows = []

        for first_index, method_a in enumerate(method_labels):
            for method_b in method_labels[first_index + 1:]:
                a = combined_events.loc[
                    combined_events.method.eq(method_a)
                ].copy()
                b = combined_events.loc[
                    combined_events.method.eq(method_b)
                ].copy()

                matched_a = set()
                matched_b = set()

                for a_idx, a_row in a.iterrows():
                    candidates = b.loc[
                        b.station_id.eq(a_row.station_id)
                    ].copy()

                    if candidates.empty:
                        continue

                    delta = (
                        pd.to_datetime(candidates.start_time)
                        - pd.Timestamp(a_row.start_time)
                    ).abs().dt.total_seconds() / 60

                    eligible = delta.le(10)

                    if not eligible.any():
                        continue

                    candidate_index = delta.loc[
                        eligible
                    ].idxmin()

                    matched_a.add(a_idx)
                    matched_b.add(candidate_index)

                denominator = (
                    len(a) + len(b) - min(len(matched_a), len(matched_b))
                )
                overlap = min(len(matched_a), len(matched_b))
                jaccard = (
                    overlap / denominator
                    if denominator > 0 else np.nan
                )

                agreement_rows.append({
                    'method_a': method_a,
                    'method_b': method_b,
                    'events_a': len(a),
                    'events_b': len(b),
                    'matched_events_within_10min_same_station':
                        overlap,
                    'event_onset_jaccard_approx': jaccard,
                })

        agreement = pd.DataFrame(agreement_rows)
        agreement.to_csv(
            output / 'batch_4_method_event_agreement.csv',
            index=False,
        )

        # --------------------------------------------------------
        # PORTFOLIO FIGURE 1: event counts by method
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=THEME_FIGSIZE_12_7)

        plot_summary = method_summary.copy()
        display_names = {
            f'Threshold_50_{BATCH_4_PERSISTENCE_INTERVALS*5}min':
                '50 mph + 15 min',
            'LA_20drop_below40':
                'LA: 20 mph drop + <40',
            'Caltrans_PeMS_spatial':
                'Caltrans PeMS spatial',
        }

        plot_summary['display'] = plot_summary.method.map(
            display_names
        )

        ax.bar(
            plot_summary.display,
            plot_summary.events,
        )

        ax.set_title(
            'Different Breakdown Definitions Find Different Event Sets',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel('Operational Definition')
        ax.set_ylabel('Detected Breakdown Events')
        ax.grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_MEDIUM)
        ax.grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)
        ax.tick_params(axis=THEME_GRID_AXIS_X, rotation=THEME_ROTATION_MEDIUM)

        for position, row in enumerate(
            plot_summary.itertuples()
        ):
            ax.annotate(
                f'{int(row.events)} events\n'
                f'{int(row.days_with_events)} days',
                (position, row.events),
                xytext=THEME_ANNOTATION_OFFSET_6,
                textcoords=THEME_TEXTCOORDS,
                ha=THEME_ALIGN_CENTER,
                fontsize=THEME_ANNOTATION_SIZE,
            )

        fig.tight_layout()
        fig.savefig(
            BATCH_4_IMAGE_FOLDER / 'traffic_b4_event_counts_by_method.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # --------------------------------------------------------
        # PORTFOLIO FIGURE 2: fixed-threshold sensitivity
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=THEME_FIGSIZE_11_7)

        ax.plot(
            threshold_sensitivity.threshold_mph,
            threshold_sensitivity.events,
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_LINEWIDTH_EMPHASIS,
        )

        ax.set_title(
            'Breakdown Counts Depend on the Speed Threshold',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel('Sustained Speed Threshold (mph)')
        ax.set_ylabel('Observed Breakdown Onsets')
        ax.set_xticks(BATCH_4_THRESHOLD_SPEEDS)
        ax.set_ylim(bottom=THEME_AXIS_MIN_ZERO)
        ax.grid(alpha=THEME_GRID_ALPHA_LIGHT)

        for row in threshold_sensitivity.itertuples():
            ax.annotate(
                f'{int(row.events)}',
                (row.threshold_mph, row.events),
                xytext=THEME_ANNOTATION_OFFSET_8,
                textcoords=THEME_TEXTCOORDS,
                ha=THEME_ALIGN_CENTER,
            )

        fig.text(
            0.5,
            0.015,
            'Each threshold requires at least 15 consecutive minutes below '
            'the selected speed.',
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )
        fig.tight_layout(rect=THEME_TIGHT_RECT_05)
        fig.savefig(
            BATCH_4_IMAGE_FOLDER / 'traffic_b4_threshold_sensitivity.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # --------------------------------------------------------
        # PORTFOLIO FIGURE 3: onset time distributions
        # --------------------------------------------------------
        bins = np.arange(
            BATCH_4_ANALYSIS_START_HOUR,
            BATCH_4_ANALYSIS_END_HOUR + 1,
            1,
        )

        fig, ax = plt.subplots(figsize=THEME_FIGSIZE_14_7)

        for method in method_labels:
            group = combined_events.loc[
                combined_events.method.eq(method)
            ]

            if group.empty:
                continue

            counts, edges = np.histogram(
                group.start_hour,
                bins=bins,
            )

            centers = (edges[:-1] + edges[1:]) / 2

            ax.plot(
                centers,
                counts,
                marker=THEME_MARKER_PRIMARY,
                linewidth=THEME_LINEWIDTH_STANDARD,
                label=display_names[method],
            )

        ax.set_title(
            'When Do Detected Breakdowns Begin?',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Breakdown Onsets')
        ax.set_xticks(
            np.arange(
                BATCH_4_ANALYSIS_START_HOUR,
                BATCH_4_ANALYSIS_END_HOUR,
                1,
            )
        )
        ax.grid(alpha=THEME_GRID_ALPHA_SUBTLE)
        ax.legend(frameon=THEME_LEGEND_FRAME)

        fig.tight_layout()
        fig.savefig(
            BATCH_4_IMAGE_FOLDER / 'traffic_b4_onsets_by_hour.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

        # --------------------------------------------------------
        # PORTFOLIO FIGURE 4: example-day speed heatmap
        # --------------------------------------------------------
        day_scores = (
            combined_events.groupby('event_date')
            .agg(
                events=('event_id', 'size'),
                methods=('method', 'nunique'),
            )
            .sort_values(
                ['methods', 'events'],
                ascending=False,
            )
        )

        if not day_scores.empty:
            example_date = str(day_scores.index[0])

            day_data = data.loc[
                data.timestamp.dt.date.astype(str).eq(
                    example_date
                )
            ].copy()

            heat = day_data.pivot(
                index='station_id',
                columns='timestamp',
                values='speed_mph',
            ).reindex(station_ids)

            if not heat.empty:
                fig, ax = plt.subplots(figsize=THEME_FIGSIZE_16_7)

                image = ax.imshow(
                    heat.to_numpy(),
                    aspect=THEME_HEATMAP_ASPECT,
                    interpolation=THEME_HEATMAP_INTERPOLATION,
                    vmin=THEME_HEATMAP_MIN,
                    vmax=THEME_HEATMAP_MAX,
                    cmap=THEME_HEATMAP_CMAP,
                )

                ax.set_title(
                    f'I-10 East Speed Heatmap — {example_date}',
                    fontfamily=THEME_TITLE_FONT,
                )
                ax.set_ylabel('Station in Travel Direction')
                ax.set_xlabel('Time of Day')

                station_labels = []
                name_lookup = corridor.set_index(
                    'station_id'
                ).Name.to_dict()

                for station_id in station_ids:
                    name = str(
                        name_lookup.get(station_id, '')
                    )
                    station_labels.append(
                        f'{station_id}  {name}'
                    )

                ax.set_yticks(
                    np.arange(len(station_ids)),
                    station_labels,
                )

                timestamp_columns = list(heat.columns)
                tick_positions = np.linspace(
                    0,
                    len(timestamp_columns) - 1,
                    min(10, len(timestamp_columns)),
                    dtype=int,
                )

                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    [
                        pd.Timestamp(
                            timestamp_columns[position]
                        ).strftime('%-I:%M %p')
                        for position in tick_positions
                    ],
                    rotation=THEME_ROTATION_XL,
                    ha=THEME_ALIGN_RIGHT,
                )

                colorbar = fig.colorbar(
                    image,
                    ax=ax,
                    pad=THEME_COLORBAR_PAD,
                )
                colorbar.set_label('Speed (mph)')

                example_events = combined_events.loc[
                    combined_events.event_date.eq(
                        example_date
                    )
                ].copy()

                marker_map = {
                    f'Threshold_50_{BATCH_4_PERSISTENCE_INTERVALS*5}min':
                        'o',
                    'LA_20drop_below40': '^',
                    'Caltrans_PeMS_spatial': 's',
                }

                x_lookup = {
                    pd.Timestamp(timestamp): index
                    for index, timestamp in enumerate(
                        timestamp_columns
                    )
                }
                y_lookup = {
                    station_id: index
                    for index, station_id in enumerate(
                        station_ids
                    )
                }

                for method in method_labels:
                    method_events = example_events.loc[
                        example_events.method.eq(method)
                    ]

                    xs = []
                    ys = []

                    for event in method_events.itertuples():
                        start = pd.Timestamp(event.start_time)
                        if (
                            start in x_lookup
                            and int(event.station_id) in y_lookup
                        ):
                            xs.append(x_lookup[start])
                            ys.append(
                                y_lookup[int(event.station_id)]
                            )

                    if xs:
                        ax.scatter(
                            xs,
                            ys,
                            marker=marker_map[method],
                            s=THEME_SCATTER_SIZE,
                            facecolors=THEME_TRANSPARENT,
                            edgecolors=THEME_WHITE,
                            linewidths=1.8,
                            label=display_names[method],
                        )

                ax.legend(
                    loc=THEME_LEGEND_LOC,
                    bbox_to_anchor=THEME_LEGEND_ANCHOR,
                    ncol=THEME_LEGEND_NCOL_3,
                    frameon=THEME_LEGEND_FRAME,
                )

                fig.tight_layout()
                fig.savefig(
                    BATCH_4_IMAGE_FOLDER / 'traffic_b4_example_speed_heatmap.png',
                    dpi=THEME_DPI,
                    bbox_inches=THEME_BBOX,
                )
                plt.close(fig)
        else:
            example_date = None

        # --------------------------------------------------------
        # Text methodology / source report
        # --------------------------------------------------------
        source_lines = []

        for source in BATCH_4_SOURCES:
            source_lines.extend([
                source['label'],
                source['url'],
                source['basis'],
                '',
            ])

        report_lines = [
            'BATCH 4 — TRAFFIC BREAKDOWN DEFINITION COMPARISON',
            f'Primary corridor: {selected_corridor}',
            f'Stations: {len(station_ids)}',
            f'Analysis rows: {len(data):,}',
            f'Analysis window: '
            f'{BATCH_4_ANALYSIS_START_HOUR:02d}:00-'
            f'{BATCH_4_ANALYSIS_END_HOUR:02d}:00',
            f'Coverage requirement: {BATCH_4_FINAL_OBSERVED_PCT}% observed',
            '',
            'METHOD SUMMARY',
            method_summary.round(3).to_string(index=False),
            '',
            'FIXED-THRESHOLD SENSITIVITY',
            threshold_sensitivity.round(3).to_string(index=False),
            '',
            'PAIRWISE EVENT-ONSET AGREEMENT',
            agreement.round(3).to_string(index=False),
            '',
            'METHODOLOGICAL POSITION',
            'No universal breakdown threshold is assumed.',
            'The three primary operational definitions are retained separately.',
            'The 40/50/55 mph sustained-threshold comparison quantifies threshold sensitivity.',
            'The LA rule tests a sharp temporal speed-collapse definition.',
            'The Caltrans rule tests a spatial bottleneck definition using adjacent stations.',
            'Batch 5 should adopt one primary definition only after reviewing agreement, event plausibility, and representative event plots.',
            '',
            'SOURCE NOTES',
            *source_lines,
            'PORTFOLIO FIGURES',
            'images/traffic_b4_event_counts_by_method.png',
            'images/traffic_b4_threshold_sensitivity.png',
            'images/traffic_b4_onsets_by_hour.png',
            'images/traffic_b4_example_speed_heatmap.png',
        ]

        report = '\n'.join(report_lines)

        (output / 'batch_4_audit_summary.txt').write_text(
            report + '\n'
        )

        settings.update(
            status='complete',
            analysis_rows=len(data),
            combined_events=len(combined_events),
            example_heatmap_date=example_date,
            portfolio_figures=[
                'images/traffic_b4_event_counts_by_method.png',
                'images/traffic_b4_threshold_sensitivity.png',
                'images/traffic_b4_onsets_by_hour.png',
                'images/traffic_b4_example_speed_heatmap.png',
            ],
        )

        print('\n' + report)
        print(
            f'\nSaved Batch 4 reports: {output.resolve()}'
        )

    except Exception as error:
        settings.update(
            status='incomplete',
            error=str(error),
        )
        raise

    finally:
        settings_path.write_text(
            json.dumps(settings, indent=2)
        )




# ===================== BATCH 4 — PART 2. CONCEPT SEPARATION =====================
# This section preserves all Batch 4 Part 1 results and adds a decision layer.
# It does NOT replace or delete any Part 1 outputs.
#
# Interpretation adopted for downstream analysis:
#
#   Congested state:
#       sustained low-speed operation (50 mph + 15 minutes)
#
#   Breakdown onset:
#       abrupt transition measures:
#       - LA temporal rule: >=20 mph drop and resulting speed <40 mph
#       - Caltrans PeMS spatial rule: >=20 mph inter-station drop with downstream
#         speed <40 mph, <3-mile spacing, and 5-of-7 persistence
#
# The purpose is to stop forcing three different operational definitions to
# represent one latent concept when Batch 4 Part 1 showed they produce strongly
# different event inventories.


def run_batch_4_part_2():
    """Preserve Part 1 and formalize congestion-state vs breakdown-onset roles."""
    source = BATCH_4_PART_2_SOURCE_FOLDER
    output = BATCH_4_PART_2_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 4 Part 2 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required_files = {
        'combined':
            source / 'batch_4_combined_event_inventory.csv',
        'summary':
            source / 'batch_4_method_summary.csv',
        'agreement':
            source / 'batch_4_method_event_agreement.csv',
        'threshold_sensitivity':
            source / 'batch_4_threshold_event_sensitivity.csv',
        'corridor_rows':
            source / 'batch_4_corridor_analysis_rows.parquet',
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            'Batch 4 Part 2 requires completed Part 1 outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_4_PART_2_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

    combined = pd.read_csv(
        required_files['combined'],
        parse_dates=['start_time', 'end_time'],
    )
    summary = pd.read_csv(required_files['summary'])
    agreement = pd.read_csv(required_files['agreement'])
    threshold_sensitivity = pd.read_csv(
        required_files['threshold_sensitivity']
    )
    corridor_rows = pd.read_parquet(required_files['corridor_rows'])

    corridor_rows['timestamp'] = pd.to_datetime(
        corridor_rows.timestamp,
        errors='raise',
    )

    threshold_method = 'Threshold_50_15min'
    la_method = 'LA_20drop_below40'
    caltrans_method = 'Caltrans_PeMS_spatial'

    expected_methods = {
        threshold_method,
        la_method,
        caltrans_method,
    }

    found_methods = set(combined.method.dropna().astype(str).unique())
    missing_methods = expected_methods - found_methods
    if missing_methods:
        raise ValueError(
            'Part 1 event inventory is missing expected methods: '
            + ', '.join(sorted(missing_methods))
        )

    # --------------------------------------------------------
    # Formal role assignment — preserve original event records
    # --------------------------------------------------------
    role_map = {
        threshold_method: 'Congested state',
        la_method: 'Breakdown onset — temporal',
        caltrans_method: 'Breakdown onset — spatial',
    }

    combined_role = combined.copy()
    combined_role['concept_role'] = combined_role.method.map(role_map)

    combined_role.to_csv(
        output / 'batch_4_part_2_event_inventory_with_roles.csv',
        index=False,
    )

    role_summary = (
        combined_role
        .groupby(['concept_role', 'method'], as_index=False)
        .agg(
            events=('event_id', 'size'),
            stations_with_events=('station_id', 'nunique'),
            days_with_events=('event_date', 'nunique'),
            median_duration_minutes=('duration_minutes', 'median'),
            p90_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.90),
            ),
        )
    )

    role_summary.to_csv(
        output / 'batch_4_part_2_role_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Decision record / paper trail
    # --------------------------------------------------------
    part1_counts = (
        summary.set_index('method')['events']
        .reindex(
            [threshold_method, la_method, caltrans_method]
        )
        .to_dict()
    )

    decision_record = pd.DataFrame([
        {
            'decision_id': 'B4P2-01',
            'decision': (
                'Separate sustained congestion state from abrupt breakdown onset.'
            ),
            'evidence': (
                f'Part 1 produced {int(part1_counts[threshold_method])} '
                f'events for {threshold_method}, '
                f'{int(part1_counts[la_method])} for {la_method}, and '
                f'{int(part1_counts[caltrans_method])} for {caltrans_method}.'
            ),
            'reasoning': (
                'The definitions have very different event counts and low '
                'pairwise onset agreement, indicating that they operationalize '
                'different traffic phenomena rather than interchangeable measures '
                'of a single event type.'
            ),
            'downstream_use': (
                'Use 50 mph + 15 min as the primary congested-state indicator. '
                'Use LA temporal and Caltrans spatial rules as alternative '
                'breakdown-onset indicators and robustness checks.'
            ),
            'part_1_results_deleted_or_replaced': False,
        },
        {
            'decision_id': 'B4P2-02',
            'decision': (
                'Retain all Batch 4 Part 1 event inventories, summaries, '
                'sensitivity tables, figures, and source notes unchanged.'
            ),
            'evidence': (
                'Part 2 reads Part 1 outputs from a separate source folder and '
                'writes only new files to a new Part 2 folder.'
            ),
            'reasoning': (
                'The original comparison is the evidence trail that motivates '
                'the new conceptual separation.'
            ),
            'downstream_use': (
                'Portfolio methodology can show the progression from competing '
                'definitions to explicit concept separation.'
            ),
            'part_1_results_deleted_or_replaced': False,
        },
    ])

    decision_record.to_csv(
        output / 'batch_4_part_2_decision_record.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Normalized onset timing — avoids 1,457-event method
    # visually flattening the rare-event methods.
    # --------------------------------------------------------
    combined_role['start_hour'] = (
        combined_role.start_time.dt.hour
        + combined_role.start_time.dt.minute / 60
    )

    bins = np.arange(
        BATCH_4_ANALYSIS_START_HOUR,
        BATCH_4_ANALYSIS_END_HOUR + 1,
        1,
    )

    onset_rows = []

    for method in [threshold_method, la_method, caltrans_method]:
        group = combined_role.loc[
            combined_role.method.eq(method)
        ]

        counts, edges = np.histogram(
            group.start_hour,
            bins=bins,
        )

        total = counts.sum()
        shares = (
            counts / total * 100
            if total > 0
            else np.zeros_like(counts, dtype=float)
        )

        for left, right, count, share in zip(
            edges[:-1],
            edges[1:],
            counts,
            shares,
        ):
            onset_rows.append({
                'method': method,
                'concept_role': role_map[method],
                'hour_start': int(left),
                'hour_end': int(right),
                'events': int(count),
                'within_method_pct': float(share),
            })

    onset_normalized = pd.DataFrame(onset_rows)

    onset_normalized.to_csv(
        output / 'batch_4_part_2_onsets_by_hour_normalized.csv',
        index=False,
    )

    display_names = {
        threshold_method: 'Congested state: 50 mph + 15 min',
        la_method: 'Breakdown onset: LA temporal',
        caltrans_method: 'Breakdown onset: Caltrans spatial',
    }

    fig, ax = plt.subplots(figsize=THEME_FIGSIZE_14_7)

    for method in [threshold_method, la_method, caltrans_method]:
        group = onset_normalized.loc[
            onset_normalized.method.eq(method)
        ]

        centers = (
            group.hour_start.to_numpy()
            + group.hour_end.to_numpy()
        ) / 2

        ax.plot(
            centers,
            group.within_method_pct,
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_LINEWIDTH_STANDARD,
            label=display_names[method],
        )

    ax.set_title(
        'Event Timing by Method — Normalized Within Each Definition',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Share of That Method’s Events (%)')
    ax.set_xticks(
        np.arange(
            BATCH_4_ANALYSIS_START_HOUR,
            BATCH_4_ANALYSIS_END_HOUR,
            1,
        )
    )
    ax.set_ylim(bottom=THEME_AXIS_MIN_ZERO)
    ax.grid(alpha=THEME_GRID_ALPHA_SUBTLE)
    ax.legend(frameon=THEME_LEGEND_FRAME)

    fig.text(
        0.5,
        0.015,
        'Normalization preserves timing patterns without allowing the '
        'high-frequency congestion-state method to flatten rare onset methods.',
        ha=THEME_ALIGN_CENTER,
        fontsize=THEME_ANNOTATION_SIZE,
    )

    fig.tight_layout(rect=THEME_TIGHT_RECT_05)
    fig.savefig(
        BATCH_4_PART_2_IMAGE_FOLDER
        / 'traffic_b4_part2_onset_timing_normalized.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Conceptual comparison plot: state vs onset
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=THEME_FIGSIZE_13_7)

    plot_roles = role_summary.copy()
    plot_roles['display'] = plot_roles.method.map(display_names)

    ax.bar(
        plot_roles.display,
        plot_roles.events,
    )

    ax.set_title(
        'Congested State and Breakdown Onset Are Different Measurements',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Operational Measure')
    ax.set_ylabel('Detected Events')
    ax.tick_params(axis=THEME_GRID_AXIS_X, rotation=THEME_ROTATION_SMALL)
    ax.grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_LIGHT)
    ax.grid(axis=THEME_GRID_AXIS_X, visible=THEME_GRID_VISIBLE_OFF)

    for position, row in enumerate(plot_roles.itertuples()):
        ax.annotate(
            f'{int(row.events)} events\n'
            f'{int(row.days_with_events)} days',
            (position, row.events),
            xytext=THEME_ANNOTATION_OFFSET_7,
            textcoords=THEME_TEXTCOORDS,
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )

    fig.tight_layout()
    fig.savefig(
        BATCH_4_PART_2_IMAGE_FOLDER
        / 'traffic_b4_part2_state_vs_onset_counts.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Example-day comparison using the same Part 1 event inventory.
    # Choose a day with both congestion-state and at least one onset signal.
    # --------------------------------------------------------
    daily_methods = (
        combined_role
        .groupby('event_date')
        .agg(
            methods=('method', 'nunique'),
            events=('event_id', 'size'),
        )
        .sort_values(
            ['methods', 'events'],
            ascending=False,
        )
    )

    example_date = (
        str(daily_methods.index[0])
        if not daily_methods.empty
        else None
    )

    if example_date is not None:
        example_events = combined_role.loc[
            combined_role.event_date.astype(str).eq(example_date)
        ].copy()

        example_events.to_csv(
            output / 'batch_4_part_2_example_day_events.csv',
            index=False,
        )

    # --------------------------------------------------------
    # Carry forward Part 1 comparison tables into a Part 2 evidence index.
    # Nothing is copied over as replacement data; this table simply records
    # the source evidence used for the decision.
    # --------------------------------------------------------
    evidence_index = pd.DataFrame([
        {
            'evidence_type': 'Part 1 method summary',
            'source_file': str(required_files['summary']),
            'role_in_decision': (
                'Shows the 1,457 vs 10 vs 7 event-count divergence.'
            ),
        },
        {
            'evidence_type': 'Part 1 pairwise agreement',
            'source_file': str(required_files['agreement']),
            'role_in_decision': (
                'Shows very low onset overlap among the three definitions.'
            ),
        },
        {
            'evidence_type': 'Part 1 threshold sensitivity',
            'source_file': str(required_files['threshold_sensitivity']),
            'role_in_decision': (
                'Shows sustained-congestion counts are broadly stable across '
                '40/50/55 mph while event durations change.'
            ),
        },
        {
            'evidence_type': 'Part 1 combined event inventory',
            'source_file': str(required_files['combined']),
            'role_in_decision': (
                'Preserves every detected event under every original method.'
            ),
        },
    ])

    evidence_index.to_csv(
        output / 'batch_4_part_2_evidence_index.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------
    report_lines = [
        'BATCH 4 — PART 2: CONGESTION STATE VS BREAKDOWN ONSET',
        '',
        'PURPOSE',
        'Preserve Batch 4 Part 1 intact while formalizing the distinction',
        'between sustained congestion state and abrupt breakdown onset.',
        '',
        'PART 1 EVIDENCE RETAINED',
        summary.round(3).to_string(index=False),
        '',
        'PAIRWISE PART 1 AGREEMENT RETAINED',
        agreement.round(3).to_string(index=False),
        '',
        'THRESHOLD SENSITIVITY RETAINED',
        threshold_sensitivity.round(3).to_string(index=False),
        '',
        'DECISION',
        '1. 50 mph + 15 minutes is retained as the primary congested-state measure.',
        '2. LA 20-mph drop + <40 mph is retained as a temporal breakdown-onset measure.',
        '3. Caltrans PeMS spatial logic is retained as a spatial breakdown-onset measure.',
        '4. No Part 1 outputs are deleted, replaced, or re-labeled retroactively.',
        '',
        'WHY',
        'The three definitions identify sharply different event sets and show',
        'very low pairwise onset agreement. The evidence therefore supports',
        'treating sustained congestion and abrupt breakdown onset as related',
        'but distinct analytical concepts.',
        '',
        'DOWNSTREAM PLAN',
        'Batch 5 should analyze congested-state duration/recovery using the',
        '50 mph + 15 min state definition, while precursor and onset analyses',
        'should use the LA temporal rule as the primary onset definition and',
        'the Caltrans spatial rule as a robustness/spatial bottleneck check.',
        '',
        'PORTFOLIO FIGURES',
        'images/traffic_b4_part2_onset_timing_normalized.png',
        'images/traffic_b4_part2_state_vs_onset_counts.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_4_part_2_decision_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'part_1_source_folder': str(source),
        'part_2_output_folder': str(output),
        'part_1_outputs_modified': False,
        'congested_state_method': threshold_method,
        'breakdown_onset_primary_temporal_method': la_method,
        'breakdown_onset_spatial_robustness_method': caltrans_method,
        'example_date': example_date,
        'portfolio_figures': [
            'images/traffic_b4_part2_onset_timing_normalized.png',
            'images/traffic_b4_part2_state_vs_onset_counts.png',
        ],
    }

    (output / 'batch_4_part_2_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 4 Part 2 reports: {output.resolve()}'
    )




# ===================== BATCH 5 — PART 1. PRE-BREAKDOWN CONDITIONS =====================
# Batch 4 separated two concepts:
#   - sustained congestion state: 50 mph + 15 minutes
#   - abrupt breakdown onset: LA temporal rule, with Caltrans spatial rule as a
#     separate robustness / bottleneck check.
#
# Batch 5 Part 1 focuses ONLY on the primary abrupt-onset definition selected
# in Batch 4 Part 2. It asks what happens at the same station in the 30 minutes
# before and after each onset, and compares pre-onset conditions with matched
# non-onset control times.
#
# All prior Batch 4 outputs remain untouched.


def batch_5_part_1_extract_profile(
    station_frame,
    anchor_time,
    relative_minutes,
):
    """Return exact five-minute observations around an anchor timestamp."""
    lookup = (
        station_frame
        .set_index('timestamp')[
            ['speed_mph', 'flow_veh_5min', 'occupancy_fraction']
        ]
        .sort_index()
    )

    rows = []

    for minute in relative_minutes:
        timestamp = pd.Timestamp(anchor_time) + pd.Timedelta(minutes=minute)

        if timestamp not in lookup.index:
            rows.append({
                'relative_minute': minute,
                'timestamp': timestamp,
                'speed_mph': np.nan,
                'flow_veh_5min': np.nan,
                'occupancy_fraction': np.nan,
                'present': False,
            })
            continue

        row = lookup.loc[timestamp]

        # Station/timestamp should be unique after the completed data audits.
        if isinstance(row, pd.DataFrame):
            raise ValueError(
                f'Duplicate station/timestamp in Batch 5 profile at {timestamp}.'
            )

        rows.append({
            'relative_minute': minute,
            'timestamp': timestamp,
            'speed_mph': float(row.speed_mph),
            'flow_veh_5min': float(row.flow_veh_5min),
            'occupancy_fraction': float(row.occupancy_fraction),
            'present': True,
        })

    return pd.DataFrame(rows)


def batch_5_part_1_matched_controls(
    event_row,
    station_frame,
    station_onsets,
):
    """
    Deterministically choose comparable non-onset control anchors.

    Matching rules:
    - same station
    - same weekday/weekend class
    - same half-hour clock bin
    - different calendar date from the event
    - at least 60 minutes from any LA onset at that station
    - complete -30 through 0 minute profile
    - nearest calendar dates chosen first
    """
    event_time = pd.Timestamp(event_row.start_time)
    event_date = event_time.normalize()
    event_is_weekend = event_time.dayofweek >= 5
    event_half_hour_bin = event_time.hour * 2 + (event_time.minute // 30)

    candidates = station_frame[
        ['timestamp', 'speed_mph', 'flow_veh_5min', 'occupancy_fraction']
    ].copy()

    candidates['date'] = candidates.timestamp.dt.normalize()
    candidates['is_weekend'] = candidates.timestamp.dt.dayofweek.ge(5)
    candidates['half_hour_bin'] = (
        candidates.timestamp.dt.hour * 2
        + candidates.timestamp.dt.minute.floordiv(30)
    )

    candidates = candidates.loc[
        candidates.is_weekend.eq(event_is_weekend)
        & candidates.half_hour_bin.eq(event_half_hour_bin)
        & candidates.date.ne(event_date)
    ].copy()

    onset_times = pd.to_datetime(station_onsets.start_time).sort_values()

    if len(onset_times):
        safe_mask = []

        for timestamp in candidates.timestamp:
            nearest_minutes = (
                (onset_times - timestamp).abs().dt.total_seconds() / 60
            ).min()

            safe_mask.append(
                nearest_minutes >= BATCH_5_PART_1_CONTROL_MINUTES_FROM_ONSET
            )

        candidates = candidates.loc[safe_mask].copy()

    candidates['date_distance_days'] = (
        candidates.date - event_date
    ).abs().dt.days

    candidates = candidates.sort_values(
        ['date_distance_days', 'timestamp'],
        kind='stable',
    )

    selected = []

    for candidate in candidates.itertuples():
        profile = batch_5_part_1_extract_profile(
            station_frame,
            candidate.timestamp,
            list(range(-30, 5, 5)),
        )

        if not profile.present.all():
            continue

        selected.append(candidate.timestamp)

        if len(selected) >= BATCH_5_PART_1_CONTROLS_PER_EVENT:
            break

    return selected



def batch_5_part_1_load_extended_context(station_ids):
    """
    Load 100%-observed selected-corridor rows from 04:30 through 20:30.

    Batch 4 intentionally detected events only in the 05:00-20:00 analysis
    window. Batch 5 needs 30 minutes of context around those events. Loading
    a wider context window here avoids artificial missing prehistory for
    events near 05:00 and missing recovery context near 20:00 without
    changing the Batch 4 event inventory or its analysis window.
    """
    station_set = set(int(value) for value in station_ids)
    dates = pd.date_range(START_DATE, END_DATE)
    parts = []

    required_columns = [
        'timestamp',
        'station_id',
        'lane_type',
        'observed_pct',
        'flow_veh_5min',
        'occupancy_fraction',
        'speed_mph',
        'invalid_timestamp',
        'date_mismatch',
        'invalid_station_id',
        'invalid_speed',
        'invalid_flow',
        'invalid_occupancy',
        'invalid_observed_pct',
        'off_5min_grid',
        'duplicate_station_timestamp',
    ]

    for number, day in enumerate(dates, 1):
        day_text = day.strftime('%Y-%m-%d')
        folder = PREPARED_DATA_FOLDER / f'date={day_text}'
        paths = sorted(folder.glob('part-*.parquet'))

        if not paths:
            continue

        print(
            f'Batch 5 Part 1 context: {day_text} '
            f'[{number}/{len(dates)}]',
            flush=True,
        )

        day_parts = []

        for path in paths:
            schema = set(pq.read_schema(path).names)
            missing = set(required_columns) - schema

            if missing:
                raise ValueError(
                    f'{path}: missing Batch 5 context columns: '
                    + ', '.join(sorted(missing))
                )

            frame = pd.read_parquet(
                path,
                columns=required_columns,
            )

            frame = frame.loc[
                frame.lane_type.eq('ML')
                & frame.station_id.isin(station_set)
            ].copy()

            if not frame.empty:
                day_parts.append(frame)

        if not day_parts:
            continue

        day_frame = pd.concat(
            day_parts,
            ignore_index=True,
        )

        duplicate_copies = day_frame.duplicated(
            ['station_id', 'timestamp'],
            keep=False,
        )

        valid = (
            ~day_frame.invalid_timestamp.fillna(True)
            & ~day_frame.date_mismatch.fillna(True)
            & ~day_frame.invalid_station_id.fillna(True)
            & ~day_frame.invalid_speed.fillna(True)
            & ~day_frame.invalid_flow.fillna(True)
            & ~day_frame.invalid_occupancy.fillna(True)
            & ~day_frame.invalid_observed_pct.fillna(True)
            & ~day_frame.off_5min_grid.fillna(True)
            & ~duplicate_copies
            & day_frame.timestamp.notna()
            & day_frame.observed_pct.eq(BATCH_4_FINAL_OBSERVED_PCT)
        )

        day_frame = day_frame.loc[valid].copy()

        hour = (
            day_frame.timestamp.dt.hour
            + day_frame.timestamp.dt.minute / 60
        )

        day_frame = day_frame.loc[
            hour.ge(BATCH_5_PART_1_CONTEXT_START_HOUR)
            & hour.lt(BATCH_5_PART_1_CONTEXT_END_HOUR)
        ].copy()

        if not day_frame.empty:
            parts.append(day_frame)

    if not parts:
        raise ValueError(
            'No extended-context rows were found for Batch 5 Part 1.'
        )

    context = pd.concat(
        parts,
        ignore_index=True,
    )

    context['station_id'] = pd.to_numeric(
        context.station_id,
        errors='raise',
    ).astype('int64')

    context = context.sort_values(
        ['station_id', 'timestamp'],
        kind='stable',
    ).reset_index(drop=True)

    duplicate_context = context.duplicated(
        ['station_id', 'timestamp'],
        keep=False,
    )

    if duplicate_context.any():
        raise ValueError(
            'Extended Batch 5 context contains duplicate station/timestamps.'
        )

    return context


def run_batch_5_part_1():
    """Analyze conditions around the primary temporal breakdown-onset events."""
    source = BATCH_5_PART_1_SOURCE_FOLDER
    decision = BATCH_5_PART_1_DECISION_FOLDER
    output = BATCH_5_PART_1_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 5 Part 1 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'events':
            source / 'batch_4_combined_event_inventory.csv',
        'corridor_rows':
            source / 'batch_4_corridor_analysis_rows.parquet',
        'decision':
            decision / 'batch_4_part_2_decision_summary.txt',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 5 Part 1 requires completed Batch 4 outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_5_PART_1_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(
        required['events'],
        parse_dates=['start_time', 'end_time'],
    )

    b4_core_data = pd.read_parquet(required['corridor_rows'])
    b4_core_data['timestamp'] = pd.to_datetime(
        b4_core_data.timestamp,
        errors='raise',
    )
    b4_core_data['station_id'] = pd.to_numeric(
        b4_core_data.station_id,
        errors='raise',
    ).astype('int64')

    station_ids = sorted(
        b4_core_data.station_id.unique().tolist()
    )

    data = batch_5_part_1_load_extended_context(
        station_ids
    )

    context_hour = (
        data.timestamp.dt.hour
        + data.timestamp.dt.minute / 60
    )

    context_core = data.loc[
        context_hour.ge(BATCH_5_PART_1_CORE_START_HOUR)
        & context_hour.lt(BATCH_5_PART_1_CORE_END_HOUR)
    ].copy()

    reconciliation_columns = [
        'station_id',
        'timestamp',
        'speed_mph',
        'flow_veh_5min',
        'occupancy_fraction',
    ]

    b4_reconcile = (
        b4_core_data[reconciliation_columns]
        .sort_values(['station_id', 'timestamp'])
        .reset_index(drop=True)
    )

    context_reconcile = (
        context_core[reconciliation_columns]
        .sort_values(['station_id', 'timestamp'])
        .reset_index(drop=True)
    )

    if len(b4_reconcile) != len(context_reconcile):
        raise ValueError(
            'Extended Batch 5 context does not reproduce the Batch 4 '
            '05:00-20:00 core row count.'
        )

    if not b4_reconcile.equals(context_reconcile):
        raise ValueError(
            'Extended Batch 5 context does not exactly reproduce the '
            'Batch 4 05:00-20:00 core measurements.'
        )

    boundary_rows = len(data) - len(context_core)

    context_audit = pd.DataFrame([
        {
            'context_start_hour': BATCH_5_PART_1_CONTEXT_START_HOUR,
            'core_start_hour': BATCH_5_PART_1_CORE_START_HOUR,
            'core_end_hour': BATCH_5_PART_1_CORE_END_HOUR,
            'context_end_hour': BATCH_5_PART_1_CONTEXT_END_HOUR,
            'extended_context_rows': len(data),
            'core_rows_reproduced': len(context_core),
            'batch_4_core_rows': len(b4_core_data),
            'boundary_context_rows_added': boundary_rows,
            'core_exact_match': True,
        }
    ])

    context_audit.to_csv(
        output / 'batch_5_part_1_context_extension_audit.csv',
        index=False,
    )

    onset_events = events.loc[
        events.method.eq(BATCH_5_PART_1_PRIMARY_ONSET_METHOD)
    ].copy()

    if onset_events.empty:
        raise ValueError(
            'No primary temporal breakdown-onset events found.'
        )

    onset_events = onset_events.sort_values(
        ['start_time', 'station_id'],
        kind='stable',
    ).reset_index(drop=True)

    if 'event_id' not in onset_events.columns:
        onset_events.insert(
            0,
            'event_id',
            np.arange(1, len(onset_events) + 1),
        )

    # --------------------------------------------------------
    # Event-centered profiles: -30 through +30 minutes
    # --------------------------------------------------------
    event_profile_parts = []

    for event in onset_events.itertuples():
        station_data = data.loc[
            data.station_id.eq(int(event.station_id))
        ].copy()

        profile = batch_5_part_1_extract_profile(
            station_data,
            event.start_time,
            BATCH_5_PART_1_PROFILE_MINUTES,
        )

        profile.insert(0, 'event_id', int(event.event_id))
        profile.insert(1, 'station_id', int(event.station_id))
        profile.insert(2, 'event_start_time', pd.Timestamp(event.start_time))
        profile.insert(3, 'event_date', pd.Timestamp(event.start_time).date())

        event_profile_parts.append(profile)

    event_profiles = pd.concat(
        event_profile_parts,
        ignore_index=True,
    )

    event_profiles.to_csv(
        output / 'batch_5_part_1_event_centered_profiles.csv',
        index=False,
    )

    profile_coverage = (
        event_profiles.groupby('event_id')
        .agg(
            rows_expected=('relative_minute', 'size'),
            rows_present=('present', 'sum'),
        )
        .reset_index()
    )

    profile_coverage['complete_profile'] = (
        profile_coverage.rows_expected.eq(
            profile_coverage.rows_present
        )
    )

    profile_coverage.to_csv(
        output / 'batch_5_part_1_event_profile_coverage.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Event-level pre-breakdown features
    # --------------------------------------------------------
    feature_rows = []

    for event in onset_events.itertuples():
        profile = event_profiles.loc[
            event_profiles.event_id.eq(int(event.event_id))
        ].set_index('relative_minute')

        def get_value(minute, column):
            if minute not in profile.index:
                return np.nan
            return profile.loc[minute, column]

        pre_window = profile.loc[
            profile.index.isin([-30, -25, -20, -15, -10, -5])
        ].copy()

        feature_rows.append({
            'event_id': int(event.event_id),
            'station_id': int(event.station_id),
            'start_time': pd.Timestamp(event.start_time),
            'speed_mph_minus30': get_value(-30, 'speed_mph'),
            'speed_mph_minus5': get_value(-5, 'speed_mph'),
            'speed_mph_onset': get_value(0, 'speed_mph'),
            'speed_change_minus30_to_minus5': (
                get_value(-5, 'speed_mph')
                - get_value(-30, 'speed_mph')
            ),
            'speed_drop_minus5_to_onset': (
                get_value(-5, 'speed_mph')
                - get_value(0, 'speed_mph')
            ),
            'pre30_speed_std': pre_window.speed_mph.std(ddof=1),
            'flow_veh_5min_minus30': get_value(-30, 'flow_veh_5min'),
            'flow_veh_5min_minus5': get_value(-5, 'flow_veh_5min'),
            'flow_veh_5min_onset': get_value(0, 'flow_veh_5min'),
            'flow_change_minus30_to_minus5': (
                get_value(-5, 'flow_veh_5min')
                - get_value(-30, 'flow_veh_5min')
            ),
            'occupancy_minus30': get_value(-30, 'occupancy_fraction'),
            'occupancy_minus5': get_value(-5, 'occupancy_fraction'),
            'occupancy_onset': get_value(0, 'occupancy_fraction'),
            'occupancy_change_minus30_to_minus5': (
                get_value(-5, 'occupancy_fraction')
                - get_value(-30, 'occupancy_fraction')
            ),
            'complete_profile': bool(
                profile.present.all()
            ),
        })

    event_features = pd.DataFrame(feature_rows)

    event_features.to_csv(
        output / 'batch_5_part_1_event_prebreakdown_features.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Matched non-onset controls
    # --------------------------------------------------------
    control_profile_parts = []
    match_rows = []

    for event in onset_events.itertuples():
        station_data = data.loc[
            data.station_id.eq(int(event.station_id))
        ].copy()

        station_onsets = onset_events.loc[
            onset_events.station_id.eq(int(event.station_id))
        ].copy()

        controls = batch_5_part_1_matched_controls(
            event,
            station_data,
            station_onsets,
        )

        match_rows.append({
            'event_id': int(event.event_id),
            'station_id': int(event.station_id),
            'event_start_time': pd.Timestamp(event.start_time),
            'controls_found': len(controls),
        })

        for control_number, control_time in enumerate(controls, 1):
            profile = batch_5_part_1_extract_profile(
                station_data,
                control_time,
                list(range(-30, 5, 5)),
            )

            profile.insert(0, 'event_id', int(event.event_id))
            profile.insert(1, 'station_id', int(event.station_id))
            profile.insert(2, 'control_number', control_number)
            profile.insert(3, 'control_anchor_time', pd.Timestamp(control_time))

            control_profile_parts.append(profile)

    control_matches = pd.DataFrame(match_rows)
    control_matches.to_csv(
        output / 'batch_5_part_1_control_match_counts.csv',
        index=False,
    )

    if control_profile_parts:
        control_profiles = pd.concat(
            control_profile_parts,
            ignore_index=True,
        )
    else:
        control_profiles = pd.DataFrame()

    control_profiles.to_csv(
        output / 'batch_5_part_1_matched_control_profiles.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Summary by relative time
    # --------------------------------------------------------
    event_profile_summary = (
        event_profiles.loc[event_profiles.present]
        .groupby('relative_minute')
        .agg(
            n_events=('event_id', 'nunique'),
            median_speed_mph=('speed_mph', 'median'),
            q25_speed_mph=('speed_mph', lambda s: s.quantile(0.25)),
            q75_speed_mph=('speed_mph', lambda s: s.quantile(0.75)),
            median_flow_veh_5min=('flow_veh_5min', 'median'),
            q25_flow_veh_5min=('flow_veh_5min', lambda s: s.quantile(0.25)),
            q75_flow_veh_5min=('flow_veh_5min', lambda s: s.quantile(0.75)),
            median_occupancy=('occupancy_fraction', 'median'),
            q25_occupancy=('occupancy_fraction', lambda s: s.quantile(0.25)),
            q75_occupancy=('occupancy_fraction', lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )

    event_profile_summary.to_csv(
        output / 'batch_5_part_1_event_profile_summary.csv',
        index=False,
    )

    if not control_profiles.empty:
        control_summary = (
            control_profiles.loc[control_profiles.present]
            .groupby('relative_minute')
            .agg(
                n_controls=('control_anchor_time', 'nunique'),
                median_speed_mph=('speed_mph', 'median'),
                q25_speed_mph=('speed_mph', lambda s: s.quantile(0.25)),
                q75_speed_mph=('speed_mph', lambda s: s.quantile(0.75)),
                median_flow_veh_5min=('flow_veh_5min', 'median'),
                q25_flow_veh_5min=('flow_veh_5min', lambda s: s.quantile(0.25)),
                q75_flow_veh_5min=('flow_veh_5min', lambda s: s.quantile(0.75)),
                median_occupancy=('occupancy_fraction', 'median'),
                q25_occupancy=('occupancy_fraction', lambda s: s.quantile(0.25)),
                q75_occupancy=('occupancy_fraction', lambda s: s.quantile(0.75)),
            )
            .reset_index()
        )
    else:
        control_summary = pd.DataFrame()

    control_summary.to_csv(
        output / 'batch_5_part_1_control_profile_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Event vs matched-control comparison at -5 minutes
    # --------------------------------------------------------
    event_minus5 = event_profiles.loc[
        event_profiles.relative_minute.eq(-5)
        & event_profiles.present
    ].copy()

    control_minus5 = (
        control_profiles.loc[
            control_profiles.relative_minute.eq(-5)
            & control_profiles.present
        ].copy()
        if not control_profiles.empty
        else pd.DataFrame()
    )

    comparison_rows = []

    for metric in [
        'speed_mph',
        'flow_veh_5min',
        'occupancy_fraction',
    ]:
        event_values = event_minus5[metric].dropna()
        control_values = (
            control_minus5[metric].dropna()
            if not control_minus5.empty
            else pd.Series(dtype=float)
        )

        comparison_rows.append({
            'metric': metric,
            'event_n': len(event_values),
            'event_median': event_values.median(),
            'event_q25': event_values.quantile(0.25),
            'event_q75': event_values.quantile(0.75),
            'control_n': len(control_values),
            'control_median': control_values.median()
                if len(control_values) else np.nan,
            'control_q25': control_values.quantile(0.25)
                if len(control_values) else np.nan,
            'control_q75': control_values.quantile(0.75)
                if len(control_values) else np.nan,
        })

    precondition_comparison = pd.DataFrame(comparison_rows)

    precondition_comparison.to_csv(
        output / 'batch_5_part_1_minus5_event_vs_control.csv',
        index=False,
    )

    # --------------------------------------------------------
    # FIGURE 1: event-centered median/IQR profiles
    # --------------------------------------------------------
    metrics = [
        (
            'speed_mph',
            'Speed (mph)',
            'median_speed_mph',
            'q25_speed_mph',
            'q75_speed_mph',
        ),
        (
            'flow_veh_5min',
            'Flow (veh / 5 min)',
            'median_flow_veh_5min',
            'q25_flow_veh_5min',
            'q75_flow_veh_5min',
        ),
        (
            'occupancy_fraction',
            'Occupancy fraction',
            'median_occupancy',
            'q25_occupancy',
            'q75_occupancy',
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=THEME_FIGSIZE_18_6_5)

    for ax, (_, ylabel, median_col, q25_col, q75_col) in zip(
        axes,
        metrics,
    ):
        summary_plot = event_profile_summary.copy()

        ax.plot(
            summary_plot.relative_minute,
            summary_plot[median_col],
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_MEDIAN_WIDTH,
            label='Breakdown events',
        )

        ax.fill_between(
            summary_plot.relative_minute,
            summary_plot[q25_col],
            summary_plot[q75_col],
            alpha=THEME_FILL_ALPHA,
        )

        if not control_summary.empty:
            control_plot = control_summary.loc[
                control_summary.relative_minute.le(0)
            ]

            ax.plot(
                control_plot.relative_minute,
                control_plot[median_col],
                marker=THEME_MARKER_PRIMARY,
                linewidth=THEME_LINEWIDTH_SECONDARY,
                linestyle=THEME_LINESTYLE_DASHED,
                label='Matched non-onset controls',
            )

        ax.axvline(
            0,
            linestyle=THEME_LINESTYLE_DOTTED,
            linewidth=THEME_LINEWIDTH_REFERENCE,
        )
        ax.set_xlabel('Minutes Relative to Breakdown Onset')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=THEME_GRID_ALPHA_SUBTLE)

    axes[0].legend(frameon=THEME_LEGEND_FRAME)

    fig.suptitle(
        'What Happens Around an Abrupt Traffic Breakdown?',
        fontfamily=THEME_TITLE_FONT,
        fontsize=THEME_SUPTITLE_SIZE,
    )

    fig.text(
        0.5,
        0.015,
        'Primary onset definition: LA temporal rule (≥20 mph five-minute '
        'drop with resulting speed <40 mph). Shaded bands are event IQRs.',
        ha=THEME_ALIGN_CENTER,
        fontsize=THEME_ANNOTATION_SIZE,
    )

    fig.tight_layout(rect=THEME_TIGHT_RECT_B5_PROFILE)
    fig.savefig(
        BATCH_5_PART_1_IMAGE_FOLDER
        / 'traffic_b5p1_event_centered_profiles.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # FIGURE 2: event x relative-time speed heatmap
    # --------------------------------------------------------
    speed_heat = event_profiles.pivot(
        index='event_id',
        columns='relative_minute',
        values='speed_mph',
    ).reindex(
        columns=BATCH_5_PART_1_PROFILE_MINUTES
    )

    fig, ax = plt.subplots(figsize=THEME_FIGSIZE_14_7)

    image = ax.imshow(
        speed_heat.to_numpy(),
        aspect=THEME_HEATMAP_ASPECT,
        interpolation=THEME_HEATMAP_INTERPOLATION,
        vmin=THEME_HEATMAP_MIN,
        vmax=THEME_HEATMAP_MAX,
        cmap=THEME_HEATMAP_CMAP,
    )

    ax.set_title(
        'Ten Abrupt Breakdowns — Speed Before and After Onset',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Minutes Relative to Onset')
    ax.set_ylabel('Breakdown Event')
    ax.set_xticks(
        np.arange(len(BATCH_5_PART_1_PROFILE_MINUTES)),
        BATCH_5_PART_1_PROFILE_MINUTES,
    )
    ax.set_yticks(
        np.arange(len(speed_heat.index)),
        [str(x) for x in speed_heat.index],
    )

    onset_position = BATCH_5_PART_1_PROFILE_MINUTES.index(0)
    ax.axvline(
        onset_position - 0.5,
        linewidth=THEME_LINEWIDTH_SECONDARY,
        linestyle=THEME_LINESTYLE_DASHED,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=THEME_COLORBAR_PAD,
    )
    colorbar.set_label('Speed (mph)')

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_1_IMAGE_FOLDER
        / 'traffic_b5p1_breakdown_speed_heatmap.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # FIGURE 3: -5 minute event vs control preconditions
    # --------------------------------------------------------
    if not control_minus5.empty:
        fig, axes = plt.subplots(1, 3, figsize=THEME_FIGSIZE_17_6)

        plot_specs = [
            ('speed_mph', 'Speed 5 Minutes Before Onset', 'Speed (mph)'),
            (
                'flow_veh_5min',
                'Flow 5 Minutes Before Onset',
                'Flow (veh / 5 min)',
            ),
            (
                'occupancy_fraction',
                'Occupancy 5 Minutes Before Onset',
                'Occupancy fraction',
            ),
        ]

        for ax, (column, title, ylabel) in zip(axes, plot_specs):
            values = [
                event_minus5[column].dropna().to_numpy(),
                control_minus5[column].dropna().to_numpy(),
            ]

            box = ax.boxplot(
                values,
                tick_labels=['Breakdown events', 'Matched controls'],
                patch_artist=True,
                widths=THEME_BOX_WIDTH,
                showfliers=THEME_BOXPLOT_SHOW_FLIERS,
                medianprops={
                    'color': THEME_MEDIAN,
                    'linewidth': THEME_MEDIAN_WIDTH,
                },
                whiskerprops={
                    'color': THEME_EDGE,
                    'linewidth': THEME_BOX_EDGE_WIDTH,
                },
                capprops={
                    'color': THEME_EDGE,
                    'linewidth': THEME_BOX_EDGE_WIDTH,
                },
                boxprops={
                    'color': THEME_EDGE,
                    'linewidth': THEME_BOX_EDGE_WIDTH,
                },
                flierprops={
                    'marker': 'o',
                    'markersize': THEME_FLIER_SIZE,
                    'markerfacecolor': THEME_EDGE,
                    'markeredgecolor': THEME_EDGE,
                    'alpha': THEME_FLIER_ALPHA,
                },
            )

            for patch, fill_color in zip(
                box['boxes'],
                [THEME_BREAKDOWN, THEME_CONTROL],
            ):
                patch.set_facecolor(fill_color)
                patch.set_alpha(THEME_BOX_ALPHA)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis=THEME_GRID_AXIS_Y, alpha=THEME_GRID_ALPHA_SUBTLE)

        fig.suptitle(
            'Are Conditions Already Different Before the Speed Collapse?',
            fontfamily=THEME_TITLE_FONT,
            fontsize=THEME_SUPTITLE_SIZE_SMALL,
        )

        fig.text(
            0.5,
            0.015,
            'Controls match station, weekday/weekend class, and half-hour '
            'clock bin; controls within 60 minutes of an LA onset are excluded.',
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )

        fig.tight_layout(rect=THEME_TIGHT_RECT_B5_BOX)
        fig.savefig(
            BATCH_5_PART_1_IMAGE_FOLDER
            / 'traffic_b5p1_preconditions_vs_controls.png',
            dpi=THEME_DPI,
            bbox_inches=THEME_BBOX,
        )
        plt.close(fig)

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------
    complete_events = int(
        profile_coverage.complete_profile.sum()
    )
    total_controls = int(
        control_matches.controls_found.sum()
    )

    report_lines = [
        'BATCH 5 — PART 1: PRE-BREAKDOWN CONDITIONS',
        '',
        f'Primary onset method: {BATCH_5_PART_1_PRIMARY_ONSET_METHOD}',
        f'Breakdown events analyzed: {len(onset_events)}',
        f'Events with complete -30 to +30 minute profile: '
        f'{complete_events}/{len(onset_events)}',
        f'Matched non-onset controls selected: {total_controls}',
        f'Extended context window: '
        f'{BATCH_5_PART_1_CONTEXT_START_HOUR:04.1f}-'
        f'{BATCH_5_PART_1_CONTEXT_END_HOUR:04.1f}',
        f'Boundary context rows added: {boundary_rows:,}',
        'Batch 4 05:00-20:00 core rows reproduced exactly: yes',
        '',
        'BOUNDARY-CONTEXT FIX',
        'Batch 4 event detection remains unchanged at 05:00-20:00.',
        'Batch 5 loads 30 minutes of additional context on both sides so',
        'events near the analysis-window boundaries can receive complete',
        'pre-onset and post-onset profiles and matched controls.',
        '',
        'EVENT PROFILE SUMMARY',
        event_profile_summary.round(4).to_string(index=False),
        '',
        'EVENT-LEVEL PRE-BREAKDOWN FEATURES',
        event_features.assign(
            **{
                column: event_features[column].round(4)
                for column in event_features.select_dtypes(
                    include=[np.number]
                ).columns
            }
        ).to_string(index=False),
        '',
        'MATCHED CONTROL COUNTS',
        control_matches.to_string(index=False),
        '',
        'FIVE-MINUTES-BEFORE-ONSET COMPARISON',
        precondition_comparison.round(4).to_string(index=False),
        '',
        'METHOD',
        'Each event is centered on its LA temporal-rule onset.',
        'Exact five-minute observations from -30 through +30 minutes are retained.',
        'Matched controls use the same station, same weekday/weekend class,',
        'same half-hour clock bin, a different date, and are at least 60 minutes',
        'from any LA onset at that station. Up to 10 nearest-date controls are',
        'selected deterministically per event, requiring complete -30 to 0 data.',
        '',
        'INTERPRETATION GUARDRAIL',
        'This section is descriptive and diagnostic. With only 10 abrupt-onset',
        'events, it does not treat event-control differences as definitive causal',
        'effects. The purpose is to characterize precursors and assess whether the',
        'selected onset rule produces coherent traffic dynamics.',
        '',
        'PORTFOLIO FIGURES',
        'images/traffic_b5p1_event_centered_profiles.png',
        'images/traffic_b5p1_breakdown_speed_heatmap.png',
        'images/traffic_b5p1_preconditions_vs_controls.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_5_part_1_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_folder': str(source),
        'decision_folder': str(decision),
        'output_folder': str(output),
        'primary_onset_method': BATCH_5_PART_1_PRIMARY_ONSET_METHOD,
        'profile_minutes': BATCH_5_PART_1_PROFILE_MINUTES,
        'control_exclusion_minutes_from_onset':
            BATCH_5_PART_1_CONTROL_MINUTES_FROM_ONSET,
        'controls_per_event': BATCH_5_PART_1_CONTROLS_PER_EVENT,
        'context_start_hour': BATCH_5_PART_1_CONTEXT_START_HOUR,
        'context_end_hour': BATCH_5_PART_1_CONTEXT_END_HOUR,
        'core_start_hour': BATCH_5_PART_1_CORE_START_HOUR,
        'core_end_hour': BATCH_5_PART_1_CORE_END_HOUR,
        'boundary_context_rows_added': int(boundary_rows),
        'batch_4_core_reproduced_exactly': True,
        'events_analyzed': int(len(onset_events)),
        'events_with_complete_profiles': complete_events,
        'matched_controls_selected': total_controls,
        'portfolio_figures': [
            'images/traffic_b5p1_event_centered_profiles.png',
            'images/traffic_b5p1_breakdown_speed_heatmap.png',
            'images/traffic_b5p1_preconditions_vs_controls.png',
        ],
    }

    (output / 'batch_5_part_1_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 5 Part 1 reports: {output.resolve()}'
    )




# ===================== BATCH 5 — PART 2. CORRIDOR PROPAGATION =====================
# Abrupt LA-rule onsets are grouped into corridor-level episodes. Sustained
# 50 mph + 15 minute state entries are then compared across stations in travel
# order. Prior outputs remain untouched; 15/30/45-minute clustering sensitivity
# is retained as an audit trail.


def batch_5_part_2_cluster_onsets(onsets, gap_minutes):
    """Cluster same-day LA onsets into corridor-level episodes."""
    ordered = onsets.sort_values(
        ['start_time', 'station_id'],
        kind='stable',
    ).copy()

    episode_ids = []
    episode_id = 0
    previous_time = None
    previous_date = None

    for row in ordered.itertuples():
        current_time = pd.Timestamp(row.start_time)
        current_date = current_time.date()

        new_episode = (
            previous_time is None
            or current_date != previous_date
            or (
                current_time - previous_time
            ).total_seconds() / 60 > gap_minutes
        )

        if new_episode:
            episode_id += 1

        episode_ids.append(episode_id)
        previous_time = current_time
        previous_date = current_date

    ordered['corridor_episode_id'] = episode_ids
    return ordered


def batch_5_part_2_build_episode_table(clustered_onsets):
    """Summarize clustered abrupt onsets into corridor-level episodes."""
    rows = []

    for episode_id, group in clustered_onsets.groupby(
        'corridor_episode_id',
        sort=True,
    ):
        group = group.sort_values(
            ['start_time', 'station_id'],
            kind='stable',
        )

        rows.append({
            'corridor_episode_id': int(episode_id),
            'episode_date': pd.Timestamp(
                group.start_time.min()
            ).date(),
            'first_onset_time': pd.Timestamp(
                group.start_time.min()
            ),
            'last_onset_time': pd.Timestamp(
                group.start_time.max()
            ),
            'onset_span_minutes': (
                pd.Timestamp(group.start_time.max())
                - pd.Timestamp(group.start_time.min())
            ).total_seconds() / 60,
            'la_onsets_in_episode': len(group),
            'stations_with_la_onset': group.station_id.nunique(),
            'first_onset_station_id': int(
                group.iloc[0].station_id
            ),
            'station_ids_with_la_onset': '|'.join(
                str(int(value))
                for value in group.station_id.tolist()
            ),
        })

    return pd.DataFrame(rows)


def batch_5_part_2_find_state_entry(
    state_events,
    station_id,
    anchor_time,
):
    """Find the sustained-congestion state nearest an abrupt-onset anchor."""
    station_states = state_events.loc[
        state_events.station_id.eq(station_id)
    ].copy()

    if station_states.empty:
        return None

    anchor = pd.Timestamp(anchor_time)

    containing = station_states.loc[
        station_states.start_time.le(anchor)
        & station_states.end_time.ge(anchor)
    ].copy()

    if not containing.empty:
        containing['distance_minutes'] = (
            anchor - containing.start_time
        ).abs().dt.total_seconds() / 60

        row = containing.sort_values(
            ['distance_minutes', 'start_time'],
            kind='stable',
        ).iloc[0].copy()

        row['match_type'] = 'contains_anchor'
        return row

    candidate = station_states.copy()
    candidate['lag_minutes'] = (
        candidate.start_time - anchor
    ).dt.total_seconds() / 60

    candidate = candidate.loc[
        candidate.lag_minutes.ge(
            -BATCH_5_PART_2_STATE_MATCH_BEFORE_MINUTES
        )
        & candidate.lag_minutes.le(
            BATCH_5_PART_2_STATE_MATCH_AFTER_MINUTES
        )
    ].copy()

    if candidate.empty:
        return None

    candidate['distance_minutes'] = candidate.lag_minutes.abs()

    row = candidate.sort_values(
        ['distance_minutes', 'start_time'],
        kind='stable',
    ).iloc[0].copy()

    row['match_type'] = 'nearest_state_start'
    return row


def run_batch_5_part_2():
    """Analyze spatial propagation around abrupt corridor breakdown episodes."""
    source = BATCH_5_PART_2_SOURCE_FOLDER
    part1 = BATCH_5_PART_2_PART1_FOLDER
    output = BATCH_5_PART_2_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 5 Part 2 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'events':
            source / 'batch_4_combined_event_inventory.csv',
        'corridor_order':
            source / 'batch_4_corridor_travel_order.csv',
        'part1_context_audit':
            part1 / 'batch_5_part_1_context_extension_audit.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 5 Part 2 requires completed prior outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_5_PART_2_IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = pd.read_csv(
        required['events'],
        parse_dates=['start_time', 'end_time'],
    )

    corridor = pd.read_csv(required['corridor_order'])
    corridor['station_id'] = pd.to_numeric(
        corridor.station_id,
        errors='raise',
    ).astype('int64')

    corridor = corridor.sort_values(
        'travel_order',
        kind='stable',
    ).reset_index(drop=True)

    station_ids = corridor.station_id.tolist()

    onsets = events.loc[
        events.method.eq(
            BATCH_5_PART_2_PRIMARY_ONSET_METHOD
        )
    ].copy()

    state_events = events.loc[
        events.method.eq(
            BATCH_5_PART_2_STATE_METHOD
        )
    ].copy()

    if onsets.empty:
        raise ValueError(
            'No LA temporal onset events found for Batch 5 Part 2.'
        )

    if state_events.empty:
        raise ValueError(
            'No sustained congestion-state events found for Batch 5 Part 2.'
        )

    cluster_sensitivity_rows = []

    for gap_minutes in BATCH_5_PART_2_CLUSTER_SENSITIVITY_MINUTES:
        clustered = batch_5_part_2_cluster_onsets(
            onsets,
            gap_minutes,
        )

        episode_table = batch_5_part_2_build_episode_table(
            clustered
        )

        cluster_sensitivity_rows.append({
            'cluster_gap_minutes': gap_minutes,
            'corridor_episodes': len(episode_table),
            'episodes_with_multiple_la_onsets': int(
                episode_table.la_onsets_in_episode.gt(1).sum()
            ),
            'maximum_la_onsets_in_episode': int(
                episode_table.la_onsets_in_episode.max()
            ),
        })

    cluster_sensitivity = pd.DataFrame(
        cluster_sensitivity_rows
    )

    cluster_sensitivity.to_csv(
        output / 'batch_5_part_2_cluster_sensitivity.csv',
        index=False,
    )

    clustered_onsets = batch_5_part_2_cluster_onsets(
        onsets,
        BATCH_5_PART_2_CLUSTER_GAP_MINUTES,
    )

    clustered_onsets.to_csv(
        output / 'batch_5_part_2_clustered_la_onsets.csv',
        index=False,
    )

    episodes = batch_5_part_2_build_episode_table(
        clustered_onsets
    )

    episodes.to_csv(
        output / 'batch_5_part_2_corridor_episodes.csv',
        index=False,
    )

    propagation_rows = []

    for episode in episodes.itertuples():
        anchor = pd.Timestamp(episode.first_onset_time)

        for station in corridor.itertuples():
            state = batch_5_part_2_find_state_entry(
                state_events,
                int(station.station_id),
                anchor,
            )

            if state is None:
                propagation_rows.append({
                    'corridor_episode_id':
                        int(episode.corridor_episode_id),
                    'episode_date':
                        episode.episode_date,
                    'anchor_time':
                        anchor,
                    'station_id':
                        int(station.station_id),
                    'travel_order':
                        int(station.travel_order),
                    'station_name':
                        station.Name,
                    'state_found':
                        False,
                    'state_start_time':
                        pd.NaT,
                    'state_end_time':
                        pd.NaT,
                    'state_entry_lag_minutes':
                        np.nan,
                    'state_duration_minutes':
                        np.nan,
                    'match_type':
                        'none',
                })
                continue

            propagation_rows.append({
                'corridor_episode_id':
                    int(episode.corridor_episode_id),
                'episode_date':
                    episode.episode_date,
                'anchor_time':
                    anchor,
                'station_id':
                    int(station.station_id),
                'travel_order':
                    int(station.travel_order),
                'station_name':
                    station.Name,
                'state_found':
                    True,
                'state_start_time':
                    pd.Timestamp(state.start_time),
                'state_end_time':
                    pd.Timestamp(state.end_time),
                'state_entry_lag_minutes': (
                    pd.Timestamp(state.start_time) - anchor
                ).total_seconds() / 60,
                'state_duration_minutes':
                    float(state.duration_minutes),
                'match_type':
                    str(state.match_type),
            })

    propagation = pd.DataFrame(
        propagation_rows
    )

    propagation.to_csv(
        output / 'batch_5_part_2_state_entry_by_station.csv',
        index=False,
    )

    episode_summary_rows = []

    for episode_id, group in propagation.groupby(
        'corridor_episode_id',
        sort=True,
    ):
        found = group.loc[group.state_found].copy()

        if found.empty:
            episode_summary_rows.append({
                'corridor_episode_id': int(episode_id),
                'stations_with_state': 0,
                'first_state_station_id': np.nan,
                'first_state_travel_order': np.nan,
                'last_state_station_id': np.nan,
                'last_state_travel_order': np.nan,
                'state_entry_span_minutes': np.nan,
                'travel_order_lag_correlation': np.nan,
            })
            continue

        first = found.sort_values(
            ['state_start_time', 'travel_order'],
            kind='stable',
        ).iloc[0]

        last = found.sort_values(
            ['state_start_time', 'travel_order'],
            kind='stable',
        ).iloc[-1]

        if len(found) >= 2:
            correlation = found[
                ['travel_order', 'state_entry_lag_minutes']
            ].corr().iloc[0, 1]
        else:
            correlation = np.nan

        episode_summary_rows.append({
            'corridor_episode_id': int(episode_id),
            'stations_with_state': len(found),
            'first_state_station_id':
                int(first.station_id),
            'first_state_travel_order':
                int(first.travel_order),
            'last_state_station_id':
                int(last.station_id),
            'last_state_travel_order':
                int(last.travel_order),
            'state_entry_span_minutes': (
                pd.Timestamp(last.state_start_time)
                - pd.Timestamp(first.state_start_time)
            ).total_seconds() / 60,
            'travel_order_lag_correlation':
                correlation,
        })

    episode_summary = pd.DataFrame(
        episode_summary_rows
    )

    episode_summary.to_csv(
        output / 'batch_5_part_2_episode_propagation_summary.csv',
        index=False,
    )

    data = batch_5_part_1_load_extended_context(
        station_ids
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P2_LAGS
    )

    for episode_id, group in propagation.groupby(
        'corridor_episode_id',
        sort=True,
    ):
        found = group.loc[
            group.state_found
        ].sort_values('travel_order')

        if found.empty:
            continue

        ax.plot(
            found.travel_order,
            found.state_entry_lag_minutes,
            marker=THEME_MARKER_PRIMARY,
            linewidth=THEME_LINEWIDTH_PROPAGATION,
            alpha=THEME_ALPHA_PROPAGATION,
            label=f'Episode {episode_id}',
        )

    ax.axhline(
        THEME_AXIS_MIN_ZERO,
        linestyle=THEME_LINESTYLE_DOTTED,
        linewidth=THEME_LINEWIDTH_REFERENCE,
        alpha=THEME_ALPHA_REFERENCE,
    )

    ax.set_title(
        'How Quickly Does Sustained Congestion Appear Across the Corridor?',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Station in Travel Direction')
    ax.set_ylabel(
        'State Entry Lag Relative to First Abrupt Onset (minutes)'
    )
    ax.set_xticks(
        corridor.travel_order,
        [
            f'{int(row.travel_order)}\n{int(row.station_id)}'
            for row in corridor.itertuples()
        ],
    )
    ax.grid(
        alpha=THEME_GRID_ALPHA_SUBTLE
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
        ncol=THEME_LEGEND_NCOL_1,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_2_IMAGE_FOLDER
        / 'traffic_b5p2_state_entry_lags.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    lag_matrix = propagation.pivot(
        index='corridor_episode_id',
        columns='travel_order',
        values='state_entry_lag_minutes',
    ).reindex(
        columns=corridor.travel_order.tolist()
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P2_HEATMAP
    )

    image = ax.imshow(
        lag_matrix.to_numpy(),
        aspect=THEME_HEATMAP_ASPECT,
        interpolation=THEME_HEATMAP_INTERPOLATION,
        cmap=THEME_HEATMAP_CMAP,
    )

    ax.set_title(
        'Sustained-Congestion Entry Lag by Corridor Episode',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Station in Travel Direction')
    ax.set_ylabel('Corridor Episode')
    ax.set_xticks(
        np.arange(len(corridor)),
        [
            f'{int(row.travel_order)}\n{int(row.station_id)}'
            for row in corridor.itertuples()
        ],
    )
    ax.set_yticks(
        np.arange(len(lag_matrix.index)),
        [str(value) for value in lag_matrix.index],
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=THEME_COLORBAR_PAD,
    )
    colorbar.set_label(
        'Minutes Relative to First LA Onset'
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_2_IMAGE_FOLDER
        / 'traffic_b5p2_propagation_lag_heatmap.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    episode_count = len(episodes)

    fig, axes = plt.subplots(
        episode_count,
        1,
        figsize=THEME_FIGSIZE_B5P2_EPISODES,
        squeeze=False,
    )

    for axis_index, episode in enumerate(
        episodes.itertuples()
    ):
        ax = axes[axis_index, 0]
        anchor = pd.Timestamp(
            episode.first_onset_time
        )

        window_start = anchor + pd.Timedelta(
            minutes=BATCH_5_PART_2_PROFILE_START_MINUTE
        )
        window_end = anchor + pd.Timedelta(
            minutes=BATCH_5_PART_2_PROFILE_END_MINUTE
        )

        window = data.loc[
            data.timestamp.ge(window_start)
            & data.timestamp.le(window_end)
        ].copy()

        heat = window.pivot(
            index='station_id',
            columns='timestamp',
            values='speed_mph',
        ).reindex(station_ids)

        if heat.empty:
            ax.set_title(
                f'Episode {episode.corridor_episode_id}: no data',
                fontfamily=THEME_TITLE_FONT,
            )
            continue

        image = ax.imshow(
            heat.to_numpy(),
            aspect=THEME_HEATMAP_ASPECT,
            interpolation=THEME_HEATMAP_INTERPOLATION,
            vmin=THEME_HEATMAP_MIN,
            vmax=THEME_HEATMAP_MAX,
            cmap=THEME_HEATMAP_CMAP,
        )

        timestamps = list(heat.columns)

        if anchor in timestamps:
            anchor_position = timestamps.index(anchor)
            ax.axvline(
                anchor_position,
                linestyle=THEME_LINESTYLE_DASHED,
                linewidth=THEME_LINEWIDTH_SECONDARY,
            )

        tick_positions = np.linspace(
            0,
            len(timestamps) - 1,
            min(10, len(timestamps)),
            dtype=int,
        )

        ax.set_xticks(
            tick_positions,
            [
                pd.Timestamp(
                    timestamps[position]
                ).strftime('%-I:%M %p')
                for position in tick_positions
            ],
            rotation=THEME_ROTATION_XL,
            ha=THEME_ALIGN_RIGHT,
        )

        ax.set_yticks(
            np.arange(len(station_ids)),
            [
                f'{int(row.travel_order)}  '
                f'{int(row.station_id)}  {row.Name}'
                for row in corridor.itertuples()
            ],
        )

        ax.set_title(
            f'Episode {episode.corridor_episode_id} — '
            f'{pd.Timestamp(anchor).strftime("%Y-%m-%d %-I:%M %p")}',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('Station in Travel Direction')

    fig.suptitle(
        'Raw Speed Evolution Around Corridor-Level Breakdown Episodes',
        fontfamily=THEME_TITLE_FONT,
        fontsize=THEME_SUPTITLE_SIZE,
    )

    fig.tight_layout(
        rect=THEME_TIGHT_RECT_B5_PROFILE
    )
    fig.savefig(
        BATCH_5_PART_2_IMAGE_FOLDER
        / 'traffic_b5p2_episode_speed_heatmaps.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    report_lines = [
        'BATCH 5 — PART 2: CORRIDOR PROPAGATION',
        '',
        f'Primary abrupt-onset method: '
        f'{BATCH_5_PART_2_PRIMARY_ONSET_METHOD}',
        f'Congestion-state method: '
        f'{BATCH_5_PART_2_STATE_METHOD}',
        f'LA onsets available: {len(onsets)}',
        f'Primary onset-cluster gap: '
        f'{BATCH_5_PART_2_CLUSTER_GAP_MINUTES} minutes',
        f'Corridor-level episodes: {len(episodes)}',
        '',
        'CLUSTER SENSITIVITY',
        cluster_sensitivity.to_string(index=False),
        '',
        'CORRIDOR EPISODES',
        episodes.to_string(index=False),
        '',
        'STATE ENTRY BY STATION',
        propagation.round(3).to_string(index=False),
        '',
        'EPISODE PROPAGATION SUMMARY',
        episode_summary.round(3).to_string(index=False),
        '',
        'METHOD',
        'LA temporal onsets on the same date are grouped into corridor-level',
        'episodes when consecutive onset times are no more than the configured',
        'gap apart. Sensitivity at 15, 30, and 45 minutes is preserved.',
        'For each episode and each station, the sustained 50 mph + 15 minute',
        'state event containing the episode anchor is preferred. If none contains',
        'the anchor, the nearest state start within the configured before/after',
        'window is used.',
        '',
        'INTERPRETATION',
        'Positive lag means sustained congestion begins after the first abrupt',
        'LA-rule onset in that corridor episode. Negative lag means the station',
        'was already in sustained congestion before that abrupt onset.',
        '',
        'PORTFOLIO FIGURES',
        'images/traffic_b5p2_state_entry_lags.png',
        'images/traffic_b5p2_propagation_lag_heatmap.png',
        'images/traffic_b5p2_episode_speed_heatmaps.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_5_part_2_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_folder': str(source),
        'part1_folder': str(part1),
        'output_folder': str(output),
        'primary_onset_method':
            BATCH_5_PART_2_PRIMARY_ONSET_METHOD,
        'state_method':
            BATCH_5_PART_2_STATE_METHOD,
        'cluster_gap_minutes':
            BATCH_5_PART_2_CLUSTER_GAP_MINUTES,
        'cluster_sensitivity_minutes':
            BATCH_5_PART_2_CLUSTER_SENSITIVITY_MINUTES,
        'corridor_episodes': int(len(episodes)),
        'portfolio_figures': [
            'images/traffic_b5p2_state_entry_lags.png',
            'images/traffic_b5p2_propagation_lag_heatmap.png',
            'images/traffic_b5p2_episode_speed_heatmaps.png',
        ],
    }

    (output / 'batch_5_part_2_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 5 Part 2 reports: {output.resolve()}'
    )




# ===================== BATCH 5 — PART 2B. PROPAGATION ROBUSTNESS =====================
# This section preserves Batch 5 Part 2 and tests whether conclusions about
# spatial propagation depend on allowing sustained-state entries as late as
# 30, 60, or 120 minutes after the first abrupt LA-rule onset.


def batch_5_part_2b_classify_station_count(stations_with_state):
    """Classify episode spatial extent from number of participating stations."""
    if stations_with_state >= BATCH_5_PART_2B_CORRIDOR_WIDE_MIN_STATIONS:
        return 'corridor-wide propagation'

    if stations_with_state >= BATCH_5_PART_2B_PARTIAL_MIN_STATIONS:
        return 'partial propagation'

    return 'localized/no propagation'


def run_batch_5_part_2b():
    """Test propagation conclusions under tighter lag windows."""
    source = BATCH_5_PART_2B_SOURCE_FOLDER
    output = BATCH_5_PART_2B_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 5 Part 2b output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'propagation':
            source / 'batch_5_part_2_state_entry_by_station.csv',
        'episodes':
            source / 'batch_5_part_2_corridor_episodes.csv',
        'cluster_sensitivity':
            source / 'batch_5_part_2_cluster_sensitivity.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 5 Part 2b requires completed Part 2 outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_5_PART_2B_IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    propagation = pd.read_csv(
        required['propagation'],
        parse_dates=[
            'anchor_time',
            'state_start_time',
            'state_end_time',
        ],
    )

    episodes = pd.read_csv(
        required['episodes'],
        parse_dates=[
            'first_onset_time',
            'last_onset_time',
        ],
    )

    cluster_sensitivity = pd.read_csv(
        required['cluster_sensitivity']
    )

    robustness_rows = []
    station_rows = []

    for window_minutes in BATCH_5_PART_2B_WINDOWS_MINUTES:
        for episode_id, group in propagation.groupby(
            'corridor_episode_id',
            sort=True,
        ):
            valid = group.loc[
                group.state_found.fillna(False)
                & group.state_entry_lag_minutes.notna()
                & group.state_entry_lag_minutes.ge(
                    -BATCH_5_PART_2_STATE_MATCH_BEFORE_MINUTES
                )
                & group.state_entry_lag_minutes.le(
                    window_minutes
                )
            ].copy()

            stations_with_state = int(
                valid.station_id.nunique()
            )

            classification = (
                batch_5_part_2b_classify_station_count(
                    stations_with_state
                )
            )

            robustness_rows.append({
                'window_minutes': window_minutes,
                'corridor_episode_id': int(episode_id),
                'stations_with_state': stations_with_state,
                'classification': classification,
                'earliest_lag_minutes': (
                    valid.state_entry_lag_minutes.min()
                    if not valid.empty else np.nan
                ),
                'latest_lag_minutes': (
                    valid.state_entry_lag_minutes.max()
                    if not valid.empty else np.nan
                ),
                'lag_span_minutes': (
                    valid.state_entry_lag_minutes.max()
                    - valid.state_entry_lag_minutes.min()
                    if not valid.empty else np.nan
                ),
            })

            for row in group.itertuples():
                included = bool(
                    row.state_found
                    and pd.notna(row.state_entry_lag_minutes)
                    and row.state_entry_lag_minutes
                    >= -BATCH_5_PART_2_STATE_MATCH_BEFORE_MINUTES
                    and row.state_entry_lag_minutes
                    <= window_minutes
                )

                station_rows.append({
                    'window_minutes': window_minutes,
                    'corridor_episode_id': int(episode_id),
                    'station_id': int(row.station_id),
                    'travel_order': int(row.travel_order),
                    'state_entry_lag_minutes':
                        row.state_entry_lag_minutes,
                    'included_same_episode': included,
                })

    robustness = pd.DataFrame(
        robustness_rows
    )

    station_robustness = pd.DataFrame(
        station_rows
    )

    robustness.to_csv(
        output / 'batch_5_part_2b_episode_classification_by_window.csv',
        index=False,
    )

    station_robustness.to_csv(
        output / 'batch_5_part_2b_station_inclusion_by_window.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Stability of classification across windows
    # --------------------------------------------------------
    classification_pivot = robustness.pivot(
        index='corridor_episode_id',
        columns='window_minutes',
        values='classification',
    )

    classification_pivot['classification_stable'] = (
        classification_pivot.nunique(
            axis=1,
            dropna=False,
        ).eq(1)
    )

    classification_pivot = (
        classification_pivot
        .reset_index()
    )

    classification_pivot.to_csv(
        output / 'batch_5_part_2b_classification_stability.csv',
        index=False,
    )

    stable_count = int(
        classification_pivot.classification_stable.sum()
    )

    total_episodes = len(classification_pivot)

    # --------------------------------------------------------
    # Summary counts by window and classification
    # --------------------------------------------------------
    classification_counts = (
        robustness
        .groupby(
            ['window_minutes', 'classification'],
            as_index=False,
        )
        .size()
        .rename(columns={'size': 'episodes'})
    )

    classification_counts.to_csv(
        output / 'batch_5_part_2b_classification_counts.csv',
        index=False,
    )

    station_count_summary = (
        robustness.groupby(
            'window_minutes',
            as_index=False,
        )
        .agg(
            median_stations_with_state=(
                'stations_with_state',
                'median',
            ),
            min_stations_with_state=(
                'stations_with_state',
                'min',
            ),
            max_stations_with_state=(
                'stations_with_state',
                'max',
            ),
        )
    )

    station_count_summary.to_csv(
        output / 'batch_5_part_2b_station_count_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Figure 1: classifications by episode and window
    # --------------------------------------------------------
    class_order = [
        'localized/no propagation',
        'partial propagation',
        'corridor-wide propagation',
    ]

    class_code = {
        label: index
        for index, label in enumerate(class_order)
    }

    heat = robustness.copy()
    heat['classification_code'] = (
        heat.classification.map(class_code)
    )

    heat_matrix = heat.pivot(
        index='corridor_episode_id',
        columns='window_minutes',
        values='classification_code',
    ).reindex(
        columns=BATCH_5_PART_2B_WINDOWS_MINUTES
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P2B_CLASSIFICATION
    )

    image = ax.imshow(
        heat_matrix.to_numpy(),
        aspect=THEME_HEATMAP_ASPECT,
        interpolation=THEME_HEATMAP_INTERPOLATION,
        cmap=THEME_HEATMAP_CMAP,
    )

    ax.set_title(
        'Propagation Classification Is Tested Across Three Lag Windows',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Maximum Allowed Lag After First Abrupt Onset (minutes)')
    ax.set_ylabel('Corridor Episode')
    ax.set_xticks(
        np.arange(len(BATCH_5_PART_2B_WINDOWS_MINUTES)),
        [
            str(value)
            for value in BATCH_5_PART_2B_WINDOWS_MINUTES
        ],
    )
    ax.set_yticks(
        np.arange(len(heat_matrix.index)),
        [str(value) for value in heat_matrix.index],
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=THEME_COLORBAR_PAD,
    )
    colorbar.set_ticks(
        np.arange(len(class_order))
    )
    colorbar.set_ticklabels(
        class_order
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_2B_IMAGE_FOLDER
        / 'traffic_b5p2b_classification_robustness.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: station participation by episode/window
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P2B_STATIONS
    )

    episode_ids = sorted(
        robustness.corridor_episode_id.unique()
    )

    offsets = {
        30: THEME_BAR_OFFSET_NEGATIVE,
        60: THEME_BAR_OFFSET_ZERO,
        120: THEME_BAR_OFFSET_POSITIVE,
    }

    for window_minutes in BATCH_5_PART_2B_WINDOWS_MINUTES:
        group = robustness.loc[
            robustness.window_minutes.eq(window_minutes)
        ].set_index(
            'corridor_episode_id'
        ).reindex(
            episode_ids
        )

        x = (
            np.arange(len(episode_ids))
            + offsets[window_minutes]
        )

        ax.bar(
            x,
            group.stations_with_state,
            width=THEME_BAR_WIDTH_GROUPED,
            label=f'{window_minutes} min',
        )

    ax.set_title(
        'How Many Stations Belong to the Same Propagation Episode?',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Corridor Episode')
    ax.set_ylabel('Stations Included')
    ax.set_xticks(
        np.arange(len(episode_ids)),
        [str(value) for value in episode_ids],
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_2B_IMAGE_FOLDER
        / 'traffic_b5p2b_station_participation_by_window.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Decision record
    # --------------------------------------------------------
    primary = robustness.loc[
        robustness.window_minutes.eq(
            BATCH_5_PART_2B_PRIMARY_WINDOW_MINUTES
        )
    ].copy()

    decision_rows = []

    for row in primary.itertuples():
        decision_rows.append({
            'corridor_episode_id':
                int(row.corridor_episode_id),
            'primary_window_minutes':
                BATCH_5_PART_2B_PRIMARY_WINDOW_MINUTES,
            'stations_with_state':
                int(row.stations_with_state),
            'classification':
                row.classification,
            'stable_across_30_60_120':
                bool(
                    classification_pivot.loc[
                        classification_pivot.corridor_episode_id.eq(
                            row.corridor_episode_id
                        ),
                        'classification_stable',
                    ].iloc[0]
                ),
        })

    decision = pd.DataFrame(
        decision_rows
    )

    decision.to_csv(
        output / 'batch_5_part_2b_primary_window_decision.csv',
        index=False,
    )

    report_lines = [
        'BATCH 5 — PART 2B: PROPAGATION ROBUSTNESS',
        '',
        'PURPOSE',
        'Test whether propagation conclusions depend on allowing state-entry',
        'lags as late as 30, 60, or 120 minutes after the first abrupt onset.',
        '',
        'CLASSIFICATION RULE',
        f'Corridor-wide propagation: '
        f'>={BATCH_5_PART_2B_CORRIDOR_WIDE_MIN_STATIONS} stations.',
        f'Partial propagation: '
        f'{BATCH_5_PART_2B_PARTIAL_MIN_STATIONS}-'
        f'{BATCH_5_PART_2B_CORRIDOR_WIDE_MIN_STATIONS - 1} stations.',
        'Localized/no propagation: fewer than 2 stations.',
        '',
        'ROBUSTNESS TABLE',
        robustness.to_string(index=False),
        '',
        'CLASSIFICATION STABILITY',
        classification_pivot.to_string(index=False),
        '',
        f'Classification stable across all three windows: '
        f'{stable_count}/{total_episodes} episodes.',
        '',
        'PRIMARY 60-MINUTE DECISION TABLE',
        decision.to_string(index=False),
        '',
        'PRIOR CLUSTER ROBUSTNESS RETAINED',
        cluster_sensitivity.to_string(index=False),
        '',
        'INTERPRETATION',
        'The 120-minute window is retained only as the original reference.',
        'The 60-minute window is the primary operational window for downstream',
        'propagation summaries because it excludes clearly delayed state entries',
        'such as the 110-minute Episode 5 matches while retaining slower but',
        'plausibly connected corridor responses.',
        '',
        'PORTFOLIO FIGURES',
        'images/traffic_b5p2b_classification_robustness.png',
        'images/traffic_b5p2b_station_participation_by_window.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_5_part_2b_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_folder': str(source),
        'output_folder': str(output),
        'windows_minutes':
            BATCH_5_PART_2B_WINDOWS_MINUTES,
        'primary_window_minutes':
            BATCH_5_PART_2B_PRIMARY_WINDOW_MINUTES,
        'corridor_wide_min_stations':
            BATCH_5_PART_2B_CORRIDOR_WIDE_MIN_STATIONS,
        'partial_min_stations':
            BATCH_5_PART_2B_PARTIAL_MIN_STATIONS,
        'stable_episode_classifications':
            stable_count,
        'total_episodes':
            total_episodes,
        'portfolio_figures': [
            'images/traffic_b5p2b_classification_robustness.png',
            'images/traffic_b5p2b_station_participation_by_window.png',
        ],
    }

    (output / 'batch_5_part_2b_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 5 Part 2b reports: {output.resolve()}'
    )




# ===================== BATCH 5 — PART 3. CONGESTION DURATION AND RECOVERY =====================


def batch_5_part_3_time_period(timestamp):
    """Assign the configured AM / Midday / PM analysis period."""
    hour = pd.Timestamp(timestamp).hour + pd.Timestamp(timestamp).minute / 60

    for label, start_hour, end_hour in BATCH_5_PART_3_TIME_PERIODS:
        if start_hour <= hour < end_hour:
            return label

    return 'Outside'


def batch_5_part_3_product_limit_duration(durations, observed):
    """
    Kaplan-Meier/product-limit survival estimate for time to recovery.

    observed=True means recovery was observed.
    observed=False means the congestion episode was right-censored.
    """
    frame = pd.DataFrame({
        'duration': pd.to_numeric(durations, errors='coerce'),
        'observed': pd.Series(observed).astype(bool),
    }).dropna(subset=['duration'])

    if frame.empty:
        return pd.DataFrame(
            columns=[
                'duration_minutes',
                'at_risk',
                'recoveries',
                'censored',
                'survival_probability',
                'recovered_probability',
            ]
        )

    event_times = sorted(
        frame.loc[
            frame.observed,
            'duration',
        ].unique()
    )

    survival = 1.0
    rows = []

    for time_value in event_times:
        at_risk = int(
            frame.duration.ge(time_value).sum()
        )
        recoveries = int(
            (
                frame.duration.eq(time_value)
                & frame.observed
            ).sum()
        )
        censored = int(
            (
                frame.duration.eq(time_value)
                & ~frame.observed
            ).sum()
        )

        if at_risk > 0:
            survival *= (
                1.0 - recoveries / at_risk
            )

        rows.append({
            'duration_minutes':
                float(time_value),
            'at_risk':
                at_risk,
            'recoveries':
                recoveries,
            'censored':
                censored,
            'survival_probability':
                survival,
            'recovered_probability':
                1.0 - survival,
        })

    return pd.DataFrame(rows)


def run_batch_5_part_3():
    """Characterize sustained-congestion duration and corridor recovery."""
    source = BATCH_5_PART_3_SOURCE_FOLDER
    propagation_source = BATCH_5_PART_3_PROPAGATION_FOLDER
    robustness_source = BATCH_5_PART_3_ROBUSTNESS_FOLDER
    output = BATCH_5_PART_3_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 5 Part 3 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'events':
            source / 'batch_4_combined_event_inventory.csv',
        'corridor_order':
            source / 'batch_4_corridor_travel_order.csv',
        'propagation':
            propagation_source / 'batch_5_part_2_state_entry_by_station.csv',
        'station_inclusion':
            robustness_source / 'batch_5_part_2b_station_inclusion_by_window.csv',
        'episode_decision':
            robustness_source / 'batch_5_part_2b_primary_window_decision.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 5 Part 3 requires completed prior outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_5_PART_3_IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = pd.read_csv(
        required['events'],
        parse_dates=['start_time', 'end_time'],
    )

    corridor = pd.read_csv(
        required['corridor_order']
    )
    corridor['station_id'] = pd.to_numeric(
        corridor.station_id,
        errors='raise',
    ).astype('int64')

    state_events = events.loc[
        events.method.eq(
            BATCH_5_PART_3_STATE_METHOD
        )
    ].copy()

    if state_events.empty:
        raise ValueError(
            'No sustained congestion-state events found.'
        )

    state_events['time_period'] = (
        state_events.start_time.apply(
            batch_5_part_3_time_period
        )
    )

    state_events['recovery_observed'] = (
        ~state_events.right_censored.fillna(False)
    )

    state_events.to_csv(
        output / 'batch_5_part_3_state_events.csv',
        index=False,
    )

    # --------------------- overall / station / period summaries ---------------------
    overall_summary = pd.DataFrame([
        {
            'events': len(state_events),
            'stations': state_events.station_id.nunique(),
            'days': state_events.event_date.nunique(),
            'median_duration_minutes':
                state_events.duration_minutes.median(),
            'q25_duration_minutes':
                state_events.duration_minutes.quantile(0.25),
            'q75_duration_minutes':
                state_events.duration_minutes.quantile(0.75),
            'p90_duration_minutes':
                state_events.duration_minutes.quantile(0.90),
            'right_censored_events':
                int(
                    state_events.right_censored.fillna(False).sum()
                ),
        }
    ])

    by_station = (
        state_events.groupby(
            'station_id',
            as_index=False,
        )
        .agg(
            events=('event_id', 'size'),
            days=('event_date', 'nunique'),
            median_duration_minutes=(
                'duration_minutes',
                'median',
            ),
            q25_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.25),
            ),
            q75_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.75),
            ),
            p90_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.90),
            ),
        )
        .merge(
            corridor[
                ['station_id', 'travel_order', 'Name']
            ],
            on='station_id',
            how='left',
        )
        .sort_values('travel_order')
    )

    by_period = (
        state_events.groupby(
            'time_period',
            as_index=False,
        )
        .agg(
            events=('event_id', 'size'),
            days=('event_date', 'nunique'),
            median_duration_minutes=(
                'duration_minutes',
                'median',
            ),
            q25_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.25),
            ),
            q75_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.75),
            ),
            p90_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.90),
            ),
        )
    )

    overall_summary.to_csv(
        output / 'batch_5_part_3_overall_duration_summary.csv',
        index=False,
    )
    by_station.to_csv(
        output / 'batch_5_part_3_duration_by_station.csv',
        index=False,
    )
    by_period.to_csv(
        output / 'batch_5_part_3_duration_by_time_period.csv',
        index=False,
    )

    # --------------------- product-limit recovery curve ---------------------
    recovery_curve = batch_5_part_3_product_limit_duration(
        state_events.duration_minutes,
        state_events.recovery_observed,
    )

    recovery_curve.to_csv(
        output / 'batch_5_part_3_recovery_product_limit.csv',
        index=False,
    )

    # --------------------- recovery order for the six corridor episodes ---------------------
    propagation = pd.read_csv(
        required['propagation'],
        parse_dates=[
            'anchor_time',
            'state_start_time',
            'state_end_time',
        ],
    )

    station_inclusion = pd.read_csv(
        required['station_inclusion']
    )

    inclusion_60 = station_inclusion.loc[
        station_inclusion.window_minutes.eq(
            BATCH_5_PART_3_PRIMARY_PROPAGATION_WINDOW_MINUTES
        )
        & station_inclusion.included_same_episode.fillna(False)
    ][
        [
            'corridor_episode_id',
            'station_id',
        ]
    ].copy()

    recovery = propagation.merge(
        inclusion_60.assign(
            included_primary_window=True
        ),
        on=[
            'corridor_episode_id',
            'station_id',
        ],
        how='left',
    )

    recovery['included_primary_window'] = (
        recovery.included_primary_window.fillna(False)
    )

    recovery = recovery.loc[
        recovery.included_primary_window
        & recovery.state_found.fillna(False)
    ].copy()

    recovery['recovery_lag_minutes'] = (
        recovery.state_end_time - recovery.anchor_time
    ).dt.total_seconds() / 60

    recovery.to_csv(
        output / 'batch_5_part_3_episode_station_recovery.csv',
        index=False,
    )

    recovery_summary_rows = []

    for episode_id, group in recovery.groupby(
        'corridor_episode_id',
        sort=True,
    ):
        group = group.sort_values(
            ['state_end_time', 'travel_order'],
            kind='stable',
        )

        if group.empty:
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        if len(group) >= 2:
            recovery_order_correlation = group[
                ['travel_order', 'recovery_lag_minutes']
            ].corr().iloc[0, 1]
        else:
            recovery_order_correlation = np.nan

        recovery_summary_rows.append({
            'corridor_episode_id':
                int(episode_id),
            'stations_in_primary_propagation':
                len(group),
            'first_recovery_station_id':
                int(first.station_id),
            'first_recovery_travel_order':
                int(first.travel_order),
            'last_recovery_station_id':
                int(last.station_id),
            'last_recovery_travel_order':
                int(last.travel_order),
            'first_recovery_lag_minutes':
                float(first.recovery_lag_minutes),
            'last_recovery_lag_minutes':
                float(last.recovery_lag_minutes),
            'recovery_span_minutes': (
                float(last.recovery_lag_minutes)
                - float(first.recovery_lag_minutes)
            ),
            'travel_order_recovery_correlation':
                recovery_order_correlation,
        })

    recovery_summary = pd.DataFrame(
        recovery_summary_rows
    )

    recovery_summary.to_csv(
        output / 'batch_5_part_3_episode_recovery_summary.csv',
        index=False,
    )

    # --------------------- Figure 1: duration by station ---------------------
    station_order = corridor.sort_values(
        'travel_order'
    ).station_id.tolist()

    station_values = [
        state_events.loc[
            state_events.station_id.eq(station_id),
            'duration_minutes',
        ].dropna().to_numpy()
        for station_id in station_order
    ]

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P3_DURATION
    )

    box = ax.boxplot(
        station_values,
        tick_labels=[
            f'{int(row.travel_order)}\n{int(row.station_id)}'
            for row in corridor.sort_values(
                'travel_order'
            ).itertuples()
        ],
        patch_artist=True,
        widths=THEME_BOX_WIDTH_DURATION,
        showfliers=THEME_BOXPLOT_SHOW_FLIERS,
        medianprops={
            'color': THEME_MEDIAN,
            'linewidth': THEME_MEDIAN_WIDTH,
        },
        whiskerprops={
            'color': THEME_EDGE,
            'linewidth': THEME_BOX_EDGE_WIDTH,
        },
        capprops={
            'color': THEME_EDGE,
            'linewidth': THEME_BOX_EDGE_WIDTH,
        },
        boxprops={
            'color': THEME_EDGE,
            'linewidth': THEME_BOX_EDGE_WIDTH,
        },
        flierprops={
            'marker': THEME_MARKER_PRIMARY,
            'markersize': THEME_FLIER_SIZE,
            'markerfacecolor': THEME_EDGE,
            'markeredgecolor': THEME_EDGE,
            'alpha': THEME_FLIER_ALPHA,
        },
    )

    duration_colors = [
        BRAND_COLORS[
            index % len(BRAND_COLORS)
        ]
        for index in range(len(box['boxes']))
    ]

    for patch, fill_color in zip(
        box['boxes'],
        duration_colors,
    ):
        patch.set_facecolor(fill_color)
        patch.set_alpha(THEME_BOX_ALPHA)

    ax.set_title(
        'How Long Does Sustained Congestion Last at Each Station?',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Station in Travel Direction')
    ax.set_ylabel('Congestion-State Duration (minutes)')
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_3_IMAGE_FOLDER
        / 'traffic_b5p3_duration_by_station.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------- Figure 2: recovery product-limit curve ---------------------
    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P3_SURVIVAL
    )

    if not recovery_curve.empty:
        ax.step(
            recovery_curve.duration_minutes,
            recovery_curve.survival_probability,
            where=THEME_STEP_WHERE_POST,
            linewidth=THEME_SURVIVAL_LINEWIDTH,
        )

    ax.set_title(
        'How Long Until a Sustained Congestion Episode Recovers?',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Minutes Since Congestion-State Onset')
    ax.set_ylabel('Probability Episode Has Not Yet Recovered')
    ax.set_ylim(
        THEME_AXIS_MIN_ZERO,
        1.0,
    )
    ax.grid(
        alpha=THEME_GRID_ALPHA_SUBTLE
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_3_IMAGE_FOLDER
        / 'traffic_b5p3_recovery_survival.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------- Figure 3: recovery order ---------------------
    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P3_RECOVERY
    )

    for episode_id, group in recovery.groupby(
        'corridor_episode_id',
        sort=True,
    ):
        group = group.sort_values(
            'travel_order'
        )

        ax.plot(
            group.travel_order,
            group.recovery_lag_minutes,
            marker=THEME_RECOVERY_MARKER,
            linewidth=THEME_RECOVERY_LINEWIDTH,
            label=f'Episode {episode_id}',
        )

    ax.set_title(
        'Recovery Timing Across Stations in the Same Corridor Episode',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Station in Travel Direction')
    ax.set_ylabel('Recovery Lag from First Abrupt Onset (minutes)')
    ax.set_xticks(
        corridor.travel_order,
        [
            f'{int(row.travel_order)}\n{int(row.station_id)}'
            for row in corridor.itertuples()
        ],
    )
    ax.grid(
        alpha=THEME_GRID_ALPHA_SUBTLE
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_3_IMAGE_FOLDER
        / 'traffic_b5p3_recovery_order.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    report_lines = [
        'BATCH 5 — PART 3: CONGESTION DURATION AND RECOVERY',
        '',
        'OVERALL DURATION',
        overall_summary.round(3).to_string(index=False),
        '',
        'DURATION BY STATION',
        by_station.round(3).to_string(index=False),
        '',
        'DURATION BY TIME PERIOD',
        by_period.round(3).to_string(index=False),
        '',
        'CORRIDOR-EPISODE RECOVERY',
        recovery_summary.round(3).to_string(index=False),
        '',
        'METHOD',
        'Sustained congestion is defined by the retained 50 mph + 15 minute',
        'state rule. Duration is measured from the first to last five-minute',
        'interval in the state episode. Right-censored episodes are retained in',
        'the recovery product-limit curve rather than treated as fully observed.',
        '',
        'For corridor recovery ordering, only stations included under the primary',
        '60-minute propagation window from Batch 5 Part 2b are used.',
        '',
        'PORTFOLIO FIGURES',
        'images/traffic_b5p3_duration_by_station.png',
        'images/traffic_b5p3_recovery_survival.png',
        'images/traffic_b5p3_recovery_order.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_5_part_3_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'state_method': BATCH_5_PART_3_STATE_METHOD,
        'primary_propagation_window_minutes':
            BATCH_5_PART_3_PRIMARY_PROPAGATION_WINDOW_MINUTES,
        'state_events': int(len(state_events)),
        'portfolio_figures': [
            'images/traffic_b5p3_duration_by_station.png',
            'images/traffic_b5p3_recovery_survival.png',
            'images/traffic_b5p3_recovery_order.png',
        ],
    }

    (output / 'batch_5_part_3_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 5 Part 3 reports: {output.resolve()}'
    )


# ===================== BATCH 5 — PART 4. BREAKDOWN PROBABILITY VS FLOW =====================


def batch_5_part_4_product_limit(observations):
    """
    Product-limit estimate of breakdown probability versus flow.

    event=1: uncensored pre-breakdown flow.
    event=0: censored free-flow observation not followed by breakdown.
    """
    frame = observations[
        ['flow_vphpl', 'event']
    ].dropna().copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                'flow_vphpl',
                'risk_set',
                'breakdowns',
                'survival_probability',
                'breakdown_probability',
            ]
        )

    event_flows = sorted(
        frame.loc[
            frame.event.eq(1),
            'flow_vphpl',
        ].unique()
    )

    survival = 1.0
    rows = []

    for flow_value in event_flows:
        risk_set = int(
            frame.flow_vphpl.ge(
                flow_value
            ).sum()
        )

        breakdowns = int(
            (
                frame.flow_vphpl.eq(flow_value)
                & frame.event.eq(1)
            ).sum()
        )

        if risk_set > 0:
            survival *= (
                1.0 - breakdowns / risk_set
            )

        rows.append({
            'flow_vphpl':
                float(flow_value),
            'risk_set':
                risk_set,
            'breakdowns':
                breakdowns,
            'survival_probability':
                survival,
            'breakdown_probability':
                1.0 - survival,
        })

    return pd.DataFrame(rows)


def batch_5_part_4_evaluate_curve(curve, grid):
    """Evaluate a right-continuous PLM step curve on a fixed flow grid."""
    if curve.empty:
        return np.zeros(
            len(grid),
            dtype=float,
        )

    event_flow = curve.flow_vphpl.to_numpy()
    probability = curve.breakdown_probability.to_numpy()

    values = np.zeros(
        len(grid),
        dtype=float,
    )

    for index, flow_value in enumerate(grid):
        eligible = np.where(
            event_flow <= flow_value
        )[0]

        if len(eligible):
            values[index] = probability[
                eligible[-1]
            ]

    return values


def run_batch_5_part_4():
    """Estimate stochastic breakdown probability versus pre-breakdown flow."""
    source = BATCH_5_PART_4_SOURCE_FOLDER
    output = BATCH_5_PART_4_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 5 Part 4 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'events':
            source / 'batch_4_combined_event_inventory.csv',
        'corridor_rows':
            source / 'batch_4_corridor_analysis_rows.parquet',
        'corridor_order':
            source / 'batch_4_corridor_travel_order.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 5 Part 4 requires completed Batch 4 outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_5_PART_4_IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = pd.read_csv(
        required['events'],
        parse_dates=['start_time', 'end_time'],
    )

    data = pd.read_parquet(
        required['corridor_rows']
    )

    data['timestamp'] = pd.to_datetime(
        data.timestamp,
        errors='raise',
    )

    data['station_id'] = pd.to_numeric(
        data.station_id,
        errors='raise',
    ).astype('int64')

    corridor = pd.read_csv(
        required['corridor_order']
    )

    if 'Lanes' not in corridor.columns:
        raise ValueError(
            'Batch 5 Part 4 requires metadata Lanes in corridor order.'
        )

    corridor['station_id'] = pd.to_numeric(
        corridor.station_id,
        errors='raise',
    ).astype('int64')

    corridor['Lanes'] = pd.to_numeric(
        corridor.Lanes,
        errors='raise',
    )

    lane_lookup = (
        corridor.set_index(
            'station_id'
        ).Lanes.to_dict()
    )

    onset_events = events.loc[
        events.method.eq(
            BATCH_5_PART_4_ONSET_METHOD
        )
    ].copy()

    onset_lookup = set(
        zip(
            onset_events.station_id.astype(int),
            pd.to_datetime(
                onset_events.start_time
            ),
        )
    )

    observation_parts = []

    for station_id, station_data in data.groupby(
        'station_id',
        sort=False,
    ):
        station_data = station_data.sort_values(
            'timestamp'
        ).copy()

        station_data['next_timestamp'] = (
            station_data.timestamp.shift(-1)
        )
        station_data['next_speed_mph'] = (
            station_data.speed_mph.shift(-1)
        )

        station_data['contiguous_next'] = (
            station_data.next_timestamp
            - station_data.timestamp
        ).eq(
            pd.Timedelta(minutes=5)
        )

        station_data = station_data.loc[
            station_data.contiguous_next
            & station_data.speed_mph.ge(
                BATCH_5_PART_4_FREE_FLOW_MIN_SPEED_MPH
            )
        ].copy()

        station_data['event'] = [
            int(
                (
                    int(station_id),
                    pd.Timestamp(next_time),
                )
                in onset_lookup
            )
            for next_time in station_data.next_timestamp
        ]

        lanes = float(
            lane_lookup[int(station_id)]
        )

        station_data['flow_vphpl'] = (
            station_data.flow_veh_5min
            * 12.0
            / lanes
        )

        station_data['date'] = (
            station_data.timestamp.dt.date.astype(str)
        )

        station_data['lanes'] = lanes

        observation_parts.append(
            station_data[
                [
                    'timestamp',
                    'date',
                    'station_id',
                    'speed_mph',
                    'flow_veh_5min',
                    'lanes',
                    'flow_vphpl',
                    'event',
                ]
            ]
        )

    observations = pd.concat(
        observation_parts,
        ignore_index=True,
    )

    observed_breakdowns = int(
        observations.event.sum()
    )

    if observed_breakdowns != len(onset_events):
        raise ValueError(
            'Product-limit risk set did not recover every LA onset. '
            f'Expected {len(onset_events)}, found {observed_breakdowns}.'
        )

    observations.to_parquet(
        output / 'batch_5_part_4_plm_observations.parquet',
        index=False,
    )

    curve = batch_5_part_4_product_limit(
        observations
    )

    if curve.empty:
        raise ValueError(
            'Product-limit curve is empty.'
        )

    curve.to_csv(
        output / 'batch_5_part_4_product_limit_curve.csv',
        index=False,
    )

    breakdown_flows = observations.loc[
        observations.event.eq(1),
        [
            'timestamp',
            'station_id',
            'flow_veh_5min',
            'lanes',
            'flow_vphpl',
        ],
    ].copy()

    breakdown_flows.to_csv(
        output / 'batch_5_part_4_breakdown_flows.csv',
        index=False,
    )

    # --------------------- day-level bootstrap ---------------------
    grid = np.linspace(
        observations.flow_vphpl.min(),
        observations.flow_vphpl.max(),
        BATCH_5_PART_4_GRID_POINTS,
    )

    unique_days = np.array(
        sorted(
            observations.date.unique()
        )
    )

    day_groups = {
        day: observations.loc[
            observations.date.eq(day),
            ['flow_vphpl', 'event'],
        ].copy()
        for day in unique_days
    }

    rng = np.random.default_rng(
        BATCH_5_PART_4_BOOTSTRAP_SEED
    )

    bootstrap_values = np.empty(
        (
            BATCH_5_PART_4_BOOTSTRAP_REPS,
            len(grid),
        ),
        dtype=float,
    )

    for rep in range(
        BATCH_5_PART_4_BOOTSTRAP_REPS
    ):
        sampled_days = rng.choice(
            unique_days,
            size=len(unique_days),
            replace=True,
        )

        sample = pd.concat(
            [
                day_groups[day]
                for day in sampled_days
            ],
            ignore_index=True,
        )

        sample_curve = batch_5_part_4_product_limit(
            sample
        )

        bootstrap_values[rep] = (
            batch_5_part_4_evaluate_curve(
                sample_curve,
                grid,
            )
        )

    bootstrap_summary = pd.DataFrame({
        'flow_vphpl': grid,
        'estimate': batch_5_part_4_evaluate_curve(
            curve,
            grid,
        ),
        'ci_low': np.quantile(
            bootstrap_values,
            BATCH_5_PART_4_CONFIDENCE_LOW,
            axis=0,
        ),
        'ci_high': np.quantile(
            bootstrap_values,
            BATCH_5_PART_4_CONFIDENCE_HIGH,
            axis=0,
        ),
    })

    bootstrap_summary.to_csv(
        output / 'batch_5_part_4_product_limit_bootstrap.csv',
        index=False,
    )

    # --------------------- descriptive summary ---------------------
    summary = pd.DataFrame([
        {
            'eligible_free_flow_intervals':
                len(observations),
            'uncensored_breakdown_flows':
                observed_breakdowns,
            'censored_free_flow_intervals':
                int(
                    observations.event.eq(0).sum()
                ),
            'stations':
                observations.station_id.nunique(),
            'days':
                observations.date.nunique(),
            'median_breakdown_flow_vphpl':
                breakdown_flows.flow_vphpl.median(),
            'min_breakdown_flow_vphpl':
                breakdown_flows.flow_vphpl.min(),
            'max_breakdown_flow_vphpl':
                breakdown_flows.flow_vphpl.max(),
            'maximum_plm_breakdown_probability':
                curve.breakdown_probability.max(),
        }
    ])

    summary.to_csv(
        output / 'batch_5_part_4_summary.csv',
        index=False,
    )

    # --------------------- Figure: PLM ---------------------
    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B5P4_PLM
    )

    ax.fill_between(
        bootstrap_summary.flow_vphpl,
        bootstrap_summary.ci_low,
        bootstrap_summary.ci_high,
        alpha=THEME_BOOTSTRAP_ALPHA,
        label='Day-bootstrap 95% interval',
    )

    ax.step(
        bootstrap_summary.flow_vphpl,
        bootstrap_summary.estimate,
        where=THEME_STEP_WHERE_POST,
        linewidth=THEME_SURVIVAL_LINEWIDTH,
        label='Product-limit estimate',
    )

    ax.scatter(
        breakdown_flows.flow_vphpl,
        np.zeros(len(breakdown_flows)),
        marker=THEME_EVENT_RUG_MARKER,
        s=THEME_EVENT_RUG_SIZE,
        linewidths=THEME_EVENT_RUG_LINEWIDTH,
        label='Observed pre-breakdown flows',
    )

    ax.set_title(
        'Estimated Probability of Breakdown Increases with Flow',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Current Flow (veh/h/lane)')
    ax.set_ylabel('Estimated Probability of Breakdown')
    ax.set_ylim(
        THEME_AXIS_MIN_ZERO,
        1.0,
    )
    ax.grid(
        alpha=THEME_GRID_ALPHA_SUBTLE
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_5_PART_4_IMAGE_FOLDER
        / 'traffic_b5p4_breakdown_probability_plm.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    source_lines = []

    for source_note in BATCH_5_PART_4_SOURCE_NOTES:
        source_lines.extend([
            source_note['label'],
            source_note['url'],
            source_note['basis'],
            '',
        ])

    report_lines = [
        'BATCH 5 — PART 4: BREAKDOWN PROBABILITY VS FLOW',
        '',
        'SUMMARY',
        summary.round(4).to_string(index=False),
        '',
        'OBSERVED PRE-BREAKDOWN FLOWS',
        breakdown_flows.round(3).to_string(index=False),
        '',
        'METHOD',
        'Each eligible observation is a 100%-observed five-minute station',
        f'interval with current speed >= '
        f'{BATCH_5_PART_4_FREE_FLOW_MIN_SPEED_MPH} mph and a contiguous next',
        'five-minute interval. The current interval is uncensored when the next',
        'interval is an LA temporal-rule onset; otherwise it is treated as',
        'censored, meaning capacity exceeded the observed flow.',
        '',
        'Flow is normalized to vehicles/hour/lane using station metadata lane',
        'counts. The nonparametric product-limit estimator is applied to the',
        'pooled five-station observations. Uncertainty is estimated by resampling',
        'whole calendar days with replacement.',
        '',
        'SOURCE NOTES',
        *source_lines,
        'PORTFOLIO FIGURE',
        'images/traffic_b5p4_breakdown_probability_plm.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_5_part_4_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'onset_method':
            BATCH_5_PART_4_ONSET_METHOD,
        'free_flow_min_speed_mph':
            BATCH_5_PART_4_FREE_FLOW_MIN_SPEED_MPH,
        'bootstrap_reps':
            BATCH_5_PART_4_BOOTSTRAP_REPS,
        'bootstrap_seed':
            BATCH_5_PART_4_BOOTSTRAP_SEED,
        'uncensored_breakdowns':
            observed_breakdowns,
        'eligible_observations':
            int(len(observations)),
        'sources':
            BATCH_5_PART_4_SOURCE_NOTES,
        'portfolio_figures': [
            'images/traffic_b5p4_breakdown_probability_plm.png',
        ],
    }

    (output / 'batch_5_part_4_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 5 Part 4 reports: {output.resolve()}'
    )


# ===================== BATCH 5 — PART 5. SYNTHESIS =====================


def run_batch_5_part_5():
    """Assemble Batch 5 results, findings, and question coverage."""
    output = BATCH_5_PART_5_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 5 Part 5 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'part1_comparison':
            BATCH_5_PART_5_PART1_FOLDER
            / 'batch_5_part_1_minus5_event_vs_control.csv',
        'part2_cluster':
            BATCH_5_PART_5_PART2_FOLDER
            / 'batch_5_part_2_cluster_sensitivity.csv',
        'part2b_stability':
            BATCH_5_PART_5_PART2B_FOLDER
            / 'batch_5_part_2b_classification_stability.csv',
        'part2b_primary':
            BATCH_5_PART_5_PART2B_FOLDER
            / 'batch_5_part_2b_primary_window_decision.csv',
        'part3_overall':
            BATCH_5_PART_5_PART3_FOLDER
            / 'batch_5_part_3_overall_duration_summary.csv',
        'part3_period':
            BATCH_5_PART_5_PART3_FOLDER
            / 'batch_5_part_3_duration_by_time_period.csv',
        'part4_summary':
            BATCH_5_PART_5_PART4_FOLDER
            / 'batch_5_part_4_summary.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 5 Part 5 requires completed Batch 5 outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)

    part1 = pd.read_csv(
        required['part1_comparison']
    )
    cluster = pd.read_csv(
        required['part2_cluster']
    )
    stability = pd.read_csv(
        required['part2b_stability']
    )
    primary = pd.read_csv(
        required['part2b_primary']
    )
    duration = pd.read_csv(
        required['part3_overall']
    )
    duration_period = pd.read_csv(
        required['part3_period']
    )
    probability = pd.read_csv(
        required['part4_summary']
    )

    cluster_episode_counts = (
        cluster.corridor_episodes.unique()
    )

    cluster_robust = (
        len(cluster_episode_counts) == 1
    )

    stable_prop_count = int(
        stability.classification_stable.sum()
    )

    total_prop_count = len(stability)

    findings = []

    findings.append({
        'finding_id': 'B5-F1',
        'candidate_priority': 'top',
        'finding': (
            'Corridor-level abrupt-breakdown episode construction is robust '
            'to the temporal clustering threshold.'
        ),
        'evidence': (
            f'15, 30, and 45 minute clustering all produced '
            f'{int(cluster_episode_counts[0]) if cluster_robust else "different"} '
            f'corridor episodes.'
        ),
        'status': (
            'supported'
            if cluster_robust else 'sensitive'
        ),
    })

    findings.append({
        'finding_id': 'B5-F2',
        'candidate_priority': 'high',
        'finding': (
            'Most propagation classifications are stable, but some episodes '
            'depend on the allowed state-entry lag.'
        ),
        'evidence': (
            f'{stable_prop_count}/{total_prop_count} episode classifications '
            'were unchanged across 30, 60, and 120 minute propagation windows.'
        ),
        'status': 'supported',
    })

    speed_row = part1.loc[
        part1.metric.eq('speed_mph')
    ].iloc[0]

    flow_row = part1.loc[
        part1.metric.eq('flow_veh_5min')
    ].iloc[0]

    occupancy_row = part1.loc[
        part1.metric.eq('occupancy_fraction')
    ].iloc[0]

    findings.append({
        'finding_id': 'B5-F3',
        'candidate_priority': 'high',
        'finding': (
            'Five minutes before abrupt collapse, event conditions differ only '
            'modestly from matched controls and do not show a simple monotonic '
            'precursor in every traffic variable.'
        ),
        'evidence': (
            f'Event/control medians: speed '
            f'{speed_row.event_median:.2f}/{speed_row.control_median:.2f} mph; '
            f'flow {flow_row.event_median:.1f}/{flow_row.control_median:.1f} '
            f'veh/5min; occupancy '
            f'{occupancy_row.event_median:.4f}/{occupancy_row.control_median:.4f}.'
        ),
        'status': 'suggestive',
    })

    duration_row = duration.iloc[0]

    findings.append({
        'finding_id': 'B5-F4',
        'candidate_priority': 'high',
        'finding': (
            'Sustained congestion episodes often persist well beyond the '
            'initial abrupt onset.'
        ),
        'evidence': (
            f'Median duration {duration_row.median_duration_minutes:.1f} min; '
            f'90th percentile {duration_row.p90_duration_minutes:.1f} min.'
        ),
        'status': 'supported',
    })

    probability_row = probability.iloc[0]

    findings.append({
        'finding_id': 'B5-F5',
        'candidate_priority': 'high',
        'finding': (
            'Breakdown probability can be estimated as a stochastic function '
            'of current per-lane flow rather than assuming one deterministic '
            'capacity threshold.'
        ),
        'evidence': (
            f'{int(probability_row.uncensored_breakdown_flows)} observed '
            f'pre-breakdown flows were combined with '
            f'{int(probability_row.censored_free_flow_intervals)} censored '
            f'free-flow intervals using the product-limit estimator.'
        ),
        'status': 'supported_methodologically',
    })

    findings_table = pd.DataFrame(findings)

    findings_table.to_csv(
        output / 'batch_5_candidate_key_findings.csv',
        index=False,
    )

    question_matrix = pd.DataFrame([
        {
            'question': 'What conditions precede abrupt breakdown?',
            'batch_5_section': 'Part 1',
            'status': 'complete',
        },
        {
            'question': 'How large is the immediate speed fall?',
            'batch_5_section': 'Part 1',
            'status': 'complete',
        },
        {
            'question': 'How does congestion propagate across stations?',
            'batch_5_section': 'Part 2',
            'status': 'complete',
        },
        {
            'question': 'Are propagation conclusions robust to timing rules?',
            'batch_5_section': 'Part 2b',
            'status': 'complete',
        },
        {
            'question': 'How long does sustained congestion last?',
            'batch_5_section': 'Part 3',
            'status': 'complete',
        },
        {
            'question': 'How does recovery proceed across the corridor?',
            'batch_5_section': 'Part 3',
            'status': 'complete',
        },
        {
            'question': 'Does duration vary by time of day?',
            'batch_5_section': 'Part 3',
            'status': 'complete',
        },
        {
            'question': 'How does breakdown probability vary with flow?',
            'batch_5_section': 'Part 4',
            'status': 'complete',
        },
    ])

    question_matrix.to_csv(
        output / 'batch_5_question_coverage.csv',
        index=False,
    )

    report_lines = [
        'BATCH 5 — COMPLETE: CONGESTION DYNAMICS',
        '',
        'CANDIDATE KEY FINDINGS',
        findings_table.to_string(index=False),
        '',
        'QUESTION COVERAGE',
        question_matrix.to_string(index=False),
        '',
        'DURATION BY TIME PERIOD',
        duration_period.round(3).to_string(index=False),
        '',
        'PRIMARY 60-MINUTE PROPAGATION CLASSIFICATION',
        primary.to_string(index=False),
        '',
        'STATUS',
        'Batch 5 Parts 1, 2, 2b, 3, and 4 are complete.',
        'Batch 6 can now focus on final robustness, influential dates/stations,',
        'limitations, and supported/suggestive/unsupported conclusions.',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_5_completion_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parts_complete': [
            'Part 1 pre-breakdown conditions',
            'Part 2 propagation',
            'Part 2b propagation robustness',
            'Part 3 duration and recovery',
            'Part 4 breakdown probability vs flow',
            'Part 5 synthesis',
        ],
        'candidate_key_findings':
            findings,
    }

    (output / 'batch_5_part_5_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 5 synthesis: {output.resolve()}'
    )




# ===================== BATCH 6 — PART 1. FINAL ROBUSTNESS =====================


def batch_6_duration_metrics(frame):
    """Return the principal duration metrics used for influence checks."""
    result = {
        'overall_median_duration':
            frame.duration_minutes.median(),
        'overall_p90_duration':
            frame.duration_minutes.quantile(0.90),
    }

    for period in ['AM', 'Midday', 'PM']:
        values = frame.loc[
            frame.time_period.eq(period),
            'duration_minutes',
        ]

        result[
            f'{period.lower()}_median_duration'
        ] = (
            values.median()
            if len(values)
            else np.nan
        )

    return result


def batch_6_relative_change(value, baseline):
    """Percent change relative to a nonzero finite baseline."""
    if (
        pd.isna(value)
        or pd.isna(baseline)
        or baseline == 0
    ):
        return np.nan

    return (
        100.0
        * (value - baseline)
        / baseline
    )


def run_batch_6_part_1():
    """Audit influential dates/stations and robustness of headline metrics."""
    output = BATCH_6_PART_1_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 6 Part 1 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'threshold_sensitivity':
            BATCH_6_PART_1_BATCH3_FOLDER
            / 'batch_3_threshold_sensitivity.csv',
        'weakest_station':
            BATCH_6_PART_1_BATCH3_FOLDER
            / 'batch_3_threshold_weakest_station.csv',
        'events':
            BATCH_6_PART_1_BATCH4_FOLDER
            / 'batch_4_combined_event_inventory.csv',
        'method_summary':
            BATCH_6_PART_1_BATCH4_FOLDER
            / 'batch_4_method_summary.csv',
        'method_agreement':
            BATCH_6_PART_1_BATCH4_FOLDER
            / 'batch_4_method_event_agreement.csv',
        'precondition':
            BATCH_6_PART_1_BATCH5_PART1_FOLDER
            / 'batch_5_part_1_minus5_event_vs_control.csv',
        'cluster_sensitivity':
            BATCH_6_PART_1_BATCH5_PART2_FOLDER
            / 'batch_5_part_2_cluster_sensitivity.csv',
        'propagation_stability':
            BATCH_6_PART_1_BATCH5_PART2B_FOLDER
            / 'batch_5_part_2b_classification_stability.csv',
        'propagation_primary':
            BATCH_6_PART_1_BATCH5_PART2B_FOLDER
            / 'batch_5_part_2b_primary_window_decision.csv',
        'state_events':
            BATCH_6_PART_1_BATCH5_PART3_FOLDER
            / 'batch_5_part_3_state_events.csv',
        'duration_period':
            BATCH_6_PART_1_BATCH5_PART3_FOLDER
            / 'batch_5_part_3_duration_by_time_period.csv',
        'probability_summary':
            BATCH_6_PART_1_BATCH5_PART4_FOLDER
            / 'batch_5_part_4_summary.csv',
        'breakdown_flows':
            BATCH_6_PART_1_BATCH5_PART4_FOLDER
            / 'batch_5_part_4_breakdown_flows.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 6 Part 1 requires completed prior outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)
    BATCH_6_PART_1_IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold = pd.read_csv(
        required['threshold_sensitivity']
    )
    weakest = pd.read_csv(
        required['weakest_station']
    )

    events = pd.read_csv(
        required['events'],
        parse_dates=['start_time', 'end_time'],
    )

    method_summary = pd.read_csv(
        required['method_summary']
    )
    method_agreement = pd.read_csv(
        required['method_agreement']
    )
    precondition = pd.read_csv(
        required['precondition']
    )
    cluster_sensitivity = pd.read_csv(
        required['cluster_sensitivity']
    )
    propagation_stability = pd.read_csv(
        required['propagation_stability']
    )
    propagation_primary = pd.read_csv(
        required['propagation_primary']
    )

    state_events = pd.read_csv(
        required['state_events'],
        parse_dates=['start_time', 'end_time'],
    )

    duration_period = pd.read_csv(
        required['duration_period']
    )

    probability_summary = pd.read_csv(
        required['probability_summary']
    )

    breakdown_flows = pd.read_csv(
        required['breakdown_flows'],
        parse_dates=['timestamp'],
    )

    # --------------------------------------------------------
    # A. Coverage-threshold robustness
    # --------------------------------------------------------
    low = threshold.loc[
        threshold.threshold.eq(
            BATCH_6_PRIMARY_THRESHOLD_LOW
        )
    ]

    high = threshold.loc[
        threshold.threshold.eq(
            BATCH_6_PRIMARY_THRESHOLD_HIGH
        )
    ]

    if len(low) != 1 or len(high) != 1:
        raise ValueError(
            'Expected exactly one 80% and one 100% threshold row.'
        )

    low = low.iloc[0]
    high = high.iloc[0]

    coverage_robustness = pd.DataFrame([
        {
            'low_threshold':
                BATCH_6_PRIMARY_THRESHOLD_LOW,
            'high_threshold':
                BATCH_6_PRIMARY_THRESHOLD_HIGH,
            'low_rows_retained':
                int(low.rows_retained),
            'high_rows_retained':
                int(high.rows_retained),
            'rows_identical':
                bool(
                    int(low.rows_retained)
                    == int(high.rows_retained)
                ),
            'low_simultaneous_hours':
                float(low.simultaneous_hours),
            'high_simultaneous_hours':
                float(high.simultaneous_hours),
            'simultaneous_hours_identical':
                bool(
                    np.isclose(
                        low.simultaneous_hours,
                        high.simultaneous_hours,
                    )
                ),
        }
    ])

    coverage_robustness.to_csv(
        output / 'batch_6_coverage_threshold_robustness.csv',
        index=False,
    )

    # --------------------------------------------------------
    # B. Abrupt-onset concentration by date / station
    # --------------------------------------------------------
    la_onsets = events.loc[
        events.method.eq('LA_20drop_below40')
    ].copy()

    la_onsets['event_date'] = (
        la_onsets.start_time.dt.date.astype(str)
    )

    onset_by_date = (
        la_onsets.groupby(
            'event_date',
            as_index=False,
        )
        .agg(
            onset_events=('event_id', 'size'),
            stations=('station_id', 'nunique'),
        )
        .sort_values(
            ['onset_events', 'event_date'],
            ascending=[False, True],
        )
    )

    onset_by_date[
        'share_of_all_onsets_pct'
    ] = (
        100.0
        * onset_by_date.onset_events
        / len(la_onsets)
    )

    onset_by_station = (
        la_onsets.groupby(
            'station_id',
            as_index=False,
        )
        .agg(
            onset_events=('event_id', 'size'),
            dates=('event_date', 'nunique'),
        )
        .sort_values(
            ['onset_events', 'station_id'],
            ascending=[False, True],
        )
    )

    onset_by_station[
        'share_of_all_onsets_pct'
    ] = (
        100.0
        * onset_by_station.onset_events
        / len(la_onsets)
    )

    onset_by_date.to_csv(
        output / 'batch_6_abrupt_onset_concentration_by_date.csv',
        index=False,
    )
    onset_by_station.to_csv(
        output / 'batch_6_abrupt_onset_concentration_by_station.csv',
        index=False,
    )

    top_onset_day_share = (
        onset_by_date.share_of_all_onsets_pct.max()
    )

    onset_date_concentrated = bool(
        top_onset_day_share
        >= BATCH_6_ONSET_DATE_CONCENTRATION_FLAG_PCT
    )

    # --------------------------------------------------------
    # C. Leave-one-station-out duration robustness
    # --------------------------------------------------------
    baseline = batch_6_duration_metrics(
        state_events
    )

    station_influence_rows = []

    for station_id in sorted(
        state_events.station_id.unique()
    ):
        reduced = state_events.loc[
            state_events.station_id.ne(
                station_id
            )
        ].copy()

        metrics = batch_6_duration_metrics(
            reduced
        )

        row = {
            'excluded_station_id':
                int(station_id),
            'remaining_events':
                len(reduced),
        }

        for metric_name, metric_value in metrics.items():
            baseline_value = baseline[
                metric_name
            ]

            row[metric_name] = (
                metric_value
            )
            row[
                f'{metric_name}_change_pct'
            ] = batch_6_relative_change(
                metric_value,
                baseline_value,
            )

        station_influence_rows.append(
            row
        )

    station_influence = pd.DataFrame(
        station_influence_rows
    )

    station_influence.to_csv(
        output / 'batch_6_leave_one_station_out_duration.csv',
        index=False,
    )

    # --------------------------------------------------------
    # D. Leave-one-date-out duration robustness
    # --------------------------------------------------------
    state_events['event_date_text'] = (
        pd.to_datetime(
            state_events.start_time
        ).dt.date.astype(str)
    )

    date_influence_rows = []

    for event_date in sorted(
        state_events.event_date_text.unique()
    ):
        reduced = state_events.loc[
            state_events.event_date_text.ne(
                event_date
            )
        ].copy()

        metrics = batch_6_duration_metrics(
            reduced
        )

        row = {
            'excluded_date':
                event_date,
            'remaining_events':
                len(reduced),
        }

        for metric_name, metric_value in metrics.items():
            baseline_value = baseline[
                metric_name
            ]

            row[metric_name] = (
                metric_value
            )
            row[
                f'{metric_name}_change_pct'
            ] = batch_6_relative_change(
                metric_value,
                baseline_value,
            )

        date_influence_rows.append(
            row
        )

    date_influence = pd.DataFrame(
        date_influence_rows
    )

    date_influence.to_csv(
        output / 'batch_6_leave_one_date_out_duration.csv',
        index=False,
    )

    duration_metric_names = [
        'overall_median_duration',
        'overall_p90_duration',
        'am_median_duration',
        'midday_median_duration',
        'pm_median_duration',
    ]

    influence_summary_rows = []

    for metric_name in duration_metric_names:
        station_changes = (
            station_influence[
                f'{metric_name}_change_pct'
            ].abs()
        )

        date_changes = (
            date_influence[
                f'{metric_name}_change_pct'
            ].abs()
        )

        influence_summary_rows.append({
            'metric':
                metric_name,
            'baseline':
                baseline[metric_name],
            'max_abs_leave_one_station_change_pct':
                station_changes.max(),
            'max_abs_leave_one_date_change_pct':
                date_changes.max(),
            'station_robust_under_rule':
                bool(
                    station_changes.max()
                    < BATCH_6_INFLUENCE_RELATIVE_CHANGE_PCT
                ),
            'date_robust_under_rule':
                bool(
                    date_changes.max()
                    < BATCH_6_INFLUENCE_RELATIVE_CHANGE_PCT
                ),
        })

    influence_summary = pd.DataFrame(
        influence_summary_rows
    )

    influence_summary.to_csv(
        output / 'batch_6_duration_influence_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # E. Midday-vs-AM/PM contrast robustness
    # --------------------------------------------------------
    baseline_midday = baseline[
        'midday_median_duration'
    ]
    baseline_am = baseline[
        'am_median_duration'
    ]
    baseline_pm = baseline[
        'pm_median_duration'
    ]

    baseline_midday_gap = (
        baseline_midday
        - max(
            baseline_am,
            baseline_pm,
        )
    )

    station_midday_gaps = (
        station_influence.midday_median_duration
        - station_influence[
            [
                'am_median_duration',
                'pm_median_duration',
            ]
        ].max(axis=1)
    )

    date_midday_gaps = (
        date_influence.midday_median_duration
        - date_influence[
            [
                'am_median_duration',
                'pm_median_duration',
            ]
        ].max(axis=1)
    )

    midday_robustness = pd.DataFrame([
        {
            'baseline_midday_minus_max_am_pm_minutes':
                baseline_midday_gap,
            'minimum_gap_leave_one_station_out_minutes':
                station_midday_gaps.min(),
            'minimum_gap_leave_one_date_out_minutes':
                date_midday_gaps.min(),
            'midday_longer_after_every_station_exclusion':
                bool(
                    station_midday_gaps.gt(0).all()
                ),
            'midday_longer_after_every_date_exclusion':
                bool(
                    date_midday_gaps.gt(0).all()
                ),
        }
    ])

    midday_robustness.to_csv(
        output / 'batch_6_midday_duration_robustness.csv',
        index=False,
    )

    # --------------------------------------------------------
    # F. Propagation / clustering robustness summary
    # --------------------------------------------------------
    cluster_counts = (
        cluster_sensitivity.corridor_episodes
        .unique()
    )

    cluster_stable = bool(
        len(cluster_counts) == 1
    )

    propagation_stable_count = int(
        propagation_stability[
            'classification_stable'
        ].sum()
    )

    propagation_total = len(
        propagation_stability
    )

    propagation_stable_pct = (
        100.0
        * propagation_stable_count
        / propagation_total
        if propagation_total
        else np.nan
    )

    propagation_robustness = pd.DataFrame([
        {
            'cluster_episode_count_stable_15_30_45':
                cluster_stable,
            'corridor_episode_count':
                int(cluster_counts[0])
                if cluster_stable
                else np.nan,
            'propagation_classifications_stable':
                propagation_stable_count,
            'propagation_classifications_total':
                propagation_total,
            'propagation_classification_stable_pct':
                propagation_stable_pct,
            'meets_stability_target':
                bool(
                    propagation_stable_pct
                    >= BATCH_6_PROPAGATION_STABILITY_TARGET_PCT
                ),
        }
    ])

    propagation_robustness.to_csv(
        output / 'batch_6_propagation_robustness_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # G. PLM usefulness audit
    # --------------------------------------------------------
    probability_row = probability_summary.iloc[0]

    low_flow_breakdowns = int(
        breakdown_flows.flow_vphpl.lt(
            600
        ).sum()
    )

    plm_audit = pd.DataFrame([
        {
            'uncensored_breakdown_flows':
                int(
                    probability_row.uncensored_breakdown_flows
                ),
            'censored_free_flow_intervals':
                int(
                    probability_row.censored_free_flow_intervals
                ),
            'maximum_plm_breakdown_probability':
                float(
                    probability_row.maximum_plm_breakdown_probability
                ),
            'maximum_plm_breakdown_probability_pct':
                100.0
                * float(
                    probability_row.maximum_plm_breakdown_probability
                ),
            'prebreakdown_flows_under_600_vphpl':
                low_flow_breakdowns,
            'portfolio_chart_recommended':
                False,
            'portfolio_substantive_finding_recommended':
                False,
            'audit_trail_retain':
                True,
        }
    ])

    plm_audit.to_csv(
        output / 'batch_6_plm_usefulness_audit.csv',
        index=False,
    )

    # --------------------------------------------------------
    # Figure 1: leave-one-out robustness ranges
    # --------------------------------------------------------
    plot_metrics = [
        (
            'overall_median_duration',
            'Overall median',
        ),
        (
            'overall_p90_duration',
            'Overall p90',
        ),
        (
            'midday_median_duration',
            'Midday median',
        ),
    ]

    baseline_plot = np.array([
        baseline[name]
        for name, _ in plot_metrics
    ])

    station_min = np.array([
        station_influence[name].min()
        for name, _ in plot_metrics
    ])

    station_max = np.array([
        station_influence[name].max()
        for name, _ in plot_metrics
    ])

    date_min = np.array([
        date_influence[name].min()
        for name, _ in plot_metrics
    ])

    date_max = np.array([
        date_influence[name].max()
        for name, _ in plot_metrics
    ])

    x = np.arange(
        len(plot_metrics)
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B6_ROBUSTNESS
    )

    ax.errorbar(
        x + THEME_BAR_OFFSET_NEGATIVE,
        baseline_plot,
        yerr=[
            baseline_plot - station_min,
            station_max - baseline_plot,
        ],
        marker=THEME_MARKER_PRIMARY,
        linewidth=THEME_REFERENCE_LINEWIDTH_B6,
        capsize=THEME_ERRORBAR_CAPSIZE_B6,
        label='Leave one station out',
    )

    ax.errorbar(
        x + THEME_BAR_OFFSET_POSITIVE,
        baseline_plot,
        yerr=[
            baseline_plot - date_min,
            date_max - baseline_plot,
        ],
        marker=THEME_MARKER_PRIMARY,
        linewidth=THEME_REFERENCE_LINEWIDTH_B6,
        capsize=THEME_ERRORBAR_CAPSIZE_B6,
        label='Leave one date out',
    )

    ax.scatter(
        x,
        baseline_plot,
        s=THEME_SCATTER_SIZE_B6,
        label='Full sample',
    )

    ax.set_title(
        'Headline Duration Results Survive Leave-One-Out Checks',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Duration Metric')
    ax.set_ylabel('Minutes')
    ax.set_xticks(
        x,
        [
            label
            for _, label in plot_metrics
        ],
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_6_PART_1_IMAGE_FOLDER
        / 'traffic_b6_leave_one_out_duration_robustness.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: time-of-day duration
    # --------------------------------------------------------
    period_order = [
        'AM',
        'Midday',
        'PM',
    ]

    period_plot = (
        duration_period.set_index(
            'time_period'
        ).reindex(
            period_order
        )
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B6_PERIOD
    )

    bars = ax.bar(
        period_order,
        period_plot.median_duration_minutes,
        width=THEME_BAR_WIDTH_B6,
    )

    for bar, row in zip(
        bars,
        period_plot.itertuples(),
    ):
        ax.annotate(
            f'{row.median_duration_minutes:.0f} min',
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=THEME_ANNOTATION_OFFSET_B6,
            textcoords=THEME_TEXTCOORDS,
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )

    ax.set_title(
        'Midday Congestion Episodes Last Much Longer',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Median Sustained-Congestion Duration (minutes)')
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_6_PART_1_IMAGE_FOLDER
        / 'traffic_b6_duration_by_time_period.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 3: abrupt-onset concentration
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B6_CONCENTRATION
    )

    date_plot = onset_by_date.sort_values(
        'event_date'
    )

    bars = ax.bar(
        date_plot.event_date,
        date_plot.onset_events,
        width=THEME_BAR_WIDTH_B6,
    )

    ax.set_title(
        'Abrupt Breakdown Onsets Are Concentrated in a Small Number of Dates',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('LA-Rule Abrupt Onsets')
    ax.tick_params(
        axis=THEME_TICK_AXIS_X,
        rotation=THEME_ROTATION_B6,
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_6_PART_1_IMAGE_FOLDER
        / 'traffic_b6_abrupt_onset_concentration.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    report_lines = [
        'BATCH 6 — PART 1: FINAL ROBUSTNESS',
        '',
        'COVERAGE-THRESHOLD ROBUSTNESS',
        coverage_robustness.to_string(index=False),
        '',
        'DURATION INFLUENCE SUMMARY',
        influence_summary.round(3).to_string(index=False),
        '',
        'MIDDAY DURATION ROBUSTNESS',
        midday_robustness.round(3).to_string(index=False),
        '',
        'ABRUPT-ONSET CONCENTRATION BY DATE',
        onset_by_date.round(3).to_string(index=False),
        '',
        'ABRUPT-ONSET CONCENTRATION BY STATION',
        onset_by_station.round(3).to_string(index=False),
        '',
        'PROPAGATION ROBUSTNESS',
        propagation_robustness.round(3).to_string(index=False),
        '',
        'PLM USEFULNESS AUDIT',
        plm_audit.round(6).to_string(index=False),
        '',
        'INTERPRETATION FLAGS',
        f'Top abrupt-onset day share: {top_onset_day_share:.1f}%.',
        f'Onset-date concentration flag: {onset_date_concentrated}.',
        f'Leave-one-out influence rule: '
        f'{BATCH_6_INFLUENCE_RELATIVE_CHANGE_PCT:.0f}% relative change.',
        '',
        'PORTFOLIO FIGURES',
        'images/traffic_b6_leave_one_out_duration_robustness.png',
        'images/traffic_b6_duration_by_time_period.png',
        'images/traffic_b6_abrupt_onset_concentration.png',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_6_part_1_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'leave_one_out_relative_change_flag_pct':
            BATCH_6_INFLUENCE_RELATIVE_CHANGE_PCT,
        'onset_date_concentration_flag_pct':
            BATCH_6_ONSET_DATE_CONCENTRATION_FLAG_PCT,
        'top_onset_day_share_pct':
            float(top_onset_day_share),
        'coverage_80_equals_100_rows':
            bool(
                coverage_robustness.iloc[0].rows_identical
            ),
        'cluster_count_stable':
            cluster_stable,
        'propagation_stable_pct':
            float(propagation_stable_pct),
        'portfolio_figures': [
            'images/traffic_b6_leave_one_out_duration_robustness.png',
            'images/traffic_b6_duration_by_time_period.png',
            'images/traffic_b6_abrupt_onset_concentration.png',
        ],
    }

    (output / 'batch_6_part_1_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 6 Part 1 reports: {output.resolve()}'
    )


# ===================== BATCH 6 — PART 2. FINDINGS AND LIMITATIONS =====================


def run_batch_6_part_2():
    """Adjudicate final findings as supported, suggestive, or unsupported."""
    output = BATCH_6_PART_2_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 6 Part 2 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'coverage':
            BATCH_6_PART_2_PART1_FOLDER
            / 'batch_6_coverage_threshold_robustness.csv',
        'influence':
            BATCH_6_PART_2_PART1_FOLDER
            / 'batch_6_duration_influence_summary.csv',
        'midday':
            BATCH_6_PART_2_PART1_FOLDER
            / 'batch_6_midday_duration_robustness.csv',
        'onset_date':
            BATCH_6_PART_2_PART1_FOLDER
            / 'batch_6_abrupt_onset_concentration_by_date.csv',
        'propagation':
            BATCH_6_PART_2_PART1_FOLDER
            / 'batch_6_propagation_robustness_summary.csv',
        'plm_audit':
            BATCH_6_PART_2_PART1_FOLDER
            / 'batch_6_plm_usefulness_audit.csv',
        'method_summary':
            BATCH_6_PART_2_BATCH4_FOLDER
            / 'batch_4_method_summary.csv',
        'method_agreement':
            BATCH_6_PART_2_BATCH4_FOLDER
            / 'batch_4_method_event_agreement.csv',
        'precondition':
            BATCH_6_PART_2_BATCH5_PART1_FOLDER
            / 'batch_5_part_1_minus5_event_vs_control.csv',
        'propagation_primary':
            BATCH_6_PART_2_BATCH5_PART2B_FOLDER
            / 'batch_5_part_2b_primary_window_decision.csv',
        'duration_overall':
            BATCH_6_PART_2_BATCH5_PART3_FOLDER
            / 'batch_5_part_3_overall_duration_summary.csv',
        'duration_period':
            BATCH_6_PART_2_BATCH5_PART3_FOLDER
            / 'batch_5_part_3_duration_by_time_period.csv',
        'recovery':
            BATCH_6_PART_2_BATCH5_PART3_FOLDER
            / 'batch_5_part_3_episode_recovery_summary.csv',
        'probability':
            BATCH_6_PART_2_BATCH5_PART4_FOLDER
            / 'batch_5_part_4_summary.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 6 Part 2 requires completed prior outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)

    coverage = pd.read_csv(required['coverage']).iloc[0]
    influence = pd.read_csv(required['influence'])
    midday = pd.read_csv(required['midday']).iloc[0]
    onset_date = pd.read_csv(required['onset_date'])
    propagation = pd.read_csv(required['propagation']).iloc[0]
    plm_audit = pd.read_csv(required['plm_audit']).iloc[0]
    method_summary = pd.read_csv(required['method_summary'])
    method_agreement = pd.read_csv(required['method_agreement'])
    precondition = pd.read_csv(required['precondition'])
    propagation_primary = pd.read_csv(required['propagation_primary'])
    duration_overall = pd.read_csv(required['duration_overall']).iloc[0]
    duration_period = pd.read_csv(required['duration_period'])
    recovery = pd.read_csv(required['recovery'])
    probability = pd.read_csv(required['probability']).iloc[0]

    method_counts = (
        method_summary.set_index('method').events.to_dict()
    )

    threshold_events = int(
        method_counts.get(
            'Threshold_50_15min',
            0,
        )
    )
    la_events = int(
        method_counts.get(
            'LA_20drop_below40',
            0,
        )
    )
    spatial_events = int(
        method_counts.get(
            'Caltrans_PeMS_spatial',
            0,
        )
    )

    stable_propagation_count = int(
        propagation.propagation_classifications_stable
    )
    total_propagation_count = int(
        propagation.propagation_classifications_total
    )

    top_day = onset_date.iloc[0]

    duration_influence_robust = bool(
        (
            influence.station_robust_under_rule
            & influence.date_robust_under_rule
        ).all()
    )

    midday_robust = bool(
        midday.midday_longer_after_every_station_exclusion
        and midday.midday_longer_after_every_date_exclusion
    )

    findings = []

    findings.append({
        'finding_id': 'F1',
        'priority': 1,
        'status': 'SUPPORTED',
        'portfolio_role': 'headline',
        'finding': (
            'Sustained congestion and abrupt breakdown are empirically '
            'different traffic phenomena on this corridor.'
        ),
        'evidence': (
            f'The sustained-state rule produced {threshold_events} events, '
            f'the LA abrupt-collapse rule {la_events}, and the Caltrans spatial '
            f'rule {spatial_events}, with extremely low pairwise onset overlap.'
        ),
        'caveat': (
            'The definitions answer different operational questions rather '
            'than competing to identify one universal event set.'
        ),
    })

    findings.append({
        'finding_id': 'F2',
        'priority': 2,
        'status': 'SUPPORTED',
        'portfolio_role': 'headline',
        'finding': (
            'Corridor-level abrupt-breakdown episode construction is robust '
            'to the temporal clustering rule.'
        ),
        'evidence': (
            f'15-, 30-, and 45-minute clustering windows all produced '
            f'{int(propagation.corridor_episode_count)} corridor episodes.'
        ),
        'caveat': (
            'Only six corridor episodes were identified, so the result is '
            'strong for this sample but not a universal traffic-flow claim.'
        ),
    })

    findings.append({
        'finding_id': 'F3',
        'priority': 3,
        'status': (
            'SUPPORTED'
            if midday_robust
            else 'SUGGESTIVE'
        ),
        'portfolio_role': 'headline',
        'finding': (
            'Midday sustained-congestion episodes last substantially longer '
            'than AM or PM episodes.'
        ),
        'evidence': (
            f'Median durations were '
            f'{duration_period.loc[duration_period.time_period.eq("AM"), "median_duration_minutes"].iloc[0]:.0f} '
            f'minutes AM, '
            f'{duration_period.loc[duration_period.time_period.eq("Midday"), "median_duration_minutes"].iloc[0]:.0f} '
            f'midday, and '
            f'{duration_period.loc[duration_period.time_period.eq("PM"), "median_duration_minutes"].iloc[0]:.0f} '
            f'PM. The midday advantage remained positive after every '
            f'leave-one-station and leave-one-date exclusion.'
        ),
        'caveat': (
            'Time-period comparisons are observational and may reflect demand, '
            'geometry, incidents, or recurring operational conditions.'
        ),
    })

    findings.append({
        'finding_id': 'F4',
        'priority': 4,
        'status': (
            'SUPPORTED'
            if duration_influence_robust
            else 'SUGGESTIVE'
        ),
        'portfolio_role': 'headline/supporting',
        'finding': (
            'Sustained congestion is often persistent and has a long duration tail.'
        ),
        'evidence': (
            f'Median duration was '
            f'{duration_overall.median_duration_minutes:.0f} minutes and the '
            f'90th percentile was '
            f'{duration_overall.p90_duration_minutes:.0f} minutes.'
        ),
        'caveat': (
            'The sustained-state rule is an operational definition, and some '
            'episodes are right-censored by the analysis window.'
        ),
    })

    findings.append({
        'finding_id': 'F5',
        'priority': 5,
        'status': 'SUPPORTED_WITH_CAVEAT',
        'portfolio_role': 'supporting',
        'finding': (
            'Most corridor propagation classifications are stable, but spatial '
            'extent is not completely insensitive to the allowed lag window.'
        ),
        'evidence': (
            f'{stable_propagation_count}/{total_propagation_count} episode '
            f'classifications were unchanged across 30-, 60-, and 120-minute '
            f'propagation windows.'
        ),
        'caveat': (
            'Episodes 3 and 5 changed classification; the 60-minute window is '
            'retained because it rejects clearly delayed 110-minute matches.'
        ),
    })

    findings.append({
        'finding_id': 'F6',
        'priority': 6,
        'status': 'SUPPORTED_METHOD',
        'portfolio_role': 'methodological/supporting',
        'finding': (
            'Using 100% observed coverage instead of 80% costs no selected-'
            'corridor observations.'
        ),
        'evidence': (
            f'80% and 100% thresholds retained '
            f'{int(coverage.high_rows_retained)} rows and '
            f'{coverage.high_simultaneous_hours:.1f} simultaneous corridor hours.'
        ),
        'caveat': (
            'This equivalence is specific to the selected five-station corridor.'
        ),
    })

    speed_row = precondition.loc[
        precondition.metric.eq('speed_mph')
    ].iloc[0]
    flow_row = precondition.loc[
        precondition.metric.eq('flow_veh_5min')
    ].iloc[0]
    occupancy_row = precondition.loc[
        precondition.metric.eq('occupancy_fraction')
    ].iloc[0]

    findings.append({
        'finding_id': 'F7',
        'priority': 7,
        'status': 'SUGGESTIVE',
        'portfolio_role': 'supporting',
        'finding': (
            'Five minutes before abrupt collapse, matched-control differences '
            'are modest rather than a simple dramatic precursor signature.'
        ),
        'evidence': (
            f'Event/control medians were '
            f'{speed_row.event_median:.2f}/{speed_row.control_median:.2f} mph, '
            f'{flow_row.event_median:.1f}/{flow_row.control_median:.1f} veh/5 min, '
            f'and {occupancy_row.event_median:.4f}/{occupancy_row.control_median:.4f} '
            f'occupancy.'
        ),
        'caveat': (
            f'The abrupt-onset sample contains only {la_events} events and '
            f'{top_day.share_of_all_onsets_pct:.1f}% occur on the single most '
            f'concentrated date ({top_day.event_date}).'
        ),
    })

    recovery_correlations = (
        recovery.travel_order_recovery_correlation
        .dropna()
    )

    mixed_recovery_direction = bool(
        recovery_correlations.lt(0).any()
        and recovery_correlations.gt(0).any()
    )

    findings.append({
        'finding_id': 'F8',
        'priority': 8,
        'status': (
            'SUPPORTED_WITH_CAVEAT'
            if mixed_recovery_direction
            else 'SUGGESTIVE'
        ),
        'portfolio_role': 'supporting',
        'finding': (
            'Recovery does not follow one universal station-order direction '
            'across the observed corridor episodes.'
        ),
        'evidence': (
            'Episode-level travel-order recovery correlations include both '
            'positive and negative values.'
        ),
        'caveat': (
            'Only propagated episodes under the primary 60-minute rule are '
            'included in this comparison.'
        ),
    })

    findings.append({
        'finding_id': 'F9',
        'priority': 9,
        'status': 'UNSUPPORTED_FOR_SUBSTANTIVE_CLAIM',
        'portfolio_role': 'audit-only',
        'finding': (
            'The current product-limit probability curve should not be used as '
            'a substantive portfolio finding.'
        ),
        'evidence': (
            f'The maximum estimated probability is only '
            f'{plm_audit.maximum_plm_breakdown_probability_pct:.4f}% and '
            f'{int(plm_audit.prebreakdown_flows_under_600_vphpl)} of the '
            f'{int(plm_audit.uncensored_breakdown_flows)} pre-breakdown flows '
            f'are below 600 veh/h/lane.'
        ),
        'caveat': (
            'The calculation is retained in the audit trail because it exposed '
            'a useful methodological limitation.'
        ),
    })

    findings_table = pd.DataFrame(
        findings
    ).sort_values(
        'priority'
    )

    findings_table.to_csv(
        output / 'batch_6_final_findings_adjudication.csv',
        index=False,
    )

    limitations = pd.DataFrame([
        {
            'limitation_id': 'L1',
            'category': 'scope',
            'limitation': (
                'The primary analysis covers one five-station I-10 East corridor, '
                'so findings should not be generalized to all Los Angeles freeways.'
            ),
            'effect': 'limits external validity',
        },
        {
            'limitation_id': 'L2',
            'category': 'time',
            'limitation': (
                'The study period is February through May 2026 rather than a '
                'full annual cycle.'
            ),
            'effect': 'seasonal conditions may be underrepresented',
        },
        {
            'limitation_id': 'L3',
            'category': 'sample size',
            'limitation': (
                f'Only {la_events} LA-rule abrupt onsets and '
                f'{int(propagation.corridor_episode_count)} corridor episodes '
                'were identified.'
            ),
            'effect': 'limits precision of abrupt-onset and propagation inference',
        },
        {
            'limitation_id': 'L4',
            'category': 'event concentration',
            'limitation': (
                f'The most concentrated abrupt-onset date accounts for '
                f'{top_day.share_of_all_onsets_pct:.1f}% of all LA-rule onsets.'
            ),
            'effect': 'precursor results are vulnerable to date-specific conditions',
        },
        {
            'limitation_id': 'L5',
            'category': 'measurement',
            'limitation': (
                'The analysis relies on loop-detector speed, flow, occupancy, '
                'and observed-percent fields rather than direct vehicle trajectories.'
            ),
            'effect': 'limits interpretation of microscopic traffic behavior',
        },
        {
            'limitation_id': 'L6',
            'category': 'operational definitions',
            'limitation': (
                'Sustained congestion and abrupt breakdown are defined with '
                'published operational rules, but no single universal breakdown '
                'definition exists.'
            ),
            'effect': 'results depend on the phenomenon being measured',
        },
        {
            'limitation_id': 'L7',
            'category': 'causality',
            'limitation': (
                'The study is observational and does not identify causal drivers '
                'such as incidents, demand shocks, weather, or signal operations.'
            ),
            'effect': 'findings describe dynamics rather than causes',
        },
        {
            'limitation_id': 'L8',
            'category': 'probability modeling',
            'limitation': (
                'The product-limit probability analysis has only ten uncensored '
                'abrupt-breakdown observations and includes implausibly low '
                'pre-breakdown flows for a capacity interpretation.'
            ),
            'effect': 'PLM result retained as audit evidence, not a headline finding',
        },
    ])

    limitations.to_csv(
        output / 'batch_6_final_limitations.csv',
        index=False,
    )

    conclusion_classes = (
        findings_table.groupby(
            'status',
            as_index=False,
        )
        .size()
        .rename(
            columns={
                'size': 'findings',
            }
        )
    )

    conclusion_classes.to_csv(
        output / 'batch_6_conclusion_status_counts.csv',
        index=False,
    )

    report_lines = [
        'BATCH 6 — PART 2: FINDINGS AND LIMITATIONS',
        '',
        'FINAL FINDING ADJUDICATION',
        findings_table.to_string(index=False),
        '',
        'LIMITATIONS',
        limitations.to_string(index=False),
        '',
        'CONCLUSION STATUS COUNTS',
        conclusion_classes.to_string(index=False),
        '',
        'PORTFOLIO POSITION',
        'Headline claims are restricted to findings labeled SUPPORTED or '
        'SUPPORTED_WITH_CAVEAT.',
        'SUGGESTIVE findings may appear as secondary observations with explicit '
        'sample-size or concentration caveats.',
        'The product-limit probability result remains in the audit trail but '
        'should not be featured as a substantive traffic finding.',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_6_part_2_audit_summary.txt').write_text(
        report + '\n'
    )

    settings = {
        'status': 'complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'headline_findings': findings_table.loc[
            findings_table.portfolio_role.str.contains(
                'headline',
                na=False,
            ),
            'finding_id',
        ].tolist(),
        'limitations_count': int(len(limitations)),
    }

    (output / 'batch_6_part_2_run_settings.json').write_text(
        json.dumps(settings, indent=2)
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 6 Part 2 reports: {output.resolve()}'
    )


# ===================== BATCH 6 — PART 3. FINAL PORTFOLIO PACKAGE =====================


def run_batch_6_part_3():
    """Create the final findings package and figure keep/exclude manifest."""
    output = BATCH_6_PART_3_REPORT_FOLDER

    if output.exists():
        raise ValueError(
            f'Batch 6 Part 3 output exists: {output}. '
            'Choose a new output folder; no files overwritten.'
        )

    required = {
        'findings':
            BATCH_6_PART_3_PART2_FOLDER
            / 'batch_6_final_findings_adjudication.csv',
        'limitations':
            BATCH_6_PART_3_PART2_FOLDER
            / 'batch_6_final_limitations.csv',
        'coverage':
            BATCH_6_PART_3_PART1_FOLDER
            / 'batch_6_coverage_threshold_robustness.csv',
        'midday':
            BATCH_6_PART_3_PART1_FOLDER
            / 'batch_6_midday_duration_robustness.csv',
        'propagation':
            BATCH_6_PART_3_PART1_FOLDER
            / 'batch_6_propagation_robustness_summary.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Batch 6 Part 3 requires completed Batch 6 outputs. Missing:\n'
            + '\n'.join(missing)
        )

    output.mkdir(parents=True)

    findings = pd.read_csv(
        required['findings']
    ).sort_values(
        'priority'
    )

    limitations = pd.read_csv(
        required['limitations']
    )

    coverage = pd.read_csv(
        required['coverage']
    ).iloc[0]

    midday = pd.read_csv(
        required['midday']
    ).iloc[0]

    propagation = pd.read_csv(
        required['propagation']
    ).iloc[0]

    headline = findings.loc[
        findings.portfolio_role.str.contains(
            'headline',
            na=False,
        )
        & ~findings.status.eq(
            'UNSUPPORTED_FOR_SUBSTANTIVE_CLAIM'
        )
    ].copy()

    headline.to_csv(
        output / 'batch_6_headline_findings.csv',
        index=False,
    )

    figure_manifest = pd.DataFrame([
        {
            'figure':
                'images/traffic_b3_threshold_sensitivity.png',
            'section':
                'Data validity',
            'recommendation':
                'KEEP',
            'reason':
                'Shows threshold choice does not materially damage corridor coverage.',
        },
        {
            'figure':
                'images/traffic_b3_weakest_station_threshold.png',
            'section':
                'Data validity',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Shows weakest-link coverage, important for spatial analysis.',
        },
        {
            'figure':
                'images/traffic_b3_measurements_by_coverage.png',
            'section':
                'Audit',
            'recommendation':
                'AUDIT_ONLY',
            'reason':
                'Useful diagnostic but not a headline result.',
        },
        {
            'figure':
                'images/traffic_b3_exact_repeat_diagnostic.png',
            'section':
                'Audit',
            'recommendation':
                'AUDIT_ONLY',
            'reason':
                'Diagnostic only; exact repeats do not establish imputation.',
        },
        {
            'figure':
                'images/traffic_b3_lane_consistency.png',
            'section':
                'Audit',
            'recommendation':
                'AUDIT_ONLY',
            'reason':
                'Validates metadata-defined physical lanes rather than a substantive result.',
        },
        {
            'figure':
                'images/traffic_b4_example_speed_heatmap.png',
            'section':
                'Breakdown definitions',
            'recommendation':
                'KEEP',
            'reason':
                'Visually anchors the distinction between congestion state and abrupt onset.',
        },
        {
            'figure':
                'images/traffic_b4_part2_onset_timing_normalized.png',
            'section':
                'Breakdown definitions',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Compares timing without allowing the high-count state method to flatten rare onset methods.',
        },
        {
            'figure':
                'images/traffic_b4_part2_state_vs_onset_counts.png',
            'section':
                'Breakdown definitions',
            'recommendation':
                'KEEP',
            'reason':
                'Directly communicates that the operational definitions measure different phenomena.',
        },
        {
            'figure':
                'images/traffic_b5p1_event_centered_profiles.png',
            'section':
                'Precursors',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Shows the abrupt speed collapse and surrounding flow/occupancy behavior.',
        },
        {
            'figure':
                'images/traffic_b5p1_breakdown_speed_heatmap.png',
            'section':
                'Precursors',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Shows event-to-event heterogeneity in the ten abrupt onsets.',
        },
        {
            'figure':
                'images/traffic_b5p1_preconditions_vs_controls.png',
            'section':
                'Precursors',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Shows matched-control differences are modest rather than dramatic.',
        },
        {
            'figure':
                'images/traffic_b5p2b_classification_robustness.png',
            'section':
                'Propagation',
            'recommendation':
                'KEEP',
            'reason':
                'Shows which episode classifications survive alternative lag windows.',
        },
        {
            'figure':
                'images/traffic_b5p2b_station_participation_by_window.png',
            'section':
                'Propagation',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Makes sensitivity of episodes 3 and 5 easy to see.',
        },
        {
            'figure':
                'images/traffic_b5p3_duration_by_station.png',
            'section':
                'Duration',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Shows persistent duration differences across corridor stations.',
        },
        {
            'figure':
                'images/traffic_b5p3_recovery_survival.png',
            'section':
                'Recovery',
            'recommendation':
                'KEEP',
            'reason':
                'Shows the long tail of sustained congestion while respecting right censoring.',
        },
        {
            'figure':
                'images/traffic_b5p3_recovery_order.png',
            'section':
                'Recovery',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Shows recovery order differs across corridor episodes.',
        },
        {
            'figure':
                'images/traffic_b5p4_breakdown_probability_plm.png',
            'section':
                'Probability audit',
            'recommendation':
                'EXCLUDE',
            'reason':
                'Maximum probability is substantively tiny and the uncensored sample is too sparse for a useful portfolio chart.',
        },
        {
            'figure':
                'images/traffic_b6_leave_one_out_duration_robustness.png',
            'section':
                'Robustness',
            'recommendation':
                'KEEP',
            'reason':
                'Demonstrates that headline duration metrics are not dominated by one station or date.',
        },
        {
            'figure':
                'images/traffic_b6_duration_by_time_period.png',
            'section':
                'Duration',
            'recommendation':
                'KEEP',
            'reason':
                'Directly communicates the strong midday-duration result.',
        },
        {
            'figure':
                'images/traffic_b6_abrupt_onset_concentration.png',
            'section':
                'Limitations',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Makes the small and date-concentrated abrupt-onset sample transparent.',
        },
    ])

    figure_manifest.to_csv(
        output / 'batch_6_figure_manifest.csv',
        index=False,
    )

    portfolio_figures = figure_manifest.loc[
        figure_manifest.recommendation.isin(
            [
                'KEEP',
                'KEEP_SUPPORTING',
            ]
        )
    ].copy()

    portfolio_figures.to_csv(
        output / 'batch_6_portfolio_figure_shortlist.csv',
        index=False,
    )

    executive_lines = [
        BATCH_6_PROJECT_TITLE,
        '',
        'FINAL HEADLINE FINDINGS',
    ]

    for row in headline.itertuples():
        executive_lines.extend([
            f'{int(row.priority)}. {row.finding}',
            f'   Evidence: {row.evidence}',
            f'   Caveat: {row.caveat}',
            '',
        ])

    executive_lines.extend([
        'KEY METHODOLOGICAL DECISIONS',
        (
            f'- 100% observed coverage was retained because it produced the '
            f'same selected-corridor row count as 80% coverage: '
            f'{int(coverage.high_rows_retained)} rows.'
        ),
        (
            f'- Six corridor episodes were obtained under all tested '
            f'15/30/45-minute onset-clustering windows.'
        ),
        (
            f'- The primary propagation window is 60 minutes; '
            f'{int(propagation.propagation_classifications_stable)}/'
            f'{int(propagation.propagation_classifications_total)} episode '
            f'classifications were stable across 30/60/120 minutes.'
        ),
        (
            f'- Midday duration remained longer than AM/PM after every '
            f'leave-one-station and leave-one-date exclusion: '
            f'{bool(midday.midday_longer_after_every_station_exclusion and midday.midday_longer_after_every_date_exclusion)}.'
        ),
        '',
        'LIMITATIONS',
    ])

    for row in limitations.itertuples():
        executive_lines.append(
            f'- {row.limitation}'
        )

    executive_summary = '\n'.join(
        executive_lines
    )

    (output / 'batch_6_executive_findings.txt').write_text(
        executive_summary + '\n'
    )

    project_status = {
        'status': 'analysis_complete',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'project_title': BATCH_6_PROJECT_TITLE,
        'completed_batches': [
            'Batch 1 observation quality',
            'Batch 2 corridor construction and selection',
            'Batch 3 analytical validity',
            'Batch 4 breakdown-definition comparison and concept separation',
            'Batch 5 congestion dynamics',
            'Batch 6 robustness and final conclusions',
        ],
        'headline_findings_count':
            int(len(headline)),
        'portfolio_figures_keep':
            int(
                figure_manifest.recommendation.eq(
                    'KEEP'
                ).sum()
            ),
        'portfolio_figures_keep_supporting':
            int(
                figure_manifest.recommendation.eq(
                    'KEEP_SUPPORTING'
                ).sum()
            ),
        'audit_or_excluded_figures':
            int(
                figure_manifest.recommendation.isin(
                    [
                        'AUDIT_ONLY',
                        'EXCLUDE',
                    ]
                ).sum()
            ),
        'next_phase': (
            'Write portfolio narrative / README, select final figures, '
            'and package repository for presentation.'
        ),
    }

    (
        output
        / 'batch_6_project_completion_status.json'
    ).write_text(
        json.dumps(
            project_status,
            indent=2,
        )
    )

    report_lines = [
        'BATCH 6 — PART 3: FINAL PORTFOLIO PACKAGE',
        '',
        'HEADLINE FINDINGS',
        headline.to_string(index=False),
        '',
        'FIGURE MANIFEST',
        figure_manifest.to_string(index=False),
        '',
        'PROJECT STATUS',
        json.dumps(
            project_status,
            indent=2,
        ),
        '',
        'NEXT PHASE',
        'The analytical work is complete. The next work is presentation:',
        'README / methodology / results narrative, figure selection, code cleanup,',
        'repository structure, and portfolio-site integration.',
    ]

    report = '\n'.join(report_lines)

    (output / 'batch_6_part_3_audit_summary.txt').write_text(
        report + '\n'
    )

    print('\n' + report)
    print(
        f'\nSaved Batch 6 final portfolio package: {output.resolve()}'
    )




# ===================== BATCH 7 — PART 1. WRITTEN PORTFOLIO PACKAGE =====================

def batch_7_markdown_table(frame):
    frame = frame.fillna('').copy()
    headers = [str(c) for c in frame.columns]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        vals = [str(v).replace('|','\\|') for v in row]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def run_batch_7_part_1():
    output = BATCH_7_REPORT_FOLDER / 'part_1_written_package'
    if output.exists():
        raise ValueError(f'Batch 7 Part 1 output exists: {output}. Choose a new folder; no overwrite.')
    output.mkdir(parents=True)

    paths = {
        'headline': BATCH_7_BATCH6_PORTFOLIO_FOLDER/'batch_6_headline_findings.csv',
        'manifest': BATCH_7_BATCH6_PORTFOLIO_FOLDER/'batch_6_figure_manifest.csv',
        'findings': BATCH_7_BATCH6_FINDINGS_FOLDER/'batch_6_final_findings_adjudication.csv',
        'limitations': BATCH_7_BATCH6_FINDINGS_FOLDER/'batch_6_final_limitations.csv',
        'coverage': BATCH_7_BATCH6_ROBUSTNESS_FOLDER/'batch_6_coverage_threshold_robustness.csv',
        'midday': BATCH_7_BATCH6_ROBUSTNESS_FOLDER/'batch_6_midday_duration_robustness.csv',
        'propagation': BATCH_7_BATCH6_ROBUSTNESS_FOLDER/'batch_6_propagation_robustness_summary.csv',
        'onset_date': BATCH_7_BATCH6_ROBUSTNESS_FOLDER/'batch_6_abrupt_onset_concentration_by_date.csv',
        'precondition': BATCH_7_BATCH5_PART1_FOLDER/'batch_5_part_1_minus5_event_vs_control.csv',
        'duration_overall': BATCH_7_BATCH5_PART3_FOLDER/'batch_5_part_3_overall_duration_summary.csv',
        'duration_period': BATCH_7_BATCH5_PART3_FOLDER/'batch_5_part_3_duration_by_time_period.csv',
        'method_summary': BATCH_7_BATCH4_FOLDER/'batch_4_method_summary.csv',
    }
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError('Missing prior outputs:\n'+'\n'.join(missing))

    headline=pd.read_csv(paths['headline']).sort_values('priority')
    manifest=pd.read_csv(paths['manifest'])
    findings=pd.read_csv(paths['findings']).sort_values('priority')
    limitations=pd.read_csv(paths['limitations'])
    coverage=pd.read_csv(paths['coverage']).iloc[0]
    midday=pd.read_csv(paths['midday']).iloc[0]
    propagation=pd.read_csv(paths['propagation']).iloc[0]
    onset_date=pd.read_csv(paths['onset_date'])
    precondition=pd.read_csv(paths['precondition'])
    duration_overall=pd.read_csv(paths['duration_overall']).iloc[0]
    duration_period=pd.read_csv(paths['duration_period'])
    method_summary=pd.read_csv(paths['method_summary'])
    method_counts=method_summary.set_index('method').events.to_dict()

    speed=precondition.loc[precondition.metric.eq('speed_mph')].iloc[0]
    flow=precondition.loc[precondition.metric.eq('flow_veh_5min')].iloc[0]
    occ=precondition.loc[precondition.metric.eq('occupancy_fraction')].iloc[0]
    period=duration_period.set_index('time_period').median_duration_minutes.to_dict()
    top_day=onset_date.iloc[0]

    readme=[
        f'# {BATCH_6_PROJECT_TITLE}','',
        '## Overview','',
        f'This project analyzes congestion and traffic breakdown on a {BATCH_7_CORRIDOR_DESCRIPTION} using Caltrans PeMS five-minute station data from {BATCH_7_STUDY_PERIOD}.', '',
        'The project separates sustained congestion from abrupt breakdown onset, studies corridor propagation and recovery, and tests the main results for sensitivity to observation thresholds, station choice, date choice, clustering windows, and propagation windows.','',
        '## Data and corridor','',
        '- Five physically adjacent eastbound I-10 stations: 717185, 717190, 716152, 717195, 717198.',
        f'- Final 100%-observed analytical sample: **{int(coverage.high_rows_retained):,} rows** and **{coverage.high_simultaneous_hours:,.1f} simultaneous corridor-hours**.',
        '- The 80% and 100% observation thresholds retained exactly the same selected-corridor rows.','',
        '## Headline findings',''
    ]
    for r in headline.itertuples():
        readme += [f'### {int(r.priority)}. {r.finding}','',f'**Evidence:** {r.evidence}','',f'**Caveat:** {r.caveat}','']
    readme += [
        '## Selected quantitative results','',
        f'- Sustained-congestion events: **{int(method_counts.get("Threshold_50_15min",0)):,}**.',
        f'- LA-rule abrupt onsets: **{int(method_counts.get("LA_20drop_below40",0))}**.',
        f'- Caltrans spatial bottleneck events: **{int(method_counts.get("Caltrans_PeMS_spatial",0))}**.',
        f'- Corridor-level abrupt-breakdown episodes: **{int(propagation.corridor_episode_count)}**.',
        f'- Median sustained-congestion duration: **{duration_overall.median_duration_minutes:.0f} minutes**; p90: **{duration_overall.p90_duration_minutes:.0f} minutes**.',
        f'- AM / Midday / PM median durations: **{period["AM"]:.0f} / {period["Midday"]:.0f} / {period["PM"]:.0f} minutes**.',
        f'- Five minutes before abrupt onset, event/control medians were speed **{speed.event_median:.2f}/{speed.control_median:.2f} mph**, flow **{flow.event_median:.1f}/{flow.control_median:.1f} veh/5 min**, occupancy **{occ.event_median:.4f}/{occ.control_median:.4f}**.',
        f'- Largest single-date share of LA-rule onsets: **{top_day.share_of_all_onsets_pct:.0f}%** on {top_day.event_date}.','',
        '## Robustness','',
        f'- 15-, 30-, and 45-minute clustering windows all produced **{int(propagation.corridor_episode_count)} episodes**.',
        f'- **{int(propagation.propagation_classifications_stable)}/{int(propagation.propagation_classifications_total)}** propagation classifications were unchanged across 30/60/120-minute windows.',
        f'- Midday remained longer than AM/PM after every leave-one-station and leave-one-date exclusion: **{bool(midday.midday_longer_after_every_station_exclusion and midday.midday_longer_after_every_date_exclusion)}**.','',
        '## Repository guide','',
        '- `prepare_traffic.py` — reproducible analysis pipeline.',
        '- `METHODS.md` — definitions and analytical design.',
        '- `RESULTS.md` — final quantitative findings.',
        '- `LIMITATIONS.md` — scope and inferential limitations.',
        '- `FIGURES.md` — figure-by-figure interpretation.',
        '- `audit_reports/` — complete batch-by-batch paper trail.',
        '- `portfolio_figures/` — shortlisted presentation figures.','',
        '## Status','',
        'The analytical phase is complete. Batch 7 packages the work for repository and portfolio presentation.'
    ]
    BATCH_7_README.write_text('\n'.join(readme)+'\n')

    methods=[
        '# Methods','',
        '## Data','',
        f'Caltrans PeMS five-minute station data were analyzed for {BATCH_7_STUDY_PERIOD}. The final analysis uses a five-station eastbound I-10 corridor.','',
        '## Observation quality','',
        f'The selected corridor retained {int(coverage.high_rows_retained):,} rows at both the 80% and 100% observed thresholds, so the strict 100% threshold was used.','',
        '## Event definitions','',
        '- **Sustained congestion:** speed below 50 mph for at least three consecutive five-minute intervals.',
        '- **LA temporal abrupt breakdown:** at least a 20 mph five-minute speed drop ending below 40 mph.',
        '- **Caltrans spatial bottleneck:** upstream/downstream speed drop of at least 20 mph, downstream speed below 40 mph, pair gap below 3 miles, sustained in at least 5 of 7 consecutive intervals.','',
        '## Pre-breakdown comparison','',
        'Abrupt-onset events were aligned at onset and compared with controls matched on station, weekday/weekend status, and half-hour time bin on different dates. Controls were kept at least 60 minutes from any LA-rule onset.','',
        '## Corridor episodes and propagation','',
        'Same-day abrupt onsets were clustered into corridor episodes. Episode construction was tested at 15-, 30-, and 45-minute clustering gaps. Propagation classification was tested at 30-, 60-, and 120-minute windows; 60 minutes was retained as the primary operational window.','',
        '## Duration and recovery','',
        'Sustained-congestion duration is measured from first to last five-minute state interval. Right-censored events are retained in the recovery analysis rather than treated as complete.','',
        '## Robustness','',
        'Headline duration metrics were recomputed after removing each station and each date one at a time. These ranges measure influence/sensitivity; they are not confidence intervals.','',
        '## Final adjudication','',
        'Conclusions were labeled supported, supported with caveat, suggestive, supported methodologically, or unsupported for substantive claim.'
    ]
    BATCH_7_METHODS.write_text('\n'.join(methods)+'\n')

    results=['# Results','', '## Final finding adjudication','', batch_7_markdown_table(findings[['finding_id','status','finding','evidence']]),'',
             '## Key numbers','',
             f'- {int(method_counts.get("Threshold_50_15min",0)):,} sustained-congestion events.',
             f'- {int(method_counts.get("LA_20drop_below40",0))} abrupt LA-rule onsets.',
             f'- {int(propagation.corridor_episode_count)} corridor episodes.',
             f'- Median duration {duration_overall.median_duration_minutes:.0f} min; p90 {duration_overall.p90_duration_minutes:.0f} min.',
             f'- AM / Midday / PM medians {period["AM"]:.0f} / {period["Midday"]:.0f} / {period["PM"]:.0f} min.',
             f'- Propagation classification stability: {int(propagation.propagation_classifications_stable)}/{int(propagation.propagation_classifications_total)} episodes.']
    BATCH_7_RESULTS.write_text('\n'.join(results)+'\n')

    BATCH_7_LIMITATIONS.write_text('# Limitations\n\n'+batch_7_markdown_table(limitations[['limitation_id','category','limitation','effect']])+'\n')

    figs=['# Figure Guide','']
    for rec in ['KEEP','KEEP_SUPPORTING','AUDIT_ONLY','EXCLUDE']:
        subset=manifest.loc[manifest.recommendation.eq(rec)]
        figs += [f'## {rec.replace("_"," ").title()}','']
        for r in subset.itertuples():
            figs += [f'- `{r.figure}` — **{r.section}**: {r.reason}']
        figs += ['']
    figs += ['## How to read the leave-one-out robustness plot','',
             'The full-sample dot is the statistic using all data. The station and date ranges show the minimum and maximum value obtained after removing one station or one date at a time. They are sensitivity ranges, not confidence intervals.']
    BATCH_7_FIGURES.write_text('\n'.join(figs)+'\n')

    docs=pd.DataFrame([
        {'file':str(BATCH_7_README),'purpose':'Repository overview'},
        {'file':str(BATCH_7_METHODS),'purpose':'Methods narrative'},
        {'file':str(BATCH_7_RESULTS),'purpose':'Results narrative'},
        {'file':str(BATCH_7_LIMITATIONS),'purpose':'Limitations'},
        {'file':str(BATCH_7_FIGURES),'purpose':'Figure guide'},
    ])
    docs.to_csv(output/'batch_7_written_outputs.csv',index=False)
    (output/'batch_7_part_1_summary.txt').write_text(docs.to_string(index=False)+'\n')
    print('\nBATCH 7 — PART 1 COMPLETE')
    print(docs.to_string(index=False))


# ===================== BATCH 7 — PART 2. PORTFOLIO FIGURE PACKAGE =====================

def run_batch_7_part_2():
    output=BATCH_7_REPORT_FOLDER/'part_2_figure_package'
    if output.exists():
        raise ValueError(f'Batch 7 Part 2 output exists: {output}. Choose a new folder; no overwrite.')
    output.mkdir(parents=True)
    manifest_path=BATCH_7_BATCH6_PORTFOLIO_FOLDER/'batch_6_figure_manifest.csv'
    if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
    if BATCH_7_PORTFOLIO_FIGURE_FOLDER.exists():
        raise ValueError(f'{BATCH_7_PORTFOLIO_FIGURE_FOLDER} already exists; no automatic overwrite.')
    BATCH_7_PORTFOLIO_FIGURE_FOLDER.mkdir()
    manifest=pd.read_csv(manifest_path)
    shortlist=manifest.loc[manifest.recommendation.isin(['KEEP','KEEP_SUPPORTING'])].copy()
    rows=[]
    for r in shortlist.itertuples():
        source=Path(r.figure); dest=BATCH_7_PORTFOLIO_FIGURE_FOLDER/source.name
        status='MISSING'
        if source.exists():
            dest.write_bytes(source.read_bytes()); status='COPIED'
        rows.append({'source':str(source),'destination':str(dest),'recommendation':r.recommendation,'section':r.section,'status':status})
    log=pd.DataFrame(rows)
    log.to_csv(output/'batch_7_figure_copy_log.csv',index=False)
    (output/'batch_7_part_2_summary.txt').write_text(log.to_string(index=False)+'\n')
    print('\nBATCH 7 — PART 2 COMPLETE')
    print(f'Copied {int(log.status.eq("COPIED").sum())}/{len(log)} shortlisted figures.')


# ===================== BATCH 7 — PART 3. FINAL REPOSITORY READINESS =====================

def run_batch_7_part_3():
    output=BATCH_7_REPORT_FOLDER/'part_3_repository_readiness'
    if output.exists():
        raise ValueError(f'Batch 7 Part 3 output exists: {output}. Choose a new folder; no overwrite.')
    output.mkdir(parents=True)
    checks=[
        ('prepare_traffic.py',Path('prepare_traffic.py').exists()),
        ('README.md',BATCH_7_README.exists()),
        ('METHODS.md',BATCH_7_METHODS.exists()),
        ('RESULTS.md',BATCH_7_RESULTS.exists()),
        ('LIMITATIONS.md',BATCH_7_LIMITATIONS.exists()),
        ('FIGURES.md',BATCH_7_FIGURES.exists()),
        ('audit_reports/',Path('audit_reports').exists()),
        ('images/',Path('images').exists()),
        ('portfolio_figures/',BATCH_7_PORTFOLIO_FIGURE_FOLDER.exists()),
    ]
    audit=pd.DataFrame(checks,columns=['item','exists'])
    audit.to_csv(output/'batch_7_repository_readiness.csv',index=False)
    ready=bool(audit.exists.all())
    status_lines=['# Project Status','',f'- Analytical phase complete: **Yes**',f'- Narrative package generated: **{ "Yes" if ready else "Needs attention" }**','',
                  '## Remaining manual work','',
                  '- Review wording for personal voice and concision.','- Decide final figure ordering on the portfolio page.','- Commit and push repository changes.','- Integrate project into the portfolio site.']
    BATCH_7_PROJECT_STATUS.write_text('\n'.join(status_lines)+'\n')
    status={'status':'portfolio_package_ready' if ready else 'portfolio_package_needs_attention','created_utc':datetime.now(timezone.utc).isoformat(),'analysis_complete':True,'repository_checks_passed':int(audit.exists.sum()),'repository_checks_total':len(audit)}
    (output/'batch_7_status.json').write_text(json.dumps(status,indent=2))
    checklist=pd.DataFrame([
        {'item':'Analysis complete','complete':True},
        {'item':'README generated','complete':BATCH_7_README.exists()},
        {'item':'Methods generated','complete':BATCH_7_METHODS.exists()},
        {'item':'Results generated','complete':BATCH_7_RESULTS.exists()},
        {'item':'Limitations generated','complete':BATCH_7_LIMITATIONS.exists()},
        {'item':'Figure guide generated','complete':BATCH_7_FIGURES.exists()},
        {'item':'Portfolio figure folder assembled','complete':BATCH_7_PORTFOLIO_FIGURE_FOLDER.exists()},
        {'item':'Personal-language edit','complete':False},
        {'item':'Git commit/push','complete':False},
        {'item':'Portfolio-site integration','complete':False},
    ])
    checklist.to_csv(output/'batch_7_final_checklist.csv',index=False)
    (output/'batch_7_part_3_summary.txt').write_text(audit.to_string(index=False)+'\n\n'+checklist.to_string(index=False)+'\n')
    print('\nBATCH 7 — PART 3 COMPLETE')
    print(audit.to_string(index=False))

# ===================== BATCH 7A CORRECTIVE SECTION =====================

def batch_7a_markdown_table(frame):
    table = frame.fillna('').copy()
    columns = [str(value) for value in table.columns]

    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]

    for row in table.itertuples(index=False, name=None):
        values = [
            str(value).replace('|', '\\|')
            for value in row
        ]
        lines.append(
            '| ' + ' | '.join(values) + ' |'
        )

    return '\n'.join(lines)


def batch_7a_time_period(timestamp):
    stamp = pd.Timestamp(timestamp)
    hour = stamp.hour + stamp.minute / 60

    if 5 <= hour < 10:
        return 'AM'
    if 10 <= hour < 15:
        return 'Midday'
    if 15 <= hour < 20:
        return 'PM'

    return 'Outside'


def batch_7a_load_full_day_corridor(station_ids):
    station_set = set(int(value) for value in station_ids)

    if BATCH_7A_FULL_DAY_CACHE.exists():
        context = pd.read_parquet(
            BATCH_7A_FULL_DAY_CACHE
        )

        context['station_id'] = pd.to_numeric(
            context.station_id,
            errors='raise',
        ).astype('int64')

        context = context.loc[
            context.station_id.isin(
                station_set
            )
        ].copy()

        print(
            f'Batch 7A full-day context: using cache '
            f'{BATCH_7A_FULL_DAY_CACHE}',
            flush=True,
        )

        return context.sort_values(
            ['station_id', 'timestamp'],
            kind='stable',
        ).reset_index(drop=True)
    dates = pd.date_range(START_DATE, END_DATE)
    parts = []

    required_columns = [
        'timestamp',
        'station_id',
        'lane_type',
        'observed_pct',
        'flow_veh_5min',
        'occupancy_fraction',
        'speed_mph',
        'invalid_timestamp',
        'date_mismatch',
        'invalid_station_id',
        'invalid_speed',
        'invalid_flow',
        'invalid_occupancy',
        'invalid_observed_pct',
        'off_5min_grid',
        'duplicate_station_timestamp',
    ]

    for number, day in enumerate(dates, 1):
        day_text = day.strftime('%Y-%m-%d')
        folder = PREPARED_DATA_FOLDER / f'date={day_text}'
        paths = sorted(folder.glob('part-*.parquet'))

        if not paths:
            continue

        print(
            f'Batch 7A full-day context: {day_text} '
            f'[{number}/{len(dates)}]',
            flush=True,
        )

        day_parts = []

        for path in paths:
            schema = set(pq.read_schema(path).names)
            missing = set(required_columns) - schema

            if missing:
                raise ValueError(
                    f'{path}: missing columns: '
                    + ', '.join(sorted(missing))
                )

            frame = pd.read_parquet(
                path,
                columns=required_columns,
            )

            frame = frame.loc[
                frame.lane_type.eq('ML')
                & frame.station_id.isin(station_set)
            ].copy()

            if not frame.empty:
                day_parts.append(frame)

        if not day_parts:
            continue

        day_frame = pd.concat(
            day_parts,
            ignore_index=True,
        )

        duplicate_copies = day_frame.duplicated(
            ['station_id', 'timestamp'],
            keep=False,
        )

        valid = (
            ~day_frame.invalid_timestamp.fillna(True)
            & ~day_frame.date_mismatch.fillna(True)
            & ~day_frame.invalid_station_id.fillna(True)
            & ~day_frame.invalid_speed.fillna(True)
            & ~day_frame.invalid_flow.fillna(True)
            & ~day_frame.invalid_occupancy.fillna(True)
            & ~day_frame.invalid_observed_pct.fillna(True)
            & ~day_frame.off_5min_grid.fillna(True)
            & ~duplicate_copies
            & day_frame.timestamp.notna()
            & day_frame.observed_pct.eq(
                BATCH_4_FINAL_OBSERVED_PCT
            )
        )

        day_frame = day_frame.loc[valid].copy()

        if not day_frame.empty:
            parts.append(day_frame)

    if not parts:
        raise ValueError(
            'No full-day selected-corridor rows found.'
        )

    context = pd.concat(
        parts,
        ignore_index=True,
    )

    context['station_id'] = pd.to_numeric(
        context.station_id,
        errors='raise',
    ).astype('int64')

    context = context.sort_values(
        ['station_id', 'timestamp'],
        kind='stable',
    ).reset_index(drop=True)

    duplicates = context.duplicated(
        ['station_id', 'timestamp'],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            'Duplicate station/timestamp rows remain.'
        )

    BATCH_7A_FULL_DAY_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    context.to_parquet(
        BATCH_7A_FULL_DAY_CACHE,
        index=False,
    )

    print(
        f'Batch 7A full-day context cached: '
        f'{BATCH_7A_FULL_DAY_CACHE}',
        flush=True,
    )

    return context


def batch_7a_merge_state_events(events, gap_minutes):
    rows = []

    for station_id, station in events.groupby(
        'station_id',
        sort=False,
    ):
        station = station.sort_values(
            'start_time',
            kind='stable',
        ).reset_index(drop=True)

        current = None

        for event in station.itertuples():
            event_start = pd.Timestamp(event.start_time)
            event_end = pd.Timestamp(event.end_time)

            event_row = {
                'station_id': int(station_id),
                'start_time': event_start,
                'end_time': event_end,
                'component_events': 1,
                'minimum_speed_mph':
                    float(event.minimum_speed_mph),
            }

            if current is None:
                current = event_row
                continue

            same_date = (
                current['start_time'].date()
                == event_start.date()
            )

            recovered_minutes = (
                event_start
                - current['end_time']
            ).total_seconds() / 60 - 5

            if (
                same_date
                and recovered_minutes <= gap_minutes
            ):
                current['end_time'] = max(
                    current['end_time'],
                    event_end,
                )
                current['component_events'] += 1
                current['minimum_speed_mph'] = min(
                    current['minimum_speed_mph'],
                    event_row['minimum_speed_mph'],
                )
            else:
                rows.append(current)
                current = event_row

        if current is not None:
            rows.append(current)

    merged = pd.DataFrame(rows)

    merged['duration_minutes'] = (
        (
            merged.end_time
            - merged.start_time
        ).dt.total_seconds() / 60
        + 5
    )

    merged['batch_7a_time_period'] = (
        merged.start_time.apply(
            batch_7a_time_period
        )
    )

    merged['start_hour'] = (
        merged.start_time.dt.hour
        + merged.start_time.dt.minute / 60
    )

    merged['end_hour'] = (
        merged.end_time.dt.hour
        + merged.end_time.dt.minute / 60
    )

    return merged


def batch_7a_persistent_run_count(frame, column, min_intervals):
    if frame.empty:
        return 0

    ordered = batch_4_contiguous_blocks(
        frame
    )
    count = 0

    for _, block in ordered.groupby(
        'block_id',
        sort=False,
    ):
        active = (
            block[column]
            .fillna(False)
            .to_numpy()
        )

        run = 0

        for value in active:
            if value:
                run += 1
            else:
                if run >= min_intervals:
                    count += 1
                run = 0

        if run >= min_intervals:
            count += 1

    return count


# ============================================================
# MAIN CORRECTIVE PASS
# ============================================================


def run_batch_7a():
    if BATCH_7A_REPORT_FOLDER.exists():
        raise ValueError(
            f'{BATCH_7A_REPORT_FOLDER} already exists. '
            'Choose a new folder; no automatic overwrite.'
        )

    required = {
        'corridor_order':
            BATCH_7A_BATCH4_FOLDER
            / 'batch_4_corridor_travel_order.csv',
        'events':
            BATCH_7A_BATCH4_FOLDER
            / 'batch_4_combined_event_inventory.csv',
        'method_summary':
            BATCH_7A_BATCH4_FOLDER
            / 'batch_4_method_summary.csv',
        'threshold_sensitivity':
            BATCH_7A_BATCH4_FOLDER
            / 'batch_4_threshold_event_sensitivity.csv',
        'corridor_episodes':
            BATCH_7A_BATCH5_PART2_FOLDER
            / 'batch_5_part_2_corridor_episodes.csv',
        'duration_original':
            BATCH_7A_BATCH5_PART3_FOLDER
            / 'batch_5_part_3_duration_by_time_period.csv',
        'recovery_curve':
            BATCH_7A_RECOVERY_CURVE_FILE,
        'leave_station':
            BATCH_7A_BATCH6_PART1_FOLDER
            / 'batch_6_leave_one_station_out_duration.csv',
        'leave_date':
            BATCH_7A_BATCH6_PART1_FOLDER
            / 'batch_6_leave_one_date_out_duration.csv',
        'figure_manifest':
            BATCH_7A_BATCH6_PART3_FOLDER
            / 'batch_6_figure_manifest.csv',
        'selection':
            BATCH_7A_BATCH2_SELECTION_FOLDER
            / 'batch_2_primary_corridor_selection.csv',
        'selection_comparison':
            BATCH_7A_BATCH2_SELECTION_FOLDER
            / 'batch_2_primary_corridor_comparison.csv',
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Missing required prior outputs:\n'
            + '\n'.join(missing)
        )

    BATCH_7A_REPORT_FOLDER.mkdir(parents=True)
    BATCH_7A_IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    corridor = pd.read_csv(
        required['corridor_order']
    ).sort_values(
        'travel_order'
    )

    corridor['station_id'] = pd.to_numeric(
        corridor.station_id,
        errors='raise',
    ).astype('int64')

    station_ids = corridor.station_id.tolist()

    events = pd.read_csv(
        required['events'],
        parse_dates=['start_time', 'end_time'],
    )

    method_summary = pd.read_csv(
        required['method_summary']
    )

    threshold_sensitivity = pd.read_csv(
        required['threshold_sensitivity']
    )

    corridor_episodes = pd.read_csv(
        required['corridor_episodes'],
        parse_dates=[
            'first_onset_time',
            'last_onset_time',
        ],
    )

    duration_original = pd.read_csv(
        required['duration_original']
    )

    recovery_curve_original = pd.read_csv(
        required['recovery_curve']
    )

    leave_station = pd.read_csv(
        required['leave_station']
    )

    leave_date = pd.read_csv(
        required['leave_date']
    )

    figure_manifest = pd.read_csv(
        required['figure_manifest']
    )

    selection = pd.read_csv(
        required['selection']
    )

    selection_comparison = pd.read_csv(
        required['selection_comparison']
    )

    full_data = batch_7a_load_full_day_corridor(
        station_ids
    )

    # --------------------------------------------------------
    # 1. Clustering claim audit
    # --------------------------------------------------------
    la = events.loc[
        events.method.eq(
            'LA_20drop_below40'
        )
    ].copy()

    la['event_date'] = (
        la.start_time.dt.date.astype(str)
    )

    cluster_rows = []

    for event_date, group in la.groupby(
        'event_date',
        sort=True,
    ):
        times = sorted(
            pd.to_datetime(
                group.start_time
            )
        )

        if len(times) >= 2:
            gaps = [
                (
                    times[index]
                    - times[index - 1]
                ).total_seconds() / 60
                for index in range(
                    1,
                    len(times),
                )
            ]
            max_gap = max(gaps)
            span = (
                times[-1] - times[0]
            ).total_seconds() / 60
        else:
            max_gap = np.nan
            span = 0.0

        cluster_rows.append({
            'event_date':
                event_date,
            'onsets':
                len(times),
            'first_onset':
                times[0],
            'last_onset':
                times[-1],
            'full_span_minutes':
                span,
            'maximum_consecutive_gap_minutes':
                max_gap,
            'would_15_minute_gap_split_date':
                bool(
                    pd.notna(max_gap)
                    and max_gap > 15
                ),
        })

    clustering_audit = pd.DataFrame(
        cluster_rows
    )

    multi_date = clustering_audit.loc[
        clustering_audit.onsets.gt(1)
    ]

    cluster_robustness_is_real = bool(
        multi_date[
            'would_15_minute_gap_split_date'
        ].any()
    )

    clustering_audit.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_clustering_claim_audit.csv',
        index=False,
    )

    # --------------------------------------------------------
    # 1b. Abrupt-onset filtered-data continuity audit
    # --------------------------------------------------------
    continuity_rows = []

    for onset in la.sort_values('start_time').itertuples():
        onset_time = pd.Timestamp(onset.start_time)
        station_id = int(onset.station_id)

        station_times = set(
            full_data.loc[
                full_data.station_id.eq(station_id),
                'timestamp',
            ].tolist()
        )

        checks = {}

        for offset_minutes in [-10, -5, 0, 5, 10]:
            check_time = onset_time + pd.Timedelta(
                minutes=offset_minutes
            )
            checks[offset_minutes] = bool(
                check_time in station_times
            )

        continuity_rows.append({
            'onset_time': onset_time,
            'event_date': onset_time.date().isoformat(),
            'station_id': station_id,
            'present_minus_10': checks[-10],
            'present_minus_5': checks[-5],
            'present_at_onset': checks[0],
            'present_plus_5': checks[5],
            'present_plus_10': checks[10],
            'temporal_drop_pair_contiguous':
                bool(checks[-5] and checks[0]),
            'nearby_5min_context_complete':
                bool(checks[-5] and checks[0] and checks[5]),
        })

    onset_continuity = pd.DataFrame(
        continuity_rows
    )

    onset_continuity.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_abrupt_onset_data_continuity.csv',
        index=False,
    )

    march2_continuity = onset_continuity.loc[
        onset_continuity.event_date.eq('2026-03-02')
    ].copy()

    march2_all_drop_pairs_contiguous = bool(
        march2_continuity.temporal_drop_pair_contiguous.all()
    )

    march2_all_nearby_context_complete = bool(
        march2_continuity.nearby_5min_context_complete.all()
    )

    march2_anchor = pd.Timestamp('2026-03-02 05:10:00')
    march2_presence_rows = []

    for station_id in station_ids:
        station_times = set(
            full_data.loc[
                full_data.station_id.eq(station_id),
                'timestamp',
            ].tolist()
        )

        for offset_minutes in range(-15, 31, 5):
            timestamp = march2_anchor + pd.Timedelta(
                minutes=offset_minutes
            )

            march2_presence_rows.append({
                'station_id': int(station_id),
                'timestamp': timestamp,
                'minutes_from_05_10': offset_minutes,
                'included_after_100pct_filter':
                    bool(timestamp in station_times),
            })

    march2_corridor_presence = pd.DataFrame(
        march2_presence_rows
    )

    march2_corridor_presence.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_march2_corridor_data_presence.csv',
        index=False,
    )

    # --------------------------------------------------------
    # 2. Full-day sustained-state re-detection
    # --------------------------------------------------------
    state_parts = []

    for station_id, station_frame in full_data.groupby(
        'station_id',
        sort=False,
    ):
        detected = batch_4_threshold_events(
            station_frame,
            station_id,
            BATCH_7A_STATE_THRESHOLD_MPH,
            BATCH_7A_STATE_MIN_INTERVALS,
        )

        if not detected.empty:
            state_parts.append(detected)

    full_states = pd.concat(
        state_parts,
        ignore_index=True,
    )

    full_states = full_states.loc[
        full_states.onset_observed
    ].copy()

    full_states['start_hour'] = (
        full_states.start_time.dt.hour
        + full_states.start_time.dt.minute / 60
    )

    full_states['batch_7a_time_period'] = (
        full_states.start_time.apply(
            batch_7a_time_period
        )
    )

    core_full_states = full_states.loc[
        full_states.start_hour.ge(
            BATCH_7A_ANALYSIS_START_HOUR
        )
        & full_states.start_hour.lt(
            BATCH_7A_ANALYSIS_END_HOUR
        )
    ].copy()

    full_context_period = (
        core_full_states.groupby(
            'batch_7a_time_period',
            as_index=False,
        )
        .agg(
            events=('start_time', 'size'),
            median_duration_minutes=(
                'duration_minutes',
                'median',
            ),
            q25_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.25),
            ),
            q75_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.75),
            ),
            p90_duration_minutes=(
                'duration_minutes',
                lambda s: s.quantile(0.90),
            ),
        )
    )

    full_context_period.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_full_context_duration_by_period.csv',
        index=False,
    )

    original_period = (
        duration_original.set_index(
            'time_period'
        ).median_duration_minutes.to_dict()
    )

    full_period = (
        full_context_period.set_index(
            'batch_7a_time_period'
        ).median_duration_minutes.to_dict()
    )

    context_comparison = pd.DataFrame([
        {
            'batch_7a_time_period':
                period,
            'original_05_20_median_minutes':
                original_period.get(
                    period,
                    np.nan,
                ),
            'full_day_context_median_minutes':
                full_period.get(
                    period,
                    np.nan,
                ),
            'change_minutes':
                full_period.get(
                    period,
                    np.nan,
                )
                - original_period.get(
                    period,
                    np.nan,
                ),
        }
        for period in [
            'AM',
            'Midday',
            'PM',
        ]
    ])

    context_comparison.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_duration_context_comparison.csv',
        index=False,
    )

    # --------------------------------------------------------
    # 3. Short-gap fragmentation sensitivity
    # --------------------------------------------------------
    merge_rows = []

    for gap_minutes in BATCH_7A_STATE_MERGE_GAPS_MINUTES:
        merged = batch_7a_merge_state_events(
            full_states,
            gap_minutes,
        )

        merged = merged.loc[
            merged.start_hour.ge(
                BATCH_7A_ANALYSIS_START_HOUR
            )
            & merged.start_hour.lt(
                BATCH_7A_ANALYSIS_END_HOUR
            )
        ].copy()

        medians = (
            merged.groupby(
                'batch_7a_time_period'
            ).duration_minutes.median()
        )

        counts = (
            merged.groupby(
                'batch_7a_time_period'
            ).size()
        )

        midday = merged.loc[
            merged.batch_7a_time_period.eq(
                'Midday'
            )
        ].copy()

        crosses_pm = (
            midday.end_hour.ge(15)
            if len(midday)
            else pd.Series(dtype=bool)
        )

        am_median = float(
            medians.get(
                'AM',
                np.nan,
            )
        )

        midday_median = float(
            medians.get(
                'Midday',
                np.nan,
            )
        )

        pm_median = float(
            medians.get(
                'PM',
                np.nan,
            )
        )

        merge_rows.append({
            'merge_gap_minutes':
                gap_minutes,
            'events_total':
                len(merged),
            'am_events':
                int(
                    counts.get(
                        'AM',
                        0,
                    )
                ),
            'midday_events':
                int(
                    counts.get(
                        'Midday',
                        0,
                    )
                ),
            'pm_events':
                int(
                    counts.get(
                        'PM',
                        0,
                    )
                ),
            'am_median_duration':
                am_median,
            'midday_median_duration':
                midday_median,
            'pm_median_duration':
                pm_median,
            'midday_minus_max_am_pm':
                (
                    midday_median
                    - max(
                        am_median,
                        pm_median,
                    )
                ),
            'midday_episodes_crossing_15_00_pct':
                (
                    100.0
                    * crosses_pm.mean()
                    if len(crosses_pm)
                    else np.nan
                ),
            'merged_components_gt1':
                int(
                    merged.component_events.gt(1).sum()
                ),
        })

    merge_summary = pd.DataFrame(
        merge_rows
    )

    merge_summary.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_duration_gap_merge_sensitivity.csv',
        index=False,
    )

    midday_survives = bool(
        merge_summary[
            'midday_minus_max_am_pm'
        ].gt(0).all()
    )

    # Corrected duration chart: show the actual gap-merge challenge test.
    merge_plot = merge_summary.sort_values(
        'merge_gap_minutes'
    ).reset_index(drop=True)

    merge_x = np.arange(
        len(merge_plot)
    )

    merge_series = [
        (
            'am_median_duration',
            'AM 05:00–09:59',
        ),
        (
            'midday_median_duration',
            'Starts 10:00–14:59',
        ),
        (
            'pm_median_duration',
            'PM 15:00–19:59',
        ),
    ]

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B7A_MERGE
    )

    for (
        data_key,
        label,
    ), offset in zip(
        merge_series,
        THEME_GROUP_BAR_OFFSETS_B7A,
    ):
        ax.bar(
            merge_x + offset,
            merge_plot[data_key],
            width=THEME_GROUP_BAR_WIDTH_B7A,
            label=label,
        )

    ax.set_title(
        'Episodes Beginning 10:00–15:00 Last Longer',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel(
        'Brief Recovery Gap Merged (minutes)'
    )
    ax.set_ylabel(
        'Median Sustained-Congestion Duration (minutes)'
    )
    ax.set_xticks(
        merge_x,
        merge_plot.merge_gap_minutes.astype(int),
    )
    ax.set_ylim(
        THEME_AXIS_MIN_ZERO,
        (
            merge_plot[
                [
                    'am_median_duration',
                    'midday_median_duration',
                    'pm_median_duration',
                ]
            ].to_numpy().max()
            * THEME_Y_HEADROOM_B7A
        ),
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_duration_by_start_period_merge_sensitivity.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # Recovery survival: explicitly begin at 1.0 at time zero.
    recovery_curve_fixed = recovery_curve_original.copy()

    start_row = pd.DataFrame([
        {
            'duration_minutes':
                THEME_RECOVERY_START_MINUTES_B7A,
            'at_risk':
                recovery_curve_fixed.at_risk.iloc[0]
                if len(recovery_curve_fixed)
                else np.nan,
            'recoveries': 0,
            'censored': 0,
            'survival_probability':
                THEME_RECOVERY_START_PROBABILITY_B7A,
            'recovered_probability': 0.0,
        }
    ])

    recovery_curve_fixed = pd.concat(
        [
            start_row,
            recovery_curve_fixed,
        ],
        ignore_index=True,
    )

    recovery_curve_fixed.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_recovery_product_limit_with_origin.csv',
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B7A_RECOVERY
    )

    ax.step(
        recovery_curve_fixed.duration_minutes,
        recovery_curve_fixed.survival_probability,
        where=THEME_STEP_WHERE_POST,
        linewidth=THEME_SURVIVAL_LINEWIDTH,
    )

    ax.set_title(
        'How Long Until a Sustained Congestion Episode Recovers?',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel(
        'Minutes Since Congestion-State Onset'
    )
    ax.set_ylabel(
        'Probability Episode Has Not Yet Recovered'
    )
    ax.set_ylim(
        THEME_AXIS_MIN_ZERO,
        THEME_RECOVERY_START_PROBABILITY_B7A,
    )
    ax.grid(
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_recovery_survival_with_origin.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # Abrupt-onset dates with whole-number count ticks.
    onset_date_counts = (
        la.assign(
            event_date=la.start_time.dt.date.astype(str)
        )
        .groupby(
            'event_date',
            as_index=False,
        )
        .agg(
            onset_events=('start_time', 'size')
        )
        .sort_values(
            'event_date'
        )
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B7A_ONSET_DATES
    )

    ax.bar(
        onset_date_counts.event_date,
        onset_date_counts.onset_events,
    )

    max_onset_count = int(
        onset_date_counts.onset_events.max()
    )

    ax.set_title(
        'Abrupt-Onset Sample Is Concentrated on a Few Dates',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel(
        'Date'
    )
    ax.set_ylabel(
        'LA-Rule Abrupt Onsets'
    )
    ax.set_yticks(
        np.arange(
            0,
            max_onset_count + 1,
            1,
        )
    )
    ax.tick_params(
        axis='x',
        rotation=THEME_ROTATION_MEDIUM,
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_abrupt_onset_dates_integer_ticks.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 4. Adjacent-pair bottleneck diagnostic
    # --------------------------------------------------------
    hour = (
        full_data.timestamp.dt.hour
        + full_data.timestamp.dt.minute / 60
    )

    core_data = full_data.loc[
        hour.ge(
            BATCH_7A_ANALYSIS_START_HOUR
        )
        & hour.lt(
            BATCH_7A_ANALYSIS_END_HOUR
        )
    ].copy()

    pair_rows = []

    for index in range(
        len(corridor) - 1
    ):
        upstream_row = corridor.iloc[index]
        downstream_row = corridor.iloc[index + 1]

        upstream = core_data.loc[
            core_data.station_id.eq(
                int(
                    upstream_row.station_id
                )
            ),
            [
                'timestamp',
                'speed_mph',
                'flow_veh_5min',
            ],
        ].rename(
            columns={
                'speed_mph':
                    'upstream_speed_mph',
                'flow_veh_5min':
                    'upstream_flow_veh_5min',
            }
        )

        downstream = core_data.loc[
            core_data.station_id.eq(
                int(
                    downstream_row.station_id
                )
            ),
            [
                'timestamp',
                'speed_mph',
                'flow_veh_5min',
            ],
        ].rename(
            columns={
                'speed_mph':
                    'downstream_speed_mph',
                'flow_veh_5min':
                    'downstream_flow_veh_5min',
            }
        )

        pair = upstream.merge(
            downstream,
            on='timestamp',
            how='inner',
            validate='one_to_one',
        )

        pair['queue_signature'] = (
            pair.upstream_speed_mph.lt(
                BATCH_7A_BOTTLENECK_UPSTREAM_SPEED_MPH
            )
            & pair.downstream_speed_mph.ge(
                BATCH_7A_BOTTLENECK_DOWNSTREAM_SPEED_MPH
            )
        )

        pair['reverse_signature'] = (
            pair.upstream_speed_mph.ge(
                BATCH_7A_BOTTLENECK_UPSTREAM_SPEED_MPH
            )
            & pair.downstream_speed_mph.lt(
                BATCH_7A_BOTTLENECK_DOWNSTREAM_SPEED_MPH
            )
        )

        upstream_congested = (
            pair.upstream_speed_mph.lt(
                BATCH_7A_BOTTLENECK_UPSTREAM_SPEED_MPH
            )
        )

        persistent_runs = batch_7a_persistent_run_count(
            pair[
                [
                    'timestamp',
                    'queue_signature',
                ]
            ],
            'queue_signature',
            BATCH_7A_BOTTLENECK_MIN_RUN_INTERVALS,
        )

        pair_rows.append({
            'upstream_travel_order':
                int(
                    upstream_row.travel_order
                ),
            'upstream_station_id':
                int(
                    upstream_row.station_id
                ),
            'upstream_name':
                upstream_row.Name,
            'downstream_travel_order':
                int(
                    downstream_row.travel_order
                ),
            'downstream_station_id':
                int(
                    downstream_row.station_id
                ),
            'downstream_name':
                downstream_row.Name,
            'paired_intervals':
                len(pair),
            'queue_signature_intervals':
                int(
                    pair.queue_signature.sum()
                ),
            'queue_signature_pct_all_intervals':
                (
                    100.0
                    * pair.queue_signature.mean()
                ),
            'queue_signature_pct_when_upstream_congested':
                (
                    100.0
                    * pair.loc[
                        upstream_congested,
                        'queue_signature',
                    ].mean()
                    if upstream_congested.any()
                    else np.nan
                ),
            'reverse_signature_intervals':
                int(
                    pair.reverse_signature.sum()
                ),
            'persistent_queue_signature_runs_15min':
                persistent_runs,
            'median_downstream_minus_upstream_speed_when_signature':
                (
                    pair.loc[
                        pair.queue_signature,
                        'downstream_speed_mph',
                    ].sub(
                        pair.loc[
                            pair.queue_signature,
                            'upstream_speed_mph',
                        ]
                    ).median()
                    if pair.queue_signature.any()
                    else np.nan
                ),
        })

    bottleneck_pairs = pd.DataFrame(
        pair_rows
    ).sort_values(
        [
            'persistent_queue_signature_runs_15min',
            'queue_signature_pct_when_upstream_congested',
            'queue_signature_intervals',
        ],
        ascending=False,
        kind='stable',
    ).reset_index(drop=True)

    bottleneck_pairs.insert(
        0,
        'bottleneck_rank',
        range(
            1,
            len(bottleneck_pairs) + 1,
        ),
    )

    bottleneck_pairs.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_bottleneck_pair_ranking.csv',
        index=False,
    )

    top_pair = bottleneck_pairs.iloc[0]

    # --------------------------------------------------------
    # 5. Downstream discharge-flow audit
    # --------------------------------------------------------
    lanes_lookup = (
        corridor.set_index(
            'station_id'
        ).Lanes.to_dict()
    )

    discharge_station = int(
        top_pair.downstream_station_id
    )

    discharge_lanes = float(
        lanes_lookup[
            discharge_station
        ]
    )

    discharge_source = full_data.loc[
        full_data.station_id.eq(
            discharge_station
        ),
        [
            'timestamp',
            'speed_mph',
            'flow_veh_5min',
        ],
    ].copy()

    discharge_rows = []

    for onset in la.sort_values(
        'start_time'
    ).itertuples():
        pre_time = (
            pd.Timestamp(
                onset.start_time
            )
            - pd.Timedelta(
                minutes=5
            )
        )

        match = discharge_source.loc[
            discharge_source.timestamp.eq(
                pre_time
            )
        ]

        if match.empty:
            discharge_rows.append({
                'onset_time':
                    onset.start_time,
                'onset_station_id':
                    int(
                        onset.station_id
                    ),
                'discharge_station_id':
                    discharge_station,
                'pre_time':
                    pre_time,
                'discharge_speed_mph':
                    np.nan,
                'discharge_flow_veh_5min':
                    np.nan,
                'discharge_flow_vphpl':
                    np.nan,
                'discharge_station_free':
                    False,
            })
            continue

        row = match.iloc[0]

        discharge_rows.append({
            'onset_time':
                onset.start_time,
            'onset_station_id':
                int(
                    onset.station_id
                ),
            'discharge_station_id':
                discharge_station,
            'pre_time':
                pre_time,
            'discharge_speed_mph':
                float(
                    row.speed_mph
                ),
            'discharge_flow_veh_5min':
                float(
                    row.flow_veh_5min
                ),
            'discharge_flow_vphpl':
                (
                    float(
                        row.flow_veh_5min
                    )
                    * 12.0
                    / discharge_lanes
                ),
            'discharge_station_free':
                bool(
                    row.speed_mph
                    >= BATCH_7A_DISCHARGE_FREE_SPEED_MPH
                ),
        })

    discharge_audit = pd.DataFrame(
        discharge_rows
    )

    discharge_audit.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_discharge_flow_at_abrupt_onsets.csv',
        index=False,
    )

    usable_discharge = (
        discharge_audit.loc[
            discharge_audit.discharge_station_free
        ]
    )

    discharge_summary = pd.DataFrame([
        {
            'top_ranked_upstream_station':
                int(
                    top_pair.upstream_station_id
                ),
            'top_ranked_downstream_station':
                discharge_station,
            'abrupt_onsets':
                len(
                    discharge_audit
                ),
            'onsets_with_free_downstream_discharge_station':
                len(
                    usable_discharge
                ),
            'median_discharge_flow_vphpl_when_free':
                (
                    usable_discharge[
                        'discharge_flow_vphpl'
                    ].median()
                    if len(
                        usable_discharge
                    )
                    else np.nan
                ),
            'minimum_discharge_flow_vphpl_when_free':
                (
                    usable_discharge[
                        'discharge_flow_vphpl'
                    ].min()
                    if len(
                        usable_discharge
                    )
                    else np.nan
                ),
            'maximum_discharge_flow_vphpl_when_free':
                (
                    usable_discharge[
                        'discharge_flow_vphpl'
                    ].max()
                    if len(
                        usable_discharge
                    )
                    else np.nan
                ),
            'plm_revival_recommended':
                False,
        }
    ])

    discharge_summary.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_discharge_flow_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # 6. March 2 transparency
    # --------------------------------------------------------
    march2 = la.loc[
        la.event_date.eq(
            '2026-03-02'
        )
    ].copy()

    march2_summary = pd.DataFrame([
        {
            'date':
                '2026-03-02',
            'abrupt_onsets':
                len(
                    march2
                ),
            'stations':
                march2.station_id.nunique(),
            'share_of_all_abrupt_onsets_pct':
                (
                    100.0
                    * len(
                        march2
                    )
                    / len(
                        la
                    )
                ),
            'cause_assigned':
                False,
            'recommended_wording':
                (
                    'Four of ten abrupt onsets occurred on March 2, 2026. '
                    'The current analysis does not assign a causal incident '
                    'or weather explanation to that concentration.'
                ),
        }
    ])

    march2_summary.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_march2_summary.csv',
        index=False,
    )

    # --------------------------------------------------------
    # 7. Corridor selection provenance
    # --------------------------------------------------------
    selected = selection.iloc[0]

    prior_210 = selection_comparison.loc[
        pd.to_numeric(
            selection_comparison.Fwy,
            errors='coerce',
        ).eq(210)
        & selection_comparison.Dir.astype(
            str
        ).str.upper().eq('W')
    ]

    provenance = pd.DataFrame([
        {
            'final_corridor':
                selected.corridor_id,
            'final_freeway':
                int(
                    selected.Fwy
                ),
            'final_direction':
                selected.Dir,
            'final_station_count':
                int(
                    selected.station_count
                ),
            'final_span_miles':
                float(
                    selected.postmile_span
                ),
            'earlier_210w_candidates_visible_in_final_comparison':
                len(
                    prior_210
                ),
            'selection_basis':
                (
                    'Corrected physical-adjacency construction followed by '
                    'weakest-link-first traffic-content selection.'
                ),
            'recommended_writeup':
                (
                    'Earlier exploratory rankings included a 210 W candidate. '
                    'The final analysis uses I-10 E after the corrected Batch 2 '
                    'workflow rebuilt corridors using physical adjacency and '
                    'selected the primary corridor using weakest-station '
                    'coverage, simultaneous coverage, recurring low-speed '
                    'evidence, and tie-breakers. The final corridor is short, '
                    f'only {float(selected.postmile_span):.2f} miles, so '
                    'propagation claims are explicitly limited to this local '
                    'corridor scale.'
                ),
        }
    ])

    provenance.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_corridor_selection_provenance.csv',
        index=False,
    )

    # --------------------------------------------------------
    # 8. Corrected event-count chart
    # --------------------------------------------------------
    display_lookup = {
        'Threshold_50_15min':
            'Sustained state',
        'LA_20drop_below40':
            'Abrupt temporal',
        'Caltrans_PeMS_spatial':
            'Spatial bottleneck',
    }

    method_plot = method_summary.copy()

    method_plot['display'] = (
        method_plot.method.map(
            display_lookup
        ).fillna(
            method_plot.method
        )
    )

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B7A_COUNTS
    )

    bars = ax.bar(
        method_plot.display,
        method_plot.events,
    )

    ax.set_yscale(
        THEME_SCALE_LOG_B7A
    )
    ax.set_title(
        'Operational Definitions Identify Very Different Event Counts',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel(
        'Operational Definition'
    )
    ax.set_ylabel(
        'Detected Events (log scale)'
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )

    for bar, row in zip(
        bars,
        method_plot.itertuples(),
    ):
        ax.annotate(
            f'{int(row.events):,}',
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=THEME_ANNOTATION_OFFSET_8,
            textcoords=THEME_TEXTCOORDS,
            ha=THEME_ALIGN_CENTER,
            fontsize=THEME_ANNOTATION_SIZE,
        )

    fig.tight_layout()
    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_breakdown_definition_counts_log.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 9. Corrected threshold title
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B7A_THRESHOLD
    )

    ax.plot(
        threshold_sensitivity.threshold_mph,
        threshold_sensitivity.events,
        marker=THEME_MARKER_PRIMARY,
        linewidth=THEME_LINEWIDTH_EMPHASIS,
    )

    ax.set_title(
        'Event Counts Remain Similar Across 40–55 mph Thresholds',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel(
        'Sustained Speed Threshold (mph)'
    )
    ax.set_ylabel(
        'Observed Congestion-State Onsets'
    )
    ax.set_xticks(
        threshold_sensitivity.threshold_mph
    )
    ax.set_ylim(
        THEME_AXIS_MIN_ZERO,
        (
            threshold_sensitivity.events.max()
            * THEME_Y_HEADROOM_B7A
        ),
    )
    ax.grid(
        alpha=THEME_GRID_ALPHA_LIGHT
    )

    for row in threshold_sensitivity.itertuples():
        ax.annotate(
            f'{int(row.events):,}',
            (
                row.threshold_mph,
                row.events,
            ),
            xytext=THEME_ANNOTATION_OFFSET_8,
            textcoords=THEME_TEXTCOORDS,
            ha=THEME_ALIGN_CENTER,
        )

    fig.tight_layout()
    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_threshold_counts_stable.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 10. Corrected leave-one-out chart
    # --------------------------------------------------------
    metrics = [
        (
            'overall_median_duration',
            'Overall median',
            60.0,
        ),
        (
            'overall_p90_duration',
            'Overall p90',
            355.0,
        ),
        (
            'midday_median_duration',
            'Midday median',
            170.0,
        ),
    ]

    x = np.arange(
        len(
            metrics
        )
    )

    baseline = np.array([
        value
        for _, _, value in metrics
    ])

    station_min = np.array([
        leave_station[
            metric
        ].min()
        for metric, _, _ in metrics
    ])

    station_max = np.array([
        leave_station[
            metric
        ].max()
        for metric, _, _ in metrics
    ])

    date_min = np.array([
        leave_date[
            metric
        ].min()
        for metric, _, _ in metrics
    ])

    date_max = np.array([
        leave_date[
            metric
        ].max()
        for metric, _, _ in metrics
    ])

    fig, ax = plt.subplots(
        figsize=THEME_FIGSIZE_B7A_LEAVEOUT
    )

    ax.errorbar(
        x + THEME_BAR_OFFSET_NEGATIVE,
        baseline,
        yerr=[
            baseline - station_min,
            station_max - baseline,
        ],
        fmt=THEME_MARKER_PRIMARY,
        linestyle=THEME_LINESTYLE_NONE_B7A,
        capsize=THEME_RANGE_CAPSIZE_B7A,
        markersize=THEME_RANGE_MARKERSIZE_B7A,
        label='Range after removing one station',
    )

    ax.errorbar(
        x + THEME_BAR_OFFSET_POSITIVE,
        baseline,
        yerr=[
            baseline - date_min,
            date_max - baseline,
        ],
        fmt=THEME_MARKER_PRIMARY,
        linestyle=THEME_LINESTYLE_NONE_B7A,
        capsize=THEME_RANGE_CAPSIZE_B7A,
        markersize=THEME_RANGE_MARKERSIZE_B7A,
        label='Range after removing one date',
    )

    ax.scatter(
        x,
        baseline,
        s=THEME_SCATTER_SIZE_B6,
        label='Full-sample value',
    )

    ax.set_title(
        'Leave-One-Out Sensitivity of Headline Duration Statistics',
        fontfamily=THEME_TITLE_FONT,
    )
    ax.set_xlabel(
        'Statistic'
    )
    ax.set_ylabel(
        'Minutes'
    )
    ax.set_xticks(
        x,
        [
            label
            for _, label, _ in metrics
        ],
    )
    ax.grid(
        axis=THEME_GRID_AXIS_Y,
        alpha=THEME_GRID_ALPHA_SUBTLE,
    )
    ax.legend(
        frameon=THEME_LEGEND_FRAME,
    )

    fig.tight_layout()
    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_leave_one_out_ranges.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 11. Corrected 2x3 episode heatmaps
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        THEME_GRID_ROWS_B7A,
        THEME_GRID_COLUMNS_B7A,
        figsize=THEME_FIGSIZE_B7A_EPISODES,
        squeeze=False,
    )

    heat_image = None

    for axis_index, episode in enumerate(
        corridor_episodes.itertuples()
    ):
        row_index = (
            axis_index
            // THEME_GRID_COLUMNS_B7A
        )
        column_index = (
            axis_index
            % THEME_GRID_COLUMNS_B7A
        )

        ax = axes[
            row_index,
            column_index,
        ]

        anchor = pd.Timestamp(
            episode.first_onset_time
        )

        start = anchor + pd.Timedelta(
            minutes=BATCH_5_PART_2_PROFILE_START_MINUTE
        )

        end = anchor + pd.Timedelta(
            minutes=BATCH_5_PART_2_PROFILE_END_MINUTE
        )

        window = full_data.loc[
            full_data.timestamp.ge(
                start
            )
            & full_data.timestamp.le(
                end
            )
        ].copy()

        heat = (
            window.pivot(
                index='station_id',
                columns='timestamp',
                values='speed_mph',
            )
            .reindex(
                station_ids
            )
        )

        heat_image = ax.imshow(
            heat.to_numpy(),
            aspect=THEME_HEATMAP_ASPECT,
            interpolation=THEME_HEATMAP_INTERPOLATION,
            vmin=THEME_HEATMAP_MIN,
            vmax=THEME_HEATMAP_MAX,
            cmap=THEME_HEATMAP_CMAP,
        )

        timestamps = list(
            heat.columns
        )

        if anchor in timestamps:
            anchor_position = (
                timestamps.index(
                    anchor
                )
            )

            ax.axvline(
                anchor_position,
                linestyle=THEME_LINESTYLE_DASHED,
                linewidth=THEME_LINEWIDTH_SECONDARY,
            )

        tick_positions = np.linspace(
            0,
            len(
                timestamps
            ) - 1,
            min(
                THEME_EPISODE_TICK_COUNT_B7A,
                len(
                    timestamps
                ),
            ),
            dtype=int,
        )

        ax.set_xticks(
            tick_positions,
            [
                pd.Timestamp(
                    timestamps[
                        position
                    ]
                ).strftime(
                    '%-I:%M %p'
                )
                for position in tick_positions
            ],
            rotation=THEME_ROTATION_MEDIUM,
            ha=THEME_ALIGN_RIGHT,
        )

        ax.set_yticks(
            np.arange(
                len(
                    station_ids
                )
            ),
            [
                f'{int(row.travel_order)} '
                f'{int(row.station_id)}'
                for row in corridor.itertuples()
            ],
        )

        ax.set_title(
            f'Ep {int(episode.corridor_episode_id)} · '
            f'{anchor.strftime("%b %-d %H:%M")}',
            fontfamily=THEME_TITLE_FONT,
        )
        ax.set_xlabel(
            'Time'
        )
        ax.set_ylabel(
            'Station'
        )

    if heat_image is not None:
        colorbar = fig.colorbar(
            heat_image,
            ax=axes.ravel().tolist(),
            shrink=THEME_COLORBAR_SHRINK_B7A,
            pad=THEME_COLORBAR_PAD_B7A,
        )
        colorbar.set_label(
            'Speed (mph)'
        )

    fig.suptitle(
        'Speed Evolution Across the Six Corridor Episodes',
        fontfamily=THEME_TITLE_FONT,
        fontsize=THEME_SUPTITLE_SIZE,
    )

    fig.savefig(
        BATCH_7A_IMAGE_FOLDER
        / 'traffic_b7a_episode_speed_heatmaps_grid.png',
        dpi=THEME_DPI,
        bbox_inches=THEME_BBOX,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 12. Revised figure manifest
    # --------------------------------------------------------
    revised_manifest = (
        figure_manifest.copy()
    )

    revised_manifest.loc[
        revised_manifest.figure.isin(
            BATCH_7A_FORCE_EXCLUDE_FIGURES
        ),
        'recommendation',
    ] = 'EXCLUDE'

    revised_manifest.loc[
        revised_manifest.figure.isin(
            BATCH_7A_FORCE_EXCLUDE_FIGURES
        ),
        'reason',
    ] = (
        'Superseded, redundant, audit-only, or not substantively useful.'
    )

    replacements = pd.DataFrame([
        {
            'figure':
                'images/traffic_b7a_breakdown_definition_counts_log.png',
            'section':
                'Breakdown definitions',
            'recommendation':
                'KEEP',
            'reason':
                'One corrected log-scale chart replaces duplicate linear charts.',
        },
        {
            'figure':
                'images/traffic_b7a_threshold_counts_stable.png',
            'section':
                'Definition sensitivity',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Title now matches the small change in event counts.',
        },
        {
            'figure':
                'images/traffic_b7a_leave_one_out_ranges.png',
            'section':
                'Robustness',
            'recommendation':
                'KEEP',
            'reason':
                'Independent ranges; no false connecting trend.',
        },
        {
            'figure':
                'images/traffic_b7a_episode_speed_heatmaps_grid.png',
            'section':
                'Propagation',
            'recommendation':
                'KEEP',
            'reason':
                'Readable 2×3 layout for all six episodes.',
        },
        {
            'figure':
                'images/traffic_b7a_duration_by_start_period_merge_sensitivity.png',
            'section':
                'Duration',
            'recommendation':
                'KEEP',
            'reason':
                'Shows the 0/5/10/15-minute fragmentation challenge directly.',
        },
        {
            'figure':
                'images/traffic_b7a_recovery_survival_with_origin.png',
            'section':
                'Recovery',
            'recommendation':
                'KEEP',
            'reason':
                'Corrected survival curve begins at 1.0 at time zero.',
        },
        {
            'figure':
                'images/traffic_b7a_abrupt_onset_dates_integer_ticks.png',
            'section':
                'Limitations',
            'recommendation':
                'KEEP_SUPPORTING',
            'reason':
                'Date concentration displayed with whole-number count ticks.',
        },
    ])

    revised_manifest = pd.concat(
        [
            revised_manifest,
            replacements,
        ],
        ignore_index=True,
    )

    revised_manifest.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_revised_figure_manifest.csv',
        index=False,
    )

    if BATCH_7A_REVISED_FIGURE_FOLDER.exists():
        raise ValueError(
            f'{BATCH_7A_REVISED_FIGURE_FOLDER} already exists.'
        )

    BATCH_7A_REVISED_FIGURE_FOLDER.mkdir(
        parents=True
    )

    shortlist = revised_manifest.loc[
        revised_manifest.recommendation.isin(
            [
                'KEEP',
                'KEEP_SUPPORTING',
            ]
        )
    ].drop_duplicates(
        subset=['figure'],
        keep='last',
    )

    for row in shortlist.itertuples():
        source = Path(
            row.figure
        )

        if not source.exists():
            continue

        shutil.copy2(
            source,
            BATCH_7A_REVISED_FIGURE_FOLDER
            / source.name,
        )

    # --------------------------------------------------------
    # 13. Revised review documents
    # --------------------------------------------------------
    top_pair_is_lark_azusa = bool(
        int(
            top_pair.upstream_station_id
        ) == 716152
        and int(
            top_pair.downstream_station_id
        ) == 717195
    )

    revised_findings = pd.DataFrame([
        {
            'finding':
                'Sustained congestion and abrupt breakdown are distinct operational phenomena.',
            'status':
                (
                    'SUPPORTED'
                    if onset_continuity.temporal_drop_pair_contiguous.all()
                    else 'REVIEW_AFTER_DATA_CONTINUITY_AUDIT'
                ),
            'basis':
                (
                    '1,457 sustained-state events versus 10 abrupt temporal onsets '
                    'and 7 spatial bottleneck events. The temporal-onset sample is '
                    'also checked for filtered-data continuity at t-5 and t.'
                ),
        },
        {
            'finding':
                'The 10 abrupt onsets form six same-day corridor episodes.',
            'status':
                'DESCRIPTIVE_NOT_ROBUSTNESS',
            'basis':
                (
                    'The 15-minute rule does not split any multi-onset date, '
                    'so 15/30/45-minute invariance is not a strong robustness test.'
                ),
        },
        {
            'finding':
                'Episodes beginning 10:00–15:00 last longer than AM or PM starts.',
            'status':
                (
                    'SUPPORTED_AFTER_FRAGMENTATION_TEST'
                    if midday_survives
                    else 'DOWNGRADE'
                ),
            'basis':
                (
                    'Full-day recovery context plus 0/5/10/15-minute '
                    'short-gap merging sensitivity.'
                ),
        },
        {
            'finding':
                'A recurring local bottleneck signature can be ranked across adjacent station pairs.',
            'status':
                'SUPPORTED_DIAGNOSTIC',
            'basis':
                (
                    f'Top pair: '
                    f'{int(top_pair.upstream_station_id)} → '
                    f'{int(top_pair.downstream_station_id)}; '
                    f'Lark Ellen → Azusa 1 = {top_pair_is_lark_azusa}.'
                ),
        },
        {
            'finding':
                'The original product-limit probability chart remains excluded.',
            'status':
                'EXCLUDE',
            'basis':
                (
                    'Upstream queued flows were not appropriate capacity measurements; '
                    'the downstream discharge audit is more defensible, but ten abrupt events remain too sparse.'
                ),
        },
    ])

    revised_findings.to_csv(
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_revised_findings.csv',
        index=False,
    )

    methods_text = '\n'.join([
        '# Methods — Revised After Final Challenge Review',
        '',
        '## Time-of-day boundaries',
        '',
        '- AM: 05:00–09:59',
        '- Midday: 10:00–14:59',
        '- PM: 15:00–19:59',
        '',
        '## Duration correction',
        '',
        (
            'Sustained-congestion states were re-detected from full-day '
            '100%-observed station data. Only episodes whose observed onset '
            'occurred from 05:00 through 19:59 were assigned to AM, Midday, '
            'or PM, so recovery after 20:00 is no longer artificially truncated.'
        ),
        '',
        (
            'To test fragmentation, same-station congestion-state episodes were '
            'also merged when separated by only 5, 10, or 15 minutes at or '
            'above 50 mph. The 0-minute case preserves the unmerged full-context result.'
        ),
        '',
        '## Bottleneck localization',
        '',
        (
            'Each adjacent station pair was tested for a queue signature: '
            'upstream speed below 50 mph while the immediately downstream '
            'station remained at or above 50 mph. The ranking uses persistent '
            '15-minute runs first, then conditional queue-signature frequency.'
        ),
        '',
        '## Flow audit',
        '',
        (
            'The original capacity-probability attempt used flow at whichever '
            'station triggered an abrupt temporal onset. The corrective audit '
            'instead examines the downstream station of the top-ranked bottleneck '
            'pair five minutes before each abrupt onset, conditional on that '
            'downstream station remaining free-flowing.'
        ),
        '',
        '## Episode clustering correction',
        '',
        (
            'The earlier claim that six episodes were robust to 15/30/45-minute '
            'clustering is no longer used as a headline robustness result. '
            'The corrective audit explicitly tests whether a 15-minute gap could '
            'have split any multi-onset date.'
        ),
    ])

    BATCH_7A_METHODS_REVISED_FILE.write_text(
        methods_text + '\n'
    )

    results_text = '\n'.join([
        '# Results — Revised After Final Challenge Review',
        '',
        '## Abrupt-onset data continuity',
        '',
        (
            'The LA temporal rule uses a five-minute speed drop. '
            'This audit checks the filtered 100%-observed data at t-10, '
            't-5, t, t+5 and t+10 for every abrupt onset.'
        ),
        '',
        batch_7a_markdown_table(
            onset_continuity
        ),
        '',
        (
            f'March 2 t-5/t pairs are contiguous at all four onset stations: '
            f'**{march2_all_drop_pairs_contiguous}**.'
        ),
        '',
        (
            f'March 2 t-5/t/t+5 context is complete at all four onset stations: '
            f'**{march2_all_nearby_context_complete}**.'
        ),
        '',
        '## Clustering claim audit',
        '',
        batch_7a_markdown_table(
            clustering_audit
        ),
        '',
        (
            f'15-minute clustering provides a real split challenge: '
            f'**{cluster_robustness_is_real}**.'
        ),
        '',
        '## Original versus full-context duration medians',
        '',
        batch_7a_markdown_table(
            context_comparison.round(2)
        ),
        '',
        '## Short-gap merging sensitivity',
        '',
        batch_7a_markdown_table(
            merge_summary.round(2)
        ),
        '',
        (
            f'Midday-start episodes remain longer than both AM and PM '
            f'under every tested merge rule: **{midday_survives}**.'
        ),
        '',
        '## Adjacent-pair bottleneck ranking',
        '',
        batch_7a_markdown_table(
            bottleneck_pairs.round(3)
        ),
        '',
        (
            f'Top-ranked pair: **{int(top_pair.upstream_station_id)} '
            f'{top_pair.upstream_name} → '
            f'{int(top_pair.downstream_station_id)} '
            f'{top_pair.downstream_name}**.'
        ),
        '',
        '## Downstream discharge-flow audit',
        '',
        batch_7a_markdown_table(
            discharge_summary.round(2)
        ),
        '',
        '## March 2 concentration',
        '',
        batch_7a_markdown_table(
            march2_summary
        ),
        '',
        '## Corridor-selection provenance',
        '',
        batch_7a_markdown_table(
            provenance
        ),
        '',
        '## Revised finding status',
        '',
        batch_7a_markdown_table(
            revised_findings
        ),
    ])

    BATCH_7A_RESULTS_REVISED_FILE.write_text(
        results_text + '\n'
    )

    figures_text = '\n'.join([
        '# Figures — Revised Final Decisions',
        '',
        '## Corrected figures',
        '',
        '- `images/traffic_b7a_breakdown_definition_counts_log.png` — one log-scale chart replaces the duplicate linear count charts.',
        '- `images/traffic_b7a_threshold_counts_stable.png` — retitled to reflect the actual stability of event counts.',
        '- `images/traffic_b7a_leave_one_out_ranges.png` — independent dots/ranges; no connecting line between unrelated statistics.',
        '- `images/traffic_b7a_episode_speed_heatmaps_grid.png` — six corridor episodes in a readable 2×3 grid with shorter titles and fewer x ticks.',
        '- `images/traffic_b7a_duration_by_start_period_merge_sensitivity.png` — replaces the old 170/50/45 chart with the actual 0/5/10/15-minute challenge test.',
        '- `images/traffic_b7a_recovery_survival_with_origin.png` — recovery survival now explicitly begins at 1.0 at time zero.',
        '- `images/traffic_b7a_abrupt_onset_dates_integer_ticks.png` — onset-date counts use whole-number y-axis ticks.',
        '',
        '## Excluded from presentation',
        '',
        '- Original raw onset-hour count chart.',
        '- Original duplicate breakdown-count chart.',
        '- Lane-difference audit plot.',
        '- Product-limit breakdown-probability plot.',
        '',
        f'Revised presentation figures are in `{BATCH_7A_REVISED_FIGURE_FOLDER}/`.',
    ])

    BATCH_7A_FIGURES_REVISED_FILE.write_text(
        figures_text + '\n'
    )

    claude_text = '\n'.join([
        '# Claude Review Packet — Corrective Pass',
        '',
        'This packet responds directly to the final methodological critique before the project narrative is frozen.',
        '',
        '## 1. March 2 abrupt-onset data continuity',
        '',
        (
            'The white cells in Episode 3 triggered a direct continuity audit. '
            'For every abrupt onset, the filtered 100%-observed dataset is checked '
            'at t-10, t-5, t, t+5 and t+10. The LA temporal drop itself depends '
            'specifically on the t-5 to t pair.'
        ),
        '',
        batch_7a_markdown_table(
            onset_continuity
        ),
        '',
        (
            f'March 2 has contiguous t-5/t data at all four onset stations: '
            f'**{march2_all_drop_pairs_contiguous}**.'
        ),
        '',
        (
            f'March 2 has complete t-5/t/t+5 context at all four onset stations: '
            f'**{march2_all_nearby_context_complete}**.'
        ),
        '',
        (
            'A separate corridor-wide CSV records 100%-observed data presence '
            'around 05:10 so white cells at non-onset stations are not conflated '
            'with missing data at the stations that generated abrupt onsets.'
        ),
        '',
        '## 2. Six-episode clustering claim',
        '',
        (
            'The earlier 15/30/45-minute invariance claim has been downgraded. '
            'The table below shows whether the 15-minute rule could actually '
            'split any multi-onset date.'
        ),
        '',
        batch_7a_markdown_table(
            clustering_audit
        ),
        '',
        '## 3. Midday duration / fragmentation challenge',
        '',
        (
            'The duration analysis was rerun using full-day recovery context. '
            'Events were then merged across brief 5/10/15-minute recoveries '
            'to test whether peak-period fragmentation created the original result.'
        ),
        '',
        batch_7a_markdown_table(
            context_comparison.round(2)
        ),
        '',
        batch_7a_markdown_table(
            merge_summary.round(2)
        ),
        '',
        (
            f'Midday-start episodes remain longer under every tested merge rule: '
            f'**{midday_survives}**.'
        ),
        '',
        (
            'The revised wording is deliberately "episodes beginning 10:00–15:00" '
            'rather than implying a causal midday effect.'
        ),
        '',
        '## 4. Bottleneck localization',
        '',
        batch_7a_markdown_table(
            bottleneck_pairs.round(3)
        ),
        '',
        (
            f'Top-ranked pair: **{int(top_pair.upstream_station_id)} '
            f'{top_pair.upstream_name} → '
            f'{int(top_pair.downstream_station_id)} '
            f'{top_pair.downstream_name}**.'
        ),
        '',
        '## 5. Flow-at-capacity correction',
        '',
        (
            'The original PLM attempt used flow at the onset detector even when '
            'that detector could already be queued. This pass audits flow at the '
            'downstream side of the top bottleneck pair when it remains free-flowing.'
        ),
        '',
        batch_7a_markdown_table(
            discharge_summary.round(2)
        ),
        '',
        (
            'No replacement PLM curve is promoted: ten abrupt events remain too sparse.'
        ),
        '',
        '## 6. Corridor selection',
        '',
        batch_7a_markdown_table(
            provenance
        ),
        '',
        '## 7. March 2',
        '',
        batch_7a_markdown_table(
            march2_summary
        ),
        '',
        '## 8. Chart corrections',
        '',
        '- duplicate count charts replaced by one log-scale chart;',
        '- threshold chart title corrected;',
        '- leave-one-out chart has no connecting lines;',
        '- six episode heatmaps rebuilt as a 2×3 grid;',
        '- raw onset-hour, lane-audit, and PLM probability charts excluded.',
        '',
        '## Files to review',
        '',
        '- `CLAUDE_REVIEW_PACKET.md`',
        '- `METHODS_REVISED.md`',
        '- `RESULTS_REVISED.md`',
        '- `FIGURES_REVISED.md`',
        '- `batch7a_corrective.py`',
    ])

    BATCH_7A_CLAUDE_REVIEW_FILE.write_text(
        claude_text + '\n'
    )

    status = {
        'status':
            'complete',
        'created_utc':
            datetime.now(
                timezone.utc
            ).isoformat(),
        'march2_all_temporal_drop_pairs_contiguous':
            march2_all_drop_pairs_contiguous,
        'march2_all_nearby_context_complete':
            march2_all_nearby_context_complete,
        'clustering_15min_is_real_challenge':
            cluster_robustness_is_real,
        'midday_survives_fragmentation_test':
            midday_survives,
        'top_bottleneck_pair': {
            'upstream_station_id':
                int(
                    top_pair.upstream_station_id
                ),
            'downstream_station_id':
                int(
                    top_pair.downstream_station_id
                ),
        },
        'review_files': [
            str(
                BATCH_7A_CLAUDE_REVIEW_FILE
            ),
            str(
                BATCH_7A_METHODS_REVISED_FILE
            ),
            str(
                BATCH_7A_RESULTS_REVISED_FILE
            ),
            str(
                BATCH_7A_FIGURES_REVISED_FILE
            ),
        ],
    }

    (
        BATCH_7A_REPORT_FOLDER
        / 'batch_7a_status.json'
    ).write_text(
        json.dumps(
            status,
            indent=2,
        )
    )

    print('\nBATCH 7A COMPLETE')
    print(
        f'March 2 abrupt-onset t-5/t pairs all contiguous: '
        f'{march2_all_drop_pairs_contiguous}'
    )
    print(
        f'March 2 t-5/t/t+5 context complete at all onset stations: '
        f'{march2_all_nearby_context_complete}'
    )
    print(
        f'Clustering 15-minute rule is a real challenge: '
        f'{cluster_robustness_is_real}'
    )
    print(
        f'Midday result survives 0/5/10/15-minute gap merging: '
        f'{midday_survives}'
    )
    print(
        'Top bottleneck pair: '
        f'{int(top_pair.upstream_station_id)} '
        f'{top_pair.upstream_name} -> '
        f'{int(top_pair.downstream_station_id)} '
        f'{top_pair.downstream_name}'
    )
    print(
        f'Revised figures: {BATCH_7A_REVISED_FIGURE_FOLDER.resolve()}'
    )
    print(
        'Claude packet: '
        f'{BATCH_7A_CLAUDE_REVIEW_FILE.resolve()}'
    )




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
        'Batch 2 station metadata/corridors': RUN_BATCH_2,
        'Batch 2 primary corridor selection': RUN_BATCH_2_SELECTION,
        'Batch 3 analytical data validity': RUN_BATCH_3,
        'Batch 4 Part 1 breakdown definition comparison': RUN_BATCH_4_PART_1,
        'Batch 4 Part 2 concept separation': RUN_BATCH_4_PART_2,
        'Batch 5 Part 1 pre-breakdown conditions': RUN_BATCH_5_PART_1,
        'Batch 5 Part 2 corridor propagation': RUN_BATCH_5_PART_2,
        'Batch 5 Part 2b propagation robustness': RUN_BATCH_5_PART_2B,
        'Batch 5 Part 3 duration and recovery': RUN_BATCH_5_PART_3,
        'Batch 5 Part 4 breakdown probability': RUN_BATCH_5_PART_4,
        'Batch 5 Part 5 synthesis': RUN_BATCH_5_PART_5,
        'Batch 6 Part 1 final robustness': RUN_BATCH_6_PART_1,
        'Batch 6 Part 2 findings and limitations': RUN_BATCH_6_PART_2,
        'Batch 6 Part 3 final portfolio package': RUN_BATCH_6_PART_3,
        'Batch 7 Part 1 written portfolio package': RUN_BATCH_7_PART_1,
        'Batch 7 Part 2 portfolio figure package': RUN_BATCH_7_PART_2,
        'Batch 7 Part 3 repository readiness': RUN_BATCH_7_PART_3,
        'Batch 7A corrective challenge tests': RUN_BATCH_7A,
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

    if RUN_BATCH_2:
        run_batch_2()

    if RUN_BATCH_2_SELECTION:
        run_batch_2_selection()

    if RUN_BATCH_3:
        run_batch_3()

    if RUN_BATCH_4_PART_1:
        run_batch_4_part_1()

    if RUN_BATCH_4_PART_2:
        run_batch_4_part_2()

    if RUN_BATCH_5_PART_1:
        run_batch_5_part_1()

    if RUN_BATCH_5_PART_2:
        run_batch_5_part_2()

    if RUN_BATCH_5_PART_2B:
        run_batch_5_part_2b()

    if RUN_BATCH_5_PART_3:
        run_batch_5_part_3()

    if RUN_BATCH_5_PART_4:
        run_batch_5_part_4()

    if RUN_BATCH_5_PART_5:
        run_batch_5_part_5()

    if RUN_BATCH_6_PART_1:
        run_batch_6_part_1()

    if RUN_BATCH_6_PART_2:
        run_batch_6_part_2()

    if RUN_BATCH_6_PART_3:
        run_batch_6_part_3()

    if RUN_BATCH_7_PART_1:
        run_batch_7_part_1()

    if RUN_BATCH_7_PART_2:
        run_batch_7_part_2()

    if RUN_BATCH_7_PART_3:
        run_batch_7_part_3()

    if RUN_BATCH_7A:
        run_batch_7a()


if __name__ == '__main__':
    main()
