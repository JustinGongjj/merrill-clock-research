"""Local financial database for the two Merrill-clock repositories.

Install into the same virtual environment as ``merrill-clock-quant`` and
``merrill-clock-factor-regression``::

    pip install -e /path/to/clockdata
    clock-data init
    clock-data check
    clock-data sync --dataset all
    clock-data status
    clock-data run --output ./outputs/local-run

On this machine the whole stack also has a one-command launcher::

    clock            # offline backtest + attribution + open the HTML dashboard
    clock menu       # interactive menu
    clock update     # refresh all datasets

``clock-data run`` writes metrics.json, clock_backtest.csv,
factor_attribution.json and dashboard.html into the output directory. Pass
``--no-dashboard`` to skip the report, or run ``clock-data dashboard
--output ./outputs/local-run --open`` to rebuild it from saved results without
recomputing anything. The dashboard is one self-contained HTML file with
inline SVG charts; it needs no network, CDN or JavaScript runtime.

The database defaults to ``~/.local/share/clockdata/finance.db`` and can be
overridden with ``--db`` or the ``CLOCK_DATA_DB`` environment variable.

Registered sources
------------------
===============  ==========================================  ==================
source_id        data                                        transport
===============  ==========================================  ==================
fred             INDPRO / CPIAUCSL (latest revisions)        Python HTTP
yahoo            ETF adjusted-close proxies                  Python / yfinance
fama_french      FF5 monthly factors + RF (decimal returns)  Python HTTP
wind_mcp         Wind Alice EDB and quotes                   WorkBuddy session
local_csv        any wide CSV                                file import
===============  ==========================================  ==================

Wind Alice is available inside the WorkBuddy session, not as a Python SDK on
this machine: export the series to CSV and use ``clock-data import`` so the
rows receive the same source/coverage/fetch-time audit trail as live sources.

All observations are stored in one long-form table keyed by
``(dataset_id, obs_date, series_id)``; writes are upserts, so repeated syncs
are idempotent. ``clock-data run`` executes three steps entirely from the local
database, with no network calls:

1. Cross-asset Merrill-clock backtest (also writes ``run_manifest.json`` and
   ``phase_diagnostics.json``).
2. FF5 time-series attribution -> ``factor_attribution.json``.
3. Cross-sectional single-factor check -> ``cross_section_report.json``,
   ``factor_ic_history.csv`` and ``synthesis_report.json``. Use
   ``--no-crosssection`` to skip this step.

Steps 2 and 3 answer different questions and keep separate reports. The
pipeline also writes ``pipeline_manifest.json``, recording datasets, coverage,
cost, benchmark, tested-variant count, parameter confirmations and data
warnings in one machine-readable contract (``clockdata/manifest.py``). A
parameter left at its parser default is flagged as unconfirmed instead of
being silently accepted.

The dashboard reads all of the above and adds: signal-layer vs portfolio-layer
net value and cumulative cost drag, phase calibration and temperature
sensitivity, the cross-sectional factor table, and the run manifest. It stays
a single self-contained HTML file with inline SVG and no external assets.

Tests: ``python -m pytest -q`` (30 tests, including integration checks against
the real local database when it exists).
