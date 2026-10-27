"""Shared pytest configuration for the casacure test-suite.

`tests/fixtures/` holds casacore-generated ground truth — a CASA image cube,
the FITS cube it came from, CASA tables and a `manifest.json` describing
them.  It is produced by `tests/make_fixtures.py`, which needs real
python-casacore, and is gitignored, so a bare checkout (or the free-threaded
CI job, which installs only numpy/pytest/maturin) has no fixtures at all.

A fixture-dependent module declares what it needs at module level:

    CASACORE_FIXTURES = ("image.image", "image.fits")

and every test in it is skipped with one clear reason when any of those
paths is missing, instead of failing with the misleading
"no such image (not a CASA table directory or a FITS file)".  Modules
without the declaration are untouched, so this is opt-in and costs a
`getattr` per collected item.

The suite is hermetic as well: nothing in it reaches the network.
`tests/test_measures.py` holds casacure to astropy's `AltAz` over
1926-2126, and for epochs beyond its bundled IERS predictive range astropy
refreshes its IERS table online (`iers.conf.auto_max_age = 30`,
`auto_download = True`).  When that fetch stalls the suite wedges inside
`ssl.read` at zero CPU with no timeout — observed at 35 minutes in
`test_azel_matches_astropy_over_a_century` — so `pytest_configure` below
disables the refresh.  `auto_max_age = None` maps to `float.max`, which
gates both `_refresh_table_as_needed` *and* the staleness error that would
otherwise be raised instead, leaving the reference at the installed
`astropy-iers-data` release: exactly what MEASURES_ACCURACY.md documents
the contract against (worst 0.026″ over the century grid).  A 20 s socket
timeout makes any other stray fetch fail fast rather than hang.
"""

import socket
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    """Keep the run hermetic — see the module docstring.  astropy is a dev
    dependency (the free-threaded CI job installs only numpy/pytest/
    maturin), so its absence must not be an error here.
    """
    socket.setdefaulttimeout(20.0)
    try:
        from astropy.utils import iers
    except ImportError:
        return
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None


def pytest_collection_modifyitems(config, items):
    for item in items:
        module = getattr(item, "module", None)
        needed = getattr(module, "CASACORE_FIXTURES", ())
        if not needed:
            continue
        missing = [name for name in needed if not (FIXTURES_DIR / name).exists()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason="casacore fixtures not generated: "
                    + ", ".join(missing)
                    + " (run `python tests/make_fixtures.py` with python-casacore)"
                )
            )
