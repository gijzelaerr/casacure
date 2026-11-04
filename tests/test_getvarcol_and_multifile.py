"""getvarcol key numbering (#31) and the MultiFile open error (#33).

* `getvarcol` numbered its `rN` keys from the requested window instead of
  from row 0 of the table: `getvarcol("V", 1, 2)` gave `r1`, `r2` where
  python-casacore gives `r2`, `r3`. The same keys feed the `{"rN": value}`
  dict-form `putcol`, so the round trip wrote the values to the wrong
  (shifted) rows.
* A MultiFile table (`table.mf` container, `table.storage.option:
  multifile` in casacore) opened with a bare missing-file os error about
  `table.f0`; the error now names the unsupported container.

The MultiFile fixture needs real python-casacore to create (an `.aipsrc`
in a private HOME, no user config touched); tests needing it are skipped
when it is not installed.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import casacure.tables as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_casacore_env(home):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    env["HOME"] = str(home)
    return env


def _real_casacore_available():
    r = subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        capture_output=True,
    )
    return r.returncode == 0


needs_casacore = pytest.mark.skipif(
    not _real_casacore_available(), reason="real python-casacore not installed"
)


def test_getvarcol_keys_numbered_from_row_zero(tmp_path):
    """#31: keys are absolute row numbers, like python-casacore."""
    path = str(tmp_path / "v.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("D", 0.0, valuetype="float")]),
        nrow=4,
        ack=False,
    )
    t.putcol("D", np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    got = t.getvarcol("D", 1, 2)
    assert list(got.keys()) == ["r2", "r3"], got
    t.close()
    # The dict-form putcol input is a different convention from the keys
    # getvarcol returns: dask-ms builds its {"r%d" % i} keys 0-based, so
    # "r2" here means row 2 (daskms/conftest.py). Verify that convention
    # is preserved (a shifted reading would land dask-ms varcol writes on
    # the wrong rows).
    t = ct.table(path, ack=False, readonly=False)
    t.putcol("D", {"r2": 99.0})
    t.flush()
    t.close()
    assert np.asarray(ct.table(path, ack=False).getcol("D")).tolist() == [
        10.0,
        20.0,
        99.0,
        40.0,
    ]


def test_getvarcol_default_window_keys_all_rows(tmp_path):
    """The default (startrow=0, whole column) keeps the r1..rN numbering."""
    path = str(tmp_path / "v2.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("D", 0.0, valuetype="float")]),
        nrow=3,
        ack=False,
    )
    t.putcol("D", np.ones(3, dtype=np.float32))
    assert list(t.getvarcol("D").keys()) == ["r1", "r2", "r3"]
    t.close()


@needs_casacore
def test_multifile_open_error_names_the_container(tmp_path):
    """#33: a MultiFile table open names the container instead of a bare
    `table.f0: No such file or directory` os error."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".aipsrc").write_text("table.storage.option: multifile\n")
    path = tmp_path / "mf.tab"
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import casacore.tables as pt, numpy as np; "
            "t = pt.table(sys.argv[1], pt.maketabdesc("
            "[pt.makescacoldesc('D', 0.0, valuetype='float')]), nrow=2, "
            "ack=False); "
            "t.putcol('D', np.ones(2, dtype='f')); t.close()",
            str(path),
        ],
        env=_real_casacore_env(home),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (path / "table.mf").is_file(), sorted(p.name for p in path.iterdir())
    with pytest.raises(RuntimeError, match="MultiFile"):
        ct.table(str(path), ack=False)
