"""Columns another writer adds are visible to a casacure reopen (issue #30).

A column added to an existing table by *another* writer — python-casacore
directly, or dask-ms 0.2.x over it — used to be invisible when the table
was reopened through the casacure wrapper in the same process: the
wrapper's process-wide per-directory write backing (``WriteData``) is
reused on reopen, and its staleness check only detected row growth, not
schema changes.  python-casacore read the data back fine; the casacure
view just didn't have the column (``KeyError: no such column``), which is
what DDFacet's ``TestPutVisColumn`` records against its casacure-backed
dask-ms write-back path.

The backing now also re-materialises when ``table.dat``'s column set
differs from the last-synced one, so the reopen adopts the foreign
schema.  A backing with *pending* writes is refused instead (flush and
close raise): flushing would rewrite ``table.dat`` from the stale
descriptor and silently drop the foreign column — refusing loses nothing
that is not already pending.

python-casacore runs in subprocesses without ``tests/shim`` on the path
(the shim stands in for it); tests needing it are skipped when it is not
installed.  The stale-view symptom requires the casacure opens and the
external write in one process, so the tests run casacure here and
python-casacore in a subprocess — a subprocess-only repro cannot see it
(a fresh process re-parses ``table.dat`` and was never affected).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import casacure.tables as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_casacore_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # drop tests/shim: real python-casacore
    return env


def _real_casacore_available():
    r = subprocess.run(
        [sys.executable, "-c", "import casacore.tables"],
        env=_real_casacore_env(),
        capture_output=True,
    )
    return r.returncode == 0


needs_casacore = pytest.mark.skipif(
    not _real_casacore_available(), reason="real python-casacore not installed"
)


def _real(code, *args):
    """Run `code` under real python-casacore; return its stdout."""
    r = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


EXTERNAL_ADDCOLS = r"""
import sys
import numpy as np
import casacore.tables as pt
path, colname = sys.argv[1], sys.argv[2]
d = pt.maketabdesc([pt.makearrcoldesc(colname, 0j, valuetype="complex",
                                      shape=[2])])
t = pt.table(path, readonly=False, ack=False)
t.addcols(d)
t.putcol(colname, np.ones((t.nrows(), 2), dtype=np.complex64))
t.flush()
t.close()
print(pt.table(path, ack=False).colnames())
"""


def _as_array(v):
    """Normalize a getcol/getcell result (dict form for string arrays,
    plain arrays otherwise) to a numpy array."""
    return np.asarray(v["array"] if isinstance(v, dict) else v)


@needs_casacore
def test_reopen_sees_column_added_by_python_casacore(tmp_path):
    """#30: create + open + close with casacure, addcols with
    python-casacore, reopen with casacure -> the column is there."""
    path = str(tmp_path / "x.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("DATA", 0.0, valuetype="float")]),
        nrow=4,
        ack=False,
    )
    t.putcol("DATA", np.arange(4, dtype=np.float32))
    t.close()
    # A first reopen in this process: registers the process-wide backing
    # that the stale view came from.
    assert ct.table(path, ack=False).colnames() == ["DATA"]

    out = _real(EXTERNAL_ADDCOLS, path, "PRED")
    assert "PRED" in out

    t2 = ct.table(path, ack=False)
    assert t2.colnames() == ["DATA", "PRED"]
    got = _as_array(t2.getcol("PRED"))
    assert got.shape == (4, 2)
    assert (got == 1).all()
    t2.close()


@needs_casacore
def test_reopen_sees_column_added_by_python_casacore_with_live_handle(tmp_path):
    """Same, with the first casacure handle still open (the dask-ms 0.2.x
    shape: the writer holds the MS while the other backend's column
    lands)."""
    path = str(tmp_path / "y.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("DATA", 0.0, valuetype="float")]),
        nrow=4,
        ack=False,
    )
    t.putcol("DATA", np.arange(4, dtype=np.float32))
    t.flush()
    _real(EXTERNAL_ADDCOLS, path, "PRED")
    t2 = ct.table(path, ack=False)
    assert t2.colnames() == ["DATA", "PRED"]
    # And the first handle sees it too after the shared backing adopts the
    # on-disk schema on the second open.
    t2.close()
    t.close()


@needs_casacore
def test_pending_writes_refuse_flush_over_foreign_column(tmp_path):
    """The write-over variant: unflushed casacure writes + an external
    addcols.  Flushing would rewrite table.dat from the stale descriptor
    and silently drop the foreign column, so flush (and close) raise, and
    the foreign column survives."""
    path = str(tmp_path / "z.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("DATA", 0.0, valuetype="float")]),
        nrow=2,
        ack=False,
    )
    t.putcol("DATA", np.array([1.0, 2.0], dtype=np.float32))
    t.flush()
    t.putcol("DATA", np.array([42.0, 2.0], dtype=np.float32))  # pending
    _real(EXTERNAL_ADDCOLS, path, "X")
    with pytest.raises(RuntimeError, match="columns changed on disk"):
        t.flush()
    assert "X" in _real(
        "import sys; import casacore.tables as pt; "
        "print(pt.table(sys.argv[1], ack=False).colnames())",
        path,
    )
    with pytest.raises(RuntimeError, match="columns changed on disk"):
        t.close()


@needs_casacore
def test_reopen_adopts_external_removecols(tmp_path):
    """The other schema direction: a column another writer removed is gone
    from the next casacure view too."""
    path = str(tmp_path / "r.tab")
    t = ct.table(
        path,
        ct.maketabdesc(
            [
                ct.makescacoldesc("D", 0.0, valuetype="float"),
                ct.makescacoldesc("E", 0.0, valuetype="float"),
            ]
        ),
        nrow=2,
        ack=False,
    )
    t.putcol("D", np.ones(2, dtype=np.float32))
    t.putcol("E", np.ones(2, dtype=np.float32))
    t.close()
    _real(
        "import sys; import casacore.tables as pt; "
        "t = pt.table(sys.argv[1], readonly=False, ack=False); "
        "t.removecols(['E']); t.close(); "
        "print(pt.table(sys.argv[1], ack=False).colnames())",
        path,
    )
    t2 = ct.table(path, ack=False)
    assert t2.colnames() == ["D"]
    t2.close()


def test_own_addcols_then_reopen_unaffected(tmp_path):
    """No false positive: casacure's own addcols (buffered, flushed) must
    still reopen through the shared backing with the columns intact."""
    path = str(tmp_path / "own.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("D", 0.0, valuetype="float")]),
        nrow=2,
        ack=False,
    )
    t.putcol("D", np.ones(2, dtype=np.float32))
    t.addcols(ct.makearrcoldesc("NEW", 0j, valuetype="complex", shape=[2]))
    t.putcol("NEW", np.ones((2, 2), dtype=np.complex64))
    t.flush()
    t2 = ct.table(path, ack=False)
    assert t2.colnames() == ["D", "NEW"]
    assert (_as_array(t2.getcol("NEW")) == 1).all()
    t.close()
    t2.close()


@needs_casacore
def test_flushed_merge_is_complete_for_casacore(tmp_path):
    """After casacure reopens over a foreign column and flushes its own
    pending write, python-casacore reads both columns' data intact."""
    path = str(tmp_path / "m.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("DATA", 0.0, valuetype="float")]),
        nrow=3,
        ack=False,
    )
    t.putcol("DATA", np.arange(3, dtype=np.float32))
    t.flush()
    _real(EXTERNAL_ADDCOLS, path, "PRED")
    t2 = ct.table(path, ack=False)
    assert t2.colnames() == ["DATA", "PRED"]
    t2.putcol("DATA", np.array([7.0, 7.0, 7.0], dtype=np.float32))
    t2.flush()
    t2.close()
    out = _real(
        "import sys; import numpy as np; import casacore.tables as pt; "
        "t = pt.table(sys.argv[1], ack=False); "
        "assert t.colnames() == ['DATA', 'PRED'], t.colnames(); "
        "assert t.getcol('DATA').tolist() == [7.0, 7.0, 7.0], t.getcol('DATA'); "
        "assert (t.getcol('PRED') == 1).all(), t.getcol('PRED'); "
        "print('OK')",
        path,
    )
    assert "OK" in out
