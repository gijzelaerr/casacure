"""The ``table.info`` marker (issue #23): tables get one, and the Python
``table`` grows ``info()``/``putinfo()``.

casacore writes a ``table.info`` into every table directory it creates — a
plain table gets ``Type = \\nSubType = \\n\\n``, a Measurement Set's main
table ``Type = Measurement Set`` plus the MS readme — and consumers branch
on it: python-casacore reading a casacure-written MS used to report
``{'type': '', ...}`` where casacore's own output says
``Measurement Set``.  Images already carried the file; tables now do too,
through the shared serialisation (``crates/casacure/tests/table_info.rs``
pins the byte-level contract in Rust).

Here, through the Python binding and against real python-casacore:

* a created table has ``table.info`` with casacore's exact bytes, and
  ``info()`` reads it back as the empty-string default;
* ``putinfo`` merges (fields absent from the dict keep their value),
  stores the readme as whole lines, and rejects unknown fields like
  python-casacore; ``addreadmeline`` appends to the readme;
* a read-only handle refuses ``putinfo``;
* ``default_ms`` marks the main table ``Measurement Set``, subtables stay
  plain;
* the files are byte-identical to python-casacore's, and each backend
  reads what the other wrote.

Real python-casacore runs in subprocesses without ``tests/shim`` on the
path (the shim stands in for it); tests needing it are skipped when it is
not installed.
"""

import os
import subprocess
import sys

import pytest

import casacure.tables as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The casacore-written golden plain-table marker tracked in this repo.
GOLDEN_PLAIN_BYTES = open(os.path.join(ROOT, "PARENT.tab", "table.info"), "rb").read()
GOLDEN_PLAIN = b"Type = \nSubType = \n\n"

MS_TYPE = "Measurement Set"
MS_README = "This is a MeasurementSet Table holding measurements from a Telescope\n"


def desc():
    return {"VALUE": {"valueType": "double", "dataManagerType": "StandardStMan",
                      "dataManagerGroup": "SSM", "option": 0}}


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
        [sys.executable, "-c", code, *map(str, args)],
        env=_real_casacore_env(),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_create_writes_the_casacore_plain_marker(tmp_path):
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=3, ack=False)
    t.putcol("VALUE", [1.0, 2.0, 3.0])
    t.close()
    assert sorted(os.listdir(path)) == ["table.dat", "table.f0", "table.info", "table.lock"]
    assert open(os.path.join(path, "table.info"), "rb").read() == GOLDEN_PLAIN
    assert GOLDEN_PLAIN_BYTES == GOLDEN_PLAIN

    t = ct.table(path, readonly=True, ack=False)
    assert t.info() == {"type": "", "subType": "", "readme": ""}
    t.close()


def test_putinfo_round_trips_and_normalises_the_readme(tmp_path):
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    t.putinfo({"type": "Measurement Set", "subType": "", "readme": MS_README})
    assert t.info() == {"type": "Measurement Set", "subType": "", "readme": MS_README}
    t.close()
    assert open(os.path.join(path, "table.info"), "rb").read() == (
        f"Type = Measurement Set\nSubType = \n\n{MS_README}".encode()
    )

    # A readme missing its newline is stored as whole lines.
    t = ct.table(path, readonly=False, ack=False)
    t.putinfo({"type": "T", "subType": "S", "readme": "hello"})
    assert t.info() == {"type": "T", "subType": "S", "readme": "hello\n"}
    t.close()
    assert open(os.path.join(path, "table.info"), "rb").read() == (
        b"Type = T\nSubType = S\n\nhello\n"
    )


def test_putinfo_merges_absent_fields(tmp_path):
    # Fields absent from the dict keep their value (python-casacore
    # semantics; `putinfo({})` changes nothing).
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    t.putinfo({"type": "T1", "subType": "S1", "readme": "R1\n"})
    t.putinfo({"type": "T2"})
    assert t.info() == {"type": "T2", "subType": "S1", "readme": "R1\n"}
    t.putinfo({})
    assert t.info() == {"type": "T2", "subType": "S1", "readme": "R1\n"}
    t.close()


def test_putinfo_rejects_unknown_fields_and_bad_shapes(tmp_path):
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    with pytest.raises(RuntimeError, match="Name of field Type must be type, subType or readme"):
        t.putinfo({"Type": "T3"})
    with pytest.raises(TypeError):
        t.putinfo("not a dict")
    with pytest.raises(RuntimeError, match="field type must be a string"):
        t.putinfo({"type": 1})
    with pytest.raises(TypeError):
        t.addreadmeline(None)
    t.close()


def test_putinfo_and_addreadmeline_on_readonly_raise(tmp_path):
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    t.putinfo({"readme": "keep\n"})
    t.close()

    t = ct.table(path, readonly=True, ack=False)
    with pytest.raises(ValueError, match="table is not writable"):
        t.putinfo({"type": "X"})
    with pytest.raises(ValueError, match="table is not writable"):
        t.addreadmeline("no")
    assert t.info() == {"type": "", "subType": "", "readme": "keep\n"}
    t.close()


def test_addreadmeline_appends_whole_lines(tmp_path):
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    t.addreadmeline("line one")
    t.addreadmeline("line two")
    assert t.info()["readme"] == "line one\nline two\n"
    t.close()
    assert open(os.path.join(path, "table.info"), "rb").read() == (
        b"Type = \nSubType = \n\nline one\nline two\n"
    )


def test_info_without_the_file_reads_empty(tmp_path):
    # A table written before casacure created the file (or whose file was
    # deleted) reads as empty strings, like python-casacore, and the file
    # is not recreated by opening or closing.
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    t.close()
    os.remove(os.path.join(path, "table.info"))
    t = ct.table(path, readonly=True, ack=False)
    assert t.info() == {"type": "", "subType": "", "readme": ""}
    t.close()
    t = ct.table(path, readonly=False, ack=False)
    t.close()
    assert not os.path.exists(os.path.join(path, "table.info"))


def test_default_ms_marks_the_main_table(tmp_path):
    msp = str(tmp_path / "test.ms")
    ct.default_ms(msp)
    t = ct.table(msp, readonly=True, ack=False)
    assert t.info() == {"type": MS_TYPE, "subType": "", "readme": MS_README}
    t.close()
    a = ct.table(os.path.join(msp, "ANTENNA"), readonly=True, ack=False)
    assert a.info() == {"type": "", "subType": "", "readme": ""}
    a.close()


@needs_casacore
def test_real_casacore_reads_what_casacure_writes(tmp_path):
    # The measured effect from the issue: an MS written through casacure
    # reports its type to python-casacore.
    msp = str(tmp_path / "test.ms")
    ct.default_ms(msp)
    out = _real(
        "import sys; from casacore.tables import table\n"
        "print(table(sys.argv[1], readonly=True, ack=False).info()['type'])\n",
        msp,
    )
    assert out.strip() == MS_TYPE

    # ... and a plain table's putinfo round-trips through the other
    # backend, byte-identically.
    path = str(tmp_path / "x.tab")
    t = ct.table(path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=1, ack=False)
    t.putinfo({"type": "Shared", "subType": "Sub", "readme": "both read this"})
    t.close()
    assert open(os.path.join(path, "table.info"), "rb").read() == (
        b"Type = Shared\nSubType = Sub\n\nboth read this\n"
    )
    out = _real(
        "import sys; from casacore.tables import table\n"
        "print(repr(table(sys.argv[1], readonly=True, ack=False).info()))\n",
        path,
    )
    import ast

    assert ast.literal_eval(out.strip()) == {
        "type": "Shared",
        "subType": "Sub",
        "readme": "both read this\n",
    }


@needs_casacore
def test_table_info_bytes_are_identical_across_backends(tmp_path):
    # A directory copied between backends must diff empty on table.info.
    code = (
        "import sys; from casacore.tables import table, makescacoldesc, maketabdesc\n"
        "t = table(sys.argv[1], maketabdesc([makescacoldesc('VALUE', 0.0)]),\n"
        "          nrow=2, ack=False)\n"
        "t.close()\n"
        "print('DONE')\n"
    )
    cc_path, cu_path = str(tmp_path / "cc.tab"), str(tmp_path / "cu.tab")
    assert _real(code, cc_path).strip() == "DONE"
    t = ct.table(cu_path, ct.maketabdesc(ct.makescacoldesc("VALUE", 0.0)), nrow=2, ack=False)
    t.close()
    cc = open(os.path.join(cc_path, "table.info"), "rb").read()
    cu = open(os.path.join(cu_path, "table.info"), "rb").read()
    assert cc == cu == GOLDEN_PLAIN

    # And python-casacore's putinfo lands in the same bytes casacure's does.
    code = (
        "import sys; from casacore.tables import table\n"
        "t = table(sys.argv[1], readonly=False, ack=False)\n"
        "t.putinfo({'type': 'Shared', 'subType': 'Sub', 'readme': 'both read this'})\n"
        "t.close()\n"
        "print('DONE')\n"
    )
    assert _real(code, cc_path).strip() == "DONE"
    t = ct.table(cu_path, readonly=False, ack=False)
    t.putinfo({"type": "Shared", "subType": "Sub", "readme": "both read this"})
    t.close()
    assert (
        open(os.path.join(cc_path, "table.info"), "rb").read()
        == open(os.path.join(cu_path, "table.info"), "rb").read()
    )
