"""`putcell` accepts arrays and coerces values like `putcol` (issue #29).

On 3.8.36 `putcell` with an array raised for float, int64, short, complex
and dcomplex columns while `putcol` on the same column worked — the
binding routed `putcell` through the keyword-value converter, which
cannot convert numeric or string ndarrays (the "string cannot be stored"
message for an array input was that converter reporting a stringified
ndarray).  python-casacore 3.8.1 accepts every case; scalars are cast to
the column's type too (a Python ``42.0`` into a ``float`` column).
"""

import numpy as np
import pytest

import casacure.tables as ct

# valueType -> (numpy dtype, one cell's expected values after the round
# trip). Every column is variable-shape ndim=1 like the issue's repro.
MATRIX = [
    ("float", np.float32, [1.0, 1.0]),
    ("int64", np.int64, [1, 1]),
    ("short", np.int16, [1, 1]),
    ("double", np.float64, [1.0, 1.0]),
    ("int", np.int32, [1, 1]),
    ("boolean", np.bool_, [True, True]),
    ("complex", np.complex64, [1.0 + 0j, 1.0 + 0j]),
    ("dcomplex", np.complex128, [1.0 + 0j, 1.0 + 0j]),
]


@pytest.mark.parametrize("valuetype,npdtype,want", MATRIX)
def test_putcell_array_of_column_dtype(valuetype, npdtype, want, tmp_path):
    path = str(tmp_path / f"{valuetype}.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makearrcoldesc("V", 0, ndim=1,
                                          valuetype=valuetype)]),
        nrow=2,
        ack=False,
    )
    t.putcol("V", np.ones((2, 2), dtype=npdtype))  # worked before, unchanged
    t.putcell("V", 0, np.ones(2, dtype=npdtype))   # raised on 3.8.36
    t.putcell("V", 1, [1, 0])                      # list coerced like putcol
    t.flush()
    t.close()
    got = ct.table(path, ack=False).getcell("V", 0)
    vals = got["array"] if isinstance(got, dict) else np.asarray(got)
    assert np.asarray(vals).tolist() == want


def test_putcell_numpy_string_array(tmp_path):
    """A numpy '<U' array cell — also went through the keyword converter."""
    path = str(tmp_path / "str.tab")
    t = ct.table(
        path,
        ct.maketabdesc(
            [ct.makearrcoldesc("S", "", shape=[2], options=5,
                               valuetype="string")]
        ),
        nrow=2,
        ack=False,
    )
    t.putcell("S", 0, np.array(["a", "bb"]))
    t.putcell("S", 1, ["ccc", "dddd"])
    t.flush()
    t.close()
    r = ct.table(path, ack=False)
    for row, want in ((0, ["a", "bb"]), (1, ["ccc", "dddd"])):
        got = r.getcell("S", row)
        vals = got["array"] if isinstance(got, dict) else np.asarray(got)
        assert np.asarray(vals).tolist() == want
    r.close()


def test_putcell_scalar_cast_to_column_type(tmp_path):
    """putcell(42.0) into a float column stores (cast), as casacore does —
    it used to raise "double cannot be stored in column F (valueType
    float)"."""
    path = str(tmp_path / "scalar.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makescacoldesc("F", 0.0, valuetype="float")]),
        nrow=2,
        ack=False,
    )
    t.putcell("F", 0, 42.0)
    t.putcell("F", 1, 1)  # int -> float
    t.flush()
    t.close()
    got = ct.table(path, ack=False).getcol("F")
    assert np.asarray(got).tolist() == [42.0, 1.0]


def test_putcell_wrong_fixed_shape_still_raises(tmp_path):
    """The fix routes through putcol's validation: a fixed-shape column
    still rejects a cell of the wrong shape, with the same message."""
    path = str(tmp_path / "fixed.tab")
    t = ct.table(
        path,
        ct.maketabdesc([ct.makearrcoldesc("V", 0.0, shape=[3],
                                          valuetype="float")]),
        nrow=2,
        ack=False,
    )
    with pytest.raises(ValueError, match="does not match column V"):
        t.putcell("V", 0, np.ones(2, dtype=np.float32))
    t.close()
