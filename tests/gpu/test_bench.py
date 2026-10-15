"""Unit tests for the IPM GPU benchmark harness. No GPU needed:
python -m unittest discover -s tests/gpu -v"""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_lib  # noqa: E402


def write_map(path, arr):
    arr = np.asarray(arr)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", arr.shape[0], arr.shape[1]))
        f.write(arr.tobytes())


class ReadMapTests(unittest.TestCase):
    def test_float32_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.bin"
            arr = np.arange(12, dtype=np.float32).reshape(3, 4)
            write_map(p, arr)
            m = bench_lib.read_map(p)
            self.assertEqual(m.dtype, np.float64)
            self.assertEqual(m.shape, (3, 4))
            np.testing.assert_array_equal(m, arr.astype(np.float64))

    def test_float64_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.bin"
            arr = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
            write_map(p, arr)
            np.testing.assert_array_equal(bench_lib.read_map(p), arr)

    def test_size_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.bin"
            with open(p, "wb") as f:
                f.write(struct.pack("ii", 3, 4))
                f.write(b"\0" * 10)
            with self.assertRaises(ValueError):
                bench_lib.read_map(p)


class ParamInfoTests(unittest.TestCase):
    def test_parses_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ipm_parameter_info.txt"
            p.write_text("kappa_tot 0.4\nmass_function equal\nt_shoot_cells 8.123\n")
            info = bench_lib.read_param_info(p)
            self.assertEqual(info["kappa_tot"], "0.4")
            self.assertEqual(info["mass_function"], "equal")
            self.assertEqual(bench_lib.t_shoot_cells(info), 8.123)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError) as ctx:
            bench_lib.t_shoot_cells({"kappa_tot": "0.4"})
        self.assertIn("t_shoot_cells", str(ctx.exception))


class RungNameTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(bench_lib.rung_label("R0p"), "R0′")
        self.assertEqual(bench_lib.rung_label("R3"), "R3")
        self.assertEqual(bench_lib.RUNGS[0], "R0")
        self.assertEqual(bench_lib.RUNGS[-1], "R5")


if __name__ == "__main__":
    unittest.main()
