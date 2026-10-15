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


import compare_maps  # noqa: E402


class MetricsTests(unittest.TestCase):
    def base(self):
        rng = np.random.default_rng(1)
        return 5.0 + rng.random((50, 40))

    def test_identical_maps_give_zero(self):
        b = self.base()
        m = compare_maps.map_metrics(b.copy(), b)
        for key in ("mean", "p999", "p9999", "max", "offmask_max", "sum_ratio"):
            self.assertEqual(m[key], 0.0, key)
        self.assertTrue(m["finite"] and m["nonneg"])
        self.assertEqual(m["n_mask"], b.size)

    def test_planted_difference(self):
        b = self.base()
        a = b.copy()
        a[3, 4] *= 1.001          # +1e-3 relative on one pixel
        m = compare_maps.map_metrics(a, b)
        self.assertAlmostEqual(m["max"], 1e-3, places=9)
        self.assertLess(m["p999"], 1e-3)       # one pixel of 2000 is below the 99.9th percentile
        self.assertGreater(m["p9999"], 0.0)
        self.assertAlmostEqual(m["mean"], 1e-3 / b.size, places=12)
        self.assertAlmostEqual(m["sum_ratio"], 1e-3 * b[3, 4] / b.sum(), places=12)

    def test_floor_excludes_low_pixels(self):
        b = self.base()
        b[0, 0] = 1e-6            # far below 1e-3 x mean
        a = b.copy()
        a[0, 0] = 2e-6            # 100 % relative, but off the mask
        m = compare_maps.map_metrics(a, b)
        self.assertEqual(m["n_mask"], b.size - 1)
        self.assertEqual(m["max"], 0.0)
        self.assertAlmostEqual(m["offmask_max"], 1e-6 / b.mean(), places=12)

    def test_nan_and_negative_fail(self):
        b = self.base()
        a = b.copy()
        a[1, 1] = np.nan
        m = compare_maps.map_metrics(a, b)
        self.assertFalse(m["finite"])
        self.assertFalse(compare_maps.evaluate_double(m, compare_maps.default_tolerances())["pass"])
        a = b.copy()
        a[1, 1] = -1.0
        m = compare_maps.map_metrics(a, b)
        self.assertTrue(m["finite"])
        self.assertFalse(m["nonneg"])
        self.assertFalse(compare_maps.evaluate_double(m, compare_maps.default_tolerances())["pass"])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            compare_maps.map_metrics(np.ones((3, 4)), np.ones((4, 3)))
        self.assertIn("(3, 4)", str(ctx.exception))
        self.assertIn("(4, 3)", str(ctx.exception))

    def test_parity_consistency(self):
        total = self.base()
        minima = 0.3 * total
        saddles = 0.7 * total
        self.assertLess(compare_maps.parity_consistency(total, minima, saddles), 1e-12)
        saddles[2, 2] += 1.0
        self.assertGreater(compare_maps.parity_consistency(total, minima, saddles), 0.1)


class EvaluateTests(unittest.TestCase):
    def metrics(self, **over):
        m = {"mean": 1e-9, "p999": 1e-7, "p9999": 1e-7, "max": 1e-6, "offmask_max": 1e-8,
             "sum_ratio": 1e-8, "n_mask": 10, "n_total": 10, "finite": True, "nonneg": True}
        m.update(over)
        return m

    def test_double_pass_and_fail(self):
        tol = compare_maps.default_tolerances()
        self.assertTrue(compare_maps.evaluate_double(self.metrics(), tol)["pass"])
        v = compare_maps.evaluate_double(self.metrics(p999=2e-5), tol)
        self.assertFalse(v["pass"])
        self.assertTrue(any("p999" in f for f in v["failures"]))
        v = compare_maps.evaluate_double(self.metrics(sum_ratio=-2e-6), tol)
        self.assertFalse(v["pass"])

    def test_float_rule_uses_ratio_to_reference_error(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        ok = compare_maps.evaluate_float({"mean": 1.2e-5, "p999": 1.4e-3, "p9999": 4e-3, "max": 1.5e-2},
                                         ref, self.metrics(), tol)
        self.assertTrue(ok["pass"])
        self.assertEqual(ok["flags"], [])
        bad = compare_maps.evaluate_float({"mean": 1.2e-5, "p999": 1.6e-3, "p9999": 4e-3, "max": 5e-2},
                                          ref, self.metrics(), tol)
        self.assertFalse(bad["pass"])
        self.assertTrue(any("p999" in f for f in bad["failures"]))
        self.assertTrue(any("max" in f for f in bad["flags"]))
        bad_sum = compare_maps.evaluate_float({"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2},
                                              ref, self.metrics(sum_ratio=2e-5), tol)
        self.assertFalse(bad_sum["pass"])


class ToleranceTests(unittest.TestCase):
    def test_load_merges_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tolerances.json"
            p.write_text(json.dumps({"double": {"p999": 2e-5}, "float_truth_error": {"nb_f": {"pixels": {"mean": 1}}}}))
            tol = compare_maps.load_tolerances(p)
            self.assertEqual(tol["double"]["p999"], 2e-5)
            self.assertEqual(tol["double"]["mean"], compare_maps.default_tolerances()["double"]["mean"])
            self.assertEqual(tol["float_truth_error"]["nb_f"]["pixels"]["mean"], 1)
            self.assertEqual(tol["floor_frac"], 1e-3)

    def test_missing_file_gives_defaults(self):
        self.assertEqual(compare_maps.load_tolerances("/nonexistent/t.json"), compare_maps.default_tolerances())


if __name__ == "__main__":
    unittest.main()
