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


class EvaluateRuleTests(unittest.TestCase):
    def metrics(self, **over):
        m = {"mean": 1e-9, "p999": 1e-7, "p9999": 1e-7, "max": 1e-6, "offmask_max": 1e-8,
             "sum_ratio": 1e-8, "n_mask": 10, "n_total": 10, "finite": True, "nonneg": True}
        m.update(over)
        return m

    def test_double_threshold_for_each_key(self):
        tol = compare_maps.default_tolerances()
        for key in ("mean", "p999", "p9999", "offmask_max"):
            thr = tol["double"][key]
            with self.subTest(key=key):
                above = compare_maps.evaluate_double(self.metrics(**{key: 1.01 * thr}), tol)
                self.assertFalse(above["pass"])
                self.assertTrue(any(f.startswith(key + "=") for f in above["failures"]), above["failures"])
                below = compare_maps.evaluate_double(self.metrics(**{key: 0.99 * thr}), tol)
                self.assertTrue(below["pass"], below["failures"])

    def test_float_common_failures_gate_the_rule(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        v = compare_maps.evaluate_float(ref, ref, self.metrics(finite=False), tol)
        self.assertFalse(v["pass"])
        self.assertIn("non-finite pixels", v["failures"])
        v = compare_maps.evaluate_float(ref, ref, self.metrics(nonneg=False), tol)
        self.assertFalse(v["pass"])
        self.assertIn("negative pixels", v["failures"])

    def test_float_max_excess_is_a_flag_not_a_failure(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        m_truth = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 4e-2}  # max above 3.0x R0's, rest within 1.5x
        v = compare_maps.evaluate_float(m_truth, ref, self.metrics(), tol)
        self.assertIs(v["pass"], True)
        self.assertEqual(v["failures"], [])
        self.assertTrue(v["flags"])
        self.assertTrue(any(f.startswith("max ") for f in v["flags"]), v["flags"])

    def test_float_mean_limit_alone_fails(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        m_truth = {"mean": 1.6e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}  # mean above 1.5x, rest within
        v = compare_maps.evaluate_float(m_truth, ref, self.metrics(), tol)
        self.assertFalse(v["pass"])
        self.assertEqual(len(v["failures"]), 1)
        self.assertTrue(v["failures"][0].startswith("mean "), v["failures"])

    def test_float_p9999_limit_alone_fails(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        m_truth = {"mean": 1e-5, "p999": 1e-3, "p9999": 5e-3, "max": 1e-2}  # p9999 above 1.5x, rest within
        v = compare_maps.evaluate_float(m_truth, ref, self.metrics(), tol)
        self.assertFalse(v["pass"])
        self.assertEqual(len(v["failures"]), 1)
        self.assertTrue(v["failures"][0].startswith("p9999 "), v["failures"])

    def test_float_sum_ratio_uses_absolute_value(self):
        tol = compare_maps.default_tolerances()
        ref = {"mean": 1e-5, "p999": 1e-3, "p9999": 3e-3, "max": 1e-2}
        v = compare_maps.evaluate_float(ref, ref, self.metrics(sum_ratio=-2e-5), tol)
        self.assertFalse(v["pass"])
        self.assertTrue(any("sum_ratio" in f for f in v["failures"]), v["failures"])

    def test_double_negative_off_mask_pixel_fails(self):
        rng = np.random.default_rng(1)
        b = 5.0 + rng.random((50, 40))
        b[0, 0] = 1e-9                  # far below 1e-3 x mean, so off the mask
        a = b.copy()
        a[0, 0] = -1e-9
        m = compare_maps.map_metrics(a, b)
        self.assertTrue(m["finite"])
        self.assertFalse(m["nonneg"])
        self.assertEqual(m["mean"], 0.0)
        self.assertEqual(m["p999"], 0.0)
        v = compare_maps.evaluate_double(m, compare_maps.default_tolerances())
        self.assertFalse(v["pass"])
        self.assertIn("negative pixels", v["failures"])

    def test_parity_consistency_hand_value_and_scale(self):
        total = np.array([[2.0, 4.0], [6.0, 8.0]])          # mean(total) = 5
        minima = np.array([[1.0, 2.0], [3.0, 4.0]])
        saddles = np.array([[1.0, 2.8], [3.0, 3.5]])
        # total - (minima + saddles) = [[0, -0.8], [0, 0.5]], so max |.| / mean(total) = 0.8 / 5
        value = compare_maps.parity_consistency(total, minima, saddles)
        self.assertAlmostEqual(value, 0.8 / 5.0, places=12)
        scaled = compare_maps.parity_consistency(10 * total, 10 * minima, 10 * saddles)
        self.assertAlmostEqual(scaled, value, places=12)


import configs  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_every_float_config_has_a_double_truth_with_identical_args(self):
        for cfg in configs.CONFIGS.values():
            self.assertIn(cfg.build, ("float", "double"), cfg.name)
            if cfg.build == "float":
                truth = configs.CONFIGS[cfg.truth]
                self.assertEqual(truth.build, "double", cfg.name)
                self.assertEqual(truth.args, cfg.args, cfg.name)
                self.assertEqual(truth.starfile_from, cfg.starfile_from, cfg.name)

    def test_ladder_and_sets(self):
        self.assertEqual(configs.LADDER, ["nb_f", "nb_d", "rect_f", "hm100_f"])
        names = [c.name for c in configs.ladder_configs()]
        self.assertEqual(names, configs.LADDER)
        self.assertEqual([c.name for c in configs.ladder_configs("R2p")], ["nb_d"])
        full = {c.name for c in configs.configs_in("full")}
        self.assertEqual(len(full), 15)
        self.assertEqual({c.name for c in configs.regression_configs("R2p", "ladder")},
                         {"nb_d", "rect_d", "rect_a0_d", "hm100_d"})
        self.assertEqual({c.name for c in configs.regression_configs("R3", "full")}, full)
        self.assertEqual({c.name for c in configs.configs_in("memory")}, {"big_f", "hm1000_f", "stars1e6_f"})

    def test_full_args_and_starfile(self):
        cfg = configs.CONFIGS["sf_f"]
        args = configs.full_args(cfg, "/ref")
        self.assertIn("-sf", args)
        self.assertEqual(args[args.index("-sf") + 1], "/ref/nb_f/run1/ipm_stars.bin")
        self.assertEqual(args[:2], ["-rs", configs.SEED])
        self.assertNotIn("-v", args)
        self.assertNotIn("-o", args)
        self.assertEqual(len(args) % 2, 0)

    def test_ordered_puts_starfile_users_last(self):
        # reversed declaration order puts every star-file user ahead of its source
        order = [c.name for c in configs.ordered(list(reversed(list(configs.CONFIGS.values()))))]
        self.assertEqual(sorted(order), sorted(configs.CONFIGS))
        users = [i for i, n in enumerate(order) if configs.CONFIGS[n].starfile_from is not None]
        others = [i for i, n in enumerate(order) if configs.CONFIGS[n].starfile_from is None]
        self.assertGreater(min(users), max(others))
        self.assertGreater(order.index("sf_f"), order.index("nb_f"))
        self.assertGreater(order.index("sf_d"), order.index("nb_f"))

    def test_exact_membership_of_each_set(self):
        self.assertEqual({c.name for c in configs.configs_in("full")}, {
            "nb_f", "nb_d", "rect_f", "hm100_f", "circ_a0_f", "rect_a0_f", "rect_d", "rect_a0_d",
            "par_f", "nry10_f", "big_f", "hm1000_f", "hm100_d", "sf_f", "stars1e6_f"})
        self.assertEqual({c.name for c in configs.configs_in("truth")}, {
            "circ_a0_d", "par_d", "nry10_d", "big_d", "hm1000_d", "sf_d", "stars1e6_d"})
        self.assertEqual({c.name for c in configs.configs_in("ladder")}, {"nb_f", "nb_d", "rect_f", "hm100_f"})

    def test_binding_constraints_hold_for_every_configuration(self):
        for cfg in configs.CONFIGS.values():
            with self.subTest(config=cfg.name):
                self.assertIn(cfg.build, ("float", "double"))
                self.assertEqual(cfg.parities, "-wp" in cfg.args)
                args = configs.full_args(cfg, "/ref")
                self.assertEqual(args[:len(configs.COMMON)], list(configs.COMMON))
                self.assertNotIn("-v", args)
                self.assertNotIn("-o", args)
                self.assertEqual(len(args) % 2, 0)

    def test_hm1000_differs_from_hm100_only_in_y(self):
        hm100 = list(configs.CONFIGS["hm100_f"].args)
        hm1000 = list(configs.CONFIGS["hm1000_f"].args)
        i = hm100.index("0.49")
        self.assertEqual(hm100[i - 1], "-y")
        expected = list(hm100)
        expected[i] = "0.499"
        self.assertEqual(hm1000, expected)

    def test_exe_for(self):
        self.assertEqual(configs.exe_for("/b", "float").name, "ipm_gpu")
        self.assertEqual(configs.exe_for("/b", "double").name, "ipm_gpu_double")


FAKE = str(HERE / "fake_ipm.py")


class RunIpmTests(unittest.TestCase):
    def run_fake(self, env_over=None, **kw):
        env = dict(os.environ)
        env.update(env_over or {})
        with tempfile.TemporaryDirectory() as d:
            rec = bench_lib.run_ipm(sys.executable, [FAKE, "-npy1", "4", "-npy2", "3"], Path(d) / "out", env=env, **kw)
            files = sorted(p.name for p in (Path(d) / "out").iterdir())
            return rec, files

    def test_success_record(self):
        rec, files = self.run_fake()
        self.assertTrue(rec["ok"], rec)
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["returncode"], 0)
        self.assertEqual(rec["t_shoot_cells"], 0.123)
        self.assertGreater(rec["wall_s"], 0.0)
        self.assertEqual(rec["param_info"]["num_stars"], "137")
        self.assertIn("ipm_magnifications.bin", files)
        self.assertIn("fake run done", rec["stdout_tail"])
        self.assertEqual(rec["args"][-2:], ["-npy2", "3"])   # -v and -o are not part of args

    def test_failure_exit_code(self):
        rec, _ = self.run_fake({"FAKE_IPM_FAIL": "1"})
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["returncode"], 3)
        self.assertIn("exit code 3", rec["error"])
        self.assertIn("fake failure", rec["stderr_tail"])

    def test_missing_param_file_not_ok(self):
        rec, _ = self.run_fake({"FAKE_IPM_NO_PARAM": "1"})
        self.assertFalse(rec["ok"])
        self.assertIn("ipm_parameter_info.txt", rec["error"])
        self.assertIsNone(rec["t_shoot_cells"])

    def test_missing_map_not_ok(self):
        rec, _ = self.run_fake({"FAKE_IPM_NO_MAP": "1"})
        self.assertFalse(rec["ok"])
        self.assertIn("ipm_magnifications.bin", rec["error"])
        rec, _ = self.run_fake({"FAKE_IPM_NO_MAP": "1"}, expect_map=False)
        self.assertTrue(rec["ok"])

    def test_old_outputs_are_removed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            out.mkdir()
            (out / "ipm_stale.bin").write_bytes(b"x")
            bench_lib.run_ipm(sys.executable, [FAKE], out)
            self.assertFalse((out / "ipm_stale.bin").exists())


class SamplerTests(unittest.TestCase):
    def test_no_index_gives_none(self):
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-b5449ec5-cd58-e6c6-e691-901514c1de2c"
            self.assertIsNone(bench_lib.gpu_index())
            self.assertIsNone(bench_lib.GpuMemorySampler().start().stop())
            os.environ.pop("CUDA_VISIBLE_DEVICES")
            self.assertIsNone(bench_lib.gpu_index())
            os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
            self.assertEqual(bench_lib.gpu_index(), "4")
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved


class HelperTests(unittest.TestCase):
    def test_median(self):
        self.assertEqual(bench_lib.median([3, 1, 2]), 2)
        self.assertEqual(bench_lib.median([4, 1, 3, 2]), 2.5)
        self.assertEqual(bench_lib.median([None, 5]), 5)
        self.assertIsNone(bench_lib.median([]))

    def test_json_roundtrip_and_build_meta(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "x.json"
            bench_lib.write_json(p, {"a": [1, 2]})
            self.assertEqual(bench_lib.read_json(p), {"a": [1, 2]})
            (Path(d) / "git_rev.txt").write_text("abc123 clean\n")
            meta = bench_lib.build_meta(d)
            self.assertEqual(meta["git_rev"], "abc123 clean")
            self.assertIsNone(meta["nvcc_version"])

    def test_environment_info_has_keys(self):
        info = bench_lib.environment_info()
        for key in ("hostname", "slurm_job_id", "cuda_visible_devices", "gpu", "nvcc", "timestamp"):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
