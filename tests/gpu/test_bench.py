"""Unit tests for the IPM GPU benchmark harness. No GPU needed:
python -m unittest discover -s tests/gpu -v"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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

    def test_hm1000_differs_from_hm100_only_in_kappa_and_shear(self):
        hm100 = list(configs.CONFIGS["hm100_f"].args)
        hm1000 = list(configs.CONFIGS["hm1000_f"].args)
        k = hm100.index("-k") + 1
        self.assertEqual(hm100[k], "0.5")
        y = hm100.index("-y") + 1
        self.assertEqual(hm100[y], "0.49")
        a = hm100.index("-a") + 1
        self.assertEqual(hm100[a], "1")
        expected = list(hm100)
        expected[k] = "0.7252"
        expected[y] = "0.2729"
        self.assertEqual(hm1000, expected)
        self.assertEqual(hm1000[a], "1")

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

    def test_timeout_gives_failed_record(self):
        rec, _ = self.run_fake({"FAKE_IPM_SLEEP": "3"}, timeout=0.5)
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["returncode"], -1)
        self.assertIn("timeout", rec["error"])
        self.assertIsInstance(rec["stdout_tail"], str)
        with tempfile.TemporaryDirectory() as d:
            bench_lib.write_json(Path(d) / "rec.json", rec)
            self.assertEqual(bench_lib.read_json(Path(d) / "rec.json")["error"], rec["error"])

    def test_timeout_keeps_partial_output_as_text(self):
        # The fake prints nothing before its sleep, so this child prints first to exercise the bytes path.
        code = "import time; print('partial', flush=True); time.sleep(30)"
        with tempfile.TemporaryDirectory() as d:
            rec = bench_lib.run_ipm(sys.executable, ["-c", code], Path(d) / "out", timeout=2)
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["stdout_tail"], "partial\n")
            bench_lib.write_json(Path(d) / "rec.json", rec)

    def test_missing_executable_gives_failed_record(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), tempfile.TemporaryDirectory() as d:
            before = threading.active_count()
            rec = bench_lib.run_ipm("/nonexistent/ipm_gpu", [], Path(d) / "out", sample_gpu=True)
            after = threading.active_count()
        self.assertFalse(rec["ok"])
        self.assertIn("cannot start", rec["error"])
        self.assertIn("/nonexistent/ipm_gpu", rec["error"])
        self.assertIsNone(rec["returncode"])
        self.assertEqual((rec["stdout_tail"], rec["stderr_tail"]), ("", ""))
        self.assertEqual(before, after)


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


import ptxas_stats  # noqa: E402


class PtxasTests(unittest.TestCase):
    LOG = """=== ipm_gpu ===
ptxas info    : Compiling entry function '_Z16histogram_kernelIfEvPT_7ComplexIiEiPii' for 'sm_80'
ptxas info    : Function properties for _Z16histogram_kernelIfEvPT_7ComplexIiEiPii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z18shoot_cells_kernelIfEvT_S0_S0_P4starIS0_ES0_P8TreeNodeIS0_Ei7ComplexIS0_EiiS7_S6_IiES7_S7_S7_S7_PS0_S9_S9_S8_Pyi' for 'sm_80'
ptxas info    : Function properties for _Z18shoot_cells_kernelIfEvT_S0_S0_P4starIS0_ES0_P8TreeNodeIS0_Ei7ComplexIS0_EiiS7_S6_IiES7_S7_S7_S7_PS0_S9_S9_S8_Pyi
    376 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 74 registers, 544 bytes cmem[0]
=== ipm_gpu_double ===
ptxas info    : Compiling entry function '_ZN8ipm_fast24shoot_cells_tiled_kernelIdfEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E' for 'sm_80'
ptxas info    : Function properties for _ZN8ipm_fast24shoot_cells_tiled_kernelIdfEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E
    48 bytes stack frame, 56 bytes spill stores, 56 bytes spill loads
ptxas info    : Used 80 registers, 6168 bytes smem, 760 bytes cmem[0]
=== lib_ipm.so ===
ptxas info    : Compiling entry function '_ZN8ipm_fast24shoot_cells_tiled_kernelIffEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E' for 'sm_80'
ptxas info    : Function properties for _ZN8ipm_fast24shoot_cells_tiled_kernelIffEEvT_S1_S1_PK4starIS1_ES1_P8TreeNodeIS1_E
    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 74 registers, 16 bytes cumulative stack size, 6168 bytes smem, 760 bytes cmem[0]
"""

    def test_parse_shoot_kernels(self):
        rows = ptxas_stats.parse(self.LOG)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["target"], "ipm_gpu")
        self.assertEqual(rows[0]["registers"], 74)
        self.assertEqual(rows[0]["stack_bytes"], 376)
        self.assertEqual(rows[0]["spill_stores"], 0)
        self.assertEqual(rows[0]["smem_bytes"], 0)
        self.assertEqual(rows[1]["target"], "ipm_gpu_double")
        self.assertEqual(rows[1]["registers"], 80)
        self.assertEqual(rows[1]["spill_loads"], 56)
        self.assertEqual(rows[1]["smem_bytes"], 6168)
        self.assertIn("tiled", rows[1]["function"])
        self.assertEqual(rows[2]["target"], "lib_ipm.so")
        self.assertEqual(rows[2]["registers"], 74)
        self.assertEqual(rows[2]["smem_bytes"], 6168)


import shutil  # noqa: E402
import make_reference  # noqa: E402


def fake_build_dir(build_dir):
    """A build directory whose executables are the fake."""
    build = Path(build_dir)
    build.mkdir(parents=True, exist_ok=True)
    for name in ("ipm_gpu", "ipm_gpu_double"):
        shutil.copy(FAKE, build / name)
        os.chmod(build / name, 0o755)
    return build


class MakeReferenceTests(unittest.TestCase):
    def test_creates_runs_then_skips_existing_runs(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            argv = ["--only", "nb_f", "--build", str(build), "--ref-root", f"{d}/ref",
                    "--scratch", f"{d}/scratch", "--results", f"{d}/results"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv), 0)
            for k in (1, 2, 3):
                self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nb_f" / f"run{k}"))
            timing = bench_lib.read_json(Path(d) / "results" / "timing_full.json")
            self.assertEqual(len(timing["records"]), 3)
            self.assertEqual(timing["records"][0]["config"], "nb_f")
            self.assertEqual(timing["rung"], "R0")
            # a truth-only configuration gets one run and no timing record
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv[:1] + ["nry10_d"] + argv[2:]), 0)
            self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nry10_d" / "run1"))
            self.assertFalse((Path(d) / "ref" / "nry10_d" / "run2").exists())
            self.assertEqual(len(bench_lib.read_json(Path(d) / "results" / "timing_full.json")["records"]), 3)
            # resume: remove run2, rerun, only run2 is regenerated and its record is replaced, not duplicated
            shutil.rmtree(Path(d) / "ref" / "nb_f" / "run2")
            stamp = (Path(d) / "ref" / "nb_f" / "run1" / "ipm_parameter_info.txt").stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv), 0)
            self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nb_f" / "run2"))
            self.assertEqual((Path(d) / "ref" / "nb_f" / "run1" / "ipm_parameter_info.txt").stat().st_mtime_ns, stamp)
            self.assertFalse((Path(d) / "scratch" / "nb_f" / "run2").exists())
            self.assertEqual(len(bench_lib.read_json(Path(d) / "results" / "timing_full.json")["records"]), 3)

    def test_failed_run_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            os.environ["FAKE_IPM_FAIL"] = "1"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = make_reference.main(["--only", "nb_f", "--build", str(build), "--ref-root", f"{d}/ref",
                                              "--scratch", f"{d}/scratch", "--results", f"{d}/results"])
            finally:
                os.environ.pop("FAKE_IPM_FAIL")
            self.assertEqual(rc, 1)
            self.assertFalse((Path(d) / "ref" / "nb_f" / "run1").exists())

    def test_force_replaces_reference_and_record(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            argv = ["--only", "nb_f", "--build", str(build), "--ref-root", f"{d}/ref",
                    "--scratch", f"{d}/scratch", "--results", f"{d}/results"]
            timing_path = Path(d) / "results" / "timing_full.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(make_reference.main(argv), 0)
            timing = bench_lib.read_json(timing_path)
            for r in timing["records"]:
                r["wall_s"] = -1.0                      # marks the first pass so a replacement is visible
            bench_lib.write_json(timing_path, timing)
            os.environ["FAKE_IPM_VALUE"] = "7.0"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(make_reference.main(argv + ["--force"]), 0)
            finally:
                os.environ.pop("FAKE_IPM_VALUE")
            np.testing.assert_array_equal(
                bench_lib.read_map(Path(d) / "ref" / "nb_f" / "run1" / "ipm_magnifications.bin"), np.full((3, 4), 7.0))
            records = bench_lib.read_json(timing_path)["records"]
            self.assertEqual(sorted((r["config"], r["repeat"]) for r in records), [("nb_f", 1), ("nb_f", 2), ("nb_f", 3)])
            self.assertTrue(all(r["wall_s"] > 0 for r in records))

    def test_failure_of_one_config_does_not_stop_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            build = fake_build_dir(Path(d) / "build")
            (build / "ipm_gpu").write_text("#!/bin/sh\nexit 3\n")   # the float build fails
            os.chmod(build / "ipm_gpu", 0o755)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = make_reference.main(["--only", "nb_f", "nb_d", "--build", str(build), "--ref-root", f"{d}/ref",
                                          "--scratch", f"{d}/scratch", "--results", f"{d}/results"])
            self.assertEqual(rc, 1)
            self.assertTrue(make_reference.is_complete(Path(d) / "ref" / "nb_d" / "run1"))
            self.assertFalse((Path(d) / "ref" / "nb_f" / "run1").exists())
            self.assertEqual(list((Path(d) / "ref").rglob("*.partial")), [])


import regress  # noqa: E402


def synthetic_reference(ref_root, name, value, shape=(30, 20), parities=False, k=1):
    d = Path(ref_root) / name / f"run{k}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base = value + rng.random(shape)
    write_map(d / "ipm_magnifications.bin", base.astype(np.float64))
    if parities:
        write_map(d / "ipm_magnifications_minima.bin", (0.4 * base).astype(np.float64))
        write_map(d / "ipm_magnifications_saddles.bin", (0.6 * base).astype(np.float64))
    (d / "ipm_parameter_info.txt").write_text("t_shoot_cells 1.0\n")
    return base


class RegressTests(unittest.TestCase):
    def tolerances(self):
        tol = compare_maps.default_tolerances()
        err = {"mean": 1e-6, "p999": 1e-5, "p9999": 2e-5, "max": 5e-5}
        tol["float_truth_error"] = {"nb_f": {"pixels": err}, "par_f": {"pixels": err, "minima": err, "saddles": err}}
        return tol

    def test_double_config_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base * (1 + 1e-9))
            res = regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R3")
            self.assertTrue(res["pass"], res["failures"])
            write_map(out / "ipm_magnifications.bin", base * (1 + 1e-4))
            res = regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R3")
            self.assertFalse(res["pass"])
            self.assertTrue(any("pixels" in f for f in res["failures"]))

    def test_float_config_uses_truth(self):
        with tempfile.TemporaryDirectory() as d:
            truth = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            synthetic_reference(f"{d}/ref", "nb_f", 5.0)      # same values: R0-float == truth here
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", (truth * (1 + 1e-6)).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertTrue(res["pass"], res["failures"])
            self.assertIn("vs_truth", res["maps"]["pixels"])
            write_map(out / "ipm_magnifications.bin", (truth * (1 + 1e-4)).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertFalse(res["pass"])

    def test_parity_maps_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "par_d", 5.0, parities=True)
            synthetic_reference(f"{d}/ref", "par_f", 5.0, parities=True)
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base.astype(np.float32))
            write_map(out / "ipm_magnifications_minima.bin", (0.4 * base).astype(np.float32))
            write_map(out / "ipm_magnifications_saddles.bin", (0.6 * base).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["par_f"], out, f"{d}/ref", self.tolerances(), "R4")
            self.assertTrue(res["pass"], res["failures"])
            self.assertEqual(set(res["maps"]), {"pixels", "minima", "saddles"})
            write_map(out / "ipm_magnifications_saddles.bin", (0.7 * base).astype(np.float32))
            res = regress.check_config(configs.CONFIGS["par_f"], out, f"{d}/ref", self.tolerances(), "R4")
            self.assertFalse(res["pass"])

    def test_missing_reference_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", np.ones((3, 3)))
            with self.assertRaises(FileNotFoundError) as ctx:
                regress.check_config(configs.CONFIGS["nb_d"], out, f"{d}/ref", self.tolerances(), "R1a")
            self.assertIn("nb_d/run1", str(ctx.exception))

    def write_fake_regression(self, d):
        """Fake build, 3x4 ladder references and a tolerances file under d; returns the build directory."""
        build = fake_build_dir(Path(d) / "build")
        # fake writes float32 maps of constant 2.0; references of the same value pass
        for name in ("nb_d", "nb_f", "rect_d", "rect_f", "hm100_d", "hm100_f"):
            ref = Path(d) / "ref" / name / "run1"
            ref.mkdir(parents=True)
            write_map(ref / "ipm_magnifications.bin", np.full((3, 4), 2.0, dtype=np.float32))
            (ref / "ipm_parameter_info.txt").write_text("t_shoot_cells 1.0\n")
        tol = self.tolerances()
        for name in ("nb_f", "rect_f", "hm100_f"):
            tol["float_truth_error"][name] = {"pixels": {"mean": 1e-6, "p999": 1e-5, "p9999": 2e-5, "max": 5e-5}}
        bench_lib.write_json(Path(d) / "tolerances.json", tol)
        return build

    def run_fake_regression(self, d, build, env):
        """regress.main over the fixture under d with the given environment switches; returns the exit code."""
        argv = ["--build", str(build), "--rung", "R1a", "--ref-root", f"{d}/ref",
                "--tolerances", f"{d}/tolerances.json", "--results", f"{d}/results", "--scratch", f"{d}/scratch"]
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            return regress.main(argv)

    def test_main_with_fake_executable(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            self.assertEqual(self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"}), 0)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            self.assertTrue(out["pass"])
            self.assertTrue(out["complete"])
            self.assertEqual(out["n_configs"], 4)
            self.assertEqual([c["config"] for c in out["configs"]], ["nb_f", "nb_d", "rect_f", "hm100_f"])
            self.assertFalse((Path(d) / "scratch" / "nb_f").exists())

    def test_check_failure_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            # 5x5 maps against the 3x4 references: every check raises a shape mismatch, and every configuration still runs
            rc = self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0", "FAKE_IPM_SHAPE": "5x5"})
            self.assertEqual(rc, 1)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            self.assertTrue(out["complete"])
            self.assertFalse(out["pass"])
            self.assertEqual(len(out["configs"]), 4)
            for c in out["configs"]:
                self.assertFalse(c["pass"])
                self.assertTrue(any(f.startswith("check failed: ValueError: shape mismatch") for f in c["failures"]),
                                c["failures"])

    def test_empty_selection_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = regress.main(["--build", f"{d}/build", "--rung", "R1a", "--only", "does_not_exist",
                                   "--ref-root", f"{d}/ref", "--tolerances", f"{d}/tolerances.json",
                                   "--results", f"{d}/results", "--scratch", f"{d}/scratch"])
            self.assertEqual(rc, 2)
            self.assertIn("no configurations selected", err.getvalue())
            self.assertFalse((Path(d) / "results").exists())

    def test_earlier_result_file_does_not_survive_an_abort(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            result = Path(d) / "results" / "regression_ladder.json"
            bench_lib.write_json(result, {"complete": True, "pass": True, "configs": []})
            # KeyboardInterrupt stands in for a killed job: it is not an Exception, so the per-configuration handler does not see it
            with mock.patch.object(bench_lib, "run_ipm", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"})
            out = bench_lib.read_json(result)
            self.assertFalse(out["complete"])
            self.assertEqual(out["configs"], [])
            # a later complete run replaces the file
            self.assertEqual(self.run_fake_regression(d, build, {"FAKE_IPM_VALUE": "2.0"}), 0)
            out = bench_lib.read_json(result)
            self.assertTrue(out["complete"])
            self.assertEqual(out["n_configs"], 4)

    def test_runs_have_a_timeout_and_failed_runs_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            build = self.write_fake_regression(d)
            with mock.patch.object(bench_lib, "run_ipm", wraps=bench_lib.run_ipm) as runs:
                rc = self.run_fake_regression(d, build, {"FAKE_IPM_FAIL": "1"})
            self.assertEqual(rc, 1)
            self.assertEqual(runs.call_count, 4)
            for call in runs.call_args_list:
                self.assertEqual(call.kwargs.get("timeout"), 3600)
            out = bench_lib.read_json(Path(d) / "results" / "regression_ladder.json")
            self.assertTrue(out["complete"])
            self.assertEqual(len(out["configs"]), 4)
            for c in out["configs"]:
                self.assertFalse(c["pass"])
                self.assertTrue(any("run failed: exit code 3" in f for f in c["failures"]), c["failures"])
            self.assertEqual(list((Path(d) / "scratch").glob("*")), [])

    def test_float_rule_is_judged_against_the_truth(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            # R0-float sits 2e-6 from the truth. The candidate is R0-float itself, so only the truth comparison can fail it.
            ref_f = Path(d) / "ref" / "nb_f" / "run1"
            ref_f.mkdir(parents=True)
            write_map(ref_f / "ipm_magnifications.bin", base * (1 + 2e-6))
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base * (1 + 2e-6))
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertFalse(res["pass"])
            self.assertTrue(res["failures"] and all("vs truth" in f for f in res["failures"]), res["failures"])

    def test_sum_ratio_is_judged_against_r0_float(self):
        with tempfile.TemporaryDirectory() as d:
            truth = synthetic_reference(f"{d}/ref", "nb_d", 5.0)
            ref_f = Path(d) / "ref" / "nb_f" / "run1"
            ref_f.mkdir(parents=True)
            write_map(ref_f / "ipm_magnifications.bin", truth * (1 + 2e-5))
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", truth)   # equal to the truth, so only the R0-float sum can fail
            res = regress.check_config(configs.CONFIGS["nb_f"], out, f"{d}/ref", self.tolerances(), "R2")
            self.assertFalse(res["pass"])
            self.assertTrue(res["failures"] and all("sum_ratio" in f for f in res["failures"]), res["failures"])

    def test_parity_consistency_fails_alone(self):
        with tempfile.TemporaryDirectory() as d:
            base = 5.0 + np.random.default_rng(3).random((30, 20))
            # self-inconsistent maps: minima + saddles = 0.9 x total, so only the parity check can fail
            maps = {"ipm_magnifications.bin": base,
                    "ipm_magnifications_minima.bin": 0.4 * base,
                    "ipm_magnifications_saddles.bin": 0.5 * base}
            for name in ("par_d", "par_f"):
                ref = Path(d) / "ref" / name / "run1"
                ref.mkdir(parents=True)
                for fname, arr in maps.items():
                    write_map(ref / fname, arr)
            out = Path(d) / "out"
            out.mkdir()
            for fname, arr in maps.items():
                write_map(out / fname, arr)
            res = regress.check_config(configs.CONFIGS["par_f"], out, f"{d}/ref", self.tolerances(), "R4")
            self.assertFalse(res["pass"])
            self.assertTrue(res["failures"] and all(f.startswith("pixels != minima + saddles") for f in res["failures"]),
                            res["failures"])
            self.assertTrue(all(entry["verdict"]["pass"] for entry in res["maps"].values()))

    def test_star_file_source_is_information_only(self):
        with tempfile.TemporaryDirectory() as d:
            base = synthetic_reference(f"{d}/ref", "sf_d", 5.0)
            synthetic_reference(f"{d}/ref", "sf_f", 5.0)
            ref_nb = Path(d) / "ref" / "nb_f" / "run1"
            ref_nb.mkdir(parents=True)
            write_map(ref_nb / "ipm_magnifications.bin", 1.5 * base)   # the source field is 1.5x the candidate
            out = Path(d) / "out"
            out.mkdir()
            write_map(out / "ipm_magnifications.bin", base)
            tol = self.tolerances()
            tol["float_truth_error"]["sf_f"] = {"pixels": {"mean": 1e-6, "p999": 1e-5, "p9999": 2e-5, "max": 5e-5}}
            res = regress.check_config(configs.CONFIGS["sf_f"], out, f"{d}/ref", tol, "R2")
            self.assertTrue(res["pass"], res["failures"])
            self.assertAlmostEqual(res["vs_source_field"]["mean"], 1 / 3, places=6)

    def test_unknown_configuration_name_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = regress.main(["--build", f"{d}/build", "--rung", "R1a", "--only", "nb_f", "nb_ff",
                                   "--ref-root", f"{d}/ref", "--tolerances", f"{d}/tolerances.json",
                                   "--results", f"{d}/results", "--scratch", f"{d}/scratch"])
            self.assertEqual(rc, 2)
            self.assertIn("unknown configuration(s): nb_ff", err.getvalue())
            self.assertFalse((Path(d) / "results").exists())


import calibrate  # noqa: E402


class CalibrateTests(unittest.TestCase):
    def test_floors_and_truth_errors(self):
        with tempfile.TemporaryDirectory() as d:
            ref = f"{d}/ref"
            for name in ("nb_f", "nb_d", "rect_f", "rect_d", "hm100_f", "hm100_d", "circ_a0_f", "circ_a0_d",
                         "rect_a0_f", "rect_a0_d", "par_f", "par_d", "nry10_f", "nry10_d", "big_f", "big_d",
                         "hm1000_f", "hm1000_d", "sf_f", "sf_d", "stars1e6_f", "stars1e6_d"):
                parities = name.startswith("par_")
                base = synthetic_reference(ref, name, 5.0, parities=parities, k=1)
                if "full" in configs.CONFIGS[name].sets:
                    for k in (2, 3):
                        dd = Path(ref) / name / f"run{k}"
                        dd.mkdir(parents=True)
                        noisy = base * (1 + 1e-9 * k)
                        write_map(dd / "ipm_magnifications.bin", noisy)
                        if parities:
                            write_map(dd / "ipm_magnifications_minima.bin", 0.4 * noisy)
                            write_map(dd / "ipm_magnifications_saddles.bin", 0.6 * noisy)
            with contextlib.redirect_stdout(io.StringIO()):   # calibrate() prints one progress line per map
                tol, lines, warnings = calibrate.calibrate(ref, compare_maps.default_tolerances())
            self.assertAlmostEqual(tol["floors"]["nb_d"]["pixels"][1]["max"], 3e-9, places=12)
            self.assertIn("nb_f", tol["float_truth_error"])
            self.assertEqual(set(tol["float_truth_error"]["par_f"]), {"pixels", "minima", "saddles"})
            self.assertEqual(warnings, [])
            self.assertTrue(any(line.startswith("nb_f") for line in lines))

    def test_warns_when_double_floor_is_close_to_threshold(self):
        tol = compare_maps.default_tolerances()
        tol["floors"] = {"nb_d": {"pixels": [{"mean": 0, "p999": 2e-6, "p9999": 0, "offmask_max": 0}]}}
        self.assertEqual(len(calibrate.sanity_warnings(tol)), 1)
        self.assertIn("nb_d", calibrate.sanity_warnings(tol)[0])


if __name__ == "__main__":
    unittest.main()
