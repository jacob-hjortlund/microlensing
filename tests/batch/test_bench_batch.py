"""Login-node tests (no GPU) of the helpers of bench_batch.py and gpu_busy.py:
python -m unittest tests.batch.test_bench_batch -v"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gpu"))

import bench_batch  # noqa: E402
import gpu_busy  # noqa: E402


def record(key, kind, config="small_f", mode="memory", gpus=1, maps_per_s=2.0, foreign=None, ok=True, **extra):
    rec = {"key": key, "kind": kind, "config": config, "mode": mode, "save_root": None, "gpus": gpus,
           "maps": 8, "wall_s": 4.0, "maps_per_s": maps_per_s, "peak_rss_mib": 1000, "ok": ok,
           "chosen_gpus": [1], "foreign_before": foreign or {}, "foreign_after": {}}
    rec.update(extra)
    return rec


class TestBenchBatch(unittest.TestCase):
    def test_median(self):
        self.assertEqual(bench_batch.median([3, 1, 2]), 2)
        self.assertEqual(bench_batch.median([1, 2]), 1.5)

    def test_speedup(self):
        self.assertEqual(bench_batch.speedup({"maps_per_s": 2.0}, {"maps_per_s": 5.0}), 2.5)

    def test_render_table_has_one_row_per_result(self):
        results = {}
        for cfg, mode in (("small_f", "memory"), ("small_f", "save"), ("nb_f", "memory")):
            for rep in (0, 1):
                k = f"{cfg}|{mode}|1|serial|{rep}"
                results[k] = record(k, "serial", config=cfg, mode=mode, maps_per_s=2.0)
                for g in (1, 2):
                    k = f"{cfg}|{mode}|{g}|batch|{rep}"
                    results[k] = record(k, "batch", config=cfg, mode=mode, gpus=g, maps_per_s=5.0 * g)
        results["nb_f|memory|2|batch|1"]["foreign_before"] = {"0": [{"pid": 7, "used_mib": 100}]}
        lines = bench_batch.render_table(results).strip().splitlines()
        rows = [ln for ln in lines if ln.startswith("|")][2:]
        self.assertEqual(len(rows), 6)    # (config, mode, gpus) groups, repeats are medians
        self.assertEqual(sum("foreign" in ln for ln in rows), 1)
        self.assertIn("2.50", [ln for ln in rows if "small_f" in ln and "memory" in ln][0])

    def test_render_table_without_serial_or_failed(self):
        k = "small_f|memory|1|batch|0"
        text = bench_batch.render_table({k: record(k, "batch", ok=False, maps_per_s=None)})
        self.assertIn("small_f", text)

    def test_choose_gpus_prefers_clean(self):
        self.assertEqual(bench_batch.choose_gpus([0, 1], 1, {0: [{"pid": 1}]}), [1])
        self.assertEqual(bench_batch.choose_gpus([0, 1], 2, {0: [{"pid": 1}]}), [0, 1])
        self.assertEqual(bench_batch.choose_gpus([0, 1, 2], 2, {}), [0, 1])
        with self.assertRaises(ValueError):
            bench_batch.choose_gpus([0], 2, {})

    def test_parse_foreign(self):
        apps = "00000000:01:00.0, 24223, 1474\n00000000:25:00.0, 5, 10\n"
        buses = "0, 00000000:01:00.0\n1, 00000000:25:00.0\n"
        got = bench_batch.parse_foreign(apps, buses, own_pids={5})
        self.assertEqual(got, {0: [{"pid": 24223, "used_mib": 1474}]})

    def test_plan_keys_and_maps(self):
        specs = bench_batch.plan(["small_f", "px40k_f"], [1, 2], ["memory"], 2, maps=4, save_root=None)
        keys = [s["key"] for s in specs]
        self.assertEqual(len(keys), len(set(keys)))
        px = [s for s in specs if s["config"] == "px40k_f" and s["kind"] == "batch" and s["gpus"] == 2]
        self.assertEqual(px[0]["maps"], 16)
        self.assertEqual([s["maps"] for s in specs if s["config"] == "small_f"], [4] * 6)
        self.assertTrue(all(s["gpus"] == 1 for s in specs if s["kind"] == "serial"))

    def test_resume_skips_existing_keys(self):
        specs = bench_batch.plan(["small_f"], [1], ["memory"], 1, maps=4, save_root=None)
        done = specs[0]["key"]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            bench_batch.write_json(out, {done: record(done, "serial")})
            launched = []

            def launch(spec):
                launched.append(spec["key"])
                return record(spec["key"], spec["kind"])

            results = bench_batch.run_plan(specs, out, launch)
            self.assertNotIn(done, launched)
            self.assertEqual(len(launched), len(specs) - 1)
            self.assertEqual(set(bench_batch.read_json(out)), {s["key"] for s in specs})
            self.assertEqual(set(results), {s["key"] for s in specs})


class TestGpuBusy(unittest.TestCase):
    def test_gpu_busy_union_and_overlap(self):
        s = gpu_busy.busy_stats([(0, 1), (0.5, 2), (3, 4)])
        self.assertAlmostEqual(s["wall_s"], 4)
        self.assertAlmostEqual(s["busy_fraction"], 3 / 4)
        self.assertAlmostEqual(s["overlap_fraction"], 0.5 / 3)

    def test_read_trace_keeps_kernels_only(self):
        text = ("Processing events...\n"
                "Start (ns),Duration (ns),GrdX,Name\n"
                "0,1000000000,8,k1\n"
                "500000000,1500000000,8,k2\n"
                "0,5,,[CUDA memcpy HtoD]\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text(text)
            iv = gpu_busy.read_trace(p)
        self.assertEqual(iv, [(0.0, 1.0), (0.5, 2.0)])


if __name__ == "__main__":
    unittest.main()
