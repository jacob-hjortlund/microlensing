"""Login-node tests (no GPU) of BatchedIPM, driven through a fake backend:
python -m unittest tests.batch.test_batched_ipm -v"""
import sys
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fake_backend import MiB, FakeBackend  # noqa: E402
from microlensing.IPM.batch import BatchedIPM, SavedMap  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError, MapFailed  # noqa: E402


def make(backend, **kw):
    kw.setdefault("kappa_tot", 0.4)
    kw.setdefault("shear", 0.3)
    with mock.patch("microlensing.IPM.batch._make_backend", return_value=backend):
        return BatchedIPM(**kw)


class FakeIPM:
    """Stands in for IPM in the serial fallback."""
    instances = []

    def __init__(self, **kw):
        self.kw = kw
        self.random_seed = kw.get("random_seed")
        self.saved = 0
        FakeIPM.instances.append(self)

    def run(self):
        if self.kw["shear"] == 0.25:
            raise RuntimeError("boom")

    def save(self):
        self.saved += 1


class TestBatchedIPM(unittest.TestCase):
    def test_constructor_validates_lengths(self):
        with self.assertRaises(ValueError):
            make(FakeBackend(), shear=[0.1, 0.2], kappa_tot=[0.3, 0.4, 0.5])

    def test_batch_level_arguments_reject_sequences(self):
        for kw in ({"is_double": [True]}, {"verbose": [1]}, {"devices": 0}, {"devices": ["a"]},
                   {"maps_per_gpu": [2]}, {"max_result_bytes": [1]}):
            with self.assertRaises(TypeError, msg=str(kw)):
                make(FakeBackend(), **kw)

    def test_num_maps_and_params(self):
        b = make(FakeBackend(), num_maps=3, outfile_prefix="r/")
        self.assertEqual(b.num_maps, 3)
        self.assertEqual(b.params(1)["outfile_prefix"], "r/1_")
        self.assertEqual(len({b.params(i)["random_seed"] for i in range(3)}), 3)
        b.close()

    def test_run_yields_index_and_result_for_every_map(self):
        with make(FakeBackend(), num_maps=5) as b:
            out = dict(b.run())
            self.assertEqual(sorted(out), list(range(5)))
            self.assertTrue(all(r.params["outfile_prefix"] == b.params(i)["outfile_prefix"]
                                for i, r in out.items()))
            self.assertIsNotNone(b.last_run_stats)

    def test_failed_map_yields_batch_map_error(self):
        fb = FakeBackend(fail_run_when=lambda p: MapFailed("bad") if p["shear"] == 0.25 else None)
        with make(fb, shear=[0.1, 0.25, 0.3]) as b:
            out = dict(b.run())
        self.assertIsInstance(out[1], BatchMapError)
        self.assertEqual(out[1].index, 1)
        self.assertFalse(isinstance(out[0], BatchMapError))

    def test_save_mode_saves_each_result_once_and_returns_records_in_order(self):
        fb = FakeBackend()
        with make(fb, num_maps=6, write_maps=True) as b:
            recs = b.run(save=True)
            self.assertEqual([r.index for r in recs], list(range(6)))
            for i, r in enumerate(recs):
                self.assertIsInstance(r, SavedMap)
                self.assertIsNone(r.error)
                self.assertEqual(r.outfile_prefix, b.params(i)["outfile_prefix"])
                self.assertEqual(r.random_seed, b.params(i)["random_seed"])
        self.assertEqual(len(fb.saved), 6)
        self.assertEqual(len({p["outfile_prefix"] for p in fb.saved}), 6)

    def test_save_mode_records_save_errors(self):
        fb = FakeBackend(save_fails_when=lambda p: p["outfile_prefix"].endswith("2_"))
        with make(fb, num_maps=4, write_maps=True) as b:
            recs = b.run(save=True)
        self.assertIsInstance(recs[2].error, BatchMapError)
        self.assertEqual(recs[2].error.index, 2)
        self.assertTrue(all(r.error is None for i, r in enumerate(recs) if i != 2))

    def test_save_mode_records_failed_maps(self):
        fb = FakeBackend(fail_run_when=lambda p: MapFailed("bad") if p["shear"] == 0.25 else None)
        with make(fb, shear=[0.1, 0.25], write_maps=True) as b:
            recs = b.run(save=True)
        self.assertIsInstance(recs[1].error, BatchMapError)
        self.assertEqual(recs[1].random_seed, b.params(1)["random_seed"])

    def test_save_mode_bounds_staging(self):
        fb = FakeBackend(run_time=0.0)
        limit = 3 * MiB
        with make(fb, num_maps=10, write_maps=True, max_result_bytes=limit) as b:
            with mock.patch("fake_backend.FakeResult.save", autospec=True,
                            side_effect=lambda self: time.sleep(0.02)):
                b.run(save=True)
            self.assertLessEqual(b.last_run_stats["peak_result_bytes"], limit)
            self.assertGreater(b.last_run_stats["peak_result_bytes"], 0)

    def test_save_without_write_flags_warns(self):
        with make(FakeBackend(), num_maps=2) as b:
            with self.assertWarnsRegex(UserWarning, "only the parameter files"):
                b.run(save=True)

    def test_save_with_write_flags_does_not_warn(self):
        with make(FakeBackend(), num_maps=2, write_stars=True) as b:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                b.run(save=True)

    def test_direct_deposit_warning(self):
        with make(FakeBackend(), num_maps=2) as b:
            b._params[0]["direct_deposit"] = True
            with self.assertWarnsRegex(UserWarning, "1 maps fell back to the direct deposit"):
                list(b.run())

    def test_dead_gpu_warning(self):
        fb = FakeBackend(devices=2, dead_after={1: 0}, run_time=0.05)
        with make(fb, num_maps=12, maps_per_gpu=1) as b:
            with self.assertWarnsRegex(UserWarning, "GPU 1"):
                list(b.run())

    def test_fallback_without_exports_runs_serially_and_warns(self):
        FakeIPM.instances = []
        with mock.patch("microlensing.IPM.batch.IPM", FakeIPM):
            with self.assertWarnsRegex(UserWarning, "no batch support") as cm:
                b = make(None, shear=[0.1, 0.25, 0.3], is_double=True, verbose=1)
            self.assertEqual(len(cm.warnings), 1)
            out = dict(b.run())
            self.assertEqual(sorted(out), [0, 1, 2])
            self.assertIsInstance(out[1], BatchMapError)
            self.assertEqual(out[1].index, 1)
            self.assertIsInstance(out[0], FakeIPM)
            self.assertTrue(out[0].kw["is_double"])
            self.assertEqual(out[0].kw["verbose"], 1)
            recs = b.run(save=True)
            self.assertEqual([r.index for r in recs], [0, 1, 2])
            self.assertIsNotNone(recs[1].error)
            self.assertIsNone(recs[0].error)
            self.assertEqual(recs[0].random_seed, b.params(0)["random_seed"])
            self.assertEqual(FakeIPM.instances[3].saved, 1)
            b.close()

    def test_close_releases_slots(self):
        fb = FakeBackend()
        with make(fb, num_maps=4) as b:
            list(b.run())
            self.assertTrue(fb.slots)
        self.assertTrue(all(s.closed for s in fb.slots))

    def test_scheduler_is_reused_between_runs(self):
        fb = FakeBackend()
        with make(fb, num_maps=3) as b:
            list(b.run())
            n = fb.open_calls
            list(b.run())
            self.assertEqual(fb.open_calls, n)

    def test_break_then_rerun(self):
        with make(FakeBackend(), num_maps=6) as b:
            for _ in b.run():
                break
            self.assertEqual(len(list(b.run())), 6)


if __name__ == "__main__":
    unittest.main()
