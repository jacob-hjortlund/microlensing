"""Tests of BatchedIPM, driven through a fake backend. These tests need no GPU:
python -m unittest tests.batch.test_batched_ipm -v"""
import _thread
import gc
import inspect
import shutil
import signal
import sys
import tempfile
import threading
import time
import unittest
import warnings
import weakref
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fake_backend import GiB, MiB, FakeBackend, FakeResult  # noqa: E402
from microlensing.IPM import ipm as ipm_module  # noqa: E402
from microlensing.IPM.batch import _BATCH_LEVEL, BatchedIPM, SavedMap  # noqa: E402
from microlensing.IPM.batch_params import derive_seeds  # noqa: E402
from microlensing.IPM.batch_scheduler import DEFAULT_MAX_RESULT_BYTES, BatchMapError, MapFailed, Scheduler  # noqa: E402


def make(backend, **kw):
    kw.setdefault("kappa_tot", 0.4)
    kw.setdefault("shear", 0.3)
    with mock.patch("microlensing.IPM.batch._make_backend", return_value=backend):
        return BatchedIPM(**kw)


def make_fallback(**kw):
    """make() for a library without the batch exports, without the warning that says so (tested on its own)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "this IPM library has no batch support", UserWarning)
        return make(None, **kw)


class FakeIPM:
    """Stands in for IPM in the serial fallback."""
    instances = []

    def __init__(self, **kw):
        self.kw = kw
        self.random_seed = kw.get("random_seed")
        self.saved = 0
        FakeIPM.instances.append(self)

    def _apply_inputs(self, **kw):
        if kw.get("kappa_tot") is not None and kw["kappa_tot"] < 0:
            raise ValueError("kappa_tot must be >= 0")

    def run(self):
        if self.kw["shear"] == 0.25:
            raise RuntimeError("boom")

    def save(self):
        self.saved += 1


class TestBatchedIPM(unittest.TestCase):
    def test_constructor_validates_lengths(self):
        with self.assertRaises(ValueError):
            make(FakeBackend(), shear=[0.1, 0.2], kappa_tot=[0.3, 0.4, 0.5])

    def test_invalid_input_raises_in_constructor(self):
        # the same ValueError IPM's constructor gives, with the backend's estimate and without the exports
        with self.assertRaisesRegex(ValueError, "kappa_tot"):
            make(FakeBackend(fail_estimate_when=lambda p: MapFailed("kappa_tot must be >= 0")
                             if p["kappa_tot"] < 0 else None), kappa_tot=-1, num_maps=3)
        with self.assertRaisesRegex(ValueError, "map 0: .*kappa_tot"):
            make_fallback(kappa_tot=-1, num_maps=3)

    def test_invalid_member_of_sequence_names_map(self):
        def reject(p):
            return MapFailed("kappa_tot must be >= 0") if p["kappa_tot"] < 0 else None

        with self.assertRaisesRegex(ValueError, "map 1: .*kappa_tot"):
            make(FakeBackend(fail_estimate_when=reject), kappa_tot=[0.4, -1, 0.4])
        with self.assertRaisesRegex(ValueError, "map 1: .*kappa_tot"):
            make_fallback(kappa_tot=[0.4, -1, 0.4])

    def test_valid_batch_is_estimated_once_per_distinct_parameter_set(self):
        fb = FakeBackend()
        with make(fb, num_maps=5):
            before = sum(e[0] == "estimate" for e in fb.log_snapshot())
            self.assertEqual(before, 1)   # the seeds and prefixes differ, the rest does not
        fb = FakeBackend()
        with make(fb, shear=[0.1, 0.2, 0.1]):
            self.assertEqual(sum(e[0] == "estimate" for e in fb.log_snapshot()), 2)

    def test_starfile_maps_are_checked_without_the_backend(self):
        # a star file is read by the estimate, which makes a CUDA context: the constructor checks such maps on
        # the host only
        fb = FakeBackend()
        with make(fb, starfile="stars.bin", num_maps=2):
            self.assertEqual([e for e in fb.log_snapshot() if e[0] == "estimate"], [])
        with self.assertRaisesRegex(ValueError, "map 0: .*kappa_tot"):
            make(FakeBackend(), starfile="stars.bin", kappa_tot=-1)

    def test_save_raises_when_every_map_fails(self):
        fb = FakeBackend(fail_run_when=lambda p: MapFailed("no map today"))
        with make(fb, num_maps=3, write_maps=True) as b:
            with self.assertRaisesRegex(BatchMapError, "no map today") as cm:
                b.run(save=True)
            self.assertEqual([r.index for r in cm.exception.records], [0, 1, 2])
            self.assertTrue(all(r.error is not None for r in cm.exception.records))

    def test_save_returns_records_when_only_some_maps_fail(self):
        fb = FakeBackend(fail_run_when=lambda p: MapFailed("bad") if p["shear"] == 0.25 else None)
        with make(fb, shear=[0.1, 0.25], write_maps=True) as b:
            self.assertEqual(len(b.run(save=True)), 2)

    def test_batch_level_arguments_reject_sequences(self):
        for kw in ({"is_double": [True]}, {"verbose": [1]}, {"devices": 0}, {"devices": ["a"]},
                   {"maps_per_gpu": [2]}, {"max_result_bytes": [1]}):
            with self.assertRaises(TypeError, msg=str(kw)):
                make(FakeBackend(), **kw)

    def test_constructor_validates_batch_level_values(self):
        # checked when the batch is built, not at its first run()
        bad = [({"maps_per_gpu": 0}, ValueError), ({"maps_per_gpu": -1}, ValueError),
               ({"maps_per_gpu": True}, TypeError), ({"maps_per_gpu": 1.5}, TypeError),
               ({"maps_per_gpu": "2"}, TypeError),
               ({"devices": []}, ValueError), ({"devices": [0, 0]}, ValueError), ({"devices": [1, -1]}, ValueError),
               ({"devices": [True]}, TypeError), ({"devices": [0.5]}, TypeError),
               # devices[0] is the GPU of the estimates: an unordered collection would pick it arbitrarily
               ({"devices": {1, 0}}, TypeError), ({"devices": frozenset({0})}, TypeError),
               ({"devices": {1: "a", 0: "b"}}, TypeError),
               ({"max_result_bytes": 0}, ValueError), ({"max_result_bytes": -1}, ValueError),
               ({"max_result_bytes": True}, TypeError), ({"max_result_bytes": 1.5}, TypeError),
               ({"max_result_bytes": None}, TypeError)]
        for kw, exc in bad:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with self.assertRaises(exc):
                    make(FakeBackend(), **kw)
        good = [({"maps_per_gpu": None}, "_maps_per_gpu", None), ({"maps_per_gpu": np.int64(3)}, "_maps_per_gpu", 3),
                ({"devices": None}, "_devices", None), ({"devices": (1, 0)}, "_devices", [1, 0]),
                ({"devices": range(2)}, "_devices", [0, 1]), ({"devices": np.array([1, 0])}, "_devices", [1, 0]),
                ({"max_result_bytes": 1}, "_max_result_bytes", 1)]
        for kw, attr, value in good:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                b = make(FakeBackend(), **kw)
                self.assertEqual(getattr(b, attr), value)
                self.assertEqual(type(getattr(b, attr)), type(value))
                b.close()

    def test_signature_lists_the_ipm_inputs(self):
        # BatchedIPM spells out IPM's inputs, so that help() and editors show them: the same names and defaults
        params = inspect.signature(BatchedIPM.__init__).parameters
        inputs = set(params) - {"self", "num_maps"} - set(_BATCH_LEVEL)
        self.assertEqual(inputs, set(ipm_module._INPUT_NAMES) | {"seed_start"})
        for name, default in ipm_module._INPUT_DEFAULTS.items():
            self.assertEqual(params[name].default, default, name)
        self.assertEqual(params["max_result_bytes"].default, DEFAULT_MAX_RESULT_BYTES)   # the scheduler's default
        with self.assertRaisesRegex(TypeError, "kapa_tot"):
            make(FakeBackend(), kapa_tot=0.4)

    def test_num_maps_and_params(self):
        b = make(FakeBackend(), num_maps=3, outfile_prefix="r/")
        self.assertEqual(b.num_maps, 3)
        self.assertEqual(b.params(1)["outfile_prefix"], "r/1_")
        self.assertEqual(len({b.params(i)["random_seed"] for i in range(3)}), 3)
        b.close()

    def test_seed_arguments(self):
        with make(FakeBackend(), random_seed=42) as b:
            self.assertEqual(b.params(0)["random_seed"], 42)
        with make(FakeBackend(), num_maps=3, seed_start=42) as b:
            self.assertEqual([b.params(i)["random_seed"] for i in range(3)], derive_seeds(42, 3))
        with self.assertRaisesRegex(ValueError, "a single random_seed is one map's seed"):
            make(FakeBackend(), num_maps=3, random_seed=42)
        with self.assertRaisesRegex(ValueError, "cannot both be given"):
            make(FakeBackend(), num_maps=2, random_seed=[1, 2], seed_start=42)
        with self.assertRaisesRegex(TypeError, "seed_start applies to the whole batch"):
            make(FakeBackend(), num_maps=2, seed_start=[1, 2])

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
        # the 2 writers are slower than the maps: the results on the host (being copied or waiting to be written)
        # never exceed the limit, while the 10 maps, which all start at once, are not counted against it while
        # they run, and the finished ones wait on the GPU. Each map stays in its run until all 10 run, and the first
        # write waits until the results on the host reach the limit, so that neither depends on thread timing
        all_run = threading.Barrier(10)

        def hold_until_all_run(p):
            try:
                all_run.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass    # fewer than 10 maps ran at once: the test fails on max_concurrency
            return None

        fb = FakeBackend(fail_run_when=hold_until_all_run)
        limit = 3 * MiB
        reached = threading.Event()

        def slow_save(result):
            if not reached.is_set():
                deadline = time.monotonic() + 5
                while fb.host_bytes < limit and time.monotonic() < deadline:
                    time.sleep(0.002)
                reached.set()
            time.sleep(0.02)

        with make(fb, num_maps=10, write_maps=True, max_result_bytes=limit) as b:
            with mock.patch.object(FakeResult, "save", slow_save):
                b.run(save=True)
            stats = b.last_run_stats
            self.assertEqual(stats["result_budget"], limit)
            self.assertEqual(stats["peak_result_bytes"], limit)
            self.assertLessEqual(fb.peak_host_bytes, limit)
            self.assertEqual(sum(fb.max_concurrency.values()), 10)

    def test_save_without_maps_copies_no_maps(self):
        # save mode collects a map whose maps are not written without them: they are never copied to the host
        # nor counted against max_result_bytes (here 1 GiB of each 1 GiB + 1 MiB result). run() collects them
        fb = FakeBackend()
        with make(fb, num_maps=6, write_maps=False, write_histograms=True) as b:
            for i, p in enumerate(b._params):
                p.update(index=i, result=GiB + MiB, maps=GiB)
            recs = b.run(save=True)
            self.assertTrue(all(r.error is None for r in recs))
            self.assertEqual(sorted(fb.collects), [(i, False, MiB) for i in range(6)])
            self.assertLessEqual(fb.peak_host_bytes, 6 * MiB)
            self.assertLessEqual(b.last_run_stats["peak_result_bytes"], 6 * MiB)
            fb.collects.clear()
            self.assertEqual(sorted(i for i, _ in b.run()), list(range(6)))
            self.assertEqual(sorted(fb.collects), [(i, True, GiB + MiB) for i in range(6)])
        # the choice is made per map
        fb = FakeBackend()
        with make(fb, num_maps=4, write_maps=[True, False, True, False]) as b:
            for i, p in enumerate(b._params):
                p.update(index=i, result=GiB + MiB, maps=GiB)
            b.run(save=True)
        self.assertEqual(sorted(fb.collects), [(0, True, GiB + MiB), (1, False, MiB), (2, True, GiB + MiB),
                                               (3, False, MiB)])

    @staticmethod
    def slow_first_map(b):
        """Map 0 of b runs for 1 s and every other map for 20 ms."""
        for i, p in enumerate(b._params):
            p["time"] = 1.0 if i == 0 else 0.02

    def test_memory_mode_keeps_no_result_the_consumer_dropped(self):
        # 4 slots on 1 GPU, and map 0 runs for 1 s: once the consumer has dropped the other 7 results, nothing
        # of BatchedIPM or the scheduler keeps them while map 0 still runs
        fb = FakeBackend(devices=1)
        with make(fb, num_maps=8, maps_per_gpu=4) as b:
            self.slow_first_map(b)
            gen = b.run()
            refs = []
            for _ in range(7):
                index, result = next(gen)
                self.assertNotEqual(index, 0)
                refs.append(weakref.ref(result))
                del result
            gc.collect()
            self.assertNotIn(0, [e[3] for e in fb.log_snapshot() if e[0] == "end"])   # map 0 still runs
            self.assertEqual(sum(r() is not None for r in refs), 0)
            self.assertEqual([i for i, _ in gen], [0])

    def test_save_mode_keeps_no_written_result(self):
        # as above in save mode: once the other 7 maps are written, nothing keeps their results while map 0 runs
        fb = FakeBackend(devices=1)
        refs = []
        alive = []
        written = threading.Event()

        def save(result):   # a plain function: a mock would keep every result in its call list
            refs.append(weakref.ref(result))
            if len(refs) == 7:
                written.set()

        def check():
            written.wait(5)
            time.sleep(0.3)   # the writer releases the map after save() returns
            gc.collect()
            alive.append(sum(r() is not None for r in refs))
            alive.append(0 in [e[3] for e in fb.log_snapshot() if e[0] == "end"])

        with make(fb, num_maps=8, maps_per_gpu=4, write_maps=True) as b:
            self.slow_first_map(b)
            checker = threading.Thread(target=check)
            with mock.patch.object(FakeResult, "save", save):
                checker.start()
                recs = b.run(save=True)
            checker.join()
        self.assertTrue(all(r.error is None for r in recs))
        self.assertEqual(len(refs), 8)
        self.assertEqual(alive, [0, False])   # no written result alive, and map 0 still ran at the check

    def test_ctrl_c_in_save_mode_cancels_queued_writes(self):
        # Ctrl-C while maps wait for the one writer: the write in progress finishes, the queued writes are dropped
        # (their maps released and freed), and run() raises KeyboardInterrupt without waiting for them. The first
        # write ends with Ctrl-C; each write takes 0.4 s, so writing all 16 maps would take more than 6 s
        if threading.current_thread() is threading.main_thread():
            # interrupt_main() does nothing while SIGINT is ignored, as it is in a process that a script starts in
            # the background
            self.addCleanup(signal.signal, signal.SIGINT, signal.signal(signal.SIGINT, signal.default_int_handler))
        n, write_time = 16, 0.4
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        lock = threading.Lock()
        started, finished, runs, interrupted = [], [], [], []

        def save(result):   # a plain function: a mock would keep every result in its call list
            with lock:
                started.append(result.params["index"])
                first = len(started) == 1
            time.sleep(write_time)
            Path(result.params["outfile_prefix"] + "map.bin").write_bytes(b"map")
            with lock:
                finished.append(result.params["index"])
            if first:
                interrupted.append(time.monotonic())
                _thread.interrupt_main()

        def begin(sched, *args):   # records the run's state, whose waiting bytes are checked afterwards
            runs.append(real_begin(sched, *args))
            return runs[-1]

        real_begin = Scheduler._begin
        fb = FakeBackend(devices=1)
        with make(fb, num_maps=n, write_maps=True, outfile_prefix=f"{tmp}/") as b:
            for i, p in enumerate(b._params):
                p["index"] = i
            tb = None
            with mock.patch.object(FakeResult, "save", save), mock.patch.object(Scheduler, "_begin", begin):
                try:
                    b.run(save=True, writers=1)
                except KeyboardInterrupt as exc:
                    tb = exc.__traceback__   # as an interactive session keeps it
                    elapsed = time.monotonic() - interrupted[0]
            self.assertIsNotNone(tb, "run() returned")
            gc.collect()
            self.assertEqual(fb.host_bytes, 0)            # the dropped maps are not kept, even by the traceback
            self.assertEqual(runs[0].waiting_bytes, 0)    # and were released
            del tb
        # the first write, the one in progress when Ctrl-C arrived, and one more if the main thread was slow
        self.assertLessEqual(len(started), 3, started)
        self.assertLess(elapsed, (n // 2) * write_time)
        self.assertEqual(sorted(finished), sorted(started))
        self.assertEqual(sorted(f.name for f in tmp.iterdir()), sorted(f"{i}_map.bin" for i in finished))
        self.assertEqual([t.name for t in threading.enumerate() if t.name.startswith(("ipm-batch-", "ipm-save"))],
                         [])

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
            with self.assertWarnsRegex(UserWarning, "no batch support.*rebuild it with `source compile`") as cm:
                b = make(None, shear=[0.1, 0.25, 0.3], is_double=True, verbose=1)
            self.assertEqual(len(cm.warnings), 1)
            FakeIPM.instances = []   # the constructor made one IPM for each distinct set of inputs
            out = dict(b.run())
            self.assertEqual(sorted(out), [0, 1, 2])
            self.assertIsInstance(out[1], BatchMapError)
            self.assertEqual(out[1].index, 1)
            self.assertIsInstance(out[0], FakeIPM)
            self.assertTrue(out[0].kw["is_double"])
            self.assertEqual(out[0].kw["verbose"], 1)
            with self.assertWarnsRegex(UserWarning, "only the parameter files"):
                recs = b.run(save=True)
            self.assertEqual([r.index for r in recs], [0, 1, 2])
            self.assertIsNotNone(recs[1].error)
            self.assertIsNone(recs[0].error)
            self.assertEqual(recs[0].random_seed, b.params(0)["random_seed"])
            self.assertEqual(FakeIPM.instances[3].saved, 1)
            b.close()

    def test_fallback_rejects_other_devices(self):
        # without the batch exports every map runs on GPU 0: other devices raise rather than being ignored, and
        # the warning names the batch arguments that have no effect
        with mock.patch("microlensing.IPM.batch.IPM", FakeIPM):
            for devices in ([1], [0, 1], (2, 0)):
                with self.subTest(devices=devices), warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with self.assertRaisesRegex(ValueError, r"devices other than \[0\] need a library with the "
                                                            r"batch exports"):
                        make(None, devices=devices)
            for devices in (None, [0], (0,)):
                with self.subTest(devices=devices):
                    with self.assertWarnsRegex(UserWarning, "one at a time on GPU 0.*maps_per_gpu and "
                                                            "max_result_bytes have no effect"):
                        make(None, devices=devices, maps_per_gpu=2).close()

    def test_fallback_keeps_no_result_the_consumer_dropped(self):
        # one map at a time: while map 1 runs, nothing keeps map 0, which the consumer has dropped (a fallback
        # IPM holds its GPU memory until it is deleted)
        class Tracked(FakeIPM):
            def run(self):
                alive.append(sum(r() is not None for r in refs))

        alive, refs = [], []
        with mock.patch("microlensing.IPM.batch.IPM", Tracked), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = make(None, num_maps=2)
            gen = b.run()
            _, result = next(gen)
            FakeIPM.instances = []
            refs.append(weakref.ref(result))
            del result
            gc.collect()
            self.assertEqual(len(list(gen)), 1)
        self.assertEqual(alive, [0, 0])

    def test_close_releases_slots(self):
        fb = FakeBackend()
        with make(fb, num_maps=4) as b:
            list(b.run())
            self.assertTrue(fb.slots)
        self.assertTrue(all(s.closed for s in fb.slots))

    def test_scheduler_is_reused_between_runs(self):
        # one GPU of one slot, so that the slots opened do not depend on which threads take the maps
        fb = FakeBackend(devices=1)
        with make(fb, num_maps=3, maps_per_gpu=1) as b:
            list(b.run())
            sched = b._scheduler
            self.assertEqual(fb.open_calls, 1)
            list(b.run())
            self.assertIs(b._scheduler, sched)
            self.assertEqual(fb.open_calls, 1)

    def test_break_then_rerun(self):
        with make(FakeBackend(), num_maps=6) as b:
            for _ in b.run():
                break
            self.assertEqual(len(list(b.run())), 6)


if __name__ == "__main__":
    unittest.main()
