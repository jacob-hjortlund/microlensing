"""Login-node tests (no GPU) of the batch scheduler, driven through a fake backend:
python -m unittest tests.batch.test_scheduler -v"""
import _thread
import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fake_backend import GiB, FakeBackend, FakeResult  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError, MapFailed, Scheduler  # noqa: E402


def maps(n, **extra):
    return [{"index": i, **extra} for i in range(n)]


def batch_threads():
    return [t for t in threading.enumerate() if t.name.startswith("ipm-batch-")]


def collect(sched, params):
    out = []
    for f in sched.results(params):
        out.append(f)
        f.release()
    return out


def runs(fake):
    """index -> list of (device, start position, end position, slot_id) in the fake's log."""
    log = fake.log_snapshot()
    open_runs = {}
    out = {}
    for pos, (event, device, slot_id, index, _) in enumerate(log):
        if event == "start":
            open_runs[(slot_id, index)] = pos
        elif event == "end":
            start = open_runs.pop((slot_id, index))
            out.setdefault(index, []).append((device, start, pos, slot_id))
    return out


def releases(fake, device, after=-1):
    """(position, slot_id) of each release() on the device that returned after the given log position."""
    return [(pos, e[2]) for pos, e in enumerate(fake.log_snapshot())
            if e[0] == "release" and e[1] == device and pos > after]


def max_running_after(fake, after, device):
    """The most maps running at once on the device at log positions after the given one."""
    running = peak = 0
    for pos, e in enumerate(fake.log_snapshot()):
        if e[1] == device and e[0] in ("start", "end"):
            running += 1 if e[0] == "start" else -1
            if pos > after:
                peak = max(peak, running)
    return peak


def overlaps(a, b):
    return a[1] < b[2] and b[1] < a[2]


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.assert_no_batch_threads)

    def assert_no_batch_threads(self):
        self.assertEqual(batch_threads(), [])

    def make(self, fake, **kwargs):
        sched = Scheduler(fake, **kwargs)
        self.addCleanup(self.check_fake, fake)
        self.addCleanup(sched.close)
        return sched

    def check_fake(self, fake):
        self.assertEqual(fake.violations, [])
        for slot in fake.slots:
            names = set(slot.run_threads)
            self.assertLessEqual(len(names), 1, f"slot {slot.slot_id} ran on several threads: {names}")
            for name in names:
                self.assertTrue(name.startswith(f"ipm-batch-{slot.device}-"), name)

    def assert_successes(self, out, n, failed=()):
        self.assertEqual(sorted(f.index for f in out), list(range(n)))
        for f in out:
            if f.index in failed:
                self.assertIsInstance(f.result, BatchMapError)
            else:
                self.assertIsInstance(f.result, FakeResult)
                self.assertEqual(f.result.params["index"], f.index)

    def assert_stopped_cleanly(self, fake, sched, n):
        self.assertEqual(batch_threads(), [])
        log = fake.log_snapshot()
        starts = [e[3] for e in log if e[0] == "start"]
        ends = [e[3] for e in log if e[0] == "end"]
        self.assertEqual(sorted(starts), sorted(ends))
        self.assertLess(len(starts), n)
        closed_at = time.monotonic()
        time.sleep(0.15)
        self.assertEqual([e for e in fake.log_snapshot() if e[0] == "start" and e[4] > closed_at], [])
        self.assert_successes(collect(sched, maps(10)), 10)

    def test_all_maps_complete_with_their_indices(self):
        fake = FakeBackend()
        sched = self.make(fake)
        params = [{"index": i, "direct_deposit": i % 5 == 0} for i in range(20)]
        out = collect(sched, params)
        self.assert_successes(out, 20)
        self.assertEqual(sum(d["maps"] for d in sched.stats["devices"].values()), 20)
        self.assertEqual(sched.stats["direct_deposit_maps"], 4)
        for f in out:
            self.assertIs(f.result.params, params[f.index])

    def test_completion_order_not_batch_order(self):
        fake = FakeBackend()
        sched = self.make(fake)
        params = maps(10, time=0.01)
        params[0]["time"] = 0.2
        out = collect(sched, params)
        self.assert_successes(out, 10)
        self.assertNotEqual(out[0].index, 0)

    def test_concurrency_limited_by_memory(self):
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(24, pixel=3 * GiB, time=0.05)), 24)
        for d in (0, 1):
            dev = sched.stats["devices"][d]
            self.assertEqual(dev["budget"], 9 * GiB)
            self.assertEqual(dev["max_concurrent"], 3)
            self.assertLessEqual(fake.max_concurrency[d], 3)
            self.assertLessEqual(fake.peak_alloc[d], 9 * GiB)

    def test_slot_cap_and_maps_per_gpu(self):
        fake = FakeBackend()
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(48, time=0.05)), 48)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 8)
            self.assertLessEqual(fake.max_concurrency[d], 8)

        fake = FakeBackend()
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(16, time=0.05)), 16)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 2)
            self.assertLessEqual(fake.max_concurrency[d], 2)

    def test_no_concurrent_managed_access_limits_to_one_slot(self):
        fake = FakeBackend(concurrent_managed=False)
        sched = self.make(fake)
        with self.assertWarns(UserWarning):
            out = collect(sched, maps(6, time=0.02))
        self.assert_successes(out, 6)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 1)
            self.assertLessEqual(fake.max_concurrency[d], 1)

    def test_devices_argument_restricts_devices(self):
        fake = FakeBackend()
        sched = self.make(fake, devices=[1])
        self.assert_successes(collect(sched, maps(12, time=0.01)), 12)
        self.assertEqual({e[1] for e in fake.log_snapshot() if e[0] == "start"}, {1})
        self.assertEqual(list(sched.stats["devices"]), [1])

    def test_result_bytes_bounded(self):
        fake = FakeBackend()
        sched = self.make(fake, max_result_bytes=3 * GiB)
        out = []
        for f in sched.results(maps(12, result=GiB, time=0.01)):
            out.append(f)
            time.sleep(0.05)
            f.release()
        self.assert_successes(out, 12)
        self.assertLessEqual(sched.stats["peak_result_bytes"], 3 * GiB)
        self.assertGreaterEqual(sched.stats["peak_result_bytes"], 2 * GiB)

        fake = FakeBackend()
        sched = self.make(fake, max_result_bytes=1)
        self.assert_successes(collect(sched, maps(8, result=GiB, time=0.01)), 8)
        self.assertEqual(sched.stats["peak_result_bytes"], GiB)
        self.assertEqual(max(fake.max_concurrency.values()), 1)

    def test_oversized_map_runs_alone_and_small_maps_continue(self):
        # 20 small maps first, so that idle slots of both GPUs hold memory when the big map comes up
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        params = maps(40, pixel=GiB, time=0.05)
        params[20].update(pixel=int(8.5 * GiB), time=0.5)
        self.assert_successes(collect(sched, params), 40)
        self.assertEqual(sched.stats["devices"][0]["budget"], 9 * GiB)
        r = runs(fake)
        (big,) = r[20]
        others = [run for i, rs in r.items() if i != 20 for run in rs]
        self.assertEqual([o for o in others if o[0] == big[0] and overlaps(o, big)], [])
        self.assertTrue([o for o in others if o[0] != big[0] and overlaps(o, big)])
        # the idle slots' memory was really freed before the big map started
        self.assertTrue([p for p, _ in releases(fake, big[0]) if p < big[1]])
        self.assertLessEqual(fake.peak_alloc[big[0]], 9 * GiB)

    def test_map_bigger_than_budget_still_attempted_alone(self):
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        params = maps(8, pixel=GiB, time=0.05)
        params[2].update(pixel=20 * GiB, time=0.1)
        self.assert_successes(collect(sched, params), 8)
        r = runs(fake)
        (big,) = r[2]
        self.assertEqual([o for i, rs in r.items() if i != 2 for o in rs if overlaps(o, big)], [])
        self.assertEqual(sched.stats["devices"][0]["peak_committed"], 20 * GiB)

    def test_slow_release_blocks_admission_on_its_gpu(self):
        # map 2 runs alone after maps 0 and 1; while their slots are still being freed (0.2 s each), maps 3 and
        # 4 must not start on top of memory that is not free yet
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, release_time=0.2)
        sched = self.make(fake, maps_per_gpu=4)
        params = [{"index": 0, "pixel": 6 * GiB, "time": 0.1},
                  {"index": 1, "pixel": 0, "star": 3 * GiB, "time": 0.1},
                  {"index": 2, "pixel": 0, "tree": 2 * GiB, "time": 0.4},
                  {"index": 3, "pixel": 3 * GiB, "time": 0.4},
                  {"index": 4, "pixel": 3 * GiB, "time": 0.4}]
        self.assert_successes(collect(sched, params), 5)
        dev = sched.stats["devices"][0]
        self.assertEqual(dev["budget"], 9 * GiB)
        self.assertLessEqual(dev["peak_committed"], dev["budget"])
        self.assertLessEqual(fake.peak_alloc[0], dev["budget"])
        r = runs(fake)
        (two,) = r[2]
        freed = [p for p, _ in releases(fake, 0) if p < two[1]]
        self.assertEqual(len(freed), 2)
        self.assertTrue(all(r[i][0][1] > max(freed) for i in (3, 4)))

    def test_grow_only_slot_gives_memory_back_after_a_big_map(self):
        # real slots keep their buffers: without giving them back, the slot that ran the big map would keep
        # re-admitting small maps alone and the GPU would stay serial for the rest of the batch
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True)
        sched = self.make(fake)
        params = [{"index": 0, "pixel": int(8.5 * GiB), "time": 0.2}] + \
                 [{"index": i, "pixel": GiB, "time": 0.05} for i in range(1, 41)]
        self.assert_successes(collect(sched, params), 41)
        (big,) = runs(fake)[0]
        self.assertTrue(releases(fake, 0, after=big[2]))
        self.assertGreaterEqual(max_running_after(fake, big[2], 0), 2)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    def test_slot_holding_more_than_needed_shrinks_while_others_run(self):
        # a 6 GiB map runs next to three 1 GiB maps (9 GiB budget) while four slots wait for memory. Small maps
        # keep running, so the run-alone release never fires: the slot that ran the big map has to give its
        # memory back while the others run, so that more than 4 maps run at once afterwards
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True)
        sched = self.make(fake)
        params = [{"index": 0, "pixel": 6 * GiB, "time": 0.3}] + \
                 [{"index": i, "pixel": GiB, "time": 0.05} for i in range(1, 61)]
        self.assert_successes(collect(sched, params), 61)
        r = runs(fake)
        (big,) = r[0]
        others = [o for i, rs in r.items() if i != 0 for o in rs]
        shrinks = [p for p, slot_id in releases(fake, 0, after=big[2]) if slot_id == big[3]]
        self.assertTrue(shrinks)
        self.assertTrue([o for o in others if o[1] < shrinks[0] < o[2]])
        self.assertGreaterEqual(max_running_after(fake, shrinks[0], 0), 6)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    def test_estimate_failure_reported_and_batch_continues(self):
        threads = []

        def fail(p):
            threads.append(threading.current_thread())
            return MapFailed("bad estimate", 1) if p["index"] == 2 else None

        fake = FakeBackend(fail_estimate_when=fail)
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.01))
        self.assert_successes(out, 10, failed={2})
        (err,) = [f.result for f in out if f.index == 2]
        self.assertEqual((err.index, err.cuda_error, str(err)), (2, 1, "bad estimate"))
        self.assertEqual(set(threads), {threading.current_thread()})
        log = fake.log_snapshot()
        first_start = min(pos for pos, e in enumerate(log) if e[0] == "start")
        self.assertTrue(all(pos < first_start for pos, e in enumerate(log) if e[0] == "estimate"))
        self.assertNotIn(2, runs(fake))

    def test_run_failure_reported_and_batch_continues(self):
        fake = FakeBackend(fail_run_when=lambda p: MapFailed("bad", 0) if p["index"] == 3 else None)
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.01))
        self.assert_successes(out, 10, failed={3})
        (err,) = [f.result for f in out if f.index == 3]
        self.assertEqual((err.index, err.cuda_error, str(err)), (3, 0, "bad"))
        self.assertEqual(sum(d["maps"] for d in sched.stats["devices"].values()), 9)

    def test_unexpected_exception_becomes_batch_map_error(self):
        fake = FakeBackend(fail_run_when=lambda p: RuntimeError("boom") if p["index"] == 4 else None)
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.01))
        self.assert_successes(out, 10, failed={4})
        (err,) = [f.result for f in out if f.index == 4]
        self.assertEqual((err.index, err.cuda_error, str(err)), (4, 0, "boom"))
        self.assertIsInstance(err.__cause__, RuntimeError)

    def test_break_stops_cleanly(self):
        fake = FakeBackend()
        sched = self.make(fake)
        gen = sched.results(maps(50, time=0.05))
        for _ in range(2):
            next(gen).release()
        with self.assertRaises(RuntimeError):
            next(sched.results(maps(1)))
        gen.close()
        self.assert_stopped_cleanly(fake, sched, 50)

    def test_exception_in_consumer_stops_cleanly(self):
        fake = FakeBackend()
        sched = self.make(fake)
        with self.assertRaises(ValueError):
            for k, f in enumerate(sched.results(maps(50, time=0.05))):
                f.release()
                if k == 1:
                    raise ValueError("consumer failed")
        self.assert_stopped_cleanly(fake, sched, 50)

    def test_interrupt_while_waiting_stops_cleanly(self):
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=2)
        timer = threading.Timer(0.3, _thread.interrupt_main)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(KeyboardInterrupt):
            for f in sched.results(maps(6, time=1.0)):
                f.release()
        # delivered while the first two maps ran, so no further map started
        self.assertEqual(len([e for e in fake.log_snapshot() if e[0] == "start"]), 2)
        self.assert_stopped_cleanly(fake, sched, 6)

    def test_slots_persist_between_runs(self):
        fake = FakeBackend()
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(16, time=0.05)), 16)
        opened = fake.open_calls
        self.assertEqual(opened, 4)
        self.assert_successes(collect(sched, maps(16, time=0.05)), 16)
        self.assertEqual(fake.open_calls, opened)
        sched.close()
        sched.close()
        self.assertEqual([s.close_calls for s in fake.slots], [1] * opened)
        with self.assertRaises(RuntimeError):
            next(sched.results(maps(1)))

    def test_held_memory_counts_against_budget(self):
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(8, pixel=4 * GiB, time=0.05)), 8)
        self.assert_successes(collect(sched, maps(24, pixel=GiB, time=0.05)), 24)
        for d in (0, 1):
            dev = sched.stats["devices"][d]
            self.assertEqual(dev["budget"], 9 * GiB)
            self.assertGreaterEqual(dev["peak_committed"], 8 * GiB)
            self.assertLessEqual(dev["peak_committed"], dev["budget"])
            self.assertLessEqual(fake.peak_alloc[d], dev["budget"])


if __name__ == "__main__":
    unittest.main()
