"""Tests of the batch scheduler, driven through a fake backend. These tests need no GPU:
python -m unittest tests.batch.test_scheduler -v"""
import _thread
import faulthandler
import gc
import signal
import sys
import threading
import time
import unittest
import warnings
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TEST_TIMEOUT = 60   # seconds; a test that runs longer fails, with a traceback of every thread (see _Watchdog)
KILL_TIMEOUT = 3 * TEST_TIMEOUT   # a test that _Watchdog cannot stop kills the run after this, with the tracebacks

from fake_backend import GiB, MiB, FakeBackend, FakeResult  # noqa: E402
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


def max_running_after(fake, after, device, before=None):
    """The most maps running at once on the device at log positions after the given one (and before `before`)."""
    running = peak = 0
    for pos, e in enumerate(fake.log_snapshot()):
        if before is not None and pos >= before:
            break
        if e[1] == device and e[0] in ("start", "end"):
            running += 1 if e[0] == "start" else -1
            if pos > after:
                peak = max(peak, running)
    return peak


def max_running(fake):
    """The most maps running at once on all devices together."""
    running = peak = 0
    for e in fake.log_snapshot():
        if e[0] in ("start", "end"):
            running += 1 if e[0] == "start" else -1
            peak = max(peak, running)
    return peak


def overlaps(a, b):
    return a[1] < b[2] and b[1] < a[2]


def events(fake, name, device=None):
    """(position, slot_id) of each event of that name in the fake's log, on the device if one is given."""
    return [(pos, e[2]) for pos, e in enumerate(fake.log_snapshot())
            if e[0] == name and (device is None or e[1] == device)]


def gpu_failed(d):
    return f"GPU {d} failed (700): illegal address"


def wait_for(ready, timeout=5.0):
    """Wait until ready() is true or the timeout has passed. A test that waits for an event this way still fails, on
    its asserts, when the event does not come."""
    deadline = time.monotonic() + timeout
    while not ready() and time.monotonic() < deadline:
        time.sleep(0.002)


def gpu_of_thread():
    """The GPU of the calling slot thread, whose name is ipm-batch-<gpu>-<slot>."""
    return int(threading.current_thread().name.split("-")[2])


def running_on(fake, device):
    with fake.lock:
        return fake.running[device]


def hold_until_running(fake, n, everywhere=False):
    """A fail_run_when: a run waits, before its sleep, until n maps run at once on its GPU (on all GPUs together with
    everywhere=True), or until wait_for's timeout. Once that has happened there, or the wait has timed out, runs no
    longer wait. An assert that n maps ran at once then rests on that event, not on how fast the slot threads start,
    and a scheduler that never runs n maps at once still fails it, after one timeout per GPU."""
    reached = set()

    def run(p):
        key = None if everywhere else gpu_of_thread()

        def ready():
            with fake.lock:
                if (sum(fake.running.values()) if everywhere else fake.running[key]) >= n:
                    reached.add(key)
                return key in reached

        wait_for(ready)
        with fake.lock:
            reached.add(key)
        return None

    return run


def out_of_memory_over(fake, limit):
    """A fail_run_when for a fake whose GPUs have limit bytes of real memory: a run runs out of memory when what its
    GPU's slots hold, its own memory included (run() counts it before it calls fail_run_when), exceeds the limit."""
    def run(p):
        d = gpu_of_thread()
        with fake.lock:
            over = fake.allocated(d) > limit
        return MapFailed("out of memory", 2) if over else None

    return run


OOM_ALONE = "out of device memory with no other map on GPU 0"


class TestTimeout(Exception):
    pass


class _Watchdog:
    """Fails a test that hangs, instead of killing the run. After TEST_TIMEOUT seconds it prints the traceback of
    every thread and sends SIGALRM to the main thread, whose handler raises TestTimeout there: the signal interrupts
    the blocking call that the test waits in (a lock, a Condition, a join or a sleep), so the test fails and the run
    goes on. It fires again every TEST_TIMEOUT seconds until cancel(), in case a cleanup hangs too. A hang that the
    signal cannot interrupt (inside C code that does not return to Python) is left to faulthandler's exit after
    KILL_TIMEOUT, which kills the run."""

    available = hasattr(signal, "pthread_kill") and hasattr(signal, "SIGALRM")   # not on Windows

    def __init__(self):
        self._stop = threading.Event()
        self._main = threading.main_thread().ident
        self._handler = signal.signal(signal.SIGALRM, self._raise)
        self._thread = threading.Thread(target=self._watch, name="test-watchdog", daemon=True)
        self._thread.start()

    @staticmethod
    def _raise(signum, frame):
        raise TestTimeout(f"the test ran for more than {TEST_TIMEOUT} s (the tracebacks of every thread are above)")

    def _watch(self):
        while not self._stop.wait(TEST_TIMEOUT):
            faulthandler.dump_traceback(all_threads=True)
            signal.pthread_kill(self._main, signal.SIGALRM)

    def cancel(self):
        self._stop.set()
        self._thread.join()
        signal.signal(signal.SIGALRM, self._handler)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        # without _Watchdog (Windows), a test that hangs kills the run after TEST_TIMEOUT
        faulthandler.dump_traceback_later(KILL_TIMEOUT if _Watchdog.available else TEST_TIMEOUT, exit=True)
        self.addCleanup(faulthandler.cancel_dump_traceback_later)
        if _Watchdog.available:
            self.addCleanup(_Watchdog().cancel)   # after the test's own cleanups, which it also covers
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

    @staticmethod
    def always_oom(index, **kwargs):
        """A fake on which map `index` always runs out of memory, and the memory allocated on its GPU at each of
        its tries (the list returned with the fake)."""
        seen = []
        fake = None

        def fail(p):
            if p["index"] != index:
                return None
            d = gpu_of_thread()
            with fake.lock:
                seen.append(fake.allocated(d))
            return MapFailed("out of memory", 2)

        fake = FakeBackend(fail_run_when=fail, **kwargs)
        return fake, seen

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
        # map 0 stays in its run until the consumer has received a map: a scheduler that hands the maps out in batch
        # order never gives it one, so map 0 ends after 5 s and comes first
        received = threading.Event()

        def hold_map_0(p):
            if p["index"] == 0:
                received.wait(5)
            return None

        fake = FakeBackend(fail_run_when=hold_map_0)
        sched = self.make(fake)
        out = []
        for f in sched.results(maps(10, time=0.01)):
            out.append(f)
            received.set()
            f.release()
        self.assert_successes(out, 10)
        self.assertNotEqual(out[0].index, 0)

    def test_concurrency_limited_by_memory(self):
        # 3 GiB maps on 9 GiB budgets: 3 at once per GPU. A GPU's first maps stay in their runs until 3 run there
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        fake.fail_run_when = hold_until_running(fake, 3)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(24, pixel=3 * GiB, time=0.1)), 24)
        for d in (0, 1):
            dev = sched.stats["devices"][d]
            self.assertEqual(dev["budget"], 9 * GiB)
            self.assertEqual(dev["max_concurrent"], 3)
            self.assertEqual(fake.max_concurrency[d], 3)
            self.assertLessEqual(fake.peak_alloc[d], 9 * GiB)

    def test_slot_cap_and_maps_per_gpu(self):
        # a GPU's first maps stay in their runs until as many run there as it has slots
        fake = FakeBackend()
        fake.fail_run_when = hold_until_running(fake, 8)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(48, time=0.1)), 48)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 8)
            self.assertEqual(fake.max_concurrency[d], 8)

        fake = FakeBackend()
        fake.fail_run_when = hold_until_running(fake, 2)
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(16, time=0.1)), 16)
        for d in (0, 1):
            self.assertEqual(sched.stats["devices"][d]["max_concurrent"], 2)
            self.assertEqual(fake.max_concurrency[d], 2)

    def test_no_concurrent_managed_access_limits_to_one_slot(self):
        # one slot per GPU, and one map at a time over both. Which GPU runs the maps depends on which slot thread
        # gets to them first, so each GPU is checked for at most one map at once, and both together for exactly one
        fake = FakeBackend(concurrent_managed=False)
        sched = self.make(fake)
        with self.assertWarns(UserWarning):
            out = collect(sched, maps(6, time=0.1))
        self.assert_successes(out, 6)
        for d in (0, 1):
            self.assertLessEqual(sched.stats["devices"][d]["max_concurrent"], 1)
            self.assertLessEqual(fake.max_concurrency[d], 1)
            self.assertLessEqual(len([s for s in fake.slots if s.device == d]), 1)
        self.assertEqual(max(dev["max_concurrent"] for dev in sched.stats["devices"].values()), 1)
        self.assertEqual(self.max_running(fake), 1)

    def max_running(self, fake):
        """The most maps that ran at once over all GPUs, from the fake's log."""
        n = best = 0
        for event, *_ in sorted((e for e in fake.log_snapshot() if e[0] in ("start", "end")), key=lambda e: e[4]):
            n += 1 if event == "start" else -1
            best = max(best, n)
        return best

    def test_no_concurrent_managed_access_runs_one_map_globally(self):
        # one of two GPUs lacks concurrent managed access: no two maps run at once, also not on different GPUs
        fake = FakeBackend(concurrent_managed=[True, False])
        sched = self.make(fake)
        with self.assertWarns(UserWarning):
            out = collect(sched, maps(6, time=0.1))
        self.assert_successes(out, 6)
        self.assertEqual(self.max_running(fake), 1)

    def test_gpu_dying_under_the_one_global_map_does_not_stall(self):
        fake = FakeBackend(concurrent_managed=[True, False], dead_after={1: 0},
                           fail_close_when=lambda s: RuntimeError("broken") if s.device == 1 else None)
        sched = self.make(fake)
        with self.assertWarns(UserWarning):
            out = collect(sched, maps(8, time=0.1))
        errors = [f for f in out if isinstance(f.result, BatchMapError)]
        self.assertLessEqual(len(errors), 1)
        self.assert_successes(out, 8, failed={f.index for f in errors})
        self.assertEqual(self.max_running(fake), 1)
        if fake.device_runs[1]:   # GPU 0 may take every map: nothing then runs on GPU 1 to break it
            self.assertTrue(sched.stats["devices"][1]["dead"])
            self.assertTrue(errors)
            with self.assertWarns(RuntimeWarning):
                sched.close()
        else:
            sched.close()

    def test_small_batches_spread_over_every_gpu(self):
        # 8 GPUs of 8 slots: 8 maps of equal length run one on each GPU and 16 maps two on each, instead of
        # filling the slots of the first GPUs. Every map stays in its run until all n run, so that every map is
        # admitted before the first one ends
        for n, per_gpu in ((8, 1), (16, 2)):
            fake = FakeBackend(devices=8)
            fake.fail_run_when = hold_until_running(fake, n, everywhere=True)
            sched = self.make(fake)
            self.assert_successes(collect(sched, maps(n, time=0.1)), n)
            self.assertEqual({d: dev["maps"] for d, dev in sched.stats["devices"].items()},
                             {d: per_gpu for d in range(8)}, n)
            self.assertEqual(fake.max_concurrency, {d: per_gpu for d in range(8)}, n)

    def test_maps_wait_for_the_slot_threads_of_a_gpu_that_runs_fewer(self):
        # the slot threads start slot 0 of every GPU first, which spreads a small batch as long as they reach their
        # first map in that order. Here GPU 1's threads reach it 0.3 s late, as on a loaded host: GPU 0's threads,
        # which could take all 4 maps at once, leave each one to GPU 1, which runs fewer maps and has idle slots, so
        # that each GPU runs 2. Every map stays in its run until all 4 run, so that none ends before GPU 1's threads
        # arrive
        class LateGpu1(Scheduler):
            def _slot_thread(self, run, st):
                if st.device == 1:
                    time.sleep(0.3)
                super()._slot_thread(run, st)

        fake = FakeBackend()
        fake.fail_run_when = hold_until_running(fake, 4, everywhere=True)
        sched = LateGpu1(fake)
        self.addCleanup(self.check_fake, fake)
        self.addCleanup(sched.close)
        self.assert_successes(collect(sched, maps(4, time=0.1)), 4)
        self.assertEqual({d: dev["maps"] for d, dev in sched.stats["devices"].items()}, {0: 2, 1: 2})
        self.assertEqual(fake.max_concurrency, {0: 2, 1: 2})

    def test_devices_argument_restricts_devices(self):
        fake = FakeBackend()
        sched = self.make(fake, devices=[1])
        self.assert_successes(collect(sched, maps(12, time=0.01)), 12)
        self.assertEqual({e[1] for e in fake.log_snapshot() if e[0] == "start"}, {1})
        self.assertEqual(list(sched.stats["devices"]), [1])

    def slow_consumer(self, sched, params, limit, delay=0.03):
        """Consume every map, releasing each one `delay` s after receiving it. Returns the items and the most
        result bytes seen waiting (queued, or received and not released) when each one was received."""
        out, seen = [], []
        for f in sched.results(params):
            out.append(f)
            seen.append(sched.stats["peak_result_bytes"])
            time.sleep(delay)
            f.release()
        self.assertTrue(all(s <= limit for s in seen), seen)
        return out

    def test_waiting_results_bounded_for_a_slow_consumer(self):
        # 2 GPUs of 8 slots, 1 GiB results and a 3 GiB limit, with a consumer slower than the maps: the finished
        # results waiting for it never exceed the limit, while the maps in progress do not count against it, so
        # all 16 slots start maps at once (the first maps stay in their runs until 16 run)
        fake = FakeBackend()
        fake.fail_run_when = hold_until_running(fake, 16, everywhere=True)
        sched = self.make(fake, max_result_bytes=3 * GiB)
        self.assert_successes(self.slow_consumer(sched, maps(40, result=GiB, time=0.1), 3 * GiB), 40)
        stats = sched.stats
        self.assertEqual(stats["result_budget"], 3 * GiB)
        self.assertEqual(stats["peak_result_bytes"], 3 * GiB)
        self.assertEqual(max_running(fake), 16)
        # a limit below one result: always room for one result, never for two, and the maps still overlap
        fake = FakeBackend(devices=1)
        fake.fail_run_when = hold_until_running(fake, 8)
        sched = self.make(fake, max_result_bytes=1)
        self.assert_successes(self.slow_consumer(sched, maps(16, result=GiB, time=0.1), GiB), 16)
        self.assertEqual(sched.stats["result_budget"], 1)
        self.assertEqual(sched.stats["peak_result_bytes"], GiB)
        self.assertEqual(fake.max_concurrency[0], 8)

    def test_host_bytes_bounded_by_cap(self):
        # 2 GPUs of 8 slots, 1 GiB results, a 2 GiB cap and a consumer slower than the maps. A finished map is
        # copied to the host (collect) only once its result fits under the cap, so the results on the host never
        # exceed the cap (the results being copied count against it) plus the one result the consumer keeps
        # after releasing it; the other finished maps wait on the GPU. The maps in progress do not count against
        # the cap, so all 16 slots start maps at once (the first maps stay in their runs until 16 run)
        fake = FakeBackend()
        fake.fail_run_when = hold_until_running(fake, 16, everywhere=True)
        sched = self.make(fake, max_result_bytes=2 * GiB)
        got = []
        for f in sched.results(maps(40, result=GiB, time=0.1)):
            got.append(f.index)   # f is the one result the consumer keeps: it is dropped when the next one comes
            time.sleep(0.03)
            f.release()
        self.assertEqual(sorted(got), list(range(40)))
        self.assertEqual(max_running(fake), 16)
        self.assertEqual(sched.stats["peak_result_bytes"], 2 * GiB)
        self.assertGreaterEqual(fake.peak_host_bytes, 2 * GiB)
        self.assertLessEqual(fake.peak_host_bytes, 2 * GiB + GiB)

    def test_collect_follows_run(self):
        # every map that ran is collected once, by the slot that ran it, after its run and before that slot runs
        # again (the fake records a violation otherwise), also with failed runs, out-of-memory retries and results
        # that wait for room
        failing = {3, 10, 17, 24}
        fake = FakeBackend(oom_above={0: 2}, grow_only=True,
                           fail_run_when=lambda p: MapFailed("bad") if p["index"] in failing else None)
        sched = self.make(fake, maps_per_gpu=4, max_result_bytes=2 * GiB)
        self.assert_successes(self.slow_consumer(sched, maps(30, result=GiB, time=0.01), 2 * GiB, 0.01), 30,
                              failed=failing)
        r = runs(fake)
        collected = {}
        for pos, (event, _, slot_id, index, _) in enumerate(fake.log_snapshot()):
            if event == "collect":
                self.assertNotIn(index, collected)
                collected[index] = (pos, slot_id)
        self.assertEqual(sorted(collected), sorted(set(range(30)) - failing))
        for index, (pos, slot_id) in collected.items():
            _, _, end, run_slot = r[index][-1]
            self.assertEqual(slot_id, run_slot)
            self.assertGreater(pos, end)
        # a stop while finished maps wait on the GPU: they are not collected, and their slots are released, so
        # that the next batch runs on slots with nothing left to collect
        gen = sched.results(maps(12, result=GiB, time=0.02))
        f = next(gen)
        time.sleep(0.2)
        before = len(events(fake, "collect"))
        gen.close()
        f.release()
        self.assertEqual(len(events(fake, "collect")), before)
        self.assertEqual([s.slot_id for s in fake.slots if s.uncollected is not None], [])
        self.assert_successes(collect(sched, maps(10)), 10, failed={3})

    def test_failed_collect_fails_only_its_map(self):
        # a map whose collect fails becomes a BatchMapError and the batch carries on; after a CUDA error its GPU is
        # checked first, and a GPU that still works stays in the batch
        fake = FakeBackend(fail_collect_when=lambda p: {3: MapFailed("copy failed"),
                                                        5: MapFailed("copy failed", 1)}.get(p["index"]))
        sched = self.make(fake)
        out = collect(sched, maps(10))
        self.assert_successes(out, 10, failed={3, 5})
        errors = sorted((f.index, str(f.result), f.result.cuda_error) for f in out if f.index in (3, 5))
        self.assertEqual(errors, [(3, "copy failed", 0), (5, "copy failed", 1)])
        self.assertEqual(len(events(fake, "check")), 1)
        self.assertFalse(any(dev["dead"] for dev in sched.stats["devices"].values()))
        self.assertEqual(sum(dev["maps"] for dev in sched.stats["devices"].values()), 8)

    def test_finished_maps_on_a_gpu_that_breaks_move_once(self):
        # 2 GPUs of 2 slots and a 1-byte limit: while the consumer keeps the first result, the other 3 finished
        # maps wait on their GPUs. The context of the GPU with 2 of them then breaks: the first whose collect fails
        # has the GPU checked and fails with the GPU's error, the other moves to the other GPU and runs there. The 4
        # maps stay in their runs until all 4 run, so that none is collected before all have started
        fake = FakeBackend()
        fake.fail_run_when = hold_until_running(fake, 4, everywhere=True)
        sched = self.make(fake, maps_per_gpu=2, max_result_bytes=1)
        gen = sched.results(maps(4, result=GiB, time=0.1))
        f = next(gen)
        self.wait_for_ends(fake, range(4))
        self.assertEqual(len(runs(fake)), 4)
        broken = 1 - f.result.device
        with fake.lock:
            fake.dead[broken] = True
        out = []
        while f is not None:
            out.append(f)
            f.release()
            f = next(gen, None)
        self.assertEqual(sorted(g.index for g in out), [0, 1, 2, 3])
        self.assertEqual([str(g.result) for g in out if isinstance(g.result, BatchMapError)], [gpu_failed(broken)])
        moved = [rr for rr in runs(fake).values() if len(rr) == 2]
        self.assertEqual([[x[0] for x in rr] for rr in moved], [[broken, 1 - broken]])
        self.assertTrue(sched.stats["devices"][broken]["dead"])

    @staticmethod
    def wait_for_ends(fake, indices, timeout=5.0):
        """Wait until the runs of the maps with these indices have ended in the fake's log."""
        deadline = time.monotonic() + timeout
        while set(indices) - {e[3] for e in fake.log_snapshot() if e[0] == "end"} and time.monotonic() < deadline:
            time.sleep(0.005)

    def test_oom_alone_retry_frees_finished_maps_once_collected(self):
        # 1 GPU of 3 slots, a 2 GiB limit, and the consumer keeps map 0's 1.5 GiB result, so X1 and X2 (1 GiB
        # each) finish and wait to be collected. W starts only once they have ended, so that B, after W, runs
        # alone; B runs out of memory whenever another slot holds memory. X1's and X2's slots hold their results,
        # so they cannot be freed then: B's retry waits until both are collected, then frees their memory and
        # runs alone, so that B is retried once. Maps 0, 1 and 2 stay in their runs until all three run, so that X1
        # and X2 start before map 0's result waits
        tries = []    # at each run of B: (the other slots that hold memory, how many other maps run)
        fake = None

        def fail(p):
            if p["index"] < 3:
                wait_for(lambda: running_on(fake, 0) >= 3)
            if p["index"] == 3:
                self.wait_for_ends(fake, (1, 2))
            elif p["index"] == 4:
                with fake.lock:
                    tries.append((sorted(s.slot_id for s in fake.slots if not s.busy and s.alloc.device_bytes() > 0),
                                  sum(s.busy for s in fake.slots) - 1))
                    return MapFailed("out of memory", 2) if tries[-1][0] else None
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail)
        sched = self.make(fake, maps_per_gpu=3, max_result_bytes=2 * GiB)
        params = [{"index": 0, "result": GiB + GiB // 2, "time": 0.01},
                  {"index": 1, "result": GiB, "time": 0.05},   # X1
                  {"index": 2, "result": GiB, "time": 0.05},   # X2
                  {"index": 3, "result": 0, "time": 0.2},      # W
                  {"index": 4, "result": 0, "time": 0.01}]     # B
        gen = sched.results(params)
        f = next(gen)
        self.assertEqual(f.index, 0)
        self.wait_for_ends(fake, (4,))   # B's first try
        self.assertEqual(sorted(e[3] for e in fake.log_snapshot() if e[0] == "collect"), [0, 3])   # X1, X2 wait
        out = []
        while f is not None:
            out.append(f)
            f.release()
            f = next(gen, None)
        self.assert_successes(out, 5)
        r = runs(fake)
        waiting = sorted(r[i][0][3] for i in (1, 2))
        self.assertEqual(tries, [(waiting, 0), ([], 0)])   # alone both times; nothing else held at the retry
        self.assertEqual(sched.stats["devices"][0]["oom_retries"], 1)

    def test_stop_leaves_the_slots_of_a_gpu_that_left_the_batch(self):
        # 1 GPU of 2 slots and a 1 GiB limit: the consumer keeps map 0's 1 GiB result, so X finishes and waits to
        # be collected. Z (no result), once X has ended, breaks the GPU's context, and the GPU leaves the batch. A
        # stop then does not free X's slot, whose GPU is not used again (close() closes it): freeing it would fail
        # and warn that the GPU leaves the batch a second time
        fake = None

        def fail(p):
            if p["index"] < 2:   # maps 0 and 1 stay in their runs until both run, so that X starts
                wait_for(lambda: running_on(fake, 0) >= 2)
            if p["index"] == 2:
                self.wait_for_ends(fake, (1,))
                with fake.lock:
                    fake.dead[0] = True
                return MapFailed("illegal address", 700)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail,
                           fail_release_when=lambda s: MapFailed("context broke", 700) if fake.dead[s.device] else None)
        sched = self.make(fake, maps_per_gpu=2, max_result_bytes=GiB)
        gen = sched.results([{"index": 0, "result": GiB, "time": 0.01}, {"index": 1, "result": GiB, "time": 0.05},
                             {"index": 2, "result": 0, "time": 0.01}])
        f = next(gen)
        self.assertEqual(f.index, 0)
        wait_for(lambda: sched.stats["devices"][0]["dead"])
        self.assertTrue(sched.stats["devices"][0]["dead"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen.close()
        f.release()
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual([e[0] for e in fake.log_snapshot() if e[0].startswith("release")], [])
        self.assertEqual(len(events(fake, "collect")), 1)

    def test_large_results_run_side_by_side(self):
        # 40000^2 float maps have 6 GiB results: one GPU with 2 slots at the default 8 GiB limit runs 2 of them at
        # once, since the maps in progress are not counted, but never holds two on the host (12 GiB > 8 GiB). The
        # first maps stay in their runs until 2 run (16 over 8 GPUs, below)
        fake = FakeBackend(devices=1)
        fake.fail_run_when = hold_until_running(fake, 2)
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(collect(sched, maps(6, result=6 * GiB, time=0.1)), 6)
        self.assertEqual(fake.max_concurrency[0], 2)
        self.assertEqual(sched.stats["result_budget"], 8 * GiB)
        self.assertEqual(sched.stats["peak_result_bytes"], 6 * GiB)
        # the same with a slow consumer: a finished map that does not fit waits on the GPU
        fake = FakeBackend(devices=1)
        fake.fail_run_when = hold_until_running(fake, 2)
        sched = self.make(fake, maps_per_gpu=2)
        self.assert_successes(self.slow_consumer(sched, maps(6, result=6 * GiB, time=0.1), 6 * GiB, 0.15), 6)
        self.assertEqual(fake.max_concurrency[0], 2)
        self.assertEqual(sched.stats["peak_result_bytes"], 6 * GiB)
        # 8 GPUs: the limit is not raised, and every GPU runs maps at once
        fake = FakeBackend(devices=8)
        fake.fail_run_when = hold_until_running(fake, 16, everywhere=True)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(16, result=6 * GiB, time=0.1)), 16)
        self.assertEqual(max_running(fake), 16)
        self.assertEqual(sched.stats["result_budget"], 8 * GiB)
        self.assertEqual(sched.stats["peak_result_bytes"], 6 * GiB)

    def test_waiting_slot_threads_keep_no_finished_result(self):
        # 4 slots on 1 GPU: map 0 runs for 1 s, maps 1-7 end within about 0.1 s. While map 0 runs, the other 3
        # slot threads wait for a map: the results that the consumer received, released and dropped must be freed
        # by then, not kept until the threads exit
        fake = FakeBackend(devices=1)
        sched = self.make(fake, maps_per_gpu=4)
        params = maps(8, time=0.02)
        params[0]["time"] = 1.0
        gen = sched.results(params)
        refs = []
        for _ in range(7):
            f = next(gen)
            self.assertNotEqual(f.index, 0)
            refs.append(weakref.ref(f.result))
            f.release()
            del f
        gc.collect()
        self.assertNotIn(0, [e[3] for e in fake.log_snapshot() if e[0] == "end"])   # map 0 still runs
        self.assertEqual(sum(r() is not None for r in refs), 0)
        f = next(gen)
        self.assertEqual(f.index, 0)
        f.release()
        gen.close()

    def test_gpu_whose_slots_wait_for_queue_room_takes_no_map_from_another(self):
        # 2 GPUs of 2 slots, 9 GiB budgets, a 3 GiB limit. The device memory places L (6 GiB) and X1 then X2 on
        # one GPU, B1 and B2 (4 GiB each) on the other. B1 and B2 finish while X1 and X2 wait (2 GiB), so both wait
        # in their slots, not yet collected. Once the consumer releases X1, S fits next to X2 but B1 and B2 do not:
        # S must start on the GPU of L at once, not be left to the other GPU, whose slots cannot start it.
        # Events keep this order: L runs until the consumer has released X2, B1 and B2 run until X2 is collected,
        # and the consumer keeps X1 until the slots of B1 and B2 wait for room (the scheduler's holding), and X2
        # until S has started. A scheduler that leaves S to the other GPU starts it only after X2's release, once
        # the consumer has waited 5 s
        fake = FakeBackend(free=10 * GiB, total=10 * GiB)
        x2_released = threading.Event()

        def logged(event, index):
            return lambda: any(e[0] == event and e[3] == index for e in fake.log_snapshot())

        def hold(p):
            if p["index"] == 0:
                x2_released.wait(10)
            elif p["index"] in (1, 2):
                wait_for(logged("collect", 4))
            return None

        def both_b_wait_for_room():
            gpu_b = next((e[1] for e in fake.log_snapshot() if e[0] == "start" and e[3] == 1), None)
            if gpu_b is None:   # B1 has not started yet
                return False
            with sched._cond:
                return sum(st.device == gpu_b for st in sched._run.holding) == 2

        fake.fail_run_when = hold
        sched = self.make(fake, maps_per_gpu=2, max_result_bytes=3 * GiB)
        params = [{"index": 0, "pixel": 6 * GiB, "result": 100 * MiB, "time": 0.05},
                  {"index": 1, "pixel": 4 * GiB, "result": int(2.5 * GiB), "time": 0.05},
                  {"index": 2, "pixel": 4 * GiB, "result": int(2.5 * GiB), "time": 0.05},
                  {"index": 3, "pixel": GiB, "result": GiB, "time": 0.05},
                  {"index": 4, "pixel": GiB, "result": GiB, "time": 0.05},
                  {"index": 5, "result": int(1.5 * GiB), "time": 0.05}]
        out, released = [], {}
        for f in sched.results(params):
            out.append(f)
            if f.index == 3:
                wait_for(both_b_wait_for_room)
            elif f.index == 4:
                wait_for(logged("start", 5))
            released[f.index] = time.monotonic()
            f.release()
            if f.index == 4:
                x2_released.set()
        self.assert_successes(out, 6)
        r = runs(fake)
        (big_l,), (b1,), (b2,), (x1,), (x2,), (s,) = (r[i] for i in range(6))
        self.assertEqual({b1[0], b2[0]}, {1 - big_l[0]})
        self.assertEqual({x1[0], x2[0], s[0]}, {big_l[0]})
        start_s = fake.log_snapshot()[s[1]][4]
        self.assertGreater(start_s, released[3])
        self.assertLess(start_s, released[4])

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

    @staticmethod
    def small_and_big_gpu(on_big_gpu, oom_on_small=False):
        """A fake with an 8 GiB GPU 0 and a 40 GiB GPU 1 (budgets 7 and 38 GiB), and a batch of 8 maps of 5 GiB and
        then a 20 GiB map (index 8), each running for 0.1 s. GPU 0 runs one 5 GiB map at a time and GPU 1 up to 7,
        and on GPU 1 each run first calls on_big_gpu(params). With oom_on_small, the 20 GiB map runs out of memory
        on GPU 0."""
        def fail(p):
            if gpu_of_thread() == 1:
                return on_big_gpu(p)
            if oom_on_small and p.get("big"):
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(free=[8 * GiB, 40 * GiB], total=[8 * GiB, 40 * GiB], fail_run_when=fail)
        params = maps(9, pixel=5 * GiB, time=0.1)
        params[8].update(pixel=20 * GiB, big=True)
        return fake, params

    def test_map_too_big_for_small_gpu_goes_to_big_gpu(self):
        # GPU 1's 5 GiB maps last 0.4 s longer than GPU 0's, so when the 20 GiB map comes up GPU 0 soon has
        # nothing to run while GPU 1 has no room for it yet. GPU 0 leaves it to GPU 1, the only GPU whose budget
        # fits it, rather than run it alone and run out of memory
        def slow(p):
            if not p.get("big"):
                time.sleep(0.4)
            return None

        fake, params = self.small_and_big_gpu(slow, oom_on_small=True)
        sched = self.make(fake)
        out = collect(sched, params)
        self.assert_successes(out, 9)
        self.assertEqual([sched.stats["devices"][d]["budget"] for d in (0, 1)], [7 * GiB, 38 * GiB])
        (big,) = [f.result for f in out if f.index == 8]
        self.assertEqual(big.device, 1)
        self.assertEqual([e for e in fake.log_snapshot() if e[3] == 8 and e[1] == 0], [])
        self.assertEqual([sched.stats["devices"][d]["oom_retries"] for d in (0, 1)], [0, 0])
        r = runs(fake)
        (run8,) = r[8]
        # GPU 0 had finished a map before the 20 GiB map started: it could have taken it
        self.assertTrue([o for i, rs in r.items() if i != 8 for o in rs if o[0] == 0 and o[2] < run8[1]])

    def test_map_too_big_for_small_gpu_runs_there_once_big_gpu_leaves(self):
        # as above, but GPU 1's 5 GiB maps break it after 0.3 s, while the 20 GiB map waits for room there. With
        # no GPU left whose budget fits it, the 20 GiB map runs alone on GPU 0 (where it does not run out here)
        def breaks(p):
            time.sleep(0.3)
            fake.device_errors[1] = 700
            return MapFailed("illegal address", 700)

        fake, params = self.small_and_big_gpu(breaks)
        sched = self.make(fake)
        out = collect(sched, params)
        errors = [f for f in out if isinstance(f.result, BatchMapError)]
        self.assertEqual([str(f.result) for f in errors], [gpu_failed(1)])
        self.assert_successes(out, 9, failed={errors[0].index})
        self.assertTrue(sched.stats["devices"][1]["dead"])
        (big,) = [f.result for f in out if f.index == 8]
        self.assertEqual(big.device, 0)
        r = runs(fake)
        (run8,) = r[8]
        self.assertEqual([o for i, rs in r.items() if i != 8 for o in rs if o[0] == 0 and overlaps(o, run8)], [])
        first_check = events(fake, "checked", 1)[0][0]
        self.assertGreater(run8[1], first_check)

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
                 [{"index": i, "pixel": GiB, "time": 0.1} for i in range(1, 41)]
        self.assert_successes(collect(sched, params), 41)
        (big,) = runs(fake)[0]
        self.assertTrue(releases(fake, 0, after=big[2]))
        self.assertGreaterEqual(max_running_after(fake, big[2], 0), 2)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    # ---- mixed map sizes: each test fails without the admission rule it names

    def test_large_map_after_small_maps_frees_idle_slots(self):
        # run alone. 16 maps of 1 GiB leave 8 slots holding 1 GiB each (8 of the 9 GiB budget) when a 6 GiB map
        # comes up, and the GPU has no more real memory than its budget. The large map waits until nothing runs,
        # has every idle slot's memory freed and starts without running out of memory. Only the 3 later maps that fit
        # beside it can start before it does: maps after it stay in their runs until it has started, so that none of
        # them ends and makes room for a fourth
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB)
        oom = out_of_memory_over(fake, 9 * GiB)

        def fail(p):
            if p["index"] > 16:
                wait_for(lambda: [e for e in fake.log_snapshot() if e[0] == "start" and e[3] == 16])
            return oom(p)

        fake.fail_run_when = fail
        sched = self.make(fake)
        params = maps(24, pixel=GiB, time=0.1)
        params[16].update(pixel=6 * GiB, time=0.2)
        self.assert_successes(collect(sched, params), 24)
        self.assertEqual(sched.stats["devices"][0]["oom_retries"], 0)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)
        r = runs(fake)
        (large,) = r[16]
        holders = {o[3] for i in range(16) for o in r[i]}
        self.assertEqual({s for p, s in releases(fake, 0) if p < large[1]}, holders)
        self.assertLessEqual(len([i for i in range(17, 24) if r[i][0][1] < large[1]]), 3)

    def test_shrink_frees_oversized_idle_slot(self):
        # shrink, with grow-only slots and a GPU with no more real memory than its 9 GiB budget. A 6 GiB map runs
        # beside a 1 GiB map that lasts 1 s and two more 1 GiB maps, and the other 1 GiB maps wait for memory. The
        # GPU never empties while the long map runs, so idle slots are never freed to run a map alone: only the slot
        # of the 6 GiB map, which keeps 6 GiB when it ends, can make room, by freeing them before it runs a 1 GiB
        # map. Then the GPU runs more than the 4 maps that fit next to 6 GiB
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True)
        fake.fail_run_when = out_of_memory_over(fake, 9 * GiB)
        sched = self.make(fake)
        params = [{"index": 0, "pixel": GiB, "time": 1.0}, {"index": 1, "pixel": 6 * GiB, "time": 0.3}] + \
                 [{"index": i, "pixel": GiB, "time": 0.1} for i in range(2, 26)]
        self.assert_successes(collect(sched, params), 26)
        self.assertEqual(sched.stats["devices"][0]["oom_retries"], 0)
        r = runs(fake)
        (long_map,), (large,) = r[0], r[1]
        self.assertGreaterEqual(max_running_after(fake, large[2], 0, before=long_map[2]), 6)
        self.assertTrue([p for p, s in releases(fake, 0) if s == large[3] and large[2] < p < long_map[2]])
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    def test_oversized_slot_keeps_its_memory_while_no_slot_waits(self):
        # mem_waiters: a slot frees memory it holds beyond its next map's needs only while another slot waits for
        # memory. With grow-only slots, 2 per GPU and a 9 GiB budget, a 6 GiB map and then 1 GiB maps always fit 2
        # at once, so no slot waits: the slot of the 6 GiB map keeps its memory instead of freeing it and growing
        # again, and the GPU frees nothing during the batch
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True)
        sched = self.make(fake, maps_per_gpu=2)
        params = [{"index": 0, "pixel": 6 * GiB, "time": 0.2}] + \
                 [{"index": i, "pixel": GiB, "time": 0.1} for i in range(1, 11)]
        self.assert_successes(collect(sched, params), 11)
        self.assertEqual(releases(fake, 0), [])
        self.assertEqual(sched.stats["devices"][0]["peak_committed"], 7 * GiB)

    # ---- running out of memory

    def test_oom_with_others_running_requeues_and_parks(self):
        # GPU 0 runs out of memory with more than 2 maps in progress. Each map that does frees its slot and goes
        # back to the front of the queue, and GPU 0 parks at least one slot each time. The first retries that
        # succeed give their slots back; when GPU 0 then runs out of memory again, it keeps its parked slots
        fake = FakeBackend(oom_above={0: 2})
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(30, time=0.05)), 30)
        dev = sched.stats["devices"][0]
        ooms = events(fake, "oom", 0)
        self.assertGreaterEqual(dev["oom_retries"], 1)
        self.assertEqual(dev["oom_retries"], len(ooms))
        self.assertLessEqual(dev["oom_retries"], 12)   # two rounds of at most 6 maps over the 2 that fit
        self.assertTrue(dev["keeps_parked_slots"])
        self.assertGreaterEqual(dev["parked_slots"], 6)
        self.assertFalse(sched.stats["devices"][1]["keeps_parked_slots"])
        self.assertEqual(sched.stats["devices"][1]["oom_retries"], 0)
        # the slot freed its memory before it did anything else
        log = fake.log_snapshot()
        for pos, slot_id in ooms:
            nxt = [e[0] for e in log[pos + 1:] if e[2] == slot_id and e[0] in ("start", "oom", "release")]
            self.assertEqual(nxt[0], "release", (pos, slot_id))

    def test_oom_retry_waits_for_fewer_maps_in_progress(self):
        # one GPU of 8 slots runs just 3 maps when map 2 runs out of memory. Its limit drops to 2 (not 7), so the
        # retry starts only once one of maps 0 and 1 has ended, never while both still run. Map 1 stays in its run
        # until the retry has started, so that it still runs then
        tries_of_2 = []
        always = []

        def fail(p):
            if p["index"] == 1 and not always:
                wait_for(lambda: len([e for e in fake.log_snapshot() if e[0] == "start" and e[3] == 2]) >= 2)
            if p["index"] != 2:
                return None
            tries_of_2.append(p["index"])
            if always or len(tries_of_2) == 1:
                time.sleep(0.02)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail)
        sched = self.make(fake)
        params = [{"index": 0, "time": 0.2}, {"index": 1, "time": 0.4}, {"index": 2, "time": 0.05}]
        self.assert_successes(collect(sched, params), 3)
        dev = sched.stats["devices"][0]
        self.assertEqual((dev["oom_retries"], dev["parked_slots"]), (1, 0))   # the retry gave its 6 slots back
        r = runs(fake)
        first, retry = r[2]
        alongside = [r[i][0] for i in (0, 1)]
        self.assertTrue(all(overlaps(o, first) for o in alongside))
        still_running = [o for o in alongside if o[1] < retry[1] < o[2]]
        self.assertEqual(len(still_running), 1)   # map 0 had ended, map 1 still ran
        self.assertLess(r[0][0][2], retry[1])
        # when map 2 always runs out, it fails alone in the end and the GPU gets back all 6 + 1 parked slots
        always.append(True)
        out = collect(sched, params)
        self.assert_successes(out, 3, failed={2})
        dev = sched.stats["devices"][0]
        self.assertGreaterEqual(dev["oom_retries"], 2)
        self.assertEqual(dev["parked_slots"], 0)

    def test_oom_while_shared_retries_with_fewer_maps(self):
        # parks: map 12 needs more memory than the other maps, but its estimate is the same as theirs, so it runs out
        # of memory, 20 ms into its run, when more than 4 other maps run beside it. Each such error parks at least
        # one of the GPU's 8 slots, so that the retry starts beside fewer maps: map 12 succeeds after at most 3
        # retries (beside at most 7, 6 and 5 others), and the GPU then gets its slots back
        beside = []   # how many other maps ran when each try of map 12 started

        def fail(p):
            if p["index"] != 12:
                return None
            with fake.lock:
                beside.append(fake.running[0] - 1)
            if beside[-1] > 4:
                time.sleep(0.02)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail)
        sched = self.make(fake)
        params = [{"index": i, "time": 0.1 + 0.01 * (i % 8)} for i in range(40)]
        self.assert_successes(collect(sched, params), 40)
        dev = sched.stats["devices"][0]
        tries = runs(fake)[12]
        self.assertEqual(dev["oom_retries"], len(tries) - 1)
        self.assertLessEqual(dev["oom_retries"], 3, beside)
        self.assertLessEqual(beside[-1], 4)
        self.assertEqual(dev["parked_slots"], 0)

    def test_parks_return_after_successful_retry(self):
        # map 5 runs out of memory on its first 3 tries that run next to other maps, so each of them parks slots of
        # the GPU, and then succeeds. The GPU gets those slots back and runs 8 maps at once again: maps 40-47 stay in
        # their runs until 8 run. The maps' times differ, so that they do not all end together and leave a retry to
        # run alone
        tries = []

        def fail(p):
            if 40 <= p["index"] < 48:
                wait_for(lambda: running_on(fake, 0) >= 8)
            if p["index"] != 5:
                return None
            with fake.lock:
                crowded = fake.running[0] > 1
                if crowded and len(tries) < 3:
                    tries.append(p["index"])
                    return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail)
        sched = self.make(fake)
        params = [{"index": i, "time": 0.1 + 0.01 * (i % 8)} for i in range(64)]
        self.assert_successes(collect(sched, params), 64)
        dev = sched.stats["devices"][0]
        self.assertEqual((dev["oom_retries"], dev["parked_slots"]), (len(tries), 0))
        self.assertGreaterEqual(len(tries), 1)
        success = runs(fake)[5][-1]
        self.assertEqual(max_running_after(fake, success[2], 0), 8)

    def test_parks_stay_once_the_gpu_runs_out_again_after_giving_them_back(self):
        # every map's estimate lets the GPU start 8 maps, but only 5 fit. The retries of the first maps that run out
        # of memory succeed and give the parked slots back; the GPU then runs out of memory beside other maps
        # again, goes back to the limit it had before and keeps it, rather than run out of memory again after each
        # successful retry. Each of the two rounds has at most 3 maps over the 5 that fit, so at most 6 errors.
        # Maps 70-77 stay in their runs until the GPU runs as many maps as its limit allows
        def hold(p):
            if 70 <= p["index"] < 78:
                wait_for(lambda: running_on(fake, 0) >= 8 - sched.stats["devices"][0]["parked_slots"])
            return None

        fake = FakeBackend(devices=1, oom_above={0: 5}, fail_run_when=hold)
        sched = self.make(fake)
        self.assert_successes(collect(sched, maps(100, time=0.1)), 100)
        dev = sched.stats["devices"][0]
        ooms = events(fake, "oom", 0)
        self.assertEqual(dev["oom_retries"], len(ooms))
        self.assertLessEqual(dev["oom_retries"], 6)
        self.assertTrue(dev["keeps_parked_slots"])
        self.assertGreaterEqual(dev["parked_slots"], 3)
        # the second half of the batch runs at the parked limit
        starts = events(fake, "start", 0)
        half = starts[len(starts) // 2][0]
        self.assertLess(ooms[-1][0], half)
        self.assertEqual(max_running_after(fake, half, 0), 8 - dev["parked_slots"])

    def test_errors_during_runs_after_a_give_back_keep_the_limit_from_before(self):
        # map 5 runs out of memory on its first 3 tries beside other maps, which parks slots, and then succeeds,
        # which gives them back. From then on only 5 maps fit: a map that starts beside 5 others runs out of memory
        # 20 ms into its run, so several maps admitted under the raised limit run out together. The first error
        # takes the GPU back to the limit it had before the give-back, and the errors of the maps admitted under
        # the raised limit do not lower it further: the GPU keeps that limit (or 5, if it was higher)
        state = {"tries": 0, "limit_before": None, "late": 0}

        def fail(p):
            with fake.lock:
                n = fake.running[0]
                if p["index"] == 5 and state["limit_before"] is None:
                    if n > 1 and state["tries"] < 3:
                        state["tries"] += 1
                        return MapFailed("out of memory", 2)
                    state["limit_before"] = 8 - sched.stats["devices"][0]["parked_slots"]
                    return None
                late = state["limit_before"] is not None and n > 5
                state["late"] += late
            if late:
                time.sleep(0.02)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail)
        sched = self.make(fake)
        params = [{"index": i, "time": 0.1 + 0.01 * (i % 8)} for i in range(40)]
        self.assert_successes(collect(sched, params), 40)
        dev = sched.stats["devices"][0]
        if state["late"] < 2:
            self.skipTest(f"{state['late']} map(s) ran out of memory after the give-back, where the test needs 2 or "
                          "more admitted under the raised limit: the slot threads started too few maps at once")
        self.assertTrue(dev["keeps_parked_slots"])
        self.assertEqual(8 - dev["parked_slots"], min(state["limit_before"], 5))

    def test_parks_return_after_a_retry_fails_with_another_error(self):
        # map 5 runs out of memory beside the other maps of the first round, which parks slots, and its pinned retry
        # then fails with another CUDA error on a GPU that still works. The map has failed, so the GPU gets the
        # parked slots back and runs 8 maps at once again: maps 30-37 stay in their runs until 8 run. Map 5's first
        # try stays in its run until 8 run, so that it runs out of memory beside other maps
        tries = []
        fake = FakeBackend(devices=1)
        first_round = hold_until_running(fake, 8)
        later_round = hold_until_running(fake, 8)

        def fail(p):
            if p["index"] == 5:
                with fake.lock:
                    tries.append(p["index"])
                    first = len(tries) == 1
                if first:
                    first_round(p)
                    return MapFailed("out of memory", 2)
                return MapFailed("invalid argument", 1)
            if 30 <= p["index"] < 38:
                wait_for(lambda: len(tries) >= 2)   # the retry has run
                later_round(p)
            return None

        fake.fail_run_when = fail
        sched = self.make(fake)
        out = collect(sched, maps(40, time=0.1))
        self.assert_successes(out, 40, failed={5})
        (err,) = [f.result for f in out if f.index == 5]
        self.assertEqual((err.cuda_error, str(err)), (1, "invalid argument"))
        dev = sched.stats["devices"][0]
        self.assertEqual((dev["oom_retries"], dev["parked_slots"]), (1, 0))
        first, retry = runs(fake)[5]
        self.assertEqual(max_running_after(fake, retry[2], 0), 8)

    def test_oom_alone_fails_map(self):
        # map 3 runs out of memory however few maps run next to it: it is retried with fewer maps in progress
        # until it runs alone with no memory held by the idle slots, and only then fails. Its GPU then gets
        # back the slots its retries parked
        fake, seen = self.always_oom(devices=1, index=3)
        sched = self.make(fake)
        out = collect(sched, maps(20, time=0.05))
        self.assert_successes(out, 20, failed={3})
        (err,) = [f.result for f in out if f.index == 3]
        self.assertEqual((err.index, err.cuda_error, str(err)), (3, 2, OOM_ALONE))
        self.assertIsInstance(err.__cause__, MapFailed)
        r = runs(fake)
        tries = r[3]
        dev = sched.stats["devices"][0]
        self.assertGreaterEqual(len(tries), 3)
        self.assertEqual(dev["oom_retries"], len(tries) - 1)
        self.assertEqual(dev["parked_slots"], 0)
        self.assertEqual([o for i, rs in r.items() if i != 3 for o in rs if overlaps(o, tries[-1])], [])
        self.assertEqual(seen[-1], MiB)   # the last try had the GPU to itself: only its own 1 MiB allocated
        self.assertGreater(seen[-2], MiB)   # the try before had idle slots holding memory, which was freed
        # every try freed its own slot
        for _, _, end, slot_id in tries:
            self.assertTrue([p for p, s in releases(fake, 0, after=end) if s == slot_id])

    def test_oom_on_every_run_fails_every_map_and_stops(self):
        # no map fits, and each one takes 20 ms to find out: every map is retried while others run, then fails
        # alone, and the batch still ends
        def oom(p):
            time.sleep(0.02)
            return MapFailed("out of memory", 2)

        fake = FakeBackend(devices=1, fail_run_when=oom)
        sched = self.make(fake)
        out = collect(sched, maps(12))
        self.assert_successes(out, 12, failed=set(range(12)))
        for f in out:
            self.assertEqual((f.result.cuda_error, str(f.result)), (2, OOM_ALONE))
        dev = sched.stats["devices"][0]
        r = runs(fake)
        self.assertGreaterEqual(dev["oom_retries"], 1)
        self.assertEqual(dev["oom_retries"], sum(len(rs) for rs in r.values()) - 12)
        self.assertEqual(dev["parked_slots"], 0)   # each map gave back the slots its retries parked
        everything = [o for rs in r.values() for o in rs]
        for i, rs in r.items():   # each map failed on a try that ran alone
            self.assertEqual([o for o in everything if o is not rs[-1] and overlaps(o, rs[-1])], [], i)

    def test_always_oom_map_stays_on_its_gpu_and_gives_slots_back(self):
        # map 15 always runs out of memory on 2 GPUs of 8 slots. It is the 16th map started, so it first runs next
        # to 7 others on its GPU. Its retries stay on that GPU, the other GPU keeps all 8 slots, and the first GPU
        # gets its parked slots back once map 15 fails. Once map 15 has run, the next 8 maps on the other GPU stay
        # in their runs until 8 run there, and so do the next 8 on map 15's GPU once the consumer has its error
        fake, seen = self.always_oom(index=15)
        oom = fake.fail_run_when
        first = []    # the GPU of map 15's first try
        failed = []   # not empty once the consumer has map 15's error
        held = {0: 0, 1: 0}

        def fail(p):
            g = gpu_of_thread()
            with fake.lock:
                if p["index"] == 15 and not first:
                    first.append(g)
                hold = p["index"] != 15 and bool(first) and (g != first[0] or bool(failed)) and held[g] < 8
                held[g] += hold
            if hold:
                wait_for(lambda: running_on(fake, g) >= 8)
            return oom(p)

        fake.fail_run_when = fail
        sched = self.make(fake)
        out = []
        for f in sched.results(maps(160, time=0.1)):
            if f.index == 15:
                failed.append(True)
            out.append(f)
            f.release()
        self.assert_successes(out, 160, failed={15})
        tries = runs(fake)[15]
        (d,) = {t[0] for t in tries}   # every try on the same GPU
        other = 1 - d
        (err,) = [f.result for f in out if f.index == 15]
        self.assertEqual((err.cuda_error, str(err)), (2, f"out of device memory with no other map on GPU {d}"))
        self.assertGreaterEqual(len(tries), 3)
        dev, odev = sched.stats["devices"][d], sched.stats["devices"][other]
        self.assertEqual(dev["oom_retries"], len(tries) - 1)
        self.assertEqual(dev["parked_slots"], 0)
        self.assertEqual((odev["oom_retries"], odev["parked_slots"]), (0, 0))
        self.assertEqual(max_running_after(fake, tries[0][1], other), 8)
        self.assertEqual(seen[-1], MiB)
        # with its slots back, the first GPU runs 8 maps at once again
        self.assertEqual(max_running_after(fake, tries[-1][2], d), 8)

    def test_oom_retry_moves_when_its_gpu_breaks(self):
        # 3 slots per GPU: a map runs out of memory on GPU 1 next to the 2 maps running there, and its retry waits
        # for GPU 1. Those 2 maps then break GPU 1, once that map has run out of memory: the waiting retry runs on
        # GPU 0. With 3 slots no other slot of GPU 1 can start a map, and run out of memory too, before the error is
        # classified
        def fail(p):
            if gpu_of_thread() == 1:
                wait_for(lambda: events(fake, "oom", 1))
                time.sleep(0.05)
                fake.device_errors[1] = 700
                return MapFailed("illegal address", 700)
            return None

        fake = FakeBackend(oom_above={1: 2}, fail_run_when=fail)
        sched = self.make(fake, maps_per_gpu=3)
        out = collect(sched, maps(40, time=0.05))
        errors = [f for f in out if isinstance(f.result, BatchMapError)]
        self.assertEqual([str(f.result) for f in errors], [gpu_failed(1)])
        self.assert_successes(out, 40, failed={errors[0].index})
        retried = {e[3] for e in fake.log_snapshot() if e[0] == "oom"}
        self.assertEqual(len(retried), 1)   # the limit, lowered for the retry, kept GPU 1 from taking other maps
        self.assertNotIn(errors[0].index, retried)
        self.assertEqual({f.result.device for f in out if f.index in retried}, {0})
        self.assertEqual(len(events(fake, "start", 1)), 2)
        self.assertTrue(sched.stats["devices"][1]["dead"])

    def test_oom_retry_moves_when_its_gpu_has_no_usable_slot(self):
        # both slots of GPU 1 run out of memory, beside each other, and then cannot free it. GPU 1 still works but
        # has no slot left, so the two retries run on GPU 0
        def fail(p):
            if gpu_of_thread() == 1:
                wait_for(lambda: running_on(fake, 1) >= 2)
                time.sleep(0.05)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(fail_run_when=fail,
                           fail_release_when=lambda s: RuntimeError("cannot free") if s.device == 1 else None)
        sched = self.make(fake, maps_per_gpu=2)
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, maps(8, time=0.05))
        self.assert_successes(out, 8)
        self.assertEqual({f.result.device for f in out}, {0})
        self.assertEqual(len(events(fake, "release_failed", 1)), 2)
        self.assertFalse(sched.stats["devices"][1]["dead"])

    def test_retry_that_is_not_pinned_parks_no_slots(self):
        # both slots of GPU 1 run out of memory beside each other and cannot free it, and each error is classified
        # only once both slots are broken: GPU 1 has no usable slot left, so the retries are not pinned and run on
        # GPU 0. They must count no slots of GPU 1 as parked, which GPU 0 would get "back" when they succeed there
        completed = []
        stuck = []    # the slots of GPU 1 whose error was classified after 10 s without both slots broken

        class Recording(Scheduler):
            def _classify(self, run, st, job, stage, exc, device_error, freed_idle=False):
                give_up = time.monotonic() + 10
                while st.device == 1 and not all(x.broken for x in self._slots[1][:2]):
                    if time.monotonic() > give_up:
                        stuck.append(st.number)
                        break
                    self._cond.wait(0.01)
                super()._classify(run, st, job, stage, exc, device_error, freed_idle)

            def _complete(self, run, st, job, outcome):
                with self._cond:
                    completed.append((job.index, st.device, run.parks[job.index]))
                super()._complete(run, st, job, outcome)

        def fail(p):
            if gpu_of_thread() == 1:
                wait_for(lambda: running_on(fake, 1) >= 2)   # the two maps of GPU 1 run beside each other
                time.sleep(0.05)
                return MapFailed("out of memory", 2)
            return None

        fake = FakeBackend(fail_run_when=fail,
                           fail_release_when=lambda s: RuntimeError("cannot free") if s.device == 1 else None)
        sched = Recording(fake, maps_per_gpu=2)
        self.addCleanup(self.check_fake, fake)
        self.addCleanup(sched.close)
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, maps(8, time=0.05))
        self.assertEqual(stuck, [], "an error on GPU 1 waited 10 s for both of its slots to break, and was then "
                                    "classified with a slot of GPU 1 still usable")
        self.assert_successes(out, 8)
        self.assertEqual(sched.stats["devices"][1]["oom_retries"], 2)
        self.assertEqual([c for c in completed if c[2]], [])

    def test_oom_alone_frees_idle_slots_before_failing(self):
        # grow-only slots keep 2 GiB each after a first batch. A map whose estimate fits next to that memory
        # runs alone and runs out of memory while the other slots still hold it: the idle slots' memory is freed
        # and the map tried again, alone, before it can fail
        def fail(p):
            if p.get("oom") == "crowded":
                with fake.lock:
                    crowded = fake.allocated(0) > 3 * GiB
                return MapFailed("out of memory", 2) if crowded else None
            return MapFailed("out of memory", 2) if p.get("oom") == "always" else None

        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB, grow_only=True, fail_run_when=fail)
        sched = self.make(fake, maps_per_gpu=4)
        warm = [{"index": i, "pixel": 2 * GiB, "time": 0.1} for i in range(4)]

        def warm_up():
            self.assert_successes(collect(sched, warm), 4)
            self.assertEqual(len({runs(fake)[i][-1][3] for i in range(4)}), 4)
            with fake.lock:
                self.assertEqual(fake.allocated(0), 8 * GiB)

        warm_up()
        (f,) = collect(sched, [{"index": 10, "pixel": GiB, "oom": "crowded"}])
        self.assertIsInstance(f.result, FakeResult)
        tries = runs(fake)[10]
        self.assertEqual(len(tries), 2)
        freed = {s for p, s in releases(fake, 0, after=tries[0][2]) if p < tries[1][1]}
        self.assertEqual(len(freed), 4)   # its own slot and the 3 idle ones
        self.assertEqual((sched.stats["devices"][0]["oom_retries"], sched.stats["devices"][0]["parked_slots"]), (1, 0))

        warm_up()
        (f,) = collect(sched, [{"index": 11, "pixel": GiB, "oom": "always"}])
        self.assertEqual((f.result.cuda_error, str(f.result)), (2, OOM_ALONE))
        tries = runs(fake)[11]
        self.assertEqual(len(tries), 2)   # once with the idle slots' memory, once without
        freed = {s for p, s in releases(fake, 0, after=tries[0][2]) if p < tries[1][1]}
        self.assertEqual(len(freed), 4)

    def test_check_device_blocks_admission_on_its_gpu(self):
        # map 0 fails with a CUDA error after 0.05 s and checking its GPU takes 0.3 s. Map 1 ends during the
        # check, but no map starts on that GPU until the check is over
        def fail(p):
            if p["index"] == 0:
                time.sleep(0.05)
                return MapFailed("invalid argument", 1)
            return None

        fake = FakeBackend(devices=1, fail_run_when=fail, check_time=0.3)
        sched = self.make(fake, maps_per_gpu=2)
        params = [{"index": 0}, {"index": 1, "time": 0.2}] + [{"index": i, "time": 0.02} for i in range(2, 8)]
        self.assert_successes(collect(sched, params), 8, failed={0})
        ((began, _),) = events(fake, "check")
        ((ended, _),) = events(fake, "checked")
        self.assertTrue(began < runs(fake)[1][0][2] < ended)
        self.assertEqual([p for p, _ in events(fake, "start") if began < p < ended], [])
        self.assertFalse(sched.stats["devices"][0]["dead"])

    def test_release_when_running_alone(self):
        # four slots hold 2 GiB each after maps 0-3. Map 4 needs 6 GiB of a 9 GiB budget, so it waits until
        # nothing runs; then every idle slot's memory is freed before it starts
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB)
        sched = self.make(fake, maps_per_gpu=4)
        params = maps(4, pixel=2 * GiB, time=0.1) + [{"index": 4, "pixel": 6 * GiB, "time": 0.1}] + \
            [{"index": i, "pixel": GiB, "time": 0.05} for i in range(5, 8)]
        self.assert_successes(collect(sched, params), 8)
        r = runs(fake)
        (big,) = r[4]
        holders = sorted(r[i][0][3] for i in range(4))
        self.assertEqual(len(set(holders)), 4)
        self.assertTrue(all(r[i][0][2] < big[1] for i in range(4)))
        self.assertEqual(sorted(slot_id for pos, slot_id in releases(fake, 0) if pos < big[1]), holders)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)

    # ---- broken GPUs

    def test_dead_device_moves_maps(self):
        # GPU 1 breaks at its 4th map. The map that reports the error first fails; the others in progress on
        # GPU 1 go back to the queue and run on GPU 0, and GPU 1 is not used again
        fake = FakeBackend(dead_after={1: 3},
                           fail_close_when=lambda s: RuntimeError("broken") if s.device == 1 else None)
        sched = self.make(fake)
        out = collect(sched, maps(40, time=0.05))
        errors = [f for f in out if isinstance(f.result, BatchMapError)]
        self.assertEqual(len(errors), 1)
        trigger = errors[0]
        self.assertEqual((trigger.result.cuda_error, str(trigger.result)), (700, gpu_failed(1)))
        self.assert_successes(out, 40, failed={trigger.index})
        ((death, _),) = events(fake, "dead")
        r = runs(fake)
        for f in out:
            if f.index != trigger.index and f.result.device != 0:
                (only,) = r[f.index]
                self.assertLess(only[2], death, f.index)   # succeeded on GPU 1 only before it broke
        self.assertTrue(all(len(rs) <= 2 for rs in r.values()))   # moved at most once
        late = [slot_id for pos, slot_id in events(fake, "start", 1) if pos > death]
        self.assertEqual(len(late), len(set(late)))   # each slot of GPU 1 started at most one map after it broke
        self.assertTrue(sched.stats["devices"][1]["dead"])
        self.assertFalse(sched.stats["devices"][0]["dead"])
        self.assertEqual(sum(d["maps"] for d in sched.stats["devices"].values()), 39)
        # closing a slot of the broken GPU raises: close() warns and closes every slot once
        with self.assertWarns(RuntimeWarning):
            sched.close()
        self.assertEqual([s.close_calls for s in fake.slots], [1] * len(fake.slots))

    def test_all_devices_dead_fails_remaining(self):
        fake = FakeBackend(devices=1, dead_after={0: 2})
        sched = self.make(fake)
        out = collect(sched, maps(10, time=0.3))
        self.assert_successes(out, 10, failed=set(range(10)))
        messages = [str(f.result) for f in out]
        self.assertEqual(messages.count(gpu_failed(0)), 1)
        self.assertEqual(messages.count("no usable GPU left: " + gpu_failed(0)), 9)
        self.assertEqual({f.result.cuda_error for f in out}, {700})
        self.assertTrue(sched.stats["devices"][0]["dead"])
        # the GPU stays out: a later batch fails at once without running anything
        started = len(events(fake, "start"))
        out = collect(sched, maps(3))
        self.assert_successes(out, 3, failed={0, 1, 2})
        self.assertEqual({(str(f.result), f.result.cuda_error) for f in out},
                         {("no usable GPU left: " + gpu_failed(0), 700)})
        self.assertEqual(len(events(fake, "start")), started)
        self.assertTrue(sched.stats["devices"][0]["dead"])

    def test_open_slot_failure_marks_device_dead(self):
        fake = FakeBackend(open_fails_for=(1,))
        sched = self.make(fake)
        out = collect(sched, maps(20, time=0.02))
        self.assert_successes(out, 20)
        self.assertEqual({f.result.device for f in out}, {0})
        self.assertEqual(events(fake, "start", 1), [])
        self.assertTrue(all(len(rs) == 1 for rs in runs(fake).values()))
        self.assertTrue(sched.stats["devices"][1]["dead"])
        self.assertEqual(sched.stats["devices"][0]["maps"], 20)
        tries = len(events(fake, "open_failed"))
        self.assertGreaterEqual(tries, 1)
        self.assertLessEqual(tries, 8)
        # GPU 1 stays out of the batch: the next call does not try to open it again
        out = collect(sched, maps(10, time=0.02))
        self.assert_successes(out, 10)
        self.assertEqual({f.result.device for f in out}, {0})
        self.assertEqual(len(events(fake, "open_failed")), tries)
        self.assertTrue(sched.stats["devices"][1]["dead"])

    def test_map_moves_once_then_fails(self):
        # one slot per GPU and no GPU can open one: the map moves from the first GPU to the second, then fails
        # with the second GPU's error
        fake = FakeBackend(open_fails_for=(0, 1))
        sched = self.make(fake, maps_per_gpu=1)
        (f,) = collect(sched, maps(1))
        tried = events(fake, "open_failed")
        self.assertEqual(len(tried), 2)
        devices = [fake.log_snapshot()[pos][1] for pos, _ in tried]
        self.assertEqual(sorted(devices), [0, 1])
        self.assertIsInstance(f.result, BatchMapError)
        self.assertEqual((f.result.cuda_error, str(f.result)),
                         (0, f"GPU {devices[1]} failed (0): cannot open a slot on fake GPU {devices[1]}"))
        self.assertTrue(all(sched.stats["devices"][d]["dead"] for d in (0, 1)))

    def test_every_map_moves_at_most_once_when_every_gpu_breaks(self):
        fake = FakeBackend(fail_run_when=lambda p: MapFailed("illegal address", 700), device_errors={0: 700, 1: 700})
        sched = self.make(fake, maps_per_gpu=2)
        out = collect(sched, maps(12, time=0.05))
        self.assert_successes(out, 12, failed=set(range(12)))
        self.assertTrue(all(len(rs) <= 2 for rs in runs(fake).values()))
        broken = {gpu_failed(0), gpu_failed(1)}
        firsts = set()
        for f in out:
            message = str(f.result)
            self.assertEqual(f.result.cuda_error, 700)
            if message.startswith("no usable GPU left: "):
                firsts.add(message[len("no usable GPU left: "):])
            else:
                self.assertIn(message, broken)
        self.assertLessEqual(len(firsts), 1)
        self.assertTrue(firsts <= broken)
        self.assertTrue(all(sched.stats["devices"][d]["dead"] for d in (0, 1)))

    def test_cuda_error_on_a_working_gpu_fails_only_its_map(self):
        fake = FakeBackend(fail_run_when=lambda p: MapFailed("invalid argument", 1) if p["index"] == 3 else None)
        sched = self.make(fake)
        out = collect(sched, maps(20, time=0.01))
        self.assert_successes(out, 20, failed={3})
        (err,) = [f.result for f in out if f.index == 3]
        self.assertEqual((err.cuda_error, str(err)), (1, "invalid argument"))
        self.assertEqual(len(events(fake, "check")), 1)
        self.assertFalse(any(d["dead"] for d in sched.stats["devices"].values()))

    # ---- release() failures

    def test_failed_release_keeps_memory_counted(self):
        # slot 0 cannot free its 2 GiB when map 4 has to run alone. The GPU still works, so slot 0 is not used
        # again and its memory still counts: the 3 GiB maps after map 4 must not run on top of it
        fake = FakeBackend(devices=1, free=10 * GiB, total=10 * GiB,
                           fail_release_when=lambda s: RuntimeError("cannot free") if s.slot_id == 0 else None)
        sched = self.make(fake, maps_per_gpu=4)
        params = maps(4, pixel=2 * GiB, time=0.1) + [{"index": 4, "pixel": 6 * GiB, "time": 0.1}] + \
            [{"index": i, "pixel": 3 * GiB, "time": 0.05} for i in range(5, 11)]
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, params)
        self.assert_successes(out, 11)
        self.assertLessEqual(fake.peak_alloc[0], 9 * GiB)
        ((failed_at, _),) = events(fake, "release_failed")
        self.assertTrue([p for p, _ in events(fake, "check", 0) if p > failed_at])
        self.assertEqual([p for p, slot_id in events(fake, "start") if slot_id == 0 and p > failed_at], [])
        self.assertFalse(sched.stats["devices"][0]["dead"])

    def test_failed_release_on_a_broken_gpu_takes_it_out(self):
        # a slot of GPU 1 runs out of memory and cannot free it because GPU 1's context broke: GPU 1 leaves the
        # batch and its map runs on GPU 0. Maps already in progress on GPU 1 may still finish there
        fake = FakeBackend(oom_above={1: 2})

        def fail(slot):
            if slot.device != 1:
                return None
            fake.device_errors[1] = 700
            return MapFailed("illegal address", 700)

        fake.fail_release_when = fail
        sched = self.make(fake)
        with self.assertWarns(RuntimeWarning):
            out = collect(sched, maps(30, time=0.05))
        self.assert_successes(out, 30)
        self.assertTrue(sched.stats["devices"][1]["dead"])
        self.assertEqual(sched.stats["devices"][1]["oom_retries"], 0)
        failed_at = events(fake, "release_failed", 1)[0][0]
        late = [slot_id for pos, slot_id in events(fake, "start", 1) if pos > failed_at]
        self.assertEqual(len(late), len(set(late)))
        moved = {e[3] for e in fake.log_snapshot() if e[0] == "oom"}
        self.assertTrue(moved)
        self.assertTrue(all(f.result.device == 0 for f in out if f.index in moved))

    # ---- other failures

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

    def hold_until_stopped(self, fake):
        """Make each map from index 2 on stay in its run until the consumer has stopped the batch, so that the
        consumer gets maps 0 and 1 first and stops while maps are still to start, however slow the threads."""
        def hold(p):
            if p["index"] >= 2:
                run = fake.sched._run
                wait_for(lambda: run is None or run.stopping)
            return None

        fake.fail_run_when = hold

    def test_break_stops_cleanly(self):
        fake = FakeBackend()
        sched = fake.sched = self.make(fake)
        self.hold_until_stopped(fake)
        gen = sched.results(maps(50, time=0.05))
        for _ in range(2):
            next(gen).release()
        with self.assertRaises(RuntimeError):
            next(sched.results(maps(1)))
        gen.close()
        fake.fail_run_when = None
        self.assert_stopped_cleanly(fake, sched, 50)

    def test_exception_in_consumer_stops_cleanly(self):
        fake = FakeBackend()
        sched = fake.sched = self.make(fake)
        self.hold_until_stopped(fake)
        with self.assertRaises(ValueError):
            for k, f in enumerate(sched.results(maps(50, time=0.05))):
                f.release()
                if k == 1:
                    raise ValueError("consumer failed")
        fake.fail_run_when = None
        self.assert_stopped_cleanly(fake, sched, 50)

    def test_break_while_results_wait_in_slots_stops_cleanly(self):
        # a 1-byte limit lets one 1 GiB result wait at a time: the consumer stops after the first map while the
        # other 3 finished maps still wait on the GPU for room. They are never copied to the host, and their slots
        # free their memory. The first 4 maps stay in their runs until all 4 run, so that none is collected before
        # all have started
        fake = FakeBackend(devices=1)
        fake.fail_run_when = hold_until_running(fake, 4)
        sched = self.make(fake, maps_per_gpu=4, max_result_bytes=1)
        gen = sched.results(maps(10, result=GiB, time=0.1))
        f = next(gen)
        self.wait_for_ends(fake, range(4))
        self.assertEqual(len(events(fake, "start")), 4)   # no room for another result, so no other map starts
        self.assertEqual(len(events(fake, "end")), 4)
        self.assertEqual(len(events(fake, "collect")), 1)
        self.assertEqual(sched.stats["peak_result_bytes"], GiB)
        self.assertEqual(fake.peak_host_bytes, GiB)
        gen.close()
        f.release()
        self.assertEqual(len(events(fake, "collect")), 1)
        self.assertEqual(sorted(s for _, s in events(fake, "release")),
                         sorted(e[2] for e in fake.log_snapshot() if e[0] == "end" and e[3] != f.index))
        self.assert_stopped_cleanly(fake, sched, 10)

    def test_interrupt_while_waiting_stops_cleanly(self):
        if threading.current_thread() is threading.main_thread():
            # interrupt_main() does nothing while SIGINT is ignored, as it is in a process that a script starts in
            # the background
            self.addCleanup(signal.signal, signal.SIGINT, signal.signal(signal.SIGINT, signal.default_int_handler))
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
