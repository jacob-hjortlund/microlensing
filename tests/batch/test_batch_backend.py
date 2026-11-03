"""Tests of the ctypes backend's failure handling, with the real library and a slot whose set_worker and run
succeed without a GPU. These tests need no GPU: python -m unittest tests.batch.test_batch_backend -v"""
import ctypes
import gc
import sys
import unittest
import weakref
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM import lib_ipm  # noqa: E402
from microlensing.IPM.batch import BatchedIPM  # noqa: E402
from microlensing.IPM.batch_backend import CtypesBackend  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError, DeviceInfo, MapFailed, MemoryEstimate  # noqa: E402
from needs_exports import NEEDS_EXPORTS  # noqa: E402

GiB = 1 << 30
MAP = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=5, half_length_y2=5, num_pixels_y1=100,
           num_pixels_y2=100)


class _HostLib:
    """The real library, except that set_worker and run succeed without touching a GPU. A worker that "ran"
    this way has no stars, so collecting its map raises in IPM._from_worker (its snapshot is null) with no CUDA
    error. host_histogram_bytes, when set, is what get_host_histogram_bytes reports for every worker."""

    def __init__(self, real):
        self._real = real
        self.host_histogram_bytes = None

    def get_host_histogram_bytes(self, obj):
        if self.host_histogram_bytes is None:
            return self._real.get_host_histogram_bytes(obj)
        return self.host_histogram_bytes

    def __getattr__(self, name):
        return getattr(self._real, name)

    def set_worker(self, obj, device):
        return True

    def run(self, obj, verbose):
        return True


class HostOnlyBackend(CtypesBackend):
    """A CtypesBackend with one fake GPU that keeps every slot it opens."""

    def __init__(self):
        super().__init__(False, 0)
        self.lib = _HostLib(self.lib)
        self.slots = []

    def device_count(self):
        return 1

    def device_info(self, device):
        return DeviceInfo(free=40 * GiB, total=40 * GiB, concurrent_managed=True)

    def check_device(self, device):
        return 0

    def open_slot(self, device):
        slot = super().open_slot(device)
        self.slots.append(slot)
        return slot


@NEEDS_EXPORTS
class FailedMapReleaseTests(unittest.TestCase):
    def run_failing_map(self, backend):
        """Run one map that fails in its slot (in run() or collect()); return (the BatchMapError, a weakref to the
        slot's worker) after close(), with the error still held."""
        with mock.patch("microlensing.IPM.batch._make_backend", return_value=backend):
            b = BatchedIPM(**MAP, num_maps=1)
        out = dict(b.run())
        self.assertEqual(len(backend.slots), 1)
        worker = weakref.ref(backend.slots[0].ipm)
        b.close()
        gc.collect()
        return out[0], worker

    def test_failed_copy_frees_the_worker_on_close(self):
        error, worker = self.run_failing_map(HostOnlyBackend())
        self.assertIsInstance(error, BatchMapError)
        self.assertIn("copying the finished map from GPU 0 failed: Error copying the IPM state", str(error))
        self.assertEqual(error.cuda_error, 0)
        self.assertIsInstance(error.__cause__, MapFailed)
        self.assertIsNotNone(error.__cause__.__cause__)   # the original exception stays the cause
        self.assertIsNone(worker(), "the error handed to the user keeps the worker (and its GPU memory) alive")

    def test_rejected_inputs_free_the_worker_on_close(self):
        backend = HostOnlyBackend()
        # the estimate passes, so that the slot's own _apply_inputs raises the ValueError
        backend.estimate = lambda params: MemoryEstimate(pixel=1 << 20, result=1 << 20)
        with mock.patch.dict(MAP, m_lower=2.0, m_upper=1.0):
            error, worker = self.run_failing_map(backend)
        self.assertIsInstance(error, BatchMapError)
        self.assertIn("m_lower must be <= m_upper", str(error))
        self.assertIsNone(worker(), "the error handed to the user keeps the worker (and its GPU memory) alive")

    def test_rejected_inputs_raise_in_the_constructor(self):
        with mock.patch("microlensing.IPM.batch._make_backend", return_value=HostOnlyBackend()):
            with self.assertRaisesRegex(ValueError, "map 1: m_lower must be <= m_upper"):
                BatchedIPM(**dict(MAP, m_lower=[1.0, 2.0], m_upper=1.0), num_maps=2)

    def test_collect_takes_only_the_last_run_once(self):
        # collect() takes the handle of the slot's last run, once (the GPU tests check that release() frees it)
        backend = HostOnlyBackend()
        slot = backend.open_slot(0)
        try:
            with self.assertRaisesRegex(MapFailed, "no finished map"):
                slot.collect(1, True)   # nothing ran
            first = slot.run(dict(MAP, random_seed=7))
            second = slot.run(dict(MAP, random_seed=7))
            self.assertEqual(second.map_bytes, 100 * 100 * 4)
            with self.assertRaisesRegex(MapFailed, "no finished map"):
                slot.collect(first.handle, True)   # the second run replaced it
            with self.assertRaisesRegex(MapFailed, "copying the finished map"):
                slot.collect(second.handle, True)   # this worker has no stars, so its snapshot fails
            with self.assertRaisesRegex(MapFailed, "no finished map"):
                slot.collect(second.handle, True)
        finally:
            slot.close()

    def test_result_bytes_count_the_stars_twice_and_the_histograms(self):
        # a result's host bytes are its maps, two copies of its stars (its snapshot's and its numpy array, 3 values
        # each) and the host copies of its histograms (which the worker reports after its run), whether or not its
        # maps are collected
        backend = HostOnlyBackend()
        backend.lib.host_histogram_bytes = 3 * 123_456
        slot = backend.open_slot(0)
        try:
            out = slot.run(dict(MAP, random_seed=7, write_histograms=True))
            self.assertGreater(slot.ipm.num_stars, 0)
            stars = 2 * slot.ipm.num_stars * 3 * 4
            self.assertEqual(out.map_bytes, 100 * 100 * 4)
            self.assertEqual(out.result_bytes, 100 * 100 * 4 + stars + 3 * 123_456)
        finally:
            slot.close()

    def test_failed_estimate_keeps_no_frames_of_the_probe(self):
        backend = HostOnlyBackend()
        with self.assertRaises(MapFailed) as cm:
            backend.estimate(dict(MAP, m_lower=2.0, m_upper=1.0))
        tb = cm.exception.__cause__.__traceback__
        while tb is not None:   # the frames below estimate() (the probe's _apply_inputs and setters) are cleared
            if tb.tb_frame.f_code.co_name != "estimate":
                self.assertEqual(tb.tb_frame.f_locals, {}, tb.tb_frame.f_code.co_name)
            tb = tb.tb_next

    def test_probe_uses_the_first_device(self):
        # a star file's estimate reads its stars on the probe's GPU: the batch's first, or GPU 0 when devices is None
        for devices, expected in (([3, 1], 3), (None, 0)):
            with self.subTest(devices=devices):
                with mock.patch.object(lib_ipm.lib, "set_device", create=True) as set_device:
                    b = BatchedIPM(**MAP, num_maps=2, seed_start=1, devices=devices)
                self.assertIsInstance(b._backend, CtypesBackend)
                set_device.assert_called_once_with(b._backend._probe.obj, expected)


if __name__ == "__main__":
    unittest.main()
