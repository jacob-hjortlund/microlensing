"""GPU tests of the worker, estimate and snapshot exports (skipped without a GPU):
python -m unittest tests.batch.test_exports_gpu -v"""
import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM import lib_ipm, lib_ipm_double  # noqa: E402
import test_exports as te  # noqa: E402
from needs_exports import NEEDS_EXPORTS  # noqa: E402

LIBS = (("float", lib_ipm.lib), ("double", lib_ipm_double.lib))
HAS_GPU = lib_ipm.lib.get_device_count() > 0 if hasattr(lib_ipm.lib, "get_device_count") else False

# the four maps that IPM<T>::tree_bytes_per_star was measured on (include/ipm.cuh)
LENS = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=25, half_length_y2=25)
CONFIGS = {
    "lens_1000": dict(LENS, num_pixels_y1=1000, num_pixels_y2=1000),
    "lens_5000": dict(LENS, num_pixels_y1=5000, num_pixels_y2=5000),
    "rectangular_1000": dict(kappa_tot=0.5, shear=0.49, kappa_star=0.1, rectangular=1, approx=1,
                             half_length_y1=12.5, half_length_y2=12.5, num_pixels_y1=1000, num_pixels_y2=1000),
    "wide_5000": dict(kappa_tot=0.3, shear=0.3, kappa_star=0.27, half_length_y1=505, half_length_y2=505,
                      num_pixels_y1=5000, num_pixels_y2=5000),
}

# high-magnification maps, run with seed 3. hm1000_like (mean magnification about 960) has a magnification range
# of about 2e3. near_critical moves the shear closer to the critical value (mean magnification about 4000): with
# parities its range is about 5300, so each histogram takes about 25 MiB with headroom, about half of what
# the estimate allows (a range of 1e4)
HM_LENS = dict(kappa_tot=0.7252, kappa_star=0.1, rectangular=1,
               half_length_y1=12.5, half_length_y2=12.5, num_pixels_y1=1000, num_pixels_y2=1000)
HIGH_MAG = (
    ("hm1000_like", dict(HM_LENS, shear=0.2729, approx=1), 0),
    ("hm1000_like", dict(HM_LENS, shear=0.2729, approx=1), 1),
    ("near_critical", dict(HM_LENS, shear=0.27435, approx=0), 1),
)


def magnification_range(lib, obj, write_parities):
    """max - min of the magnifications of the last run's maps, rounded to the histograms' 1/1000 steps."""
    n = lib.get_num_pixels_y1(obj) * lib.get_num_pixels_y2(obj)
    getters = (lib.get_pixels, lib.get_pixels_minima, lib.get_pixels_saddles) if write_parities else (lib.get_pixels,)
    maps = [np.ctypeslib.as_array(get(obj), shape=(n,)) for get in getters]
    lo = min(round(float(m.min()) * 1000) for m in maps)
    hi = max(round(float(m.max()) * 1000) for m in maps)
    return (hi - lo) / 1000


def host_result_bytes(lib, obj, itemsize, write_parities):
    """The host bytes of a result of the last run: its maps, two copies of its stars (3 values each: the snapshot's
    and the numpy copy that IPM._from_worker makes) and the host copies of its histograms that a snapshot holds."""
    n = lib.get_num_pixels_y1(obj) * lib.get_num_pixels_y2(obj)
    return (n * (3 if write_parities else 1) + 2 * lib.get_num_stars(obj) * 3) * itemsize \
        + lib.get_host_histogram_bytes(obj)


def apply(lib, obj, params, write_parities):
    """Set the inputs of params on obj, with histograms and with or without parities."""
    for name, value in params.items():
        getattr(lib, "set_" + name)(obj, value)
    lib.set_write_histograms(obj, 1)
    lib.set_write_parities(obj, write_parities)


@NEEDS_EXPORTS
@unittest.skipUnless(HAS_GPU, "no GPU visible")
class ExportGpuTests(unittest.TestCase):
    def test_estimate_bounds_allocation(self):
        names = ("pixel", "star", "tree")
        for label, lib in LIBS:
            for name, params in CONFIGS.items():
                for parities in (0, 1):
                    for seed in (3, 4, 5):
                        with self.subTest(precision=label, config=name, parities=parities, seed=seed):
                            obj = lib.IPM_init()
                            try:
                                self.assertTrue(lib.set_worker(obj, 0))
                                apply(lib, obj, params, parities)
                                lib.set_random_seed(obj, seed)
                                est = (ctypes.c_ulonglong * 4)()
                                alloc = (ctypes.c_ulonglong * 3)()
                                self.assertTrue(lib.estimate_memory(obj, est))
                                self.assertTrue(lib.run(obj, 0))
                                lib.get_allocated_bytes(obj, alloc)
                                for k in range(3):
                                    self.assertGreaterEqual(
                                        est[k], alloc[k],
                                        f"{names[k]}: est={est[k]} alloc={alloc[k]} ratio={est[k] / alloc[k]:.4f}")
                                # the histogram term is a fixed bound, far above these maps' histograms, so the
                                # estimate's upper bound is checked without it: the term itself is checked
                                # exactly, and the histograms the run allocated (the pixel bytes beyond the maps
                                # and the step buffer) must fit into it
                                n_maps = 3 if parities else 1
                                itemsize = 4 if label == "float" else 8
                                map_bytes = params["num_pixels_y1"] * params["num_pixels_y2"] * itemsize
                                hist_est = n_maps * te.HIST_BYTES_PER_KIND
                                self.assertEqual(est[0], map_bytes * (n_maps + 1) + hist_est)
                                hist_alloc = alloc[0] - map_bytes * (n_maps + lib.get_perimeter_deposit_used(obj))
                                self.assertLessEqual(hist_alloc, hist_est)
                                rest_est, rest_alloc = sum(est[:3]) - hist_est, sum(alloc) - hist_alloc
                                self.assertLessEqual(
                                    rest_est, 1.5 * rest_alloc + (64 << 20),
                                    f"total without the histograms: est={rest_est} alloc={rest_alloc} "
                                    f"ratio={rest_est / rest_alloc:.4f}")
                                # the result's host bytes, its histograms included, are within the estimate
                                hist_host = lib.get_host_histogram_bytes(obj)
                                self.assertGreater(hist_host, 0)
                                self.assertLessEqual(hist_host, hist_alloc)
                                self.assertGreaterEqual(est[3], host_result_bytes(lib, obj, itemsize, parities))
                                print(f"ratios {label} {name} p{parities} s{seed}: "
                                      + " ".join(f"{names[k]}={est[k] / alloc[k]:.3f}" for k in range(3)))
                            finally:
                                lib.IPM_delete(obj)

    def test_estimate_bounds_high_magnification_allocation(self):
        names = ("pixel", "star", "tree")
        for label, lib in LIBS:
            for name, params, parities in HIGH_MAG:
                with self.subTest(precision=label, config=name, parities=parities):
                    obj = lib.IPM_init()
                    try:
                        self.assertTrue(lib.set_worker(obj, 0))
                        apply(lib, obj, params, parities)
                        lib.set_random_seed(obj, 3)
                        est = (ctypes.c_ulonglong * 4)()
                        alloc = (ctypes.c_ulonglong * 3)()
                        self.assertTrue(lib.estimate_memory(obj, est))
                        self.assertTrue(lib.run(obj, 0))
                        lib.get_allocated_bytes(obj, alloc)
                        mag_range = magnification_range(lib, obj, parities)
                        print(f"{label} {name} p{parities}: magnification range {mag_range:.0f}, "
                              + " ".join(f"{names[k]}={est[k] / alloc[k]:.3f}" for k in range(3)))
                        if name == "near_critical":
                            self.assertTrue(4400 < mag_range < 8000, f"magnification range {mag_range}")
                        for k in range(3):
                            self.assertGreaterEqual(
                                est[k], alloc[k],
                                f"{names[k]}: est={est[k]} alloc={alloc[k]} ratio={est[k] / alloc[k]:.4f}")
                        # the host copies of the histograms: per map kind, at least the bins of the magnification
                        # range (one fewer for rounding) and the log histogram's, a few thousand bins at most
                        n_kinds = 3 if parities else 1
                        hist_host = lib.get_host_histogram_bytes(obj)
                        span = round(mag_range * 1000)
                        self.assertGreaterEqual(hist_host, n_kinds * 4 * span)
                        self.assertLess(hist_host, n_kinds * 4 * (span + 20_000))
                        itemsize = 4 if label == "float" else 8
                        self.assertGreaterEqual(est[3], host_result_bytes(lib, obj, itemsize, parities))
                    finally:
                        lib.IPM_delete(obj)

    def test_snapshot_copies_inputs(self):
        for label, lib in LIBS:
            obj = lib.IPM_init()
            try:
                self.assertTrue(lib.set_worker(obj, 0))
                te.set_valid_map(lib, obj)
                lib.set_half_length_y1(obj, 5)
                lib.set_half_length_y2(obj, 5)
                lib.set_num_pixels_y1(obj, 100)
                lib.set_num_pixels_y2(obj, 80)
                lib.set_outfile_prefix(obj, b"snap_")
                lib.set_write_parities(obj, 1)
                lib.set_random_seed(obj, 99)
                self.assertTrue(lib.run(obj, 0))
                snap = lib.IPM_snapshot(obj)
                self.assertTrue(snap)
                try:
                    self.assertEqual(te.read_all(lib, snap), te.read_all(lib, obj))
                    self.assertEqual(lib.get_num_stars(snap), lib.get_num_stars(obj))
                    # the snapshot holds the host copies of the six histograms that the worker reports
                    self.assertGreater(lib.get_host_histogram_bytes(obj), 0)
                    self.assertEqual(lib.get_host_histogram_bytes(obj) % (3 * 4), 0)
                    self.assertEqual(lib.get_host_histogram_bytes(snap), lib.get_host_histogram_bytes(obj))
                finally:
                    lib.IPM_delete(snap)
            finally:
                lib.IPM_delete(obj)

    def test_take_last_cuda_error_reports_oom(self):
        lib = lib_ipm.lib
        lib.take_last_cuda_error()
        obj = lib.IPM_init()
        try:
            self.assertTrue(lib.set_worker(obj, 0))
            # three 100000 x 100000 float maps need 111 GiB, more than any GPU holds (a single map of 37 GiB fits on an A100)
            te.set_valid_map(lib, obj, write_parities=1)
            lib.set_num_pixels_y1(obj, 100000)
            lib.set_num_pixels_y2(obj, 100000)
            self.assertFalse(lib.run(obj, 0))
            self.assertEqual(lib.take_last_cuda_error(), 2)
            self.assertEqual(lib.take_last_cuda_error(), 0)
        finally:
            lib.IPM_delete(obj)

    def test_set_worker_rejects_invalid_device(self):
        for label, lib in LIBS:
            obj = lib.IPM_init()
            try:
                self.assertFalse(lib.set_worker(obj, lib.get_device_count()))
                self.assertFalse(lib.set_worker(obj, -1))
            finally:
                lib.IPM_delete(obj)

    def test_check_device_healthy(self):
        self.assertEqual(lib_ipm.lib.check_device(0), 0)

    def test_get_device_info(self):
        out = (ctypes.c_ulonglong * 2)()
        cm = ctypes.c_int(-1)
        self.assertTrue(lib_ipm.lib.get_device_info(0, out, ctypes.byref(cm)))
        self.assertGreater(out[0], 0)
        self.assertLessEqual(out[0], out[1])
        self.assertEqual(cm.value, 1)
        self.assertFalse(lib_ipm.lib.get_device_info(lib_ipm.lib.get_device_count(), out, ctypes.byref(cm)))


if __name__ == "__main__":
    unittest.main()
