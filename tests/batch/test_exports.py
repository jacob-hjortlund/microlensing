"""Tests of the worker, estimate and snapshot exports of lib_ipm and lib_ipm_double. These tests need no GPU:
python -m unittest tests.batch.test_exports -v"""
import ctypes
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM import lib_ipm, lib_ipm_double  # noqa: E402
from needs_exports import NEEDS_EXPORTS  # noqa: E402

# name -> a value that differs from the default
NUMERIC = {
    "kappa_tot": 0.55, "shear": 0.31, "kappa_star": 0.21, "theta_star": 1.5, "m_solar": 0.5,
    "m_lower": 0.2, "m_upper": 30.0, "light_loss": 0.01, "safety_scale": 2.5,
    "center_y1": 0.25, "center_y2": -0.5, "half_length_y1": 7.0, "half_length_y2": 9.0,
    "rectangular": 1, "approx": 1, "num_pixels_y1": 123, "num_pixels_y2": 77, "num_rays_y": 11,
    "random_seed": 4242, "write_stars": 1, "write_maps": 1, "write_parities": 1, "write_histograms": 1,
}
STRINGS = {"mass_function": "salpeter", "starfile": "some_stars.bin", "outfile_prefix": "pre_"}

# the histogram term of the estimate per map kind: a magnification range of 1e4 (IPM<T>::HIST_RANGE_BOUND) at
# 1000 bins per unit, 4-byte bins and 1.25 headroom; the host copies of the result are counted without the headroom
HIST_BYTES_PER_KIND = 10_000 * 1000 * 4 * 5 // 4
HIST_RESULT_BYTES_PER_KIND = 10_000 * 1000 * 4


def read_all(lib, obj):
    values = {n: getattr(lib, "get_" + n)(obj) for n in NUMERIC}
    for n in STRINGS:
        values[n] = getattr(lib, "get_" + n)(obj)
    return values


def set_non_defaults(lib, obj):
    for n, v in NUMERIC.items():
        getattr(lib, "set_" + n)(obj, v)
    for n, v in STRINGS.items():
        getattr(lib, "set_" + n)(obj, v.encode())


def set_valid_map(lib, obj, write_parities=0, write_histograms=1):
    lib.set_kappa_tot(obj, 0.4)
    lib.set_shear(obj, 0.4)
    lib.set_kappa_star(obj, 0.2)
    lib.set_half_length_y1(obj, 10)
    lib.set_half_length_y2(obj, 10)
    lib.set_num_pixels_y1(obj, 1000)
    lib.set_num_pixels_y2(obj, 1000)
    lib.set_write_histograms(obj, write_histograms)
    lib.set_write_parities(obj, write_parities)


@NEEDS_EXPORTS
class ExportTests(unittest.TestCase):
    libs = ((lib_ipm.lib, 4), (lib_ipm_double.lib, 8))

    def test_reset_inputs_restores_defaults(self):
        for lib, _ in self.libs:
            fresh = lib.IPM_init()
            obj = lib.IPM_init()
            try:
                defaults = read_all(lib, fresh)
                set_non_defaults(lib, obj)
                self.assertNotEqual(read_all(lib, obj), defaults)
                lib.reset_inputs(obj)
                self.assertEqual(read_all(lib, obj), defaults)
            finally:
                lib.IPM_delete(fresh)
                lib.IPM_delete(obj)

    def test_snapshot_copies_inputs(self):
        # a snapshot of an object that has not run is refused (nullptr), so only the refusal is checked here
        for lib, _ in self.libs:
            obj = lib.IPM_init()
            try:
                set_non_defaults(lib, obj)
                snap = lib.IPM_snapshot(obj)
                self.assertFalse(snap)
                self.assertEqual(lib.get_perimeter_deposit_used(obj), 0)
            finally:
                lib.IPM_delete(obj)

    def test_estimate_memory_pixel_and_result_terms(self):
        for lib, size in self.libs:
            obj = lib.IPM_init()
            out = (ctypes.c_ulonglong * 4)()
            try:
                set_valid_map(lib, obj)
                before = read_all(lib, obj), lib.get_num_stars(obj)
                self.assertTrue(lib.estimate_memory(obj, out))
                # the map and the step buffer, and the histograms
                self.assertEqual(out[0], 1000 * 1000 * size * 2 + HIST_BYTES_PER_KIND)
                # the result: the map, two copies of the stars (3 values each, see the next test) and the histograms
                stars = out[3] - 1000 * 1000 * size - HIST_RESULT_BYTES_PER_KIND
                self.assertEqual(stars % (2 * 3 * size), 0)
                self.assertGreater(stars, 0)
                self.assertGreaterEqual(out[2], 1 << 20)
                # the estimate does not change the object
                self.assertEqual((read_all(lib, obj), lib.get_num_stars(obj)), before)
                set_valid_map(lib, obj, write_parities=1)
                self.assertTrue(lib.estimate_memory(obj, out))
                self.assertEqual(out[0], 1000 * 1000 * size * 4 + 3 * HIST_BYTES_PER_KIND)
                self.assertEqual(out[3], 1000 * 1000 * size * 3 + stars + 3 * HIST_RESULT_BYTES_PER_KIND)
                set_valid_map(lib, obj, write_parities=1, write_histograms=0)
                self.assertTrue(lib.estimate_memory(obj, out))
                self.assertEqual(out[0], 1000 * 1000 * size * 4)
                self.assertEqual(out[3], 1000 * 1000 * size * 3 + stars)
            finally:
                lib.IPM_delete(obj)

    def test_estimate_memory_counts_the_stars_twice_in_the_result(self):
        # a collected result holds two host copies of its stars: its snapshot's, and the numpy copy that
        # IPM._from_worker makes of it. The number of stars comes from the star terms of the two libraries: each
        # counts per star a curand state and two device star arrays of 3 values, so they differ by 2 * 3 * (8 - 4)
        # bytes per star
        star_term, result_stars = {}, {}
        for lib, size in self.libs:
            obj = lib.IPM_init()
            out = (ctypes.c_ulonglong * 4)()
            try:
                set_valid_map(lib, obj, write_histograms=0)
                self.assertTrue(lib.estimate_memory(obj, out))
                star_term[size], result_stars[size] = out[1], out[3] - 1000 * 1000 * size
            finally:
                lib.IPM_delete(obj)
        num_stars, rest = divmod(star_term[8] - star_term[4], 2 * 3 * (8 - 4))
        self.assertEqual(rest, 0)
        self.assertGreater(num_stars, 0)
        for size in (4, 8):
            self.assertEqual(result_stars[size], 2 * num_stars * 3 * size, f"{size}-byte values")

    def test_host_histogram_bytes_before_a_run(self):
        # an object that has not run holds no histograms, whatever its write flags
        for lib, _ in self.libs:
            obj = lib.IPM_init()
            try:
                set_valid_map(lib, obj, write_parities=1, write_histograms=1)
                self.assertEqual(lib.get_host_histogram_bytes(obj), 0)
            finally:
                lib.IPM_delete(obj)

    def test_estimate_memory_rejects_invalid(self):
        for lib, _ in self.libs:
            obj = lib.IPM_init()
            out = (ctypes.c_ulonglong * 4)()
            try:
                lib.set_kappa_tot(obj, 0.4)
                lib.set_kappa_star(obj, 0.5)
                self.assertFalse(lib.estimate_memory(obj, out))
            finally:
                lib.IPM_delete(obj)


if __name__ == "__main__":
    unittest.main()
