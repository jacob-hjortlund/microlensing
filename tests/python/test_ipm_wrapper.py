"""Tests of IPM.run() in microlensing/IPM/ipm.py against a fake library, without a GPU. Importing microlensing.IPM loads
the IPM libraries, so they must load (build them if the included ones do not; see Compiling in the README). From the repository root:
python -m unittest discover -s tests/python -v"""
import ctypes
import importlib
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MAPS = ("pixels", "pixels_minima", "pixels_saddles")


class FakeIpmLib:
    """Stands in for lib_ipm.lib or lib_ipm_double.lib behind IPM(): set_* store values and get_* return them, and the
    map exports are logged in calls. get_pixels* return host buffers holding 0.5 + offset + i. copy_pixels* exist only
    with copy_exports; they fill dst with offset + i and return True, or return False for the maps in failing. The
    offsets are 0, 1000 and 2000 for the total, minima and saddles maps."""

    def __init__(self, ctype, copy_exports=True, failing=()):
        self.ctype = ctype
        self.calls = []
        self.values = {"m_lower": 0.01, "m_upper": 50.0, "num_stars": 2, "rectangular": 0,
                       "corner_x1": 3.0, "corner_x2": 4.0, "theta_star": 1.0}
        self.stars = (ctype * 6)(*range(6))
        self.host_maps = {}
        if copy_exports:
            for k, which in enumerate(MAPS):
                setattr(self, "copy_" + which, self._copy_export(which, 1000 * k, which not in failing))

    def num_pixels(self):
        return self.values["num_pixels_y1"] * self.values["num_pixels_y2"]

    def _copy_export(self, which, offset, ok):
        def copy(obj, dst):
            self.calls.append("copy_" + which)
            if not isinstance(dst, ctypes.POINTER(self.ctype)):   # what ctypes refuses against the bound argtypes
                raise ctypes.ArgumentError(f"copy_{which}: dst is a {type(dst).__name__}")
            if ok:
                n = self.num_pixels()
                np.ctypeslib.as_array(dst, shape=(n,))[:] = np.arange(offset, offset + n)
            return ok
        return copy

    def _host_map(self, which, offset):
        self.calls.append("get_" + which)
        n = self.num_pixels()
        self.host_maps[which] = (self.ctype * n)(*(0.5 + offset + i for i in range(n)))
        return ctypes.cast(self.host_maps[which], ctypes.POINTER(self.ctype))

    def get_pixels(self, obj):
        return self._host_map("pixels", 0)

    def get_pixels_minima(self, obj):
        return self._host_map("pixels_minima", 1000)

    def get_pixels_saddles(self, obj):
        return self._host_map("pixels_saddles", 2000)

    def get_stars(self, obj):
        return ctypes.cast(self.stars, ctypes.POINTER(self.ctype))

    def IPM_init(self):
        return 1

    def IPM_delete(self, obj):
        pass

    def run(self, obj, verbose):
        self.calls.append("run")
        return True

    def __getattr__(self, name):   # the other set_* and get_* exports
        if name.startswith("set_"):
            return lambda obj, value: self.values.__setitem__(name[4:], value)
        if name.startswith("get_") and name[4:] in self.values:
            return lambda obj: self.values[name[4:]]
        raise AttributeError(name)


class FakeCDLL:
    """Takes the argtypes/restype bindings of lib_ipm*.py for any export; the copy_* exports exist only with has_copy."""

    def __init__(self, has_copy):
        self._has_copy = has_copy
        self._funcs = {}

    def __getattr__(self, name):
        if name.startswith("_") or (name.startswith("copy_") and not self._has_copy):
            raise AttributeError(name)
        return self._funcs.setdefault(name, types.SimpleNamespace())


class PythonWrapperTests(unittest.TestCase):
    """IPM.run() of microlensing/IPM/ipm.py against a fake library."""

    @classmethod
    def setUpClass(cls):
        # another install of the package (such as an editable one of a different checkout) would test other code: the
        # modules imported must be this checkout's, or copies of them
        from microlensing.IPM import ipm
        imported = Path(ipm.__file__).resolve().parent
        for name in ("ipm.py", "lib_ipm.py", "lib_ipm_double.py"):
            try:
                same = (imported / name).read_bytes() == (REPO / "microlensing" / "IPM" / name).read_bytes()
            except OSError:   # a file missing from either package
                same = False
            if not same:
                raise AssertionError(f"imported {imported / name}, which differs from this checkout's; "
                                     f"run from {REPO} or with PYTHONPATH={REPO}")
        cls.ipm = ipm

    def make_ipm(self, lib, is_double=False, parities=False):
        module = self.ipm.lib_ipm_double if is_double else self.ipm.lib_ipm
        with mock.patch.object(module, "lib", lib):
            obj = self.ipm.IPM(num_pixels_y1=5, num_pixels_y2=3, write_parities=parities, is_double=is_double)
        self.assertIs(obj.lib, lib)
        return obj

    @staticmethod
    def maps(obj):
        return {"pixels": obj.magnifications, "pixels_minima": obj.magnifications_minima,
                "pixels_saddles": obj.magnifications_saddles}

    def test_run_copies_maps_through_the_copy_exports(self):
        for is_double, parities in ((False, False), (False, True), (True, True)):
            with self.subTest(is_double=is_double, parities=parities):
                ctype, dtype = (ctypes.c_double, np.float64) if is_double else (ctypes.c_float, np.float32)
                lib = FakeIpmLib(ctype)
                obj = self.make_ipm(lib, is_double, parities)
                obj.run()
                for k, (which, arr) in enumerate(self.maps(obj).items()):
                    if k > 0 and not parities:
                        self.assertIsNone(arr, which)
                        continue
                    self.assertEqual(arr.shape, (3, 5), which)   # (num_pixels_y2, num_pixels_y1)
                    self.assertEqual(arr.dtype, dtype, which)
                    self.assertTrue(arr.flags.c_contiguous and arr.flags.writeable, which)
                    np.testing.assert_array_equal(arr, np.arange(1000 * k, 1000 * k + 15).reshape(3, 5))
                copies = ["copy_pixels"] + (["copy_pixels_minima", "copy_pixels_saddles"] if parities else [])
                self.assertEqual([c for c in lib.calls if c.startswith("copy_")], copies)
                self.assertEqual([c for c in lib.calls if c.startswith("get_pixels")], [])

    def test_run_falls_back_to_get_pixels_without_copy_exports(self):
        lib = FakeIpmLib(ctypes.c_float, copy_exports=False)
        self.assertIsNone(getattr(lib, "copy_pixels", None))
        obj = self.make_ipm(lib, parities=True)
        obj.run()
        for k, (which, arr) in enumerate(self.maps(obj).items()):
            self.assertEqual(arr.shape, (3, 5), which)
            self.assertEqual(arr.dtype, np.float32, which)
            np.testing.assert_array_equal(arr, 0.5 + 1000 * k + np.arange(15).reshape(3, 5))
            lib.host_maps[which][0] = -1   # the wrapper keeps a copy, not a view of the library's buffer
            self.assertEqual(arr[0, 0], 0.5 + 1000 * k, which)
        self.assertEqual([c for c in lib.calls if c.startswith("get_pixels")], ["get_" + w for w in MAPS])

    def test_copy_failure_raises(self):
        for which in MAPS:
            with self.subTest(which):
                obj = self.make_ipm(FakeIpmLib(ctypes.c_float, failing=(which,)), parities=True)
                with self.assertRaisesRegex(Exception, "^Error copying magnification map$"):
                    obj.run()

    def test_copy_exports_are_bound_only_when_the_library_has_them(self):
        for name, ctype in (("lib_ipm", ctypes.c_float), ("lib_ipm_double", ctypes.c_double)):
            module = getattr(self.ipm, name)
            real = module.lib
            for has_copy in (True, False):
                with self.subTest(name, has_copy=has_copy):
                    fake = FakeCDLL(has_copy)
                    try:
                        with mock.patch("ctypes.CDLL", return_value=fake):
                            importlib.reload(module)
                        self.assertIs(module.lib, fake)
                        self.assertIs(fake.get_pixels.restype, ctypes.POINTER(ctype))
                        if has_copy:
                            for which in MAPS:
                                export = getattr(fake, "copy_" + which)
                                self.assertEqual(export.argtypes, [ctypes.c_void_p, ctypes.POINTER(ctype)], which)
                                self.assertIs(export.restype, ctypes.c_bool, which)
                        else:
                            self.assertFalse(hasattr(fake, "copy_pixels"))
                    finally:
                        module.lib = real


if __name__ == "__main__":
    unittest.main()
