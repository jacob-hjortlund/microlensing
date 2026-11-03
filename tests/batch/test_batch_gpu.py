"""GPU tests of batches run through the ctypes backend (skipped without a GPU): batched maps against serial
IPM.run() maps, save mode against IPM.save(), the memory estimates, every visible GPU, clean stops, the host
memory of save mode, and a batch on GPU 1 leaving GPU 0 and the calling thread's device alone. Uses every visible
GPU (CUDA_VISIBLE_DEVICES selects them):
python -m unittest tests.batch.test_batch_gpu -v

Stars of two runs are compared after sorting by (x1, x2, mass), as their order after the tree sort is not
reproducible. Maps are judged by batch_rules against the noise between serial runs of the same map (the larger
of two serial runs' differences from a third, the reference)."""
import ast
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import batch_rules  # noqa: E402
from microlensing.IPM import lib_ipm  # noqa: E402
from microlensing.IPM.batch import BatchedIPM  # noqa: E402
from microlensing.IPM.batch_backend import CtypesBackend  # noqa: E402
from microlensing.IPM.batch_params import expand  # noqa: E402
from microlensing.IPM.batch_scheduler import BatchMapError, MapFailed  # noqa: E402
from microlensing.IPM.ipm import IPM  # noqa: E402
from needs_exports import HAS_EXPORTS, NO_EXPORTS  # noqa: E402

LENS = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2)
MAP_NAMES = ("magnifications", "magnifications_minima", "magnifications_saddles")
MiB = 1 << 20
SERIAL_RUNS = 3   # per configuration: the reference and two runs whose larger difference from it is the noise
HIST_FACTOR = 3   # a histogram's L1 difference from the reference may be 3 times the serial runs' plus 1e-3 per pixel

TMP = None        # the module's scratch directory
STARFILE = None   # a star file written by a serial IPM run (setUpModule)


def lens_map(n, hl=10.0, **extra):
    """The lens kappa_tot=0.4, shear=0.4, kappa_star=0.2 on an n x n map with half-lengths hl."""
    return dict(LENS, half_length_y1=hl, half_length_y2=hl, num_pixels_y1=n, num_pixels_y2=n, **extra)


SALPETER = dict(rectangular=True, approx=True, mass_function="Salpeter", m_solar=0.5, m_lower=0.08, m_upper=1.5)


def star_file_map():
    """A map whose stars come from STARFILE (a field written for half-lengths 10, used at 8)."""
    return dict(kappa_tot=0.4, shear=0.4, half_length_y1=8.0, half_length_y2=8.0, num_pixels_y1=700,
                num_pixels_y2=700, starfile=STARFILE)


def device_count():
    return lib_ipm.lib.get_device_count() if hasattr(lib_ipm.lib, "get_device_count") else 0


def setUpModule():
    if not HAS_EXPORTS:
        raise unittest.SkipTest(NO_EXPORTS)
    if device_count() == 0:
        raise unittest.SkipTest("no GPU visible")
    global TMP, STARFILE
    TMP = tempfile.mkdtemp(prefix="ipm_batch_gpu_")
    ipm = IPM(**lens_map(600), random_seed=77, write_stars=True, outfile_prefix=TMP + "/sf_")
    ipm.run()
    ipm.save()
    STARFILE = TMP + "/sf_ipm_stars.bin"
    if not Path(STARFILE).is_file():
        raise RuntimeError(f"the serial run wrote no star file {STARFILE}")


def tearDownModule():
    if TMP is not None:
        shutil.rmtree(TMP, ignore_errors=True)


def batch_of(maps, **batch_kw):
    """A BatchedIPM of the given per-map dicts (keys a map leaves out are None, i.e. IPM's default), checked to
    use the ctypes backend rather than the one-at-a-time fallback."""
    keys = sorted(set().union(*maps))
    seqs = {k: [m.get(k) for m in maps] for k in keys}
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # the fallback warns that the library has no batch support
        b = BatchedIPM(**seqs, **batch_kw)
    assert isinstance(b._backend, CtypesBackend), type(b._backend)
    return b


def serial(params, is_double):
    ipm = IPM(**params, is_double=is_double)
    ipm.run()
    return ipm


def sorted_stars(arr):
    arr = np.asarray(arr)
    return arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]


def child_env():
    """The environment of a child Python process that imports the microlensing package this process uses (and the
    modules of this directory), whatever its working directory."""
    package_root = Path(lib_ipm.__file__).resolve().parents[2]
    path = [str(package_root), str(HERE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(path))


def batch_threads():
    return [t.name for t in threading.enumerate() if t.name.startswith("ipm-batch-")]


def device_free(d):
    out = (ctypes.c_ulonglong * 2)()
    cm = ctypes.c_int(0)
    assert lib_ipm.lib.get_device_info(d, out, ctypes.byref(cm))
    return out[0]


def cuda_driver():
    """The CUDA driver library, initialized, or None if it cannot be loaded."""
    for name in (("nvcuda.dll",) if os.name == "nt" else ("libcuda.so.1", "libcuda.so")):
        try:
            drv = ctypes.CDLL(name)
        except OSError:
            continue
        if drv.cuInit(0) == 0:
            return drv
    return None


def current_device(drv):
    """The device of the calling thread's current CUDA context, None if it has none: the device that CuPy or
    PyTorch code on this thread would use."""
    ctx = ctypes.c_void_p()
    if drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0:
        raise RuntimeError("cuCtxGetCurrent failed")
    if not ctx.value:
        return None
    dev = ctypes.c_int(-1)
    if drv.cuCtxGetDevice(ctypes.byref(dev)) != 0:
        raise RuntimeError("cuCtxGetDevice failed")
    return dev.value


def gpu_processes():
    """({CUDA device: MiB this process uses}, {CUDA device: MiB other processes use}) from nvidia-smi, for the
    devices of a numeric CUDA_VISIBLE_DEVICES (CUDA_DEVICE_ORDER=PCI_BUS_ID); (None, None) if unavailable."""
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not visible or not all(x.isdigit() for x in visible) or shutil.which("nvidia-smi") is None:
        return None, None
    try:
        listed = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    buses = {}
    for line in listed.splitlines():
        index, bus = (x.strip() for x in line.split(","))
        buses[bus.lower()] = index
    cuda_of = {idx: d for d, idx in enumerate(visible)}
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    mine, others = {}, {}
    for line in out.splitlines():
        pid, bus, used = (x.strip() for x in line.split(","))
        d = cuda_of.get(buses.get(bus.lower()))
        if d is not None and used.isdigit():
            target = mine if int(pid) == os.getpid() else others
            target[d] = target.get(d, 0) + int(used)
    return mine, others


class _Equivalence:
    """Checks of batched results against serial runs of the same parameters."""

    def serial_refs(self, batch, is_double):
        """SERIAL_RUNS serial IPM.run() objects per map, shared by maps whose parameters differ only in their
        prefix: the first is the reference, the others give the noise."""
        refs, cache = [], {}
        for i in range(batch.num_maps):
            p = batch.params(i)
            key = repr(sorted((k, v) for k, v in p.items() if k != "outfile_prefix"))
            if key not in cache:
                cache[key] = tuple(serial(p, is_double) for _ in range(SERIAL_RUNS))
            refs.append(cache[key])
        return refs

    def assert_matches(self, label, params, result, refs, is_double):
        ref_a = refs[0]
        self.assertNotIsInstance(result, BatchMapError, f"{label}: {result}")
        self.assertTrue(result._host_only, label)
        for ref in refs[1:]:
            self.assertTrue(np.array_equal(sorted_stars(ref_a.stars.stars), sorted_stars(ref.stars.stars)),
                            f"{label}: the serial runs differ in their stars")
        self.assertEqual(result.num_stars, ref_a.num_stars, label)
        self.assertTrue(np.array_equal(sorted_stars(result.stars.stars), sorted_stars(ref_a.stars.stars)),
                        f"{label}: stars differ from the serial run")
        self.assertEqual(result.corner, ref_a.corner, label)
        expected_seed = 0 if params["starfile"] is not None else params["random_seed"]
        self.assertEqual(result.random_seed, expected_seed, label)
        self.assertEqual(ref_a.random_seed, expected_seed, label)
        for name in MAP_NAMES:
            a, c = getattr(ref_a, name), getattr(result, name)
            if a is None:
                self.assertIsNone(c, f"{label}: {name}")
                continue
            self.assertEqual(c.dtype, a.dtype, f"{label}: {name}")
            self.assertEqual(c.shape, a.shape, f"{label}: {name}")
            noise = batch_rules.noise(a, *(getattr(ref, name) for ref in refs[1:]))
            fails = batch_rules.judge(c, a, noise, is_double, parity=name != "magnifications")
            self.assertEqual(fails, [], f"{label}: {name}")


class BatchEquivalenceTests(_Equivalence, unittest.TestCase):
    def test_batch_matches_serial_runs(self):
        # float: more maps than slots (2 per GPU), so slots run several different maps; map 5 repeats map 0's seed
        float_maps = [
            lens_map(1000, random_seed=1001),
            lens_map(600, random_seed=1002, **SALPETER),
            lens_map(800, random_seed=1003, write_parities=True),
            lens_map(600, random_seed=1004, num_rays_y=4),
            star_file_map(),
            lens_map(1000, random_seed=1001),
        ]
        # double (is_double applies to a whole batch): 4 maps on the 2 slots of GPU 0, with and without parities
        double_maps = [lens_map(600, random_seed=2001), lens_map(600, random_seed=2002, write_parities=True),
                       lens_map(600, random_seed=2003), lens_map(600, random_seed=2004, write_parities=True)]
        for label, maps, is_double, kw in (("float", float_maps, False, {}),
                                           ("double", double_maps, True, {"devices": [0]})):
            with batch_of(maps, is_double=is_double, maps_per_gpu=2, **kw) as b:
                refs = self.serial_refs(b, is_double)
                out = dict(b.run())
                stats = b.last_run_stats
            print(f"\n{label}: maps per GPU {({d: s['maps'] for d, s in stats['devices'].items()})}, "
                  f"max concurrent {({d: s['max_concurrent'] for d, s in stats['devices'].items()})}")
            self.assertEqual(sorted(out), list(range(len(maps))))
            for i in range(len(maps)):
                with self.subTest(precision=label, map=i):
                    self.assert_matches(f"{label} map {i}", b.params(i), out[i], refs[i], is_double)
            self.assertEqual(stats["direct_deposit_maps"], 0)
            if not is_double:   # equal seeds and parameters give equal star fields within a batch too
                self.assertTrue(np.array_equal(sorted_stars(out[5].stars.stars), sorted_stars(out[0].stars.stars)))

    def test_judge_rejects_another_seed(self):
        # the rule still discriminates: a serial map of the same configuration with another seed fails it
        for label, p, is_double in (("float parities", lens_map(600, write_parities=True), False),
                                    ("double", lens_map(600), True)):
            refs = [serial(dict(p, random_seed=9101), is_double) for _ in range(SERIAL_RUNS)]
            other = serial(dict(p, random_seed=9102), is_double)
            for name in MAP_NAMES:
                a = getattr(refs[0], name)
                if a is None:
                    continue
                with self.subTest(case=label, map=name):
                    noise = batch_rules.noise(a, *(getattr(r, name) for r in refs[1:]))
                    fails = batch_rules.judge(getattr(other, name), a, noise, is_double,
                                              parity=name != "magnifications")
                    print(f"  another seed, {label} {name}: {fails}")
                    self.assertNotEqual(fails, [])

    def test_reused_slot_matches_fresh_runs(self):
        # one slot runs every map: Salpeter with m_solar=0.5 (C++ scales m_lower/m_upper in place) alternates with
        # a star-file map (which frees and reallocates the stars) and differs in size from it
        maps = []
        for seed in (3001, 3002, 3003):
            maps += [lens_map(600, random_seed=seed, **SALPETER), star_file_map()]
        with batch_of(maps, devices=[0], maps_per_gpu=1) as b:
            refs = self.serial_refs(b, False)
            out = dict(b.run())
            stats = b.last_run_stats
        self.assertEqual(stats["devices"][0]["maps"], len(maps))
        self.assertEqual(stats["devices"][0]["max_concurrent"], 1)
        print()
        for i in range(len(maps)):
            with self.subTest(map=i):
                self.assert_matches(f"reused map {i}", b.params(i), out[i], refs[i], False)


def read_star_file(path, itemsize):
    """(header bytes, stars sorted by x1, x2, mass) of a star file: int num_stars, int rectangular, the complex
    corner and theta_star, then x1, x2 and mass of each star."""
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data, dtype=np.int32, count=1)[0])
    head = 8 + 3 * itemsize
    dtype = np.float64 if itemsize == 8 else np.float32
    stars = np.frombuffer(data, dtype=dtype, offset=head)
    if stars.size != 3 * n:
        raise ValueError(f"{path}: {stars.size} values for {n} stars")
    return data[:head], sorted_stars(stars.reshape(n, 3))


def read_map(path):
    """A map file of write_array(): int32 rows and columns, then the values (float32 or float64)."""
    data = Path(path).read_bytes()
    nrows, ncols = (int(x) for x in np.frombuffer(data, dtype=np.int32, count=2))
    n = nrows * ncols
    if len(data) - 8 == 4 * n:
        dtype = np.float32
    elif len(data) - 8 == 8 * n:
        dtype = np.float64
    else:
        raise ValueError(f"{path}: {len(data) - 8} bytes of values for {nrows}x{ncols} pixels")
    return np.frombuffer(data, dtype=dtype, count=n, offset=8).reshape(nrows, ncols)


def read_histogram(path):
    counts = {}
    for line in Path(path).read_text().splitlines():
        value, count = line.split()
        counts[int(value)] = int(count)
    return counts


def histogram_l1(a, b):
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


class SaveModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="save_", dir=TMP))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_mode_files_match_ipm_save(self):
        flags = dict(write_stars=True, write_maps=True, write_parities=True, write_histograms=True)
        maps = [lens_map(600, random_seed=4001, **flags), lens_map(800, random_seed=4002, **flags),
                lens_map(600, random_seed=4003, **SALPETER, **flags)]
        serial_dirs = "BCD"[:SERIAL_RUNS]   # B is the reference, the others give the noise
        dirs = {k: self.tmp / k for k in "A" + serial_dirs}
        for d in dirs.values():
            d.mkdir()
        with batch_of(maps, outfile_prefix=f"{dirs['A']}/", maps_per_gpu=2) as b:
            recs = b.run(save=True)
            params = [b.params(i) for i in range(len(maps))]
        self.assertEqual([r.index for r in recs], list(range(len(maps))))
        for i, r in enumerate(recs):
            self.assertIsNone(r.error, f"map {i}: {r.error}")
            self.assertEqual(r.random_seed, params[i]["random_seed"])
            self.assertEqual(r.outfile_prefix, f"{dirs['A']}/{i}_")
        for i, p in enumerate(params):
            for k in serial_dirs:
                ipm = IPM(**dict(p, outfile_prefix=f"{dirs[k]}/{i}_"))
                ipm.run()
                ipm.save()
        names = sorted(f.name for f in dirs["A"].iterdir())
        for k in serial_dirs:
            self.assertEqual(names, sorted(f.name for f in dirs[k].iterdir()))
        per_map = 1 + 1 + 3 + 6   # parameter info, stars, 3 maps, 6 histograms
        self.assertEqual(len(names), per_map * len(maps), names)
        print()
        for name in names:
            a, b = dirs["A"] / name, dirs["B"] / name
            others = [dirs[k] / name for k in serial_dirs[1:]]
            i = int(name.split("_", 1)[0])
            npix = params[i]["num_pixels_y1"] * params[i]["num_pixels_y2"]
            with self.subTest(file=name):
                if name.endswith("_ipm_parameter_info.txt"):
                    la, lb = a.read_text().splitlines(), b.read_text().splitlines()
                    self.assertEqual(len(la), len(lb))
                    self.assertEqual([x for x in la if not x.startswith("t_shoot_cells ")],
                                     [x for x in lb if not x.startswith("t_shoot_cells ")])
                    self.assertEqual(sum(x.startswith("t_shoot_cells ") for x in la), 1)
                elif name.endswith("_ipm_stars.bin"):
                    ha, sa = read_star_file(a, 4)
                    hb, sb = read_star_file(b, 4)
                    self.assertEqual(ha, hb)
                    self.assertTrue(np.array_equal(sa, sb))
                elif name.endswith(".bin"):
                    self.assertEqual(a.read_bytes()[:8], b.read_bytes()[:8])
                    self.assertEqual(a.stat().st_size, b.stat().st_size)
                    ref, cand = read_map(b), read_map(a)
                    noise = batch_rules.noise(ref, *(read_map(x) for x in others))
                    parity = not name.endswith("_ipm_magnifications.bin")
                    self.assertEqual(batch_rules.judge(cand, ref, noise, False, parity=parity), [])
                elif name.endswith(".txt"):
                    ha, hb = read_histogram(a), read_histogram(b)
                    self.assertEqual(sum(ha.values()), npix)
                    self.assertEqual(sum(hb.values()), npix)
                    # the bins are 1/1000 of a magnification wide, so serial noise alone moves a few per cent of
                    # the pixels into a neighbouring bin: the limit is relative to the serial runs' L1
                    l1 = histogram_l1(ha, hb)
                    l1_noise = max(histogram_l1(read_histogram(x), hb) for x in others)
                    limit = HIST_FACTOR * l1_noise + 1e-3 * npix
                    print(f"  {name}: L1 batch {l1}, L1 serial {l1_noise}, limit {limit:.0f} ({npix} pixels)")
                    self.assertLessEqual(l1, limit)
                else:
                    self.fail(f"unexpected file {name}")

    def result_bytes_from_files(self, i):
        """The host bytes of the result of map i without its maps, from its files: its stars twice (the result holds
        its snapshot's copy and its numpy array), and its 6 histograms, each with at least as many bins as the values
        in its file span (the three of a kind share their range)."""
        _, stars = read_star_file(self.tmp / f"{i}_ipm_stars.bin", 4)
        nbytes = 2 * stars.size * 4
        for kind in ("mags", "log_mags"):
            values = [v for part in ("", "_minima", "_saddles")
                      for v in read_histogram(self.tmp / f"{i}_ipm_{kind}_numpixels{part}.txt")]
            nbytes += 3 * 4 * (max(values) - min(values) + 1)
        return nbytes

    def test_save_mode_without_maps_copies_no_maps(self):
        # maps that are not written are never copied to the host: a result holds two copies of its stars and its
        # histograms. With max_result_bytes=1 one result is on the host at a time, so the peak is the largest result:
        # it counts both copies of the stars, the histograms and no map
        flags = dict(write_stars=True, write_parities=True, write_histograms=True)
        with batch_of([lens_map(1000, random_seed=4101 + i, **flags) for i in range(4)], outfile_prefix=f"{self.tmp}/",
                      devices=[0], max_result_bytes=1) as b:
            recs = b.run(save=True)
            stats = b.last_run_stats
        self.assertEqual([r.error for r in recs], [None] * 4)
        names = sorted(f.name for f in self.tmp.iterdir())
        self.assertEqual(len(names), 4 * (1 + 1 + 6), names)   # parameter info, stars, 6 histograms
        self.assertFalse([n for n in names if "_ipm_magnifications" in n])
        largest = max(self.result_bytes_from_files(i) for i in range(4))
        print(f"\npeak_result_bytes {stats['peak_result_bytes']}, largest result from the files {largest}")
        self.assertGreaterEqual(stats["peak_result_bytes"], largest)
        self.assertLess(stats["peak_result_bytes"], largest + 1000 * 1000 * 4)   # less than one map more

    def test_save_mode_host_memory_flat(self):
        # max_result_bytes holds 4 of these 16 MB maps: however slowly the files are written, the results on the host
        # stay within it, so the peak does not grow with the number of maps (the default 8 GiB would hold all 128)
        child = (
            "import sys\n"
            "from microlensing.IPM.batch import BatchedIPM\n"
            "n, out = int(sys.argv[1]), sys.argv[2]\n"
            "with BatchedIPM(kappa_tot=0.4, shear=0.4, kappa_star=0.2, half_length_y1=10, half_length_y2=10,\n"
            "                num_pixels_y1=2000, num_pixels_y2=2000, num_maps=n, seed_start=5001,\n"
            "                write_maps=True, outfile_prefix=out + '/', max_result_bytes=64 << 20) as b:\n"
            "    recs = b.run(save=True)\n"
            "bad = [r.index for r in recs if r.error is not None]\n"
            "if bad or len(recs) != n:\n"
            "    sys.exit(f'failed maps {bad} of {len(recs)}')\n"
            "with open('/proc/self/status') as f:\n"
            "    print('VmHWM_kB', next(l.split()[1] for l in f if l.startswith('VmHWM:')))\n"
        )
        peaks = {}
        for n in (16, 128):
            out = self.tmp / f"n{n}"
            out.mkdir()
            proc = subprocess.run([sys.executable, "-c", child, str(n), str(out)], cwd=str(self.tmp),
                                  capture_output=True, text=True, timeout=900, env=child_env())
            self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
            self.assertEqual(len(list(out.glob("*_ipm_magnifications.bin"))), n)
            line = next(x for x in proc.stdout.splitlines() if x.startswith("VmHWM_kB"))
            peaks[n] = int(line.split()[1]) / 1024
            shutil.rmtree(out)
        print(f"\nsave mode VmHWM: N=16 {peaks[16]:.0f} MiB, N=128 {peaks[128]:.0f} MiB, "
              f"difference {peaks[128] - peaks[16]:.0f} MiB")
        self.assertLess(abs(peaks[128] - peaks[16]), 512)


class BackendTests(unittest.TestCase):
    def test_open_slot_failure_names_the_gpu_and_cause(self):
        backend = CtypesBackend(False, 0)
        missing = device_count()
        with self.assertRaisesRegex(RuntimeError, f"GPU {missing}: there is no GPU {missing} among the"):
            backend.open_slot(missing)

    def test_released_slot_holds_nothing_and_runs_again(self):
        backend = CtypesBackend(False, 0)
        p = expand(dict(lens_map(600), random_seed=[9001]))[0]
        est = backend.estimate(p)
        slot = backend.open_slot(0)
        try:
            first = slot.run(p)
            self.assertGreater(first.held.device_bytes(), 0)
            for k in ("pixel", "star", "tree"):
                self.assertLessEqual(getattr(first.held, k), getattr(est, k), k)
            # the result: the map of a map without parities, two copies of its stars (3 values each) and the host
            # copies of its histograms (expand leaves write_histograms to the library's default, on). The estimate
            # allows the histograms a range of HIST_RANGE_BOUND, far more than this map's
            hist = backend.lib.get_host_histogram_bytes(slot.ipm.obj)
            self.assertGreater(hist, 0)
            self.assertEqual(first.result_bytes, 600 * 600 * 4 + 2 * slot.ipm.num_stars * 3 * 4 + hist)
            self.assertLessEqual(first.result_bytes, est.result)
            self.assertEqual((first.map_bytes, est.maps), (600 * 600 * 4, 600 * 600 * 4))
            r1 = slot.collect(first.handle, True)
            # what the result holds: its numpy map and stars, its snapshot's copy of the stars and the histograms
            self.assertEqual(r1.magnifications.nbytes + r1.stars.stars.nbytes + r1.num_stars * 3 * 4 + hist,
                             first.result_bytes)
            with self.assertRaises(MapFailed):
                slot.collect(first.handle, True)   # a map is collected once
            slot.release()
            alloc = (ctypes.c_ulonglong * 3)()
            backend.lib.get_allocated_bytes(slot.ipm.obj, alloc)
            self.assertEqual(list(alloc), [0, 0, 0])
            second = slot.run(p)
            r2 = slot.collect(second.handle, False)   # without its maps
            self.assertIsNone(r2.magnifications)
            self.assertTrue(np.array_equal(sorted_stars(r2.stars.stars), sorted_stars(r1.stars.stars)))
            self.assertEqual(r2.random_seed, 9001)
            third = slot.run(p)
            slot.release()
            with self.assertRaises(MapFailed):
                slot.collect(third.handle, True)   # release() freed the map
        finally:
            slot.close()
        self.assertIsNone(slot.ipm)


class SchedulingTests(unittest.TestCase):
    def test_estimates_prevent_direct_deposit(self):
        with batch_of([lens_map(5000, hl=25.0)] * 32, seed_start=6001, devices=[0]) as b:
            for i, r in b.run():
                self.assertNotIsInstance(r, BatchMapError, f"map {i}: {r}")
            stats = b.last_run_stats
        dev = stats["devices"][0]
        print(f"\n5000^2 x 32 on GPU 0: max concurrent {dev['max_concurrent']}, peak committed "
              f"{dev['peak_committed'] / MiB:.0f} MiB of a {dev['budget'] / MiB:.0f} MiB budget, "
              f"direct deposit maps {stats['direct_deposit_maps']}")
        self.assertEqual(dev["maps"], 32)
        self.assertEqual(stats["direct_deposit_maps"], 0)
        self.assertLessEqual(dev["peak_committed"], dev["budget"])
        self.assertGreater(dev["max_concurrent"], 1)

    def test_uses_every_visible_gpu(self):
        n_dev = device_count()
        if n_dev < 2:
            self.skipTest("fewer than 2 GPUs visible")
        with batch_of([lens_map(600)] * 16, seed_start=7001) as b:
            out = dict(b.run())
            stats = b.last_run_stats
        self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
        per_gpu = {d: s["maps"] for d, s in stats["devices"].items()}
        print(f"\n16 maps on every GPU: {per_gpu}")
        self.assertEqual(sorted(per_gpu), list(range(n_dev)))
        self.assertTrue(all(v > 0 for v in per_gpu.values()), per_gpu)
        self.assertEqual(sum(per_gpu.values()), 16)
        with batch_of([lens_map(600)] * 4, seed_start=7002, devices=[1]) as b:
            out = dict(b.run())
            stats = b.last_run_stats
        self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
        self.assertEqual(list(stats["devices"]), [1])
        self.assertEqual(stats["devices"][1]["maps"], 4)

    def test_break_then_rerun(self):
        # 2000^2 maps, so that the slots hold some 270 MiB per GPU: a slot that close() left open would show. The
        # driver keeps a cache of device memory that nvidia-smi gives to no process and that moves in steps of
        # about 128 MiB (after earlier tests put managed memory on a GPU, measured 2026-10-22), so the free memory
        # may move by up to 256 MiB. This process's own usage must come back within 64 MiB after close(), and must
        # not grow over 3 further close()/rerun cycles
        free_tolerance = 256 * MiB
        maps = [lens_map(2000)] * 20
        n_dev = device_count()
        with batch_of(maps[:4], seed_start=8000) as warm:   # each GPU's first batch also loads code onto it
            list(warm.run())
        before = {d: device_free(d) for d in range(n_dev)}
        before_mine, before_others = gpu_processes()
        b = batch_of(maps, seed_start=8001)
        try:
            got = []
            for i, r in b.run():
                self.assertNotIsInstance(r, BatchMapError, f"map {i}: {r}")
                got.append(i)
                if len(got) == 2:
                    break
            self.assertEqual(batch_threads(), [])
            out = dict(b.run())   # the next run of the same batch, on the slots the first one opened
            self.assertEqual(sorted(out), list(range(20)))
            self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
            self.assertEqual(batch_threads(), [])
            holding_mine, _ = gpu_processes()
        finally:
            b.close()
        after = {d: device_free(d) for d in range(n_dev)}
        after_mine, after_others = gpu_processes()
        cycles_mine = []   # this process's usage after each further close()
        for _ in range(3):
            try:
                out = dict(b.run())   # a closed batch opens new slots
                self.assertEqual(sorted(out), list(range(20)))
                self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
            finally:
                b.close()
            self.assertEqual(batch_threads(), [])
            cycles_mine.append(gpu_processes()[0])
        print(f"\nfree memory before / after close (MiB): "
              + ", ".join(f"GPU {d} {before[d] / MiB:.0f} / {after[d] / MiB:.0f}" for d in range(n_dev))
              + f"\nthis process (nvidia-smi, MiB): before {before_mine}, holding {holding_mine}, "
              f"after {after_mine}, after 3 more close()/rerun cycles {cycles_mine}; "
              f"other processes before {before_others}, after {after_others}")
        if before_mine is None:   # no per-process memory from nvidia-smi: only the free memory can be checked
            for d in range(n_dev):
                self.assertLessEqual(abs(after[d] - before[d]), free_tolerance, f"GPU {d}")
            return
        for d in range(n_dev):
            # this process's own memory on each GPU, which other processes on a shared GPU cannot disturb
            self.assertGreater(holding_mine.get(d, 0), before_mine.get(d, 0) + 100, f"GPU {d}: slots not seen")
            self.assertLessEqual(abs(after_mine.get(d, 0) - before_mine.get(d, 0)), 64, f"GPU {d}")
            for k, mine in enumerate(cycles_mine):   # no growth from one close() to the next
                self.assertLessEqual(mine.get(d, 0) - after_mine.get(d, 0), 64, f"GPU {d}, cycle {k + 1}")
            # the free memory, on a GPU that no other process uses: another process's allocations would move it
            if not before_others.get(d) and not after_others.get(d):
                self.assertLessEqual(abs(after[d] - before[d]), free_tolerance, f"GPU {d}")
            else:
                print(f"GPU {d}: free-memory check skipped, another process uses this GPU "
                      f"({before_others.get(d)} / {after_others.get(d)} MiB before / after)")


class DeviceTests(unittest.TestCase):
    def setUp(self):
        if device_count() < 2:
            self.skipTest("fewer than 2 GPUs visible")

    def test_plain_ipm_after_closed_batch(self):
        # a plain run makes GPU 0 this thread's device. Batches on GPU 1 leave it so, give back their slots'
        # memory when closed, and a plain run works afterwards. The first batch on GPU 1 also loads code onto it,
        # which stays, so the memory is compared with this process's usage after that batch
        drv = cuda_driver()
        if drv is None:
            self.skipTest("the CUDA driver library cannot be loaded")
        serial(lens_map(600, random_seed=9201), False)
        self.assertEqual(current_device(drv), 0)
        with batch_of([lens_map(2000)] * 2, seed_start=9202, devices=[1]) as warm:
            list(warm.run())
        self.assertEqual(current_device(drv), 0, "after the first batch")
        base_mine, _ = gpu_processes()
        with batch_of([lens_map(2000)] * 8, seed_start=9203, devices=[1]) as b:
            out = dict(b.run())
            holding_mine, _ = gpu_processes()
        self.assertTrue(all(not isinstance(r, BatchMapError) for r in out.values()), out)
        self.assertEqual(current_device(drv), 0, "after the second batch")
        ipm = serial(lens_map(600, random_seed=9204), False)
        self.assertEqual(ipm.magnifications.shape, (600, 600))
        after_mine, _ = gpu_processes()
        print(f"\nthis process on GPU 1 (nvidia-smi, MiB): after the first batch {base_mine}, holding {holding_mine}, "
              f"after the second batch and a plain run {after_mine}")
        if not holding_mine:
            return    # no per-process memory from nvidia-smi, or it cannot see this process (a PID namespace)
        self.assertGreater(holding_mine.get(1, 0), base_mine.get(1, 0) + 100, "slots not seen")
        self.assertLessEqual(after_mine.get(1, 0), base_mine.get(1, 0) + 256)

    def test_star_file_batch_on_device1(self):
        # a batch on GPU 1 that mixes a star-file map with seeded maps makes no context on GPU 0. A new process
        # runs it, from a directory without the package, and prints the GPUs that nvidia-smi lists it on
        child = (
            "import sys\n"
            "import test_batch_gpu as t\n"
            "t.STARFILE = sys.argv[1]\n"
            "maps = [t.lens_map(600, random_seed=9301), t.star_file_map(), t.lens_map(600, random_seed=9302)]\n"
            "with t.batch_of(maps, devices=[1]) as b:\n"
            "    out = dict(b.run())\n"
            "bad = [i for i, r in out.items() if isinstance(r, t.BatchMapError)]\n"
            "if bad or len(out) != 3:\n"
            "    sys.exit(f'failed maps {bad} of {len(out)}')\n"
            "mine, _ = t.gpu_processes()\n"
            "print('GPUS', None if mine is None else sorted(mine))\n"
        )
        proc = subprocess.run([sys.executable, "-c", child, STARFILE], cwd=TMP, capture_output=True, text=True,
                              timeout=600, env=child_env())
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        listed = ast.literal_eval(next(x for x in proc.stdout.splitlines() if x.startswith("GPUS ")).split(" ", 1)[1])
        print(f"\nthe batch process is on GPUs {listed}")
        if not listed:   # None: no per-process list from nvidia-smi; empty: it cannot see the process (a PID namespace)
            self.skipTest("nvidia-smi lists no GPU for the batch process")
        self.assertNotIn(0, listed)
        self.assertIn(1, listed)


if __name__ == "__main__":
    unittest.main()
