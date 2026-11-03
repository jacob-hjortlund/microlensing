"""Batches of IPM maps: BatchedIPM runs many maps on one or more GPUs and hands them back as they finish, or
writes each one to disk as it finishes. Libraries without batch support run the maps one at a time."""
import operator
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np

from .batch_backend import CtypesBackend
from .batch_params import expand
from .batch_scheduler import DEFAULT_MAX_RESULT_BYTES, BatchMapError, Scheduler
from .ipm import IPM, _INPUT_NAMES

_BATCH_LEVEL = ("verbose", "is_double", "devices", "maps_per_gpu", "max_result_bytes")
_WRITE_FLAGS = ("write_stars", "write_maps", "write_parities", "write_histograms")


@dataclass(frozen=True)
class SavedMap:
    index: int
    outfile_prefix: str
    random_seed: "int | None"   # the result's seed (0 for a star-file map); the map's own seed if no result exists
    error: "BatchMapError | None"


def _make_backend(is_double: bool, verbose: int, device: int):
    """Return the ctypes backend for the library, with its estimates on GPU device, or None when the library lacks
    the batch exports."""
    from . import lib_ipm, lib_ipm_double
    lib = (lib_ipm_double if is_double else lib_ipm).lib
    if not hasattr(lib, "set_worker"):
        return None
    return CtypesBackend(is_double, verbose, device)


def _is_sequence(value):
    return isinstance(value, (list, tuple, range, set, frozenset, dict)) or \
        (isinstance(value, np.ndarray) and value.ndim >= 1)


def _check_devices(devices):
    """devices as a list of ints, or None. Only ordered sequences are accepted (a list, tuple, range or numpy array),
    as devices[0] is the GPU of the estimates: a set or a dict raises TypeError."""
    if devices is None:
        return None
    if not (isinstance(devices, (list, tuple, range)) or (isinstance(devices, np.ndarray) and devices.ndim >= 1)):
        raise TypeError(f"devices must be a list, tuple, range or numpy array of GPU numbers, got {devices!r}")
    items = list(devices)
    for d in items:
        if isinstance(d, bool):
            raise TypeError(f"devices must hold integers, got {d!r}")
        try:
            operator.index(d)
        except TypeError:
            raise TypeError(f"devices must hold integers, got {d!r}") from None
    items = [operator.index(d) for d in items]
    if not items:
        raise ValueError("devices must list at least one GPU")
    if any(d < 0 for d in items):
        raise ValueError(f"devices must hold GPU numbers of at least 0, got {items}")
    if len(set(items)) != len(items):
        raise ValueError(f"devices has repeated entries: {items}")
    return items


def _check_positive(name, value):
    """value as an int of at least 1; a bool or a non-integer raises TypeError."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}")
    return value


class BatchedIPM:
    """Many IPM maps from one description. Every IPM input may be a scalar (shared by all maps) or a sequence
    with one value per map; num_maps repeats scalars. verbose, is_double, devices, maps_per_gpu and
    max_result_bytes apply to the whole batch. See batch_params.expand for seeds and output prefixes. devices is
    a list, tuple, range or numpy array of GPU numbers (its first GPU reads the star files of the memory
    estimates), and a set or a dict raises TypeError. max_result_bytes caps the host bytes of the finished results
    that wait for the consumer: their maps, their histograms and two copies of their stars (a result holds them in
    its C++ object and in its stars attribute).

    Seeds: a sequence random_seed gives each map its seed. A single random_seed is one map's seed, so it is
    accepted only when at most one map has no starfile, and raises ValueError otherwise. seed_start=s, an integer,
    gives the maps the seeds derive_seeds(s, num_maps) (map i's seed does not depend on num_maps); it cannot be
    combined with random_seed. With neither, the seeds are drawn at random.

    The inputs are validated when the batch is built, once for each distinct set of inputs (the seeds and output
    prefixes aside), without using a GPU: the constructor raises ValueError("map i: ...") for the first map i
    with invalid inputs, as IPM's constructor does. A map with a starfile is checked on the host only (its file
    when it runs). run(save=True) raises the first map's BatchMapError, with every map's SavedMap in its
    `records` attribute, when every map failed; when only some failed, it returns the records, with the errors
    set.

    Without the batch exports (an older library) the maps run one at a time on GPU 0: devices other than [0]
    raise ValueError, maps_per_gpu and max_result_bytes have no effect, and the constructor checks the inputs
    with IPM's Python setters only."""

    def __init__(self, *, num_maps=None, kappa_tot=None, shear=None, smooth_fraction=None, kappa_star=None,
                 theta_star=None, mass_function=None, m_solar=None, m_lower=None, m_upper=None, light_loss=None,
                 rectangular=None, approx=None, safety_scale=None, starfile=None, center_y1=None,
                 center_y2=None, half_length_y1=None, half_length_y2=None, num_pixels_y1=None,
                 num_pixels_y2=None, num_rays_y=None, random_seed=None, seed_start=None, write_stars=False,
                 write_maps=False, write_parities=False, write_histograms=False, outfile_prefix=None, verbose=0,
                 is_double=False, devices=None, maps_per_gpu=None, max_result_bytes=DEFAULT_MAX_RESULT_BYTES):
        args = dict(locals())
        if _is_sequence(seed_start):
            raise TypeError("seed_start applies to the whole batch and cannot be a sequence")
        for name in _BATCH_LEVEL:
            if name != "devices" and _is_sequence(args[name]):
                raise TypeError(f"{name} applies to the whole batch and cannot be a sequence")
        self._devices = _check_devices(devices)
        self._maps_per_gpu = None if maps_per_gpu is None else _check_positive("maps_per_gpu", maps_per_gpu)
        self._max_result_bytes = _check_positive("max_result_bytes", max_result_bytes)
        self._verbose = verbose
        self._is_double = is_double
        self._params = expand({k: args[k] for k in _INPUT_NAMES}, num_maps, seed_start)
        self._lock = threading.Lock()
        self._scheduler = None
        self._warned_dead = set()
        self.last_run_stats = None
        self._backend = _make_backend(is_double, verbose, self._devices[0] if self._devices else 0)
        if self._backend is None:
            if self._devices is not None and self._devices != [0]:
                raise ValueError(f"devices other than [0] need a library with the batch exports, got "
                                 f"{self._devices}")
            warnings.warn("this IPM library has no batch support (it predates the batch exports; rebuild it with "
                          "`source compile`, or compile.bat on Windows); the maps run one at a time on GPU 0, and "
                          "maps_per_gpu and max_result_bytes have no effect", UserWarning, stacklevel=2)
        self._validate()

    def _validate(self):
        """Raise ValueError for the first map whose inputs are invalid, checking one map of each distinct set of
        inputs. With the batch exports that is the backend's estimate (the checks of IPM's Python setters, then the
        C++ input checks), otherwise the checks of IPM's Python setters only. A map with a starfile gets the checks
        of IPM's Python setters only: its estimate reads the file with a CUDA call, which would make a context on the
        batch's first GPU here, so the file is checked when the map runs."""
        seen = set()
        for i, params in enumerate(self._params):
            key = tuple(sorted((k, v) for k, v in params.items() if k not in ("random_seed", "outfile_prefix")))
            if key in seen:
                continue
            seen.add(key)
            try:
                if self._backend is not None and not params.get("starfile"):
                    self._backend.estimate(dict(params))
                else:
                    IPM(is_double=self._is_double)._apply_inputs(**params)
            except Exception as exc:
                raise ValueError(f"map {i}: {str(exc) or type(exc).__name__}") from exc

    @property
    def num_maps(self) -> int:
        return len(self._params)

    def params(self, index: int) -> dict:
        """The resolved inputs of one map, random seed and output prefix included."""
        return dict(self._params[index])

    def run(self, save: bool = False, writers: int = None):
        """save=False: a generator of (index, IPM or BatchMapError) in completion order. save=True: write every
        map as it finishes and return a list of SavedMap sorted by index. Ctrl-C, or any other exception, during
        run(save=True) stops the batch as it stops a save=False loop (the maps in progress finish), then drops the
        maps still waiting for a writer, whose files are not written, lets the writes in progress finish and
        raises."""
        if save:
            return self._run_saving(writers)
        return self._run_memory()

    def close(self) -> None:
        sched = getattr(self, "_scheduler", None)
        self._scheduler = None
        if sched is not None:
            sched.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- runs

    def _results(self, want_maps=True):
        """A generator of Finished (scheduler) or Finished-like items (serial fallback). want_maps (one value, or
        one per map) says whether the scheduler copies a map's maps to the host; the fallback always does."""
        if self._backend is None:
            return self._serial()
        with self._lock:
            if self._scheduler is None:
                self._scheduler = Scheduler(self._backend, devices=self._devices, maps_per_gpu=self._maps_per_gpu,
                                            max_result_bytes=self._max_result_bytes)
            sched = self._scheduler
        return sched.results([dict(p) for p in self._params], want_maps)

    def _serial(self):
        self.last_run_stats = None
        for i in range(len(self._params)):
            yield self._serial_done(i)   # no local keeps map i (and its GPU memory) while map i + 1 runs

    def _serial_done(self, i):
        ipm, error = self._serial_map(i)
        return _Done(i, ipm if error is None else error)

    def _serial_map(self, i, save=False):
        ipm = None
        try:
            ipm = IPM(**self._params[i], verbose=self._verbose, is_double=self._is_double)
            ipm.run()
            if save:
                ipm.save()
            return ipm, None
        except Exception as exc:
            return ipm, BatchMapError(i, str(exc) or type(exc).__name__)

    def _after_run(self):
        if self._backend is not None and self._scheduler is not None:
            self.last_run_stats = self._scheduler.stats   # results() replaces the dict when it starts
        stats = self.last_run_stats
        if not stats:
            return
        for d, info in stats["devices"].items():
            if info["dead"] and d not in self._warned_dead:
                self._warned_dead.add(d)
                warnings.warn(f"GPU {d} left the batch after a CUDA error. The map whose error revealed the break "
                              "fails with the GPU's error; other maps that fail there are retried once on the "
                              "remaining GPUs, if any", UserWarning, stacklevel=3)
        n = stats["direct_deposit_maps"]
        if n > 0:
            warnings.warn(f"{n} maps fell back to the direct deposit (slower) because the step buffer did not fit",
                          UserWarning, stacklevel=3)

    def _run_memory(self):
        gen = self._results()
        try:
            for fin in gen:
                fin.release()
                yield _take(fin)
        finally:
            gen.close()
            self._after_run()

    def _record(self, index, result, error=None):
        seed = getattr(result, "random_seed", None) if result is not None else None
        if seed is None:
            seed = self._params[index]["random_seed"]
        return SavedMap(index, self._params[index]["outfile_prefix"], seed, error)

    def _run_saving(self, writers):
        if not any(bool(p[f]) for p in self._params for f in _WRITE_FLAGS):
            warnings.warn("only the parameter files will be written: none of write_stars, write_maps, "
                          "write_parities and write_histograms is set", UserWarning, stacklevel=3)
        if writers is None:
            used = len(self._devices) if self._devices is not None else \
                (self._backend.device_count() if self._backend is not None else 1)
            writers = min(8, max(1, used))
        if self._backend is None:
            self.last_run_stats = None
            recs = []
            for i in range(len(self._params)):
                ipm, error = self._serial_map(i, save=True)
                recs.append(self._record(i, ipm, error))
            return self._records_or_raise(recs)
        records = {}

        def save_one(fin):
            result, fin.result = fin.result, None   # so that the result is freed before its bytes are released
            try:
                try:
                    result.save()
                    records[fin.index] = self._record(fin.index, result)
                except Exception as exc:
                    records[fin.index] = self._record(fin.index, result,
                                                      BatchMapError(fin.index, str(exc) or type(exc).__name__))
            finally:
                result = None
                fin.release()

        gen = self._results([bool(p["write_maps"]) for p in self._params])   # maps not written are not copied
        executor = ThreadPoolExecutor(max_workers=writers, thread_name_prefix="ipm-save")
        writes = []   # (future, Finished) of each map handed to the writers; a writer empties the item's result
        try:
            for fin in gen:
                if isinstance(fin.result, BatchMapError):
                    records[fin.index] = self._record(fin.index, None, fin.result)
                    fin.release()
                else:
                    try:
                        writes.append((executor.submit(save_one, fin), fin))
                    except BaseException:
                        fin.release()
                        raise
                fin = None    # a writer has it now: keep nothing of it while waiting for the next map
            pending = [f for f, _ in writes]
            while pending:
                pending = wait(pending, timeout=0.1).not_done   # wakes up regularly so that Ctrl-C is delivered
        except BaseException:
            # Ctrl-C or an error: drop the writes that have not started, with their maps (the writes in progress
            # finish in shutdown below)
            executor.shutdown(wait=False, cancel_futures=True)
            for future, fin in writes:
                if future.cancelled():
                    fin.result = None
                    fin.release()
            raise
        finally:
            gen.close()
            executor.shutdown(wait=True)
            self._after_run()
        for f, _ in writes:
            f.result()
        return self._records_or_raise([records[i] for i in sorted(records)])

    @staticmethod
    def _records_or_raise(recs):
        """recs, unless every map failed: then the first map's error, with the records on it."""
        if recs and all(r.error is not None for r in recs):
            err = recs[0].error
            err.records = recs
            raise err
        return recs


def _take(fin):
    """(index, result) of a finished map, dropping the item's own reference to the result, so that the memory-mode
    generator keeps nothing of the map while it waits for the next one."""
    result, fin.result = fin.result, None
    return fin.index, result


class _Done:
    """A finished map of the serial fallback, with the interface of the scheduler's Finished."""
    __slots__ = ("index", "result")

    def __init__(self, index, result):
        self.index = index
        self.result = result

    def release(self):
        pass
