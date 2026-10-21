"""Scheduling of a batch of maps over worker slots on one or more GPUs.

Each slot runs one map at a time and keeps its GPU memory between maps. During results(), one thread per
slot drives it. A map is admitted to a GPU when the extra memory its slot needs, on top of what that GPU's
slots already hold, fits the GPU's budget. A map that does not fit runs alone once nothing else runs on the
GPU. Finished results wait in a queue bounded in bytes until the consumer releases them. A failing map
becomes a BatchMapError result and the batch carries on.

The scheduler knows nothing of IPM. A backend estimates the maps, describes the GPUs and opens the slots,
and the parameter dicts pass through untouched."""
import operator
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Protocol

CUDA_ERROR_MEMORY_ALLOCATION = 2
RESERVE_MIN = 1 << 30
RESERVE_FRACTION = 0.05
DEFAULT_SLOT_CAP = 8
DEFAULT_MAX_RESULT_BYTES = 8 << 30


@dataclass(frozen=True)
class MemoryEstimate:
    """Device memory by component (pixel, star, tree), and the host bytes of a map's result."""
    pixel: int = 0
    star: int = 0
    tree: int = 0
    result: int = 0

    def device_bytes(self) -> int:
        return self.pixel + self.star + self.tree

    def cmax(self, other: "MemoryEstimate") -> "MemoryEstimate":
        """The componentwise maximum: what a slot with grow-only buffers holds after running both."""
        return MemoryEstimate(max(self.pixel, other.pixel), max(self.star, other.star),
                              max(self.tree, other.tree), max(self.result, other.result))


_NOTHING = MemoryEstimate()


@dataclass(frozen=True)
class DeviceInfo:
    free: int
    total: int
    concurrent_managed: bool


class MapFailed(Exception):
    """Raised by a backend when a map cannot be estimated or run. cuda_error is 0 when no CUDA error is known."""

    def __init__(self, message: str, cuda_error: int = 0):
        super().__init__(message)
        self.cuda_error = cuda_error

    def __reduce__(self):
        return type(self), (str(self), self.cuda_error)


class BatchMapError(Exception):
    """The result of a map that failed. index is the map's position in the batch."""

    def __init__(self, index: int, message: str, cuda_error: int = 0):
        super().__init__(message)
        self.index = index
        self.cuda_error = cuda_error

    def __reduce__(self):
        return type(self), (self.index, str(self), self.cuda_error)


@dataclass
class SlotOutcome:
    """What a slot's run() returns: the result, its host bytes, the device memory the slot holds afterwards,
    and whether the map fell back to the direct deposit."""
    result: object
    result_bytes: int
    held: MemoryEstimate
    direct_deposit: bool = False


class Slot(Protocol):
    device: int

    def run(self, params: dict) -> SlotOutcome: ...   # raises MapFailed; always called from the slot's own thread

    def release(self) -> None: ...                    # frees the slot's GPU memory

    def close(self) -> None: ...


class Backend(Protocol):
    def device_count(self) -> int: ...

    def device_info(self, device: int) -> DeviceInfo: ...

    def estimate(self, params: dict) -> MemoryEstimate: ...   # raises MapFailed

    def open_slot(self, device: int) -> Slot: ...

    def check_device(self, device: int) -> int: ...


@dataclass
class Finished:
    """One map's result: an IPM-like object, or a BatchMapError. Its bytes count against the scheduler's
    result budget until release() is called."""
    index: int
    result: object
    _nbytes: int = field(default=0, repr=False, compare=False)
    _run: object = field(default=None, repr=False, compare=False)

    def release(self) -> None:
        """Return the result's bytes to the result budget. Calling it again does nothing."""
        run = self._run
        if run is not None:
            run.release(self)


def _map_error(index, exc):
    error = BatchMapError(index, str(exc) or type(exc).__name__, exc.cuda_error if isinstance(exc, MapFailed) else 0)
    error.__cause__ = exc
    return error


def _empty_stats(devices, dead):
    return {"devices": {d: {"budget": 0, "maps": 0, "max_concurrent": 0, "peak_committed": 0, "oom_retries": 0,
                            "parked_slots": 0, "dead": d in dead} for d in devices},
            "direct_deposit_maps": 0, "peak_result_bytes": 0}


class _SlotState:
    """A slot position on a device. It lives as long as the scheduler; the slot opens on first use."""

    def __init__(self, device, number):
        self.device = device
        self.number = number
        self.slot = None
        self.held = _NOTHING      # device memory the slot holds, or will hold while it runs its admitted map
        self.releasing = False    # another thread is freeing the slot's memory


class _Job(NamedTuple):
    index: int
    estimate: MemoryEstimate
    victims: list               # idle slots of the device whose memory is freed before this map runs


class _Run:
    """The state of one results() call. Every field is read and written under the scheduler's condition."""

    def __init__(self, cond, params, devices, dead):
        self.cond = cond
        self.params = params
        self.estimates = [None] * len(params)
        self.pending = deque()    # indices waiting to run, in batch order
        self.queue = deque()      # Finished items not yet yielded
        self.stopping = False
        self.error = None         # an exception that escaped the scheduler's own code in a slot thread
        self.threads = []
        self.budget = {}
        self.n_slots = {}
        self.slot_limit = {}
        self.running = {}
        self.committed = {}
        self.total_running = 0
        self.reserved_result = 0  # host bytes of results running, queued, or yielded and not released
        self.stats = _empty_stats(devices, dead)

    def release(self, item):
        with self.cond:
            if item._run is not self:
                return
            item._run = None
            self.reserved_result -= item._nbytes
            self.cond.notify_all()


class Scheduler:
    """Runs batches of maps on worker slots: maps_per_gpu slots per GPU (slot_cap when it is None), each
    driven by its own thread while results() runs.

    stats describes the latest results() call and is updated while it runs:
    {"devices": {d: {"budget", "maps", "max_concurrent", "peak_committed", "oom_retries", "parked_slots",
    "dead"}}, "direct_deposit_maps": int, "peak_result_bytes": int}. "maps" counts the maps that succeeded
    on that GPU."""

    def __init__(self, backend: Backend, devices=None, maps_per_gpu=None,
                 max_result_bytes=DEFAULT_MAX_RESULT_BYTES, slot_cap=DEFAULT_SLOT_CAP):
        count = backend.device_count()
        if devices is None:
            if count < 1:
                raise RuntimeError("no GPU is visible")
            devices = range(count)
        devices = [operator.index(d) for d in devices]
        if not devices:
            raise ValueError("devices is empty")
        if len(set(devices)) != len(devices):
            raise ValueError(f"devices has repeated entries: {devices}")
        missing = [d for d in devices if not 0 <= d < count]
        if missing:
            raise ValueError(f"no GPU {missing} among the {count} visible")
        if maps_per_gpu is not None and operator.index(maps_per_gpu) < 1:
            raise ValueError(f"maps_per_gpu must be at least 1, got {maps_per_gpu!r}")
        if operator.index(slot_cap) < 1:
            raise ValueError(f"slot_cap must be at least 1, got {slot_cap!r}")
        if max_result_bytes < 0:
            raise ValueError(f"max_result_bytes must not be negative, got {max_result_bytes!r}")
        self._backend = backend
        self._devices = devices
        self._maps_per_gpu = maps_per_gpu
        self._slot_cap = slot_cap
        self._max_result_bytes = max_result_bytes
        self._cond = threading.Condition()
        self._slots = {d: [] for d in devices}
        self._dead = {}           # device -> (cuda error, message); a dead device stays dead
        self._run = None          # the results() call whose threads may still be alive
        self._closed = False
        self.stats = _empty_stats(devices, self._dead)

    def results(self, params: list[dict]) -> Iterator[Finished]:
        """Run every map and yield one Finished per map, in completion order.

        Release each Finished when its result is no longer needed: at most max_result_bytes of results that
        are running, queued or unreleased are held at a time (always at least one map). Closing the generator
        (break, an exception in the consumer, Ctrl-C) lets the maps in progress finish, drops their results
        and joins every slot thread."""
        params = list(params)
        run = self._begin(params)
        try:
            self._estimate(run)
            self._start(run)
            for _ in range(len(params)):
                yield self._next_finished(run)
        finally:
            self._stop(run)

    def close(self) -> None:
        """Stop a results() call in progress, then close every slot. Calling it again does nothing."""
        with self._cond:
            self._closed = True
            run = self._run
            if run is not None:
                run.stopping = True
            self._cond.notify_all()
        if run is not None and not self._finish_stop(run):
            return    # called from a slot thread, which cannot wait for itself: leave the slots open
        with self._cond:
            slots = [st.slot for states in self._slots.values() for st in states if st.slot is not None]
            for states in self._slots.values():
                for st in states:
                    st.slot = None
                    st.held = _NOTHING
            self._cond.notify_all()
        first = None
        for slot in slots:
            try:
                slot.close()
            except Exception as exc:
                first = first or exc
        if first is not None:
            raise first

    # ---- the consumer's side

    def _begin(self, params):
        with self._cond:
            if self._closed:
                raise RuntimeError("the scheduler is closed")
            prev = self._run
            if prev is not None and not prev.stopping:
                raise RuntimeError("another results() call is still in progress on this scheduler")
        if prev is not None:
            self._finish_stop(prev)   # a stop that could not wait for its threads: wait now
        with self._cond:
            if self._run is not None:
                raise RuntimeError("another results() call is still in progress on this scheduler")
            run = _Run(self._cond, params, self._devices, self._dead)
            self._run = run
            self.stats = run.stats
            self._cond.notify_all()
            return run

    def _estimate(self, run):
        """Estimate every map on the calling thread, before any slot thread starts: an estimate can read a
        star file and allocate, so it must not overlap running maps."""
        for i, p in enumerate(run.params):
            try:
                estimate = self._backend.estimate(p)
            except Exception as exc:
                with self._cond:
                    run.queue.append(Finished(i, _map_error(i, exc)))
                    self._cond.notify_all()
            else:
                with self._cond:
                    run.estimates[i] = estimate
                    run.pending.append(i)
                    self._cond.notify_all()

    def _start(self, run):
        with self._cond:
            devices = [d for d in self._devices if d not in self._dead]
        infos = {d: self._backend.device_info(d) for d in devices}
        unmanaged = []
        with self._cond:
            for d in devices:
                info = infos[d]
                held = self._held_on(d)
                run.budget[d] = info.free + held - max(RESERVE_MIN, int(RESERVE_FRACTION * info.total))
                n = self._maps_per_gpu or self._slot_cap
                if not info.concurrent_managed:
                    unmanaged.append(d)
                    n = 1
                states = self._slots[d]
                while len(states) < n:
                    states.append(_SlotState(d, len(states)))
                run.n_slots[d] = run.slot_limit[d] = n
                run.running[d] = 0
                run.committed[d] = held
                dev = run.stats["devices"][d]
                dev["budget"] = run.budget[d]
                dev["peak_committed"] = held
                for st in states[:n]:
                    run.threads.append(threading.Thread(target=self._slot_thread, args=(run, st),
                                                        name=f"ipm-batch-{d}-{st.number}", daemon=True))
            self._cond.notify_all()
        for d in unmanaged:
            warnings.warn(f"GPU {d} has no concurrent managed memory access, so its maps run one at a time",
                          stacklevel=3)
        for t in run.threads:
            t.start()

    def _next_finished(self, run):
        with self._cond:
            while not run.queue:
                if run.error is not None:
                    raise RuntimeError("a batch scheduler thread failed") from run.error
                if run.stopping:
                    raise RuntimeError("the scheduler was closed while results() was running")
                if not any(t.is_alive() for t in run.threads):
                    self._fail_stranded(run)
                    if not run.queue:
                        raise RuntimeError("the batch scheduler lost track of a map")
                    break
                self._cond.wait(timeout=0.1)   # wakes up regularly so that Ctrl-C is delivered
            item = run.queue.popleft()
            self._cond.notify_all()
            return item

    def _fail_stranded(self, run):
        """Fail the maps still waiting when no slot thread is left to run them (lock held)."""
        while run.pending:
            i = run.pending.popleft()
            run.queue.append(Finished(i, BatchMapError(i, "no usable GPU left")))
        self._cond.notify_all()

    def _stop(self, run):
        with self._cond:
            run.stopping = True
            self._cond.notify_all()
        self._finish_stop(run)

    def _finish_stop(self, run):
        """Wait for the stopping run's threads, then drop its unyielded results. Returns False, without
        waiting, when called from one of those threads."""
        if threading.current_thread() in run.threads:
            return False
        for t in run.threads:
            if t.ident is not None:
                t.join()
        with self._cond:
            leftovers = list(run.queue)
            run.queue.clear()
            if self._run is run:
                self._run = None
            self._cond.notify_all()
        for item in leftovers:
            item.release()
        return True

    # ---- the slot threads' side

    def _slot_thread(self, run, st):
        try:
            while True:
                job = self._admit(run, st)
                if job is None:
                    return
                outcome, exc = self._execute(run, st, job)
                self._complete(run, st, job, outcome, exc)
        except BaseException as exc:   # a fault in the scheduler itself: stop the batch and tell the consumer
            with self._cond:
                if run.error is None:
                    run.error = exc
                run.stopping = True
                self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def _admit(self, run, st):
        """Wait until the map at the head of pending can start in this slot. Returns None when the thread
        should exit: the run is stopping, the device is dead, or nothing is left to run anywhere."""
        with self._cond:
            while True:
                if run.stopping or st.device in self._dead:
                    return None
                if not run.pending:
                    if run.total_running == 0:
                        return None
                elif not st.releasing:
                    job = self._try_admit(run, st)
                    if job is not None:
                        self._cond.notify_all()
                        return job
                self._cond.wait()

    def _try_admit(self, run, st):
        """The admission rules (lock held). Returns a _Job, or None to wait."""
        d = st.device
        i = run.pending[0]
        e = run.estimates[i]
        if run.reserved_result > 0 and run.reserved_result + e.result > self._max_result_bytes:
            return None
        if run.running[d] >= run.slot_limit[d]:
            return None
        need = st.held.cmax(e)
        growth = need.device_bytes() - st.held.device_bytes()
        victims = []
        if run.committed[d] + growth > run.budget[d]:
            if run.running[d] > 0:
                return None
            # Nothing runs on d: free what its idle slots hold and run this map alone.
            victims = [x for x in self._slots[d] if x.held.device_bytes() > 0]
            for x in victims:
                x.held = _NOTHING
                if x is not st:
                    x.releasing = True
            need = st.held.cmax(e)
        run.pending.popleft()
        st.held = need
        run.committed[d] = self._held_on(d)
        run.running[d] += 1
        run.total_running += 1
        run.reserved_result += e.result
        self._update_peaks(run, d)
        return _Job(i, e, victims)

    def _execute(self, run, st, job):
        """Free the victims' memory, open the slot if needed and run the map, all outside the lock. Never
        raises: returns (outcome, None) or (None, exception)."""
        try:
            try:
                for x in job.victims:
                    if x.slot is not None:
                        x.slot.release()
            finally:
                if job.victims:
                    with self._cond:
                        for x in job.victims:
                            x.releasing = False
                        self._cond.notify_all()
            slot = st.slot
            if slot is None:
                slot = self._backend.open_slot(st.device)
                with self._cond:
                    st.slot = slot
                    self._cond.notify_all()
            return slot.run(run.params[job.index]), None
        except BaseException as exc:
            return None, exc

    def _complete(self, run, st, job, outcome, exc):
        d = st.device
        with self._cond:
            run.running[d] -= 1
            run.total_running -= 1
            if exc is None:
                st.held = outcome.held
                run.committed[d] = self._held_on(d)
                run.reserved_result += outcome.result_bytes - job.estimate.result
                run.queue.append(Finished(job.index, outcome.result, outcome.result_bytes, run))
                run.stats["devices"][d]["maps"] += 1
                if outcome.direct_deposit:
                    run.stats["direct_deposit_maps"] += 1
            else:
                self._map_failed(run, st, job, exc)
            self._update_peaks(run, d)
            self._cond.notify_all()

    def _map_failed(self, run, st, job, exc):
        """Classify a failed run (lock held). This is the one place that decides what a failure means; for
        now every failure fails its map."""
        if st.slot is None:
            st.held = _NOTHING    # the slot never opened, so it holds nothing
            run.committed[st.device] = self._held_on(st.device)
        run.reserved_result -= job.estimate.result
        run.queue.append(Finished(job.index, _map_error(job.index, exc)))

    # ---- helpers (lock held)

    def _held_on(self, d):
        return sum(st.held.device_bytes() for st in self._slots[d])

    def _update_peaks(self, run, d):
        dev = run.stats["devices"][d]
        dev["max_concurrent"] = max(dev["max_concurrent"], run.running[d])
        dev["peak_committed"] = max(dev["peak_committed"], run.committed[d])
        run.stats["peak_result_bytes"] = max(run.stats["peak_result_bytes"], run.reserved_result)
