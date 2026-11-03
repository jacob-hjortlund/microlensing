"""Scheduling of a batch of maps over worker slots on one or more GPUs.

Each slot runs one map at a time and keeps its GPU memory between maps. During results(), one thread per slot drives
it. A slot's work on a map has two phases: run() leaves the result in the slot's device memory, and collect() copies
it to the host. A failing map becomes a BatchMapError result and the batch carries on. The scheduler knows nothing of
IPM: a backend estimates the maps, describes the GPUs and opens the slots, and the parameter dicts pass through
untouched.

The state of a results() call (_Run) is read and written under one condition, and each change that may let a waiting
slot go on notifies it. For each GPU d, budget[d] is its free memory at the start plus what its slots hold, less a
reserve. committed[d] is what its slots hold, or will hold while their admitted maps run (held, per slot: a slot's
buffers only grow until it is freed, so it holds the componentwise maximum of the maps it ran). running[d] counts
the maps admitted on d that have not ended. A map's growth on a slot is what admitting it there adds to committed[d].

Admission (_try_admit). A slot starts the map at the head of the queue (pending) only when each rule below allows
it, checked in this order; otherwise it waits. Each rule names the case it serves and tests that fail without it.
 1. Nothing is being freed or checked on the GPU (releases, checks): committed[d] already leaves out memory whose
    release() is still in flight, and a check after a CUDA error decides whether the GPU stays in the batch. Tests:
    test_slow_release_blocks_admission_on_its_gpu, test_check_device_blocks_admission_on_its_gpu.
 2. Pinning (pinned): the slot takes the first map in pending that is not pinned to another GPU. An out-of-memory retry
    is pinned to the GPU where it ran out (rule 10, which also parks slots for it, or rule 11) until that GPU leaves the
    batch or has no usable slot. Otherwise admission is first in, first out on every GPU: no map is admitted before an
    earlier one that its GPU may run, so later small maps do not overtake a large map (only those admitted after it that
    fit beside it can start before its run does). Tests: test_oom_with_others_running_requeues_and_parks (pinning),
    test_large_map_after_small_maps_frees_idle_slots (first in, first out).
 3. Result bytes (_room_for, waiting_bytes, holding): the map's result estimate (estimates, want_maps) fits next to
    the finished results on the host (being collected, queued, or yielded and not released) within
    max_result_bytes, or none waits. A finished map is collected under the same test, and until then its slot is in
    holding and keeps the result in device memory, so that the results on the host stay within the larger of
    max_result_bytes and one result, besides those the consumer released and keeps. Maps in progress do not count.
    Case: a consumer slower than the maps, and large results. Tests (the start, then the collect):
    test_gpu_whose_slots_wait_for_queue_room_takes_no_map_from_another, test_host_bytes_bounded_by_cap.
 4. Slot limit: running[d] < slot_limit[d], which is n_slots[d] less the slots parked by rule 10. n_slots[d] is
    maps_per_gpu, else slot_cap, and 1 on a GPU without concurrent managed access; it is also the number of slot
    threads, so that only parked slots make this rule bind. Test: test_oom_retry_waits_for_fewer_maps_in_progress.
 5. One map at a time (max_total_running, total_running): when a GPU lacks concurrent managed access, at most one map
    runs over all GPUs, as the host writes managed memory of every map it starts. Slots in holding run no kernel and
    do not count. Test: test_no_concurrent_managed_access_runs_one_map_globally.
 6. Spread (_taken_elsewhere): the slot leaves the map to another GPU that runs fewer maps and would start it at once:
    below its slot_limit, nothing freed or checked there (releases, checks), no retry pinned to it ahead of this map
    (pinned), and an idle slot (not active, not in holding, not broken) whose growth fits its budget (committed). Case:
    a small batch spreads over every GPU. Starting slot 0 of every GPU first spreads it while the threads reach their
    first map in that order; this rule spreads it also when they do not, as on a loaded host. With mixed sizes, a GPU
    that cannot start the map at once does not take it away. Tests:
    test_maps_wait_for_the_slot_threads_of_a_gpu_that_runs_fewer,
    test_gpu_whose_slots_wait_for_queue_room_takes_no_map_from_another.
 7. Memory: committed[d] + growth <= budget[d]. Case: maps whose memory, not the slot limit, bounds how many run at
    once. Test: test_concurrency_limited_by_memory. A map that does not fit:
    a. is left to another usable GPU whose budget fits it, if it needs more than this GPU's whole budget and is not
       pinned (_fits_elsewhere). Budgets do not change during a call, so it does not bounce between GPUs. Case:
       GPUs of different sizes or free memory. Test: test_map_too_big_for_small_gpu_goes_to_big_gpu.
    b. waits while other maps run on the GPU, and its slot joins mem_waiters[d], which only rule 9 reads. Tests:
       test_large_map_after_small_maps_frees_idle_slots (the wait), test_shrink_frees_oversized_idle_slot (mem_waiters).
    c. otherwise is admitted once nothing runs on the GPU and no finished map there waits to be collected (holding: that
       map's slot keeps the result in its memory). Its slot first frees what every idle slot of the GPU holds (the
       victims, counted in releases); once those frees are done, maps that fit can start beside it. Case: a large map
       after small maps whose idle slots hold the memory it needs; a map larger than every budget is still tried. Tests:
       test_large_map_after_small_maps_frees_idle_slots and test_release_when_running_alone (the frees),
       test_map_bigger_than_budget_still_attempted_alone (the wait for collects).
 8. Retry alone (retry_alone): the retry of a map that ran out of memory alone while idle slots held memory (rule 11),
    when it fits by its estimate and comes up while nothing runs on its GPU, first frees what the idle slots then hold,
    once no finished map there waits to be collected (when it does not fit, rule 7c does the same). Maps that fit can
    still start beside it. Case: idle slots holding memory of larger maps, which the estimate leaves room for but the
    run does not. Test: test_oom_alone_retry_frees_finished_maps_once_collected.
 9. Shrink: a slot that holds more than the map needs (the grow-only buffers of a larger map) while another slot of
    the GPU waits for memory (mem_waiters) frees its memory before it runs the map; while no slot waits, it keeps
    its buffers rather than free and grow them again. Case: small maps after a large one, while other maps keep the
    GPU from emptying, which rule 7c needs. Tests: test_shrink_frees_oversized_idle_slot,
    test_oversized_slot_keeps_its_memory_while_no_slot_waits.

Failures (_failed, _classify, _gpu_failure).
10. Parks: a map that runs out of memory while another map ran on its GPU (shared) goes back to the front of pending,
    pinned to the GPU, and the GPU parks slots: slot_limit drops below the number of maps in progress when the error
    is classified, and at least one below its value, so that the retry starts beside fewer maps (parks[i] counts the
    slots). Case: estimates that run low, for one map of a mixed batch or for all. Tests:
    test_oom_while_shared_retries_with_fewer_maps, test_oom_retry_waits_for_fewer_maps_in_progress.
    - Give-back: once the map succeeds or fails, the GPU gets those slots back (_unpark), so that one outlier does not
      slow the rest of the batch. Tests: test_parks_return_after_successful_retry,
      test_parks_return_after_a_retry_fails_with_another_error.
    - Keep: the first such error on a GPU after successful retries gave slots back (gave_back) sets slot_limit back to
      its value before them instead, and from then on successes give none back (keeps_parks), so that estimates low
      for every map do not run out of memory after each successful retry. This happens once per GPU and call. Test:
      test_parks_stay_once_the_gpu_runs_out_again_after_giving_them_back.
    - A map admitted under a higher limit than the current one (job.limit) lowers it only to below the maps in
      progress, not to one below its value: maps admitted together under the higher limit, which run out of
      memory together, do not each take one more slot. Test:
      test_errors_during_runs_after_a_give_back_keep_the_limit_from_before.
    - A retry that is not pinned (its GPU has no usable slot left) parks nothing. Test:
      test_retry_that_is_not_pinned_parks_no_slots.
11. A map that runs out of memory alone while idle slots held memory has that memory freed (not yet that of a slot
    in holding) and is retried, pinned and in retry_alone (rule 8). Otherwise it fails, and its GPU gets back what
    its retries parked. Tests: test_oom_alone_frees_idle_slots_before_failing, test_oom_alone_fails_map.
12. A GPU whose check after a CUDA error reports an error, or where a slot cannot open, leaves the batch for the
    scheduler's lifetime (_dead): its threads exit, its retries are unpinned, the map whose error led to the check fails
    with the GPU's error, and each other map that fails there goes back to the front of pending, once (attempts); the
    second time, or with no GPU left, it fails; when no GPU is left, the maps still waiting fail. A slot whose release()
    fails is not used again, its memory stays counted, and its GPU is checked. Tests: test_dead_device_moves_maps,
    test_map_moves_once_then_fails, test_oom_retry_moves_when_its_gpu_breaks, test_failed_release_keeps_memory_counted.

Known limits:
- An out-of-memory error beside other maps sets slot_limit from the maps in progress when it is classified. Where
  maps of equal length end together, that count can be low, and the limit drops further than needed.
- Between an out-of-memory error and its classification, idle slots on that GPU can still start maps under the old
  limit, and each of them that also runs out of memory is retried too.
- Admission is first in, first out, so a GPU runs nothing while the map at the head of the queue fits only on a
  larger GPU (rule 7a).
- The keep rule cannot tell two outlier maps from estimates that are low for every map: once a second map runs out of
  memory beside others after the first one's retry gave slots back, the GPU keeps the lower limit.
- Idle slots' memory is freed only when nothing runs on the GPU (rules 7c, 8, 11) or by a slot that shrinks (rule 9):
  a map that lacks only that memory waits until the GPU empties, behind one long map if need be."""
import itertools
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
    """Device memory by component (pixel, star, tree), and the host bytes of a map's result (result), of which
    maps are its maps: a result collected without its maps has result - maps bytes."""
    pixel: int = 0
    star: int = 0
    tree: int = 0
    result: int = 0
    maps: int = 0

    def device_bytes(self) -> int:
        return self.pixel + self.star + self.tree

    def cmax(self, other: "MemoryEstimate") -> "MemoryEstimate":
        """The componentwise maximum: what a slot with grow-only buffers holds after running both."""
        return MemoryEstimate(max(self.pixel, other.pixel), max(self.star, other.star),
                              max(self.tree, other.tree), max(self.result, other.result),
                              max(self.maps, other.maps))


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
    """What a slot's run() returns: the handle that collect() takes, the host bytes of the result (result_bytes,
    of which map_bytes are its maps, which a collect without the maps does not copy), the device memory the slot
    holds afterwards, and whether the map fell back to the direct deposit."""
    handle: object
    result_bytes: int
    held: MemoryEstimate
    direct_deposit: bool = False
    map_bytes: int = 0


class Slot(Protocol):
    """run() and collect() are always called from the slot's own thread. After a run() that returns, the
    scheduler calls collect() with its handle, or frees the result with release(), before the slot runs again."""
    device: int

    def run(self, params: dict) -> SlotOutcome: ...   # raises MapFailed; the result stays in device memory

    def collect(self, handle, want_maps: bool) -> object: ...   # copies the result to the host; raises MapFailed

    def release(self) -> None: ...                    # frees the slot's GPU memory, and a result not collected

    def close(self) -> None: ...


class Backend(Protocol):
    def device_count(self) -> int: ...

    def device_info(self, device: int) -> DeviceInfo: ...

    def estimate(self, params: dict) -> MemoryEstimate: ...   # raises MapFailed

    def open_slot(self, device: int) -> Slot: ...

    def check_device(self, device: int) -> int: ...


@dataclass
class Finished:
    """One map's result: an IPM-like object, or a BatchMapError. Its bytes count as waiting, against the
    scheduler's max_result_bytes, until release() is called."""
    index: int
    result: object
    _nbytes: int = field(default=0, repr=False, compare=False)
    _run: object = field(default=None, repr=False, compare=False)

    def release(self) -> None:
        """Stop counting the result's bytes as waiting. Calling it again does nothing."""
        run = self._run
        if run is not None:
            run.release(self)


def _describe(exc):
    return str(exc) or type(exc).__name__


def _cuda_error(exc):
    return exc.cuda_error if isinstance(exc, MapFailed) else 0


def _map_error(index, exc):
    error = BatchMapError(index, _describe(exc), _cuda_error(exc))
    error.__cause__ = exc
    return error


def _result_bytes(total, maps, want_maps):
    """The host bytes of a result of total bytes, of which maps are its maps, collected with or without them."""
    return total if want_maps else total - maps


def _growth(held, e):
    """The device bytes a slot that holds `held` adds to run a map of estimate e (its buffers only grow)."""
    return held.cmax(e).device_bytes() - held.device_bytes()


def _empty_stats(devices, dead, max_result_bytes):
    return {"devices": {d: {"budget": 0, "maps": 0, "max_concurrent": 0, "peak_committed": 0, "oom_retries": 0,
                            "parked_slots": 0, "keeps_parked_slots": False, "dead": d in dead} for d in devices},
            "direct_deposit_maps": 0, "peak_result_bytes": 0, "result_budget": max_result_bytes}


class _SlotState:
    """A slot position on a device. It lives as long as the scheduler; the slot opens on first use."""

    def __init__(self, device, number):
        self.device = device
        self.number = number
        self.slot = None
        self.held = _NOTHING      # device memory the slot holds, or will hold while it runs its admitted map
        self.broken = False       # its release() raised: it is not used again, and its memory stays counted
        self.shared = False       # another map was admitted on the device while this slot's latest map was


class _Job(NamedTuple):
    index: int
    frees: list                 # (slot state, slot, held before the admission): release() each before the map runs
    limit: int                  # the device's slot_limit when the map was admitted


class _Run:
    """The state of one results() call. Every field is read and written under the scheduler's condition."""

    def __init__(self, cond, params, want_maps, devices, dead, max_result_bytes):
        self.cond = cond
        self.params = params
        self.want_maps = want_maps    # per map: whether collect() copies its maps
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
        self.releases = {}        # device -> release() calls in flight on it; admission on it waits for them
        self.mem_waiters = {}     # device -> slots whose latest admission attempt waited for memory
        self.checks = {}          # device -> check_device() calls in flight on it; admission on it waits for them
        self.active = {}          # device -> slots with an admitted map
        self.holding = set()      # slots whose finished map waits for room to be collected, or is being collected
        self.attempts = [0] * len(params)   # how often each map moved off a GPU that left the batch
        self.pinned = {}          # index -> the GPU that alone may run this out-of-memory retry
        # how much each map's retries lowered slot_limit on its pinned GPU, until the map succeeds or fails (_unpark)
        self.parks = [0] * len(params)
        self.gave_back = {}       # device -> how many parked slots successful retries gave back on it
        self.keeps_parks = {}     # device -> whether it ran out of memory beside other maps after that: its parks stay
        # indices of the out-of-memory retries of maps that ran out of memory alone: such a retry that comes up while
        # nothing runs on its pinned GPU frees what the idle slots hold before it starts (see _try_admit)
        self.retry_alone = set()
        self.total_running = 0
        # most maps running at once over all GPUs, or None: set when a GPU lacks concurrent managed access (_start)
        self.max_total_running = None
        self.waiting_bytes = 0    # host bytes of finished results being collected, queued, or yielded and not released
        self.stats = _empty_stats(devices, dead, max_result_bytes)

    def release(self, item):
        with self.cond:
            if item._run is not self:
                return
            item._run = None
            self.waiting_bytes -= item._nbytes
            self.cond.notify_all()


class Scheduler:
    """Runs batches of maps on worker slots: maps_per_gpu slots per GPU (slot_cap when it is None), each
    driven by its own thread while results() runs.

    stats describes the latest results() call and is updated while it runs:
    {"devices": {d: {"budget", "maps", "max_concurrent", "peak_committed", "oom_retries", "parked_slots",
    "keeps_parked_slots", "dead"}}, "direct_deposit_maps": int, "peak_result_bytes": int, "result_budget": int}.
    "maps" counts the maps that succeeded on that GPU, "oom_retries" the maps sent back to the queue after running
    out of memory there, "parked_slots" how many of its slots run no maps until the out-of-memory retries that
    parked them have succeeded or failed, "keeps_parked_slots" whether it ran out of memory beside other maps after
    successful retries gave it slots back, so that successful retries give it no more slots back in this call (a
    map that fails still gives back its own), and "dead" whether the GPU has left the batch.
    "peak_result_bytes" is the most bytes of finished results on the host at once (being collected, queued, or
    yielded and not released), and "result_budget" is max_result_bytes, the limit on them."""

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
        self.stats = _empty_stats(devices, self._dead, max_result_bytes)

    def results(self, params: list[dict], want_maps=True) -> Iterator[Finished]:
        """Run every map and yield one Finished per map, in completion order. want_maps says whether a map's
        result is collected with its maps: one value for every map, or one per map. Without them its slot copies
        no maps, and the result counts result - maps bytes (MemoryEstimate.maps, SlotOutcome.map_bytes).

        Release each Finished when its result is no longer needed: until then it counts as waiting. At most
        max_result_bytes of finished results are on the host at a time (always at least one), counting those being
        collected: a finished map is collected only once its result fits, and waits in its slot's device memory
        until then, and while the waiting results leave no room for a map's result, no slot starts that map. The
        host therefore holds at most max_result_bytes of results (or one result, if that is larger), plus the
        results that the consumer released and still keeps. A consumer that waits for the next result while it
        keeps unreleased results that leave no room for it waits forever. The maps in progress are not counted.
        Closing the generator (break, an exception in the consumer, Ctrl-C) lets the maps in progress finish,
        frees the device memory of the slots whose results were not collected, drops the results and joins every
        slot thread."""
        params = list(params)
        try:
            want_maps = [bool(w) for w in want_maps]
        except TypeError:
            want_maps = [bool(want_maps)] * len(params)
        if len(want_maps) != len(params):
            raise ValueError(f"want_maps has {len(want_maps)} entries for {len(params)} maps")
        run = self._begin(params, want_maps)
        try:
            self._estimate(run)
            self._start(run)
            for _ in range(len(params)):
                yield self._next_finished(run)
        finally:
            self._stop(run)

    def close(self) -> None:
        """Stop a results() call in progress, then close every slot. Calling it again does nothing. A slot of a
        GPU that left the batch, or one whose memory could not be freed, that fails to close gives a warning;
        any other slot that fails to close raises, after every slot was closed."""
        with self._cond:
            self._closed = True
            run = self._run
            if run is not None:
                run.stopping = True
            self._cond.notify_all()
        if run is not None and not self._finish_stop(run):
            return    # called from a slot thread, which cannot wait for itself: leave the slots open
        with self._cond:
            slots = [(st.device, st.slot, st.broken or st.device in self._dead)
                     for states in self._slots.values() for st in states if st.slot is not None]
            for states in self._slots.values():
                for st in states:
                    st.slot = None
                    st.held = _NOTHING
            self._cond.notify_all()
        first = None
        for d, slot, broken in slots:
            try:
                slot.close()
            except Exception as exc:
                if broken:    # a slot of a GPU that left the batch, or one whose memory could not be freed
                    warnings.warn(f"closing a slot on GPU {d} failed: {_describe(exc)}", RuntimeWarning,
                                  stacklevel=2)
                elif first is None:
                    first = exc
        if first is not None:
            raise first

    # ---- the consumer's side

    def _begin(self, params, want_maps):
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
            run = _Run(self._cond, params, want_maps, self._devices, self._dead, self._max_result_bytes)
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
                run.gave_back[d] = 0
                run.keeps_parks[d] = False
                run.running[d] = 0
                run.committed[d] = held
                run.releases[d] = 0
                run.checks[d] = 0
                run.mem_waiters[d] = set()
                run.active[d] = set()
                dev = run.stats["devices"][d]
                dev["budget"] = run.budget[d]
                dev["peak_committed"] = held
            # slot 0 of every GPU first, then slot 1, and so on, so that the first maps find a thread on every GPU
            for k in range(max(run.n_slots.values(), default=0)):
                for d in devices:
                    st = self._slots[d][k] if k < run.n_slots[d] else None
                    if st is not None and not st.broken:
                        run.threads.append(threading.Thread(target=self._slot_thread, args=(run, st),
                                                            name=f"ipm-batch-{d}-{st.number}", daemon=True))
            if unmanaged:
                # Where the host cannot touch managed memory while a kernel runs, it must not do so while a kernel
                # runs on any GPU: the host writes managed memory (the tree, the counters) of every map it starts.
                # Slots in holding do not count: they run no kernel, and collecting reads device memory through the
                # CUDA runtime, not managed memory from the host. total_running leaves a slot when its map ends,
                # also when its GPU breaks, so the GPU of the one running map dying does not stall the others.
                run.max_total_running = 1
            self._cond.notify_all()
        for d in unmanaged:
            warnings.warn(f"GPU {d} has no concurrent managed memory access, so maps run one at a time over all "
                          f"GPUs", stacklevel=3)
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
        """Fail the maps still waiting when no GPU or slot thread is left to run them (lock held)."""
        while run.pending:
            self._fail(run, run.pending.popleft(), *self._no_gpu())
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
            while self._one_map(run, st):
                pass
        except BaseException as exc:   # a fault in the scheduler itself: stop the batch and tell the consumer
            with self._cond:
                if run.error is None:
                    run.error = exc
                run.stopping = True
                self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def _one_map(self, run, st):
        """Admit, run and hand on one map in this slot; False when the thread should exit. The map's job, outcome,
        result and exception are locals of this call only, so that nothing of a finished map stays referenced
        while the thread waits in _admit for its next map."""
        job = self._admit(run, st)
        if job is None:
            return False
        stage, outcome, exc = self._execute(run, st, job)
        if stage == "done":
            self._complete(run, st, job, outcome)
        else:
            self._failed(run, st, job, stage, exc)
        return True

    def _admit(self, run, st):
        """Wait until the map at the head of pending can start in this slot. Returns None when the thread
        should exit: the run is stopping, the device is dead, the slot is broken, or nothing is left to run
        anywhere (no map waits or runs, and no finished map waits to be collected: a failed collect can send its
        map back to pending)."""
        with self._cond:
            waiters = run.mem_waiters[st.device]
            while True:
                was_waiting = st in waiters
                waiters.discard(st)
                if run.stopping or st.device in self._dead or st.broken or \
                        (not run.pending and run.total_running == 0 and not run.holding):
                    self._cond.notify_all()
                    return None
                job = self._try_admit(run, st) if run.pending else None
                if job is not None or was_waiting != (st in waiters):
                    self._cond.notify_all()
                if job is not None:
                    return job
                self._cond.wait()

    def _try_admit(self, run, st):
        """Admission rules 1-9 of the module docstring, in that order (lock held), for the first map in pending that
        this slot's device may run (maps pinned to another GPU are skipped). Returns a _Job, or None to wait."""
        d = st.device
        if run.releases[d] > 0 or run.checks[d] > 0:
            return None    # memory of d is still being freed (committed[d] is not free yet), or d is being checked
        pos = next((k for k, j in enumerate(run.pending) if run.pinned.get(j, d) == d), None)
        if pos is None:
            return None
        i = run.pending[pos]
        e = run.estimates[i]
        if not self._room_for(run, _result_bytes(e.result, e.maps, run.want_maps[i])):
            return None
        if run.running[d] >= run.slot_limit[d]:
            return None
        if run.max_total_running is not None and run.total_running >= run.max_total_running:
            return None
        if self._taken_elsewhere(run, d, pos):
            return None    # a GPU that runs fewer maps starts this one at once: leave it to that GPU
        need = st.held.cmax(e)
        victims = []
        if run.committed[d] + _growth(st.held, e) > run.budget[d]:
            if e.device_bytes() > run.budget[d] and i not in run.pinned and self._fits_elsewhere(run, d, e):
                # too big for d even alone, but another GPU's budget fits it: leave it to that GPU. d runs nothing
                # while this map is at the head of the queue, as the maps behind it keep their batch order
                return None
            if run.running[d] > 0:
                run.mem_waiters[d].add(st)
                return None
            if self._collecting_on(run, d):
                return None    # a finished map's slot on d holds memory that can be freed only once it is collected
            # Nothing runs on d: free what its idle slots hold, then start this map (once the frees are done, maps
            # that fit can start beside it).
            victims = [x for x in self._slots[d] if x.held.device_bytes() > 0 and not x.broken]
        elif run.running[d] == 0 and i in run.retry_alone:
            # The retry of a map that ran out of memory alone: free what d's idle slots hold, which _failed could
            # not do for a slot whose finished map waited to be collected, and start it while nothing runs on d.
            if self._collecting_on(run, d):
                return None    # a finished map's slot on d holds memory that can be freed only once it is collected
            victims = [x for x in self._slots[d] if x.held.device_bytes() > 0 and not x.broken]
        elif need.device_bytes() > e.device_bytes() and any(w is not st for w in run.mem_waiters[d]):
            # The map fits, but this slot keeps more than the map needs (grow-only buffers from a bigger map)
            # while another slot of d waits for memory: free this slot first, then hold just the map's estimate.
            victims = [st]
        frees = [(x, x.slot, x.held) for x in victims if x.slot is not None]
        for x in victims:
            x.held = _NOTHING
        need = st.held.cmax(e)
        del run.pending[pos]
        run.retry_alone.discard(i)
        st.held = need
        st.shared = bool(run.active[d])
        for x in run.active[d]:
            x.shared = True
        run.active[d].add(st)
        run.committed[d] = self._held_on(d)
        run.running[d] += 1
        run.total_running += 1
        run.releases[d] += len(frees)
        self._update_peaks(run, d)
        return _Job(i, frees, run.slot_limit[d])

    def _execute(self, run, st, job):
        """Free the job's slots, open this slot if needed and run the map, all outside the lock. Never raises.
        Returns (stage, outcome, exception): stage is "done" when the map ran, "free" when it did not run because
        freeing memory took this slot or its device out of the batch, "open" when open_slot raised and "run"
        when the run raised."""
        stage = "free"
        try:
            error = self._free(run, st.device, job.frees)
            with self._cond:
                if st.broken or st.device in self._dead:
                    return stage, None, error
            stage = "open"
            slot = st.slot
            if slot is None:
                slot = self._backend.open_slot(st.device)
                with self._cond:
                    st.slot = slot
                    self._cond.notify_all()
            stage = "run"
            return "done", slot.run(run.params[job.index]), None
        except BaseException as exc:
            return stage, None, exc

    def _free(self, run, d, frees):
        """Call release() on each slot in frees, outside the lock. The caller counted each call in
        run.releases[d] and set the slot's held to what it holds once freed, so admission on d waits until the
        count is back down. A release() that raises leaves the slot's memory counted as before and the slot
        unusable, and makes the device checked: a device that reports an error leaves the batch. Returns the
        first exception raised, or None."""
        first = None
        for state, slot, held_before in frees:
            error = None
            failed, code = False, 0
            try:
                try:
                    slot.release()
                except BaseException as exc:
                    error = exc
                    failed, code = self._check(d, exc)
            finally:
                with self._cond:
                    if error is not None:
                        state.held = held_before    # the memory may still be allocated: keep counting it
                        state.broken = True
                        run.committed[d] = self._held_on(d)
                        if failed:
                            self._mark_dead(run, d, code, _describe(error))
                        elif not self._usable(run, d):
                            self._unpin(run, d)     # no slot of d can run its retries any more
                        self._update_peaks(run, d)
                    run.releases[d] -= 1
                    self._cond.notify_all()
            if error is not None:
                if first is None:
                    first = error
                warnings.warn(f"freeing the memory of a slot on GPU {d} failed: {_describe(error)}; "
                              + ("the GPU leaves the batch" if failed else "the slot is not used again"),
                              RuntimeWarning, stacklevel=2)
        return first

    def _check(self, d, exc):
        """Ask the backend whether device d still works, outside the lock. Returns (failed, CUDA error). A
        device that cannot be checked counts as failed, with the code of the error that made us check."""
        try:
            code = int(self._backend.check_device(d))
        except Exception:
            return True, _cuda_error(exc)
        return code != 0, code

    def _complete(self, run, st, job, outcome):
        """A map that ran. Its slot is idle on its GPU at once, and the GPU gets back the slots that the map's
        out-of-memory retries parked (_unpark), unless it keeps its parks (keeps_parks, see _classify). Its result
        stays in the slot's device memory until it fits next to the waiting results (_room_for). Then its bytes
        count as waiting, and this thread collects it (outside the lock) and queues it. While the result waits for
        room or is being collected, the slot is in holding. A stopping run never collects it: the slot frees its
        memory, and with it the result (release, counted as in _free), unless its GPU has left the batch."""
        d = st.device
        i = job.index
        want_maps = run.want_maps[i]
        nbytes = _result_bytes(outcome.result_bytes, outcome.map_bytes, want_maps)
        frees = []
        with self._cond:
            run.running[d] -= 1
            run.total_running -= 1
            run.active[d].discard(st)
            st.held = outcome.held
            run.committed[d] = self._held_on(d)
            if run.parks[i] and run.keeps_parks[d]:
                run.parks[i] = 0    # d ran out of memory again after it gave parked slots back: they stay parked
            elif run.parks[i]:
                limit = run.slot_limit[d]
                self._unpark(run, d, i)
                run.gave_back[d] += run.slot_limit[d] - limit
            self._update_peaks(run, d)
            self._cond.notify_all()
            run.holding.add(st)
            while not run.stopping and not self._room_for(run, nbytes):
                self._cond.wait()
            slot = st.slot
            stopping = run.stopping
            if stopping:
                run.holding.discard(st)
                if d not in self._dead:    # a GPU that left the batch is not used again: close() closes the slot
                    frees = [(st, slot, st.held)]
                    st.held = _NOTHING
                    run.committed[d] = self._held_on(d)
                    run.releases[d] += 1
            else:
                run.waiting_bytes += nbytes    # counted before the copy, so that it stays within the limit
                self._update_peaks(run, d)
            self._cond.notify_all()
        if stopping:
            if frees:
                self._free(run, d, frees)
            return
        try:
            result = slot.collect(outcome.handle, want_maps)
        except BaseException as exc:
            self._collect_failed(run, st, i, nbytes, exc)
            return
        with self._cond:
            run.holding.discard(st)
            run.stats["devices"][d]["maps"] += 1
            if outcome.direct_deposit:
                run.stats["direct_deposit_maps"] += 1
            if run.stopping:
                run.waiting_bytes -= nbytes
            else:
                run.queue.append(Finished(i, result, nbytes, run))
            self._cond.notify_all()

    def _collect_failed(self, run, st, i, nbytes, exc):
        """A map that ran but whose result could not be collected. After a CUDA error its device is checked,
        outside the lock and counted so that nothing is admitted on it meanwhile. Then _gpu_failure decides, as
        after a failed run, and otherwise the map fails."""
        d = st.device
        code = _cuda_error(exc)
        with self._cond:
            run.waiting_bytes -= nbytes
            # Out of memory (2) is checked as well, unlike after a run, where it means that the map lacked device
            # memory and is retried with fewer maps beside it: a collect allocates no device memory.
            checking = code != 0 and d not in self._dead
            if checking:
                run.checks[d] += 1
            self._cond.notify_all()
        failed, device_code = False, 0
        try:
            if checking:
                failed, device_code = self._check(d, exc)
        finally:
            with self._cond:
                try:
                    run.holding.discard(st)
                    if not self._gpu_failure(run, i, d, exc, device_code if failed else None):
                        run.queue.append(Finished(i, _map_error(i, exc)))
                finally:
                    if checking:
                        run.checks[d] -= 1
                    self._update_peaks(run, d)
                    self._cond.notify_all()

    def _failed(self, run, st, job, stage, exc):
        """A job whose map did not succeed. After an out-of-memory error the slot's memory is freed, and with it
        the memory of the device's idle slots if the map ran alone (but not yet that of a slot whose finished map
        waits to be collected); after another CUDA error the device is checked. Both happen outside the lock and
        are counted so that nothing is admitted on the device meanwhile. Then _classify decides."""
        d = st.device
        code = _cuda_error(exc)
        device_error = None
        frees = []
        freed_idle = False
        checking = False
        if stage == "run":
            with self._cond:
                dead = d in self._dead
                if not dead and code == CUDA_ERROR_MEMORY_ALLOCATION and st.slot is not None:
                    frees = [(st, st.slot, st.held)]
                    st.held = _NOTHING
                    if not st.shared:
                        # it ran alone: the idle slots' memory may be what it lacked, so free that too. A slot whose
                        # finished map waits to be collected keeps it until then: the retry frees it (retry_alone)
                        for x in self._slots[d]:
                            if x is not st and not x.broken and x not in run.active[d] and x.held.device_bytes() > 0:
                                freed_idle = True
                                if x in run.holding:
                                    continue
                                if x.slot is not None:
                                    frees.append((x, x.slot, x.held))
                                x.held = _NOTHING
                    run.committed[d] = self._held_on(d)
                    run.releases[d] += len(frees)
                    self._cond.notify_all()
                elif not dead and code not in (0, CUDA_ERROR_MEMORY_ALLOCATION):
                    checking = True
                    run.checks[d] += 1
                    self._cond.notify_all()
            if frees:
                self._free(run, d, frees)
            elif checking:
                try:
                    failed, device_code = self._check(d, exc)
                except BaseException:
                    with self._cond:
                        run.checks[d] -= 1
                        self._cond.notify_all()
                    raise
                if failed:
                    device_error = device_code
        with self._cond:
            try:
                self._classify(run, st, job, stage, exc, device_error, freed_idle)
            finally:
                if checking:
                    run.checks[d] -= 1    # only now, so that d is not used before its check's verdict is in
                self._update_peaks(run, d)
                self._cond.notify_all()

    def _classify(self, run, st, job, stage, exc, device_error, freed_idle=False):
        """Decide what a failed job means (lock held). It classifies the failures of runs, and _collect_failed
        those of collects; the part that the GPU decides is _gpu_failure, which both use:
        - a GPU where a slot cannot open leaves the batch;
        - a GPU whose check after a CUDA error reports an error leaves the batch, and the map fails with the GPU's
          error, and a map that fails on a GPU that has left the batch moves to another GPU, once (_gpu_failure);
        - a map that did not run because its slot could not free its memory goes back to the front of the queue;
        - a map that runs out of memory goes back to the front of the queue, pinned to its GPU, if another map ran
          on that GPU while it did, or if it ran alone while idle slots held memory, which _failed has freed. In the
          first case the GPU's slot_limit drops below the number of maps in progress when the failure is classified
          (this one included), and below its current value, so that the retry starts only once fewer maps run there;
          if the map was admitted under a higher limit that an error has lowered since, slot_limit need not drop
          below its current value. The first such error on a GPU after successful retries gave it parked slots back
          instead takes slot_limit back to its value before them, and from then on successful retries on that GPU
          give no parked slots back (keeps_parks). A retry that is not pinned, because its GPU has no usable slot
          left, parks nothing. In the second case, the retry is in retry_alone: if it comes up while nothing runs on
          the GPU, it first frees what the idle slots then hold, including a slot whose finished map waited to be
          collected, which _failed could not free (the retry waits until that map is collected). It is not kept
          alone: a map ahead of it in the queue, such as one moved to the front off a GPU that left the batch, can
          start first, and maps that fit can start beside it. Otherwise the map fails, and its GPU gets back the
          slots its retries parked, as it does when a retry succeeds (_complete), even if it keeps its parks: this
          map was too big even alone;
        - any other failure fails the map, and its GPU gets back the slots that the map's out-of-memory retries parked
          there (_unpark)."""
        d = st.device
        i = job.index
        run.running[d] -= 1
        run.total_running -= 1
        run.active[d].discard(st)
        if st.slot is None:
            st.held = _NOTHING    # the slot never opened, so it holds nothing
            run.committed[d] = self._held_on(d)
        if stage == "open":
            self._mark_dead(run, d, _cuda_error(exc), _describe(exc))
        if self._gpu_failure(run, i, d, exc, device_error):
            return
        if stage == "free" and st.broken:
            run.pending.appendleft(i)
        elif stage == "run" and _cuda_error(exc) == CUDA_ERROR_MEMORY_ALLOCATION:
            dev = run.stats["devices"][d]
            if st.shared or freed_idle:
                run.pending.appendleft(i)
                if self._usable(run, d):
                    run.pinned[i] = d
                    if not st.shared:
                        run.retry_alone.add(i)
                dev["oom_retries"] += 1
                if st.shared and run.pinned.get(i) == d:    # an unpinned retry runs elsewhere and parks nothing
                    in_progress = run.running[d] + 1    # running[d] no longer counts this map
                    if job.limit > run.slot_limit[d]:
                        # admitted under a higher limit, which an error since has lowered: the retry already starts
                        # with fewer maps in progress than this map had, so the limit need not drop below its value
                        limit = min(run.slot_limit[d], max(1, in_progress - 1))
                    elif run.gave_back[d] and not run.keeps_parks[d]:
                        # the first such error since d gave parked slots back: d goes back to the limit it ran at
                        # before (a count taken now can miss maps that ended since the error) and keeps it
                        run.keeps_parks[d] = dev["keeps_parked_slots"] = True
                        limit = max(1, min(run.slot_limit[d] - 1, run.slot_limit[d] - run.gave_back[d]))
                    else:
                        limit = max(1, min(run.slot_limit[d] - 1, in_progress - 1))
                    run.parks[i] += run.slot_limit[d] - limit
                    run.slot_limit[d] = limit
            else:
                self._unpark(run, d, i)
                self._fail(run, i, CUDA_ERROR_MEMORY_ALLOCATION,
                           f"out of device memory with no other map on GPU {d}", exc)
            dev["parked_slots"] = run.n_slots[d] - run.slot_limit[d]
        else:
            self._unpark(run, d, i)
            run.queue.append(Finished(i, _map_error(i, exc)))

    def _gpu_failure(self, run, i, d, exc, device_error):
        """The part of a failed map's verdict that its GPU d decides (lock held); returns whether it decided. A GPU
        whose check reported device_error (not None) leaves the batch, and the map, whose error made us check,
        fails with the GPU's error; a map that failed on a GPU that has left the batch moves (_move)."""
        if device_error is not None and d not in self._dead:
            self._mark_dead(run, d, device_error, _describe(exc))
            self._fail(run, i, *self._dead[d], exc)
            return True
        if d in self._dead:
            self._move(run, i, d, exc)
            return True
        return False

    def _move(self, run, i, d, cause):
        """A map that failed on GPU d, which has left the batch, goes back to the front of the queue to run on
        another GPU, once (lock held). The second time it fails with d's error; with no GPU left, it fails."""
        if run.attempts[i] > 0:
            self._fail(run, i, *self._dead[d], cause)
        elif self._all_dead():
            self._fail(run, i, *self._no_gpu(), cause)
        else:
            run.attempts[i] += 1
            run.pending.appendleft(i)

    def _mark_dead(self, run, d, code, detail):
        """Take GPU d out of the batch for the scheduler's lifetime (lock held): its slot threads exit and it is
        not used again. When no GPU is left, the maps still waiting fail."""
        if d in self._dead:
            return
        self._dead[d] = (code, f"GPU {d} failed ({code}): {detail}")
        run.stats["devices"][d]["dead"] = True
        self._unpin(run, d)
        if self._all_dead():
            self._fail_stranded(run)
        self._cond.notify_all()

    # ---- helpers (lock held)

    def _room_for(self, run, nbytes):
        """Whether a result of nbytes fits next to the waiting results: within max_result_bytes, or nothing
        waits (always room for one). It decides both whether a map may start and whether a finished one may be
        collected, so the results on the host never exceed max(max_result_bytes, one result), apart from those the
        consumer released and still keeps."""
        return run.waiting_bytes == 0 or run.waiting_bytes + nbytes <= self._max_result_bytes

    def _taken_elsewhere(self, run, d, pos):
        """Whether another GPU that runs fewer maps than d would start pending[pos], d's next map, at once: the map
        is not pinned to d, no map ahead of it is pinned to that GPU, the GPU is below its slot limit, no memory
        is being freed on it and it is not being checked, and one of its idle slots fits the map in its budget (a
        slot whose finished map waits to be collected, or is being collected, is not idle).
        Each GPU on the way to the one that starts the map runs fewer maps than the one before, so it is found."""
        i = run.pending[pos]
        e = run.estimates[i]
        for d2, running in run.running.items():
            if d2 == d or d2 in self._dead or running >= run.running[d] or running >= run.slot_limit[d2] or \
                    run.releases[d2] > 0 or run.checks[d2] > 0 or run.pinned.get(i, d2) != d2:
                continue
            if any(run.pinned.get(j) == d2 for j in itertools.islice(run.pending, pos)):
                continue    # d2 would start a retry pinned to it first
            room = run.budget[d2] - run.committed[d2]
            if any(not x.broken and x not in run.active[d2] and x not in run.holding and _growth(x.held, e) <= room
                   for x in self._slots[d2][:run.n_slots[d2]]):
                return True
        return False

    def _fits_elsewhere(self, run, d, e):
        """Whether a GPU other than d that can still run maps in this call has a budget that fits a map of estimate
        e. Budgets do not change during a call, so such a GPU never leaves the map in turn and the map does not
        bounce between GPUs; once no such GPU is left (it left the batch, or has no usable slot), d runs the map."""
        return any(d2 != d and budget >= e.device_bytes() and self._usable(run, d2)
                   for d2, budget in run.budget.items())

    def _unpark(self, run, d, i):
        """Give GPU d back the slots that map i's out-of-memory retries parked there, once its last try has ended."""
        run.slot_limit[d] = min(run.n_slots[d], run.slot_limit[d] + run.parks[i])
        run.parks[i] = 0
        run.stats["devices"][d]["parked_slots"] = run.n_slots[d] - run.slot_limit[d]

    def _collecting_on(self, run, d):
        """Whether a finished map on d waits to be collected or is being collected: its slot's device memory holds
        the result, so it cannot be freed until then."""
        return any(x.device == d for x in run.holding)

    def _usable(self, run, d):
        """Whether d can still run maps in this call: it is not dead and one of its slots is not broken."""
        return d not in self._dead and not all(x.broken for x in self._slots[d][:run.n_slots[d]])

    def _unpin(self, run, d):
        """Let any GPU run the retries pinned to d, which can no longer run them."""
        for i in [i for i, x in run.pinned.items() if x == d]:
            del run.pinned[i]
            run.parks[i] = 0
            run.retry_alone.discard(i)
        self._cond.notify_all()

    def _all_dead(self):
        return all(d in self._dead for d in self._devices)

    def _no_gpu(self):
        """(code, message) for a map that no GPU is left to run: the error of the first GPU that failed."""
        if not self._dead:
            return 0, "no usable GPU left"
        code, message = next(iter(self._dead.values()))
        return code, "no usable GPU left: " + message

    def _fail(self, run, i, code, message, cause=None):
        error = BatchMapError(i, message, code)
        error.__cause__ = cause
        run.queue.append(Finished(i, error))

    def _held_on(self, d):
        return sum(st.held.device_bytes() for st in self._slots[d])

    def _update_peaks(self, run, d):
        dev = run.stats["devices"][d]
        dev["max_concurrent"] = max(dev["max_concurrent"], run.running[d])
        dev["peak_committed"] = max(dev["peak_committed"], run.committed[d])
        run.stats["peak_result_bytes"] = max(run.stats["peak_result_bytes"], run.waiting_bytes)
