"""Shared helpers for the IPM GPU benchmark and regression harness.

Map files (`ipm_magnifications*.bin`) are two int32 (rows = num_pixels_y2,
columns = num_pixels_y1) followed by the pixel values in the build's precision.
`ipm_parameter_info.txt` has one `name value` pair per line.
"""
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

BENCH_ROOT = Path(os.environ["IPM_BENCH_ROOT"]) if os.environ.get("IPM_BENCH_ROOT") else None
SCRATCH = Path(os.environ.get("IPM_BENCH_TMP", "/tmp/ipm_bench"))

RUNGS = ["R0", "R0p", "R1a", "R1b-i", "R1b-ii", "R2", "R2p", "R3", "R4", "R5"]
RUNG_LABELS = {"R0p": "R0′", "R2p": "R2′"}


def rung_label(rung):
    return RUNG_LABELS.get(rung, rung)


def bench_path(*parts):
    """A path under $IPM_BENCH_ROOT, the directory that holds builds/, ref/ and results/."""
    if BENCH_ROOT is None:
        raise SystemExit("IPM_BENCH_ROOT is not set: point it at the directory that holds builds/, ref/ and results/, "
                         "or pass the paths as arguments")
    return BENCH_ROOT.joinpath(*parts)


def read_map(path):
    """Read a map written by write_array() as a float64 array of shape (rows, cols)."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: shorter than the 8-byte header")
    nrows, ncols = struct.unpack("ii", data[:8])
    if nrows < 1 or ncols < 1:
        raise ValueError(f"{path}: invalid header {nrows}x{ncols}")
    n = nrows * ncols
    payload = len(data) - 8
    if payload == 4 * n:
        dtype = np.float32
    elif payload == 8 * n:
        dtype = np.float64
    else:
        raise ValueError(f"{path}: {payload} payload bytes do not match {nrows}x{ncols} float32 or float64")
    return np.frombuffer(data, dtype=dtype, count=n, offset=8).reshape(nrows, ncols).astype(np.float64)


def read_param_info(path):
    info = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            info[parts[0]] = parts[1].strip()
    return info


def t_shoot_cells(info):
    try:
        return float(info["t_shoot_cells"])
    except KeyError:
        raise KeyError("t_shoot_cells missing from ipm_parameter_info.txt "
                       "(the run did not reach write_files)") from None


def gpu_index():
    """Physical GPU index for nvidia-smi from CUDA_VISIBLE_DEVICES; None if unset or not numeric."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return first if first.isdigit() else None


def nvidia_smi(query, index=None):
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    if index is not None:
        cmd += ["-i", str(index)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class GpuMemorySampler:
    """Polls nvidia-smi memory.used (MiB) on the visible GPU in a thread; peak() is None without a usable GPU index."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.index = gpu_index()
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.index is not None and shutil.which("nvidia-smi") is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            out = nvidia_smi("memory.used", self.index)
            if out.isdigit():
                self.samples.append(int(out))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.peak()

    def peak(self):
        return max(self.samples) if self.samples else None


class HostMemorySampler:
    """Polls VmHWM (MiB) of a process from /proc/<pid>/status in a thread; stop() returns the last peak seen,
    or None if it was never readable (a child that exits before the first poll gives None). A vanished process ends the loop."""

    def __init__(self, pid, interval=0.05):
        self.pid = pid
        self.interval = interval
        self.peak_mib = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        with open(f"/proc/{self.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024
        return None

    def _loop(self):
        while not self._stop.is_set():
            try:
                mib = self._read()
            except (OSError, ValueError):
                return
            if mib is not None and (self.peak_mib is None or mib > self.peak_mib):
                self.peak_mib = mib
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.peak_mib


def _extra_gpus():
    """GPU indices (as strings) from IPM_BENCH_EXTRA_GPUS that count as allocated besides the Slurm ones."""
    return {g.strip() for g in os.environ.get("IPM_BENCH_EXTRA_GPUS", "").split(",") if g.strip()}


def check_gpu_allocation():
    """Error message if the first CUDA_VISIBLE_DEVICES entry differs from the Slurm-allocated GPU, else None."""
    allocated = os.environ.get("SLURM_STEP_GPUS")
    if allocated is None:
        allocated = os.environ.get("SLURM_JOB_GPUS")
    if not allocated:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    want = allocated.split(",")[0].strip()
    if visible != want and visible not in _extra_gpus():
        return (f"CUDA_VISIBLE_DEVICES={visible!r} does not match the Slurm-allocated GPU {want!r} "
                f"(SLURM_STEP_GPUS/SLURM_JOB_GPUS={allocated!r}); only the allocated GPU may be used")
    return None


def check_all_gpus_allocated():
    """Error message if any CUDA_VISIBLE_DEVICES entry is not among the Slurm-allocated GPUs (or
    CUDA_VISIBLE_DEVICES is unset, so that every GPU of the node is visible), else None. None when Slurm sets
    neither SLURM_STEP_GPUS nor SLURM_JOB_GPUS. For tests that use several GPUs."""
    allocated = os.environ.get("SLURM_STEP_GPUS")
    if allocated is None:
        allocated = os.environ.get("SLURM_JOB_GPUS")
    if not allocated:
        return None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return (f"CUDA_VISIBLE_DEVICES is unset, so every GPU of the node is visible, but Slurm allocated only "
                f"{allocated!r} (SLURM_STEP_GPUS/SLURM_JOB_GPUS)")
    want = {g.strip() for g in allocated.split(",") if g.strip()} | _extra_gpus()
    missing = [g.strip() for g in raw.split(",") if g.strip() and g.strip() not in want]
    if missing:
        return (f"CUDA_VISIBLE_DEVICES={raw!r} has entries {missing} that Slurm did not allocate "
                f"(SLURM_STEP_GPUS/SLURM_JOB_GPUS={allocated!r}); only the allocated GPUs may be used")
    return None


def _as_text(data):
    """Decode the raw bytes that TimeoutExpired carries even with text=True; None becomes ''."""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""


def run_ipm(exe, args, outdir, verbose=1, sample_gpu=False, expect_map=True, timeout=None, env=None,
            sample_host=False, keep_stdout=False):
    """Run one IPM executable with `-v <verbose> -o <outdir>/`; wall time covers process start to exit.
    keep_stdout=True adds the whole standard output as record["stdout"] (stdout_tail keeps only its last 2000 characters)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("ipm_*"):
        old.unlink()
    cmd = [str(exe)] + [str(a) for a in args] + ["-v", str(verbose), "-o", str(outdir) + "/"]
    sampler = GpuMemorySampler().start() if sample_gpu else None
    t0 = time.perf_counter()
    timed_out, start_error = False, None
    host_sampler = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if sample_host:
            host_sampler = HostMemorySampler(proc.pid).start()
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
            returncode, stdout, stderr = -1, _as_text(stdout), _as_text(stderr)
        except BaseException:
            proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()
            raise
    except OSError as exc:
        start_error = f"cannot start {exe}: {exc}"
        returncode, stdout, stderr = None, "", ""
    finally:
        wall = time.perf_counter() - t0
        peak = sampler.stop() if sampler else None
        peak_host = host_sampler.stop() if host_sampler else None
    record = {"exe": str(exe), "args": [str(a) for a in args], "outdir": str(outdir),
              "returncode": returncode, "wall_s": wall, "t_shoot_cells": None, "param_info": {},
              "peak_gpu_mem_mib": peak, "peak_host_mem_mib": peak_host, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
              "ok": False, "error": None}
    if keep_stdout:
        record["stdout"] = stdout
    param_path = outdir / "ipm_parameter_info.txt"
    if start_error is not None:
        record["error"] = start_error
    elif timed_out:
        record["error"] = f"timeout after {timeout} s"
    elif returncode != 0:
        record["error"] = f"exit code {returncode}"
    elif not param_path.exists():
        record["error"] = "ipm_parameter_info.txt not written"
    else:
        info = read_param_info(param_path)
        record["param_info"] = info
        try:
            record["t_shoot_cells"] = t_shoot_cells(info)
        except KeyError as exc:
            record["error"] = str(exc)
        if record["error"] is None and expect_map and not (outdir / "ipm_magnifications.bin").exists():
            record["error"] = "ipm_magnifications.bin not written"
    record["ok"] = record["error"] is None
    return record


def _descendants(root):
    """root and the PIDs of all its descendants (from `ps -e -o pid=,ppid=`); just {root} if ps fails."""
    try:
        text = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {root}
    kids = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, todo = {root}, [root]
    while todo:
        for kid in kids.get(todo.pop(), []):
            if kid not in seen:
                seen.add(kid)
                todo.append(kid)
    return seen


def _smi_text(*args):
    """stdout of nvidia-smi; raises RuntimeError with the reason when it is missing or fails."""
    try:
        proc = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"nvidia-smi unavailable: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi exit code {proc.returncode}: {proc.stderr.strip()[-200:]}")
    return proc.stdout


def gpu_foreign_processes():
    """Compute processes on the GPU this process uses (first entry of CUDA_VISIBLE_DEVICES, a physical index with
    CUDA_DEVICE_ORDER=PCI_BUS_ID) other than this process and its descendants, as ([{pid, used_memory, process_name}],
    None); (None, reason) when that cannot be determined. Parsing is bench_batch.parse_foreign."""
    # imported here: bench_batch imports bench_lib at its top level
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch"))
    from bench_batch import parse_foreign
    idx = gpu_index()
    if idx is None:
        return None, "CUDA_VISIBLE_DEVICES does not start with a GPU index"
    try:
        apps = _smi_text("--query-compute-apps=gpu_bus_id,pid,used_memory,process_name", "--format=csv,noheader,nounits")
        buses = _smi_text("--query-gpu=index,pci.bus_id", "--format=csv,noheader")
        found = parse_foreign(apps, buses, _descendants(os.getpid()), with_names=True).get(int(idx), [])
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)
    return [{"pid": p["pid"], "used_memory": p["used_mib"], "process_name": p["name"]} for p in found], None


_SMI_WARNED = set()


def run_ipm_checked(exe, args, outdir, foreign_retries=3, **kwargs):
    """run_ipm with proof of a free GPU: the compute processes of the GPU are listed just before and just after the
    run (own PID tree excluded), with os.getloadavg(), as gpu_foreign_before/after and loadavg_before/after. Foreign
    processes in either list mark the record gpu_clean=False and the run is repeated, up to foreign_retries times;
    the last record is kept (with a warning) if it is still not clean. gpu_clean is None, with gpu_check_error, when
    nvidia-smi cannot say. gpu_attempts counts the runs made."""
    attempts = 0
    while True:
        attempts += 1
        before, why_before = gpu_foreign_processes()
        load_before = os.getloadavg()
        rec = run_ipm(exe, args, outdir, **kwargs)
        after, why_after = gpu_foreign_processes()
        rec.update({"gpu_index": gpu_index(), "gpu_foreign_before": before, "gpu_foreign_after": after,
                    "loadavg_before": list(load_before), "loadavg_after": list(os.getloadavg()),
                    "gpu_attempts": attempts})
        why = why_before or why_after
        if why is not None:
            rec["gpu_clean"], rec["gpu_check_error"] = None, why
            if why not in _SMI_WARNED:
                _SMI_WARNED.add(why)
                print(f"WARNING: cannot verify that the GPU is free ({why}); timings are recorded with gpu_clean=null",
                      file=sys.stderr, flush=True)
            return rec
        rec["gpu_clean"] = not (before or after)
        if rec["gpu_clean"]:
            return rec
        seen = sorted({p["pid"] for p in (before or []) + (after or [])})
        if attempts > foreign_retries:
            print(f"WARNING: foreign process(es) {seen} on GPU {rec['gpu_index']} around {Path(str(exe)).name} "
                  f"{[str(a) for a in args][:3]}...; still not clean after {foreign_retries} retries, keeping the "
                  f"record with gpu_clean=false", file=sys.stderr, flush=True)
            return rec
        print(f"foreign process(es) {seen} on GPU {rec['gpu_index']}; discarding the timing and repeating "
              f"(retry {attempts}/{foreign_retries})", file=sys.stderr, flush=True)


def unclean_records(records):
    """The records a timing must not use: those with gpu_clean == False (old records without the field are kept)."""
    return [r for r in records if r.get("gpu_clean") is False]


def environment_info():
    smi = nvidia_smi("name,driver_version,memory.total,clocks.sm,clocks.max.sm", gpu_index())
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except (OSError, IndexError):
        nvcc = ""
    return {"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": smi, "nvcc": nvcc,
            "gpu_pci_bus_id": nvidia_smi("pci.bus_id", gpu_index()),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def build_meta(build_dir):
    build_dir = Path(build_dir)
    meta = {}
    for name in ("git_rev.txt", "nvcc_version.txt", "flags.txt"):
        p = build_dir / name
        meta[name.rsplit(".", 1)[0]] = p.read_text().strip() if p.exists() else None
    return meta


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def clear_outputs(outdir):
    shutil.rmtree(outdir, ignore_errors=True)
