#!/usr/bin/env python3
"""Exit 0 when the given GPUs are idle (no compute process, at most 100 MiB in use), else 1 with the reason.

usage: gpu_idle.py [index ...]   (default: the indices in CUDA_VISIBLE_DEVICES)
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch"))
from bench_batch import parse_foreign  # noqa: E402

MAX_IDLE_MIB = 100


def verdict(apps_csv, gpus_csv, indices):
    """(True, "") when every GPU in indices is idle, else (False, reason).
    apps_csv: `--query-compute-apps=gpu_bus_id,pid`; gpus_csv: `--query-gpu=index,pci.bus_id,memory.used`."""
    # parse_foreign wants a third (used memory) column that is irrelevant here
    apps = "\n".join(f"{ln.strip()}, 0" for ln in apps_csv.strip().splitlines() if ln.strip())
    buses = "\n".join(",".join(ln.split(",")[:2]) for ln in gpus_csv.strip().splitlines())
    procs = parse_foreign(apps, buses)
    used = {}
    for line in gpus_csv.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        used[int(parts[0])] = int(float(parts[2]))
    reasons = []
    for i in indices:
        if i not in used:
            reasons.append(f"GPU {i} is not listed by nvidia-smi")
            continue
        if procs.get(i):
            reasons.append(f"GPU {i} has compute process(es) {[p['pid'] for p in procs[i]]}")
        if used[i] > MAX_IDLE_MIB:
            reasons.append(f"GPU {i} has {used[i]} MiB in use")
    return (not reasons, "; ".join(reasons))


def _smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=True).stdout


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        indices = [int(x) for x in argv]
    else:
        indices = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not indices:
        print("no GPU indices given and CUDA_VISIBLE_DEVICES is empty", file=sys.stderr)
        return 2
    try:
        apps = _smi("--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader,nounits")
        gpus = _smi("--query-gpu=index,pci.bus_id,memory.used", "--format=csv,noheader,nounits")
    except (OSError, subprocess.SubprocessError) as e:
        print(f"nvidia-smi failed: {e}", file=sys.stderr)
        return 1
    ok, why = verdict(apps, gpus, indices)
    if not ok:
        print(why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
