#!/usr/bin/env python3
"""GPU busy and kernel-overlap fractions from an Nsight Systems trace:

    nsys stats -r cuda_gpu_trace --format csv myrun.nsys-rep > trace.csv
    python tests/batch/gpu_busy.py trace.csv

Prints wall_s (first kernel start to last kernel end), busy_fraction (the union of the kernel intervals divided by
wall_s) and overlap_fraction (time with two or more kernels running divided by the busy time). Memory copies and
sets are ignored: only rows with a grid size count as kernels."""
import csv
import io
import sys
from pathlib import Path

GRID_COLUMNS = ("GrdX", "Grid X")


def read_trace(path):
    """Kernel intervals (start_s, end_s) of a cuda_gpu_trace CSV; text before the header line is skipped."""
    lines = Path(path).read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"Start (ns)"') or ln.startswith("Start (ns)")), None)
    if start is None:
        raise ValueError(f"{path}: no 'Start (ns)' header line")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    grid = next((c for c in GRID_COLUMNS if rows and c in rows[0]), None)
    intervals = []
    for row in rows:
        if grid is not None and not (row[grid] or "").strip():
            continue
        t0 = float(row["Start (ns)"]) * 1e-9
        intervals.append((t0, t0 + float(row["Duration (ns)"]) * 1e-9))
    return intervals


def busy_stats(intervals):
    """wall_s, busy_fraction and overlap_fraction of (start, end) intervals in seconds."""
    if not intervals:
        raise ValueError("no kernel intervals")
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals], key=lambda x: (x[0], x[1]))
    wall0, wall1 = min(s for s, _ in intervals), max(e for _, e in intervals)
    busy = overlap = 0.0
    depth, last = 0, wall0
    for t, step in events:
        if depth >= 1:
            busy += t - last
        if depth >= 2:
            overlap += t - last
        depth += step
        last = t
    wall = wall1 - wall0
    return {"wall_s": wall, "busy_fraction": busy / wall if wall > 0 else 0.0,
            "overlap_fraction": overlap / busy if busy > 0 else 0.0}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    stats = busy_stats(read_trace(argv[0]))
    for k in ("wall_s", "busy_fraction", "overlap_fraction"):
        print(f"{k} {stats[k]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
