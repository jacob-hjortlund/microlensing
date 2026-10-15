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
import threading
import time
from pathlib import Path

import numpy as np

BENCH_ROOT = Path(os.environ.get("IPM_BENCH_ROOT", "/cfs/home/jahj0154/microlensing_bench"))
SCRATCH = Path(os.environ.get("IPM_BENCH_TMP", "/tmp/ipm_bench"))

RUNGS = ["R0", "R0p", "R1a", "R1b-i", "R1b-ii", "R2", "R2p", "R3", "R4", "R5"]
RUNG_LABELS = {"R0p": "R0′", "R2p": "R2′"}


def rung_label(rung):
    return RUNG_LABELS.get(rung, rung)


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
