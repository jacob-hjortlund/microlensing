#!/usr/bin/env python3
"""Throughput of a serial IPM.run() loop against BatchedIPM, in memory and in save mode, on 1 and several GPUs.

    python tests/batch/bench_batch.py --configs nb_f small_f --gpus 1 2 --modes memory save \\
        --save-root /tmp/ipm_bench/save --maps 64 --repeats 3 --out results.json

Every measurement runs in a fresh child process (`--one <key>`) that prints one JSON line. The run is resumable:
keys already in --out are skipped. The GPUs of this node are not isolated, so each measurement picks its GPUs
from the allocated ones (CUDA_VISIBLE_DEVICES of this process), preferring GPUs without a foreign compute
process, runs its child with CUDA_VISIBLE_DEVICES set to them (devices 0..g-1 inside the child) and records the
foreign processes on them before and after.

The timed region of a child is: serial, the loop of IPM(...).run() [+ save()] over all maps; batch, building the
BatchedIPM, running it (to the last map consumed, or the last file written in save mode) but not close().
Before timing, the child creates the CUDA context and runs an untimed 200x200 map of the same lens; the batch
kind also runs an untimed BatchedIPM of g 200x200 maps, which brings up the workers of the devices.
The peak RSS (VmHWM) of a child includes the warm-up. The serial kind is measured on one GPU only (g = 1).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu"))

from bench_lib import check_all_gpus_allocated, median, read_json, write_json  # noqa: E402

LENS_NB = dict(kappa_tot=0.4, shear=0.4, kappa_star=0.2)
CONFIGS = {
    "nb_f": dict(LENS_NB, half_length_y1=25, half_length_y2=25, num_pixels_y1=5000, num_pixels_y2=5000),
    "small_f": dict(LENS_NB, half_length_y1=25, half_length_y2=25, num_pixels_y1=1000, num_pixels_y2=1000),
    "stars1e6_f": dict(kappa_tot=0.3, shear=0.3, kappa_star=0.27, half_length_y1=505, half_length_y2=505,
                       num_pixels_y1=5000, num_pixels_y2=5000),
    "px40k_f": dict(LENS_NB, half_length_y1=25, half_length_y2=25, num_pixels_y1=40000, num_pixels_y2=40000),
}
WARMUP_PIXELS = 200
STDERR_TAIL = 2000


def speedup(serial, batch):
    return batch["maps_per_s"] / serial["maps_per_s"]


# ---- GPU choice

def parse_foreign(apps_csv, buses_csv, own_pids=()):
    """{gpu index: [{"pid", "used_mib"}]} from `--query-compute-apps=gpu_bus_id,pid,used_memory` and
    `--query-gpu=index,pci.bus_id` output (csv, noheader, nounits); pids in own_pids are not foreign."""
    index_of = {}
    for line in buses_csv.strip().splitlines():
        idx, bus = [x.strip() for x in line.split(",", 1)]
        index_of[bus.upper()] = int(idx)
    out = {}
    for line in apps_csv.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        bus, pid, used = parts[0].upper(), int(parts[1]), parts[2]
        if pid in own_pids or bus not in index_of:
            continue
        out.setdefault(index_of[bus], []).append({"pid": pid, "used_mib": int(used) if used.isdigit() else None})
    return out


def _smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def foreign_processes(own_pids=()):
    return parse_foreign(_smi("--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"),
                         _smi("--query-gpu=index,pci.bus_id", "--format=csv,noheader"), own_pids)


def allocated_gpus():
    """Physical indices in CUDA_VISIBLE_DEVICES of this process (every GPU nvidia-smi lists when it is unset)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return [int(ln.split(",")[0]) for ln in _smi("--query-gpu=index", "--format=csv,noheader").split()]
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"CUDA_VISIBLE_DEVICES={raw!r} must list GPU indices, not UUIDs")


def choose_gpus(allocated, g, foreign):
    """g of the allocated GPUs, those with the fewest foreign processes first (ties by order); sorted."""
    if g > len(allocated):
        raise ValueError(f"{g} GPUs requested but only {len(allocated)} allocated: {allocated}")
    ranked = sorted(allocated, key=lambda i: len(foreign.get(i, [])))   # sorted() is stable
    return sorted(ranked[:g])


def _on(foreign, gpus):
    return {str(i): foreign[i] for i in gpus if foreign.get(i)}


# ---- the plan and the table

def plan(configs, gpus, modes, repeats, maps, save_root):
    """Measurement specs in run order. Serial measurements use one GPU whatever --gpus says."""
    specs = []
    for cfg in configs:
        for mode in modes:
            for rep in range(repeats):
                for kind, g in [("serial", 1)] + [("batch", g) for g in gpus]:
                    n = max(maps, 8 * g) if cfg == "px40k_f" else maps
                    specs.append({"key": f"{cfg}|{mode}|{g}|{kind}|{rep}", "config": cfg, "mode": mode, "kind": kind,
                                  "gpus": g, "rep": rep, "maps": n,
                                  "save_root": str(save_root) if mode == "save" else None})
    return specs


def run_plan(specs, out, launch):
    """Run the specs whose keys are not in the results file `out`, saving it after each; returns all results."""
    out = Path(out)
    results = read_json(out) if out.exists() else {}
    for spec in specs:
        if spec["key"] in results:
            print(f"skip {spec['key']} (done)", file=sys.stderr)
            continue
        print(f"run  {spec['key']}", file=sys.stderr, flush=True)
        results[spec["key"]] = launch(spec)
        write_json(out, results)
    return results


def _ok_rates(recs):
    return [r["maps_per_s"] for r in recs if r.get("ok") and r.get("maps_per_s") is not None]


def _has_foreign(recs):
    return any(any(r.get(k, {}).values()) for r in recs for k in ("foreign_before", "foreign_after"))


def render_table(results):
    """Markdown table, one row per (config, mode, save root, GPUs) with the medians of the repeats."""
    recs = list(results.values()) if isinstance(results, dict) else list(results)
    groups = {}
    for r in recs:
        groups.setdefault((r["config"], r["mode"], r.get("save_root"), r["gpus"] if r["kind"] == "batch" else None), []).append(r)
    rows = []
    for (cfg, mode, root, g), batch in groups.items():
        if g is None:
            continue
        serial = groups.get((cfg, mode, root, None), [])
        rows.append((cfg, mode, root, g, serial, batch))
    for (cfg, mode, root, g), serial in groups.items():     # serial-only configurations still get a row
        if g is None and not any(r[:3] == (cfg, mode, root) for r in rows):
            rows.append((cfg, mode, root, 1, serial, []))
    rows.sort(key=lambda r: (r[0], r[1], str(r[2]), r[3]))
    head = ["config", "mode", "save root", "GPUs", "GPU ids", "serial maps/s", "batch maps/s", "speedup",
            "batch peak RSS MiB", "foreign"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for cfg, mode, root, g, serial, batch in rows:
        s, b = median(_ok_rates(serial)), median(_ok_rates(batch))
        rss = median([r.get("peak_rss_mib") for r in batch if r.get("ok")])
        ids = sorted({i for r in serial + batch for i in r.get("chosen_gpus", [])})
        cells = [cfg, mode, root or "-", str(g), ",".join(map(str, ids)) or "-",
                 f"{s:.3f}" if s else "-", f"{b:.3f}" if b else "-",
                 f"{speedup({'maps_per_s': s}, {'maps_per_s': b}):.2f}" if s and b else "-",
                 f"{rss:.0f}" if rss is not None else "-", "foreign" if _has_foreign(serial + batch) else ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---- the child

def _vmhwm_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    return None


def measure(config, kind, mode, g, maps, save_root):
    """Warm up, then time one measurement; returns the record (without the parent's fields)."""
    from microlensing.IPM.batch import BatchedIPM, BatchMapError
    from microlensing.IPM.ipm import IPM

    kw = dict(CONFIGS[config])
    warm = dict(kw, num_pixels_y1=WARMUP_PIXELS, num_pixels_y2=WARMUP_PIXELS)
    save = mode == "save"
    flags = dict(write_maps=True, write_stars=False, write_parities=False, write_histograms=False) if save else {}
    seeds = list(range(1, maps + 1))
    rec = {"maps": maps, "map_errors": 0}

    IPM(**warm, random_seed=1).run()     # creates the CUDA context
    if kind == "batch":
        with BatchedIPM(num_maps=g, **warm, random_seed=list(range(1, g + 1)), devices=list(range(g))) as w:
            for _ in w.run():
                pass

    outdir = tempfile.mkdtemp(prefix="bench_", dir=save_root) if save else None
    try:
        if kind == "serial":
            t0 = time.perf_counter()
            for s in seeds:
                m = IPM(**kw, random_seed=s, **flags, **({"outfile_prefix": f"{outdir}/{s}_"} if save else {}))
                m.run()
                if save:
                    m.save()
                del m
            wall = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            b = BatchedIPM(num_maps=maps, **kw, random_seed=seeds, devices=list(range(g)), **flags,
                           **({"outfile_prefix": f"{outdir}/"} if save else {}))
            try:
                if save:
                    rec["map_errors"] = sum(1 for sm in b.run(save=True) if sm.error is not None)
                else:
                    for _, res in b.run():
                        if isinstance(res, BatchMapError):
                            rec["map_errors"] += 1
                        del res
                wall = time.perf_counter() - t0
                rec["scheduler_stats"] = b.last_run_stats
            finally:
                b.close()
        if save:
            rec["files_written"] = sum(1 for p in Path(outdir).iterdir())
    finally:
        if outdir:
            shutil.rmtree(outdir, ignore_errors=True)
    rec.update(wall_s=wall, maps_per_s=maps / wall, peak_rss_mib=_vmhwm_mib())
    return rec


def child_main(args):
    # the C++ library prints to fd 1: send it to stderr and keep the real stdout for the one JSON line
    real = os.dup(1)
    os.dup2(2, 1)
    try:
        rec = measure(args.config, args.kind, args.mode, args.gpu_count, args.map_count, args.save_root)
        rec["ok"] = rec["map_errors"] == 0
        if not rec["ok"]:
            rec["error"] = f"{rec['map_errors']} maps failed"
    except BaseException as exc:    # noqa: BLE001 - the parent must get a record whatever happens
        import traceback
        traceback.print_exc()
        rec = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "maps_per_s": None}
    line = json.dumps(rec, default=str) + "\n"
    os.write(real, line.encode())
    return 0


# ---- the parent

def launch_child(spec, timeout):
    allocated = allocated_gpus()
    own = {os.getpid()}
    before = foreign_processes(own)
    chosen = choose_gpus(allocated, spec["gpus"], before)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, chosen)), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", spec["key"], "--config", spec["config"],
           "--kind", spec["kind"], "--mode", spec["mode"], "--gpu-count", str(spec["gpus"]),
           "--map-count", str(spec["maps"])]
    if spec["save_root"]:
        cmd += ["--save-root", spec["save_root"]]
    rec, stderr = None, ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        stderr = proc.stderr
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        try:
            rec = json.loads(lines[-1])
        except (IndexError, ValueError):
            rec = {"ok": False, "error": f"no JSON from the child (exit code {proc.returncode})", "maps_per_s": None}
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr
        stderr = err.decode(errors="replace") if isinstance(err, bytes) else (err or "")
        rec = {"ok": False, "error": f"timeout after {timeout} s", "maps_per_s": None}
    after = foreign_processes(own)
    rec.update({k: spec[k] for k in ("key", "config", "mode", "kind", "gpus", "rep", "save_root")})
    rec.update(chosen_gpus=chosen, foreign_before=_on(before, chosen), foreign_after=_on(after, chosen),
               stderr_tail=stderr[-STDERR_TAIL:])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["nb_f"], choices=sorted(CONFIGS))
    ap.add_argument("--gpus", nargs="+", type=int, default=[1])
    ap.add_argument("--modes", nargs="+", default=["memory"], choices=["memory", "save"])
    ap.add_argument("--save-root", default=None, help="directory under which each save measurement makes (and deletes) its own")
    ap.add_argument("--maps", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=7200, help="seconds per measurement")
    ap.add_argument("--one", default=None, help="child protocol: run one measurement and print its JSON")
    ap.add_argument("--config", choices=sorted(CONFIGS), help=argparse.SUPPRESS)
    ap.add_argument("--kind", choices=["serial", "batch"], help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=["memory", "save"], help=argparse.SUPPRESS)
    ap.add_argument("--gpu-count", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--map-count", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.one is not None:
        return child_main(args)

    if args.out is None:
        ap.error("--out is required")
    if "save" in args.modes:
        if not args.save_root:
            ap.error("--save-root is required for save mode")
        Path(args.save_root).mkdir(parents=True, exist_ok=True)
    problem = check_all_gpus_allocated()
    if problem:
        sys.exit(problem)
    specs = plan(args.configs, args.gpus, args.modes, args.repeats, args.maps, args.save_root)
    results = run_plan(specs, args.out, lambda spec: launch_child(spec, args.timeout))
    mine = {k: v for k, v in results.items() if k in {s["key"] for s in specs}}
    print(render_table(mine))
    for r in mine.values():
        if not r.get("ok"):
            print(f"FAILED {r['key']}: {r.get('error')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
