#!/usr/bin/env python3
"""Compare the total, minima and saddles maps of a parity configuration between two sources.

usage: parity_compare.py --config parnry10_d --a <A> --b <B> [--exe-suffix-a S] [--exe-suffix-b S]
                         [--max-rel 1e-4] [--shift] [--results DIR] [--scratch DIR]
A and B are each a build directory (the configuration's executable is run into the scratch directory) or a
reference run directory holding ipm_magnifications{,_minima,_saddles}.bin.
--shift adds configs.SHIFT to the arguments of every build-directory run; it is refused with a reference
directory, because the stored references are unshifted.
Exits 1 if the max relative difference of any map (or its change below the reference's floor, relative to the
mean) exceeds --max-rel, 2 on a usage or run error, else 0.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402

MAPS = {"total": "ipm_magnifications.bin", "minima": "ipm_magnifications_minima.bin",
        "saddles": "ipm_magnifications_saddles.bin"}


def maps_of(source, cfg, suffix, scratch, label, shift=False):
    """{map name: array} of a reference run directory, or of a run of cfg's executable from a build directory."""
    source = Path(source)
    if (source / MAPS["total"]).exists():
        run_dir = source
    else:
        exe = cf.exe_for(source, cfg.build, suffix)
        rec = bl.run_ipm(exe, (cf.shifted_args if shift else cf.full_args)(cfg, "."), Path(scratch) / label, timeout=3600)
        if not rec["ok"]:
            raise RuntimeError(f"{label}: {rec['error']}\n{rec['stderr_tail']}")
        run_dir = Path(scratch) / label
    return {name: bl.read_map(run_dir / fname) for name, fname in MAPS.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--exe-suffix-a", default="")
    ap.add_argument("--exe-suffix-b", default="")
    ap.add_argument("--max-rel", type=float, default=1e-4)
    ap.add_argument("--shift", action="store_true", help="add configs.SHIFT to the arguments of build-directory runs")
    ap.add_argument("--results")
    ap.add_argument("--scratch", default=str(bl.SCRATCH / "parity_compare"))
    a = ap.parse_args(argv)
    cfg = cf.CONFIGS.get(a.config)
    if cfg is None or not cfg.parities:
        print(f"{a.config}: not a configuration with parities", file=sys.stderr)
        return 2
    if a.shift:
        refs = [s for s in (a.a, a.b) if (Path(s) / MAPS["total"]).exists()]
        if refs:
            print(f"--shift: {refs[0]} is a reference run directory; the stored references are unshifted, "
                  "pass the build directory instead (e.g. --b $B/U0)", file=sys.stderr)
            return 2
    try:
        ma = maps_of(a.a, cfg, a.exe_suffix_a, a.scratch, "a", a.shift)
        mb = maps_of(a.b, cfg, a.exe_suffix_b, a.scratch, "b", a.shift)
        metrics = {name: cm.map_metrics(ma[name], mb[name]) for name in MAPS}
    except (RuntimeError, ValueError, OSError) as e:
        print(e, file=sys.stderr)
        return 2
    failed = False
    for name, m in metrics.items():
        bad = not (max(m["max"], m["offmask_max"]) <= a.max_rel)   # offmask: pixels below the reference's floor
        failed |= bad
        print(f"{name:8s} mean {m['mean']:.2e} p99.99 {m['p9999']:.2e} max {m['max']:.2e} offmask {m['offmask_max']:.2e} "
              f"{'FAIL' if bad else 'ok'} (max-rel {a.max_rel:.0e})")
    if a.results:
        bl.write_json(Path(a.results) / f"{cfg.name}.json", {"a": a.a, "b": a.b, "max_rel": a.max_rel, **metrics})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
