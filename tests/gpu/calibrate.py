#!/usr/bin/env python3
"""Noise floors and float truth errors from the R0 references; writes tolerances.json (spec section 8)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402
from regress import reference_map  # noqa: E402


def sanity_warnings(tol):
    """Double thresholds must be at least 10x the measured double noise floor."""
    warnings = []
    for name, per_map in tol["floors"].items():
        if cf.CONFIGS[name].build != "double":
            continue
        for which, floors in per_map.items():
            for key in ("mean", "p999", "p9999", "offmask_max"):
                worst = max(f[key] for f in floors)
                if worst * 10 > tol["double"][key]:
                    warnings.append(f"{name} {which}: double floor {key}={worst:.2e} is within 10x of the threshold {tol['double'][key]:.1e}")
    return warnings


def calibrate(ref_root, tol):
    tol["floors"] = {}
    tol["float_truth_error"] = {}
    lines = []
    for cfg in cf.configs_in("full"):
        which_list = ["pixels", "minima", "saddles"] if cfg.parities else ["pixels"]
        for which in which_list:
            ref1 = reference_map(ref_root, cfg.name, 1, which)
            floors = [cm.map_metrics(reference_map(ref_root, cfg.name, k, which), ref1, tol["floor_frac"]) for k in (2, 3)]
            tol["floors"].setdefault(cfg.name, {})[which] = floors
            line = (f"{cfg.name:12s} {which:7s} floor mean {max(f['mean'] for f in floors):.2e} "
                    f"p99.9 {max(f['p999'] for f in floors):.2e} max {max(f['max'] for f in floors):.2e}")
            if cfg.build == "float":
                err = cm.map_metrics(ref1, reference_map(ref_root, cfg.truth, 1, which), tol["floor_frac"])
                tol["float_truth_error"].setdefault(cfg.name, {})[which] = err
                line += (f" | vs truth mean {err['mean']:.2e} p99.9 {err['p999']:.2e} "
                         f"p99.99 {err['p9999']:.2e} max {err['max']:.2e}")
            lines.append(line)
            print(line, flush=True)
    warnings = sanity_warnings(tol)
    tol["warnings"] = warnings
    for w in warnings:
        print("WARNING", w)
    return tol, lines, warnings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", default=str(bl.BENCH_ROOT / "ref"))
    ap.add_argument("--out", default=str(bl.BENCH_ROOT / "results" / "tolerances.json"))
    a = ap.parse_args(argv)
    tol, lines, warnings = calibrate(a.ref_root, cm.default_tolerances())
    bl.write_json(a.out, tol)
    summary = Path(a.out).with_name("tolerances_summary.md")
    summary.write_text("```\n" + "\n".join(lines + warnings) + "\n```\n")
    print("wrote", a.out, "and", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
