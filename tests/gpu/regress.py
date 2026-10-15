#!/usr/bin/env python3
"""Compare a rung's maps with the R0 references (spec section 8).

Double build: fixed thresholds against R0's run1. Float build: error against the double truth
must stay within tol["float"]["ratio"] of R0-float's own error (from tolerances.json), and the
total deposited area must agree with R0-float.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import compare_maps as cm  # noqa: E402
import configs as cf  # noqa: E402

MAP_FILES = {"pixels": "ipm_magnifications.bin",
             "minima": "ipm_magnifications_minima.bin",
             "saddles": "ipm_magnifications_saddles.bin"}


def reference_map(ref_root, name, k, which="pixels"):
    p = Path(ref_root) / name / f"run{k}" / MAP_FILES[which]
    if not p.exists():
        raise FileNotFoundError(f"reference map missing: {p} (run make_reference.py first)")
    return bl.read_map(p)


def check_config(cfg, outdir, ref_root, tol, rung):
    result = {"config": cfg.name, "build": cfg.build, "rung": rung, "maps": {}, "pass": True, "failures": [], "flags": []}
    which_list = ["pixels", "minima", "saddles"] if cfg.parities else ["pixels"]
    loaded = {}
    for which in which_list:
        cand = bl.read_map(Path(outdir) / MAP_FILES[which])
        loaded[which] = cand
        ref = reference_map(ref_root, cfg.name, 1, which)
        m_r0 = cm.map_metrics(cand, ref, tol["floor_frac"])
        entry = {"vs_r0": m_r0}
        if cfg.build == "double":
            verdict = cm.evaluate_double(m_r0, tol)
        else:
            truth = reference_map(ref_root, cfg.truth, 1, which)
            m_truth = cm.map_metrics(cand, truth, tol["floor_frac"])
            try:
                ref_truth = tol["float_truth_error"][cfg.name][which]
            except KeyError:
                raise KeyError(f"tolerances.json has no float_truth_error for {cfg.name}/{which}; run calibrate.py") from None
            entry["vs_truth"] = m_truth
            verdict = cm.evaluate_float(m_truth, ref_truth, m_r0, tol)
        entry["verdict"] = verdict
        result["maps"][which] = entry
        result["pass"] = result["pass"] and verdict["pass"]
        result["failures"] += [f"{which}: {f}" for f in verdict["failures"]]
        result["flags"] += [f"{which}: {f}" for f in verdict["flags"]]
    if cfg.parities:
        pc = cm.parity_consistency(loaded["pixels"], loaded["minima"], loaded["saddles"])
        result["parity_consistency"] = pc
        if pc > tol["parity_consistency"]:
            result["pass"] = False
            result["failures"].append(f"pixels != minima + saddles: {pc:.3e} > {tol['parity_consistency']:.1e}")
    if cfg.starfile_from:   # information only: the same stars read back from a file
        result["vs_source_field"] = cm.map_metrics(loaded["pixels"], reference_map(ref_root, cfg.starfile_from, 1), tol["floor_frac"])
    return result


def summarize(res):
    m = res.get("maps", {}).get("pixels")
    if not m:
        return "; ".join(res["failures"])
    key = "vs_truth" if "vs_truth" in m else "vs_r0"
    s = m[key]
    txt = f"{key} mean {s['mean']:.2e} p99.9 {s['p999']:.2e} max {s['max']:.2e} sum {m['vs_r0']['sum_ratio']:+.1e}"
    if res["failures"]:
        txt += " | " + "; ".join(res["failures"])
    if res["flags"]:
        txt += " | flags: " + "; ".join(res["flags"])
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--set", default="ladder", choices=["ladder", "full"])
    ap.add_argument("--ref-root", default=str(bl.BENCH_ROOT / "ref"))
    ap.add_argument("--tolerances", default=str(bl.BENCH_ROOT / "results" / "tolerances.json"))
    ap.add_argument("--results", default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep-outputs", action="store_true")
    a = ap.parse_args(argv)
    tol = cm.load_tolerances(a.tolerances)
    results_dir = Path(a.results or bl.BENCH_ROOT / "results" / a.rung)
    scratch = Path(a.scratch or bl.SCRATCH / a.rung / "regress")
    cfgs = cf.regression_configs(a.rung, a.set)
    if a.only:
        cfgs = [c for c in cfgs if c.name in a.only]
    out = {"rung": a.rung, "set": a.set, "env": bl.environment_info(), "build": bl.build_meta(a.build),
           "tolerances": {k: tol[k] for k in ("floor_frac", "double", "float", "parity_consistency")},
           "configs": [], "pass": True}
    for cfg in cf.ordered(cfgs):
        outdir = scratch / cfg.name
        rec = bl.run_ipm(cf.exe_for(a.build, cfg.build), cf.full_args(cfg, a.ref_root), outdir)
        if not rec["ok"]:
            res = {"config": cfg.name, "build": cfg.build, "rung": a.rung, "pass": False,
                   "failures": [f"run failed: {rec['error']}"], "flags": [], "maps": {}}
        else:
            res = check_config(cfg, outdir, a.ref_root, tol, a.rung)
        res["run"] = {k: rec[k] for k in ("wall_s", "t_shoot_cells", "returncode", "stderr_tail")}
        out["configs"].append(res)
        out["pass"] = out["pass"] and res["pass"]
        print(f"{'PASS' if res['pass'] else 'FAIL':4s} {cfg.name:12s} {summarize(res)}", flush=True)
        if not a.keep_outputs:
            bl.clear_outputs(outdir)
        bl.write_json(results_dir / f"regression_{a.set}.json", out)
    print("REGRESSION", "PASS" if out["pass"] else "FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
