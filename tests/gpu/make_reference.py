#!/usr/bin/env python3
"""Run the R0 build over the matrix and store the reference maps (spec sections 5 and 8).

Three runs per full-set configuration (run1 is the reference, run2/run3 give the noise floor),
one run per truth-only configuration. Runs execute in the scratch directory and are moved to
ref/<config>/run<k>/ afterwards, so their wall times double as R0's full-matrix timing.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import configs as cf  # noqa: E402


def runs_for(cfg):
    return 3 if "full" in cfg.sets else 1


def reference_dir(ref_root, cfg, k):
    return Path(ref_root) / cfg.name / f"run{k}"


def is_complete(d):
    return (d / "ipm_magnifications.bin").exists() and (d / "ipm_parameter_info.txt").exists()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(bl.BENCH_ROOT / "builds" / "R0"))
    ap.add_argument("--ref-root", default=str(bl.BENCH_ROOT / "ref"))
    ap.add_argument("--scratch", default=str(bl.SCRATCH / "ref"))
    ap.add_argument("--results", default=str(bl.BENCH_ROOT / "results" / "R0"))
    ap.add_argument("--only", nargs="*", help="configuration names (default: every full and truth configuration)")
    ap.add_argument("--force", action="store_true", help="re-run even if the reference exists")
    a = ap.parse_args(argv)
    wanted = [cf.CONFIGS[n] for n in a.only] if a.only else cf.configs_in("full") + cf.configs_in("truth")
    timing_path = Path(a.results) / "timing_full.json"
    if timing_path.exists():
        timing = bl.read_json(timing_path)
    else:
        timing = {"rung": "R0", "set": "full", "env": bl.environment_info(), "build": bl.build_meta(a.build), "records": []}
    done = {(r["config"], r["repeat"]) for r in timing["records"]}
    failures = []
    for cfg in cf.ordered(wanted):
        exe = cf.exe_for(a.build, cfg.build)
        for k in range(1, runs_for(cfg) + 1):
            dest = reference_dir(a.ref_root, cfg, k)
            if is_complete(dest) and not a.force:
                print(f"skip {cfg.name} run{k} (exists)", flush=True)
                continue
            scratch = Path(a.scratch) / cfg.name / f"run{k}"
            print(f"run  {cfg.name} run{k}", flush=True)
            rec = bl.run_ipm(exe, cf.full_args(cfg, a.ref_root), scratch)
            rec.update({"config": cfg.name, "repeat": k, "rung": "R0"})
            if not rec["ok"]:
                failures.append(f"{cfg.name} run{k}: {rec['error']}\n{rec['stderr_tail']}")
                print(failures[-1], flush=True)
                continue
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(scratch), str(dest))
            if "full" in cfg.sets and (cfg.name, k) not in done:
                timing["records"].append(rec)
                bl.write_json(timing_path, timing)
            print(f"     wall {rec['wall_s']:.2f} s, t_shoot_cells {rec['t_shoot_cells']:.3f} s", flush=True)
    if failures:
        print("\nFAILED:\n" + "\n".join(failures))
        return 1
    print("all references present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
