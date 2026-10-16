#!/usr/bin/env python3
"""Render the round-2 result tables (spec section 9) from the JSON results.

Kernel time (t_shoot_cells) of the base comes from round 1's files; wall and Python time of the base come from the
round-2 runs. A median more than 3 % slower than the base is marked with SLOW_MARK and listed at the end.
Anything missing is shown as a dash and listed under "Missing inputs"."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as bl  # noqa: E402
import configs as cf  # noqa: E402
from report import caption, fmt, regression_table, speedup  # noqa: E402
from time_rung import summarize  # noqa: E402

SLOW_MARK = " ⚠"
SLOW_LIMIT = 1.03
ROUND2_RUNGS = ["A1", "A2", "A4"]
PYTHON_RUNGS = ["base", "base-repeat", "A1", "A2", "A4"]
HM100_NOTE = ("¹ Round 1's full-matrix median for hm100_f was disturbed (0.339 s; two of its three repeats were slow, "
              "the ladder gives {ladder} s for the same binary). The speedup and the slowdown flag for this row use the ladder value.")


def is_slow(base, value):
    return base is not None and value is not None and value > base * SLOW_LIMIT


class Ctx:
    """Reads result files, remembering what is missing and which cells were flagged."""

    def __init__(self, root):
        self.root = Path(root)
        self.missing = []
        self.flags = []

    def load(self, rel):
        p = self.root / rel
        if not p.exists():
            self.missing.append(rel)
            return None
        return bl.read_json(p)

    def summary(self, rel):
        d = self.load(rel)
        if d is None:
            return None, {}
        return d, (d.get("summary") or summarize(d["records"]))

    def cell(self, where, base, value, nd=3):
        """Formatted value, marked and logged when more than 3 % slower than base."""
        if is_slow(base, value):
            self.flags.append(f"{where}: {value:.{nd}f} against base {base:.{nd}f} ({100 * (value / base - 1):+.1f} %)")
            return fmt(value, nd) + SLOW_MARK
        return fmt(value, nd)


def run_note(env):
    if not env:
        return "unknown run"
    return f"GPU {env.get('cuda_visible_devices', '?')}, job {env.get('slurm_job_id', '?')}"


def rung_table(ctx, title, data, rungs, names, base_ref, flag=True):
    """data: rung -> config -> value. base_ref: config -> value of the base, the reference for speedups and flags."""
    head = "| rung | " + " | ".join(f"{n} [s] | vs base | vs prev" for n in names) + " |"
    rows = [head, "|---|" + "---|---|---|" * len(names)]
    prev = {n: None for n in names}
    for rung in rungs:
        cells = []
        for n in names:
            v = data.get(rung, {}).get(n)
            b = base_ref.get(n)
            shown = ctx.cell(f"{title}, {rung}, {n}", b, v) if flag and rung != "base" else fmt(v)
            cells.append(f"{shown} | {fmt(speedup(b, v), 2)} | {fmt(speedup(prev[n], v), 2)}")
            if v is not None:
                prev[n] = v
        rows.append(f"| {bl.rung_label(rung)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def values(summary, key):
    return {c: s.get(key) for c, s in summary.items()}


def kernel_and_wall_tables(ctx, set_name, base_kernel_file, inter_file, names, title):
    """Two sections (kernel, wall) for one configuration set; returns the markdown and the interleaved env."""
    _, base_k = ctx.summary(base_kernel_file)
    inter = ctx.load(inter_file)
    summ = inter["summary"] if inter else {}
    rungs = [r for r in (inter["rungs"] if inter else ["base"] + ROUND2_RUNGS)]
    if "base" not in rungs:
        rungs = ["base"] + rungs
    kernel = {r: values(summ.get(r, {}), "median_t_shoot_cells") for r in rungs}
    kernel["base"] = values(base_k, "median_t_shoot_cells")
    wall = {r: values(summ.get(r, {}), "median_wall_s") for r in rungs}
    base_note = run_note(bl.read_json(ctx.root / base_kernel_file)["env"]) if (ctx.root / base_kernel_file).exists() else "missing"
    r2_note = run_note(inter["env"]) if inter else "missing"
    out = [f"## {title}: median shooting time t_shoot_cells", "",
           f"Base from round 1 ({base_kernel_file}; {base_note}); other rungs from {inter_file} ({r2_note}).", "",
           rung_table(ctx, f"{title} kernel", kernel, rungs, names, kernel["base"]), "",
           f"## {title}: median end-to-end wall time", "",
           f"All rungs from {inter_file} ({r2_note}); the base is re-measured in the same run.", "",
           rung_table(ctx, f"{title} wall", wall, rungs, names, wall["base"]), ""]
    return out, (inter["env"] if inter else None)


def full_table(ctx, key, title, base_values, inter, extra_note=""):
    """Configuration rows, one value and speedup per rung; base_values supplies the base column."""
    summ = inter["summary"] if inter else {}
    rungs = [r for r in (inter["rungs"] if inter else []) if r != "base"] or ROUND2_RUNGS
    names = sorted({n for n in base_values if not n.startswith("_ref_")} | {c for r in rungs for c in summ.get(r, {})},
                   key=lambda n: list(cf.CONFIGS).index(n) if n in cf.CONFIGS else 99)
    head = "| configuration | base [s] | " + " | ".join(f"{r} [s] | {r} vs base" for r in rungs) + " |"
    rows = [head, "|---|---|" + "---|---|" * len(rungs)]
    for n in names:
        b = base_values.get(n)
        ref = base_values.get(f"_ref_{n}", b)
        cells = [fmt(b) + (f" (ladder {fmt(ref)}) ¹" if f"_ref_{n}" in base_values else "")]
        for r in rungs:
            v = summ.get(r, {}).get(n, {}).get(key)
            cells.append(f"{ctx.cell(f'{title}, {r}, {n}', ref, v)} | {fmt(speedup(ref, v), 2)}")
        rows.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def python_section(ctx):
    data = {}
    for rung in PYTHON_RUNGS:
        d = ctx.load(f"{rung}/python.json")
        if d:
            data[rung] = d["configs"]
    base = data.get("base", {})
    rows = ["| configuration | rung | run() median [s] | vs base | t_shoot_cells median [s] | check |", "|---|---|---|---|---|---|"]
    for rung in PYTHON_RUNGS:
        for name, e in data.get(rung, {}).items():
            b = base.get(name, {})
            run = ctx.cell(f"Python run(), {rung}, {name}", b.get("median_run_wall_s"), e["median_run_wall_s"]) if rung != "base" else fmt(e["median_run_wall_s"])
            shoot = ctx.cell(f"Python t_shoot_cells, {rung}, {name}", b.get("median_t_shoot_cells"), e["median_t_shoot_cells"]) if rung != "base" else fmt(e["median_t_shoot_cells"])
            rows.append(f"| {name} | {rung} | {run} | {fmt(speedup(b.get('median_run_wall_s'), e['median_run_wall_s']), 2)} | {shoot} | "
                        f"{'pass' if e['verdict']['pass'] else 'FAIL'} |")
    if not data:
        rows.append("| – | – | – | – | – | – |")
    return "\n".join(rows)


def peaks(d, field):
    out = {}
    for r in (d or {}).get("records", []):
        if r.get("ok") and r.get(field) is not None:
            out[r["config"]] = max(out.get(r["config"], 0), r[field])
    return out


def memory_section(ctx):
    r1 = ctx.load("R5/timing_memory.json")
    mem = {"base": ctx.load("base/timing_memory.json"), "A2": ctx.load("A2/timing_memory.json"), "A4": ctx.load("A4/timing_memory.json")}
    dev = {k: peaks(v, "peak_gpu_mem_mib") for k, v in mem.items()}
    host = {k: peaks(v, "peak_host_mem_mib") for k, v in mem.items()}
    base_dev = {**peaks(r1, "peak_gpu_mem_mib"), **dev["base"]}   # round 1 figures for round 1's configs
    for name, v in peaks(r1, "peak_gpu_mem_mib").items():
        base_dev[name] = v
    names = sorted(set(base_dev) | set(dev["A2"]) | set(dev["A4"]) | set(host["base"]) | set(host["A2"]) | set(host["A4"]),
                   key=lambda n: list(cf.CONFIGS).index(n) if n in cf.CONFIGS else 99)
    rows = ["| configuration | base device [MiB] | A2 device | A4 device | base host [MiB] | A2 host | A4 host |", "|---|---|---|---|---|---|---|"]
    show = lambda x: "–" if x is None else str(x)
    for n in names:
        rows.append(f"| {n} | {show(base_dev.get(n))} | {show(dev['A2'].get(n))} | {show(dev['A4'].get(n))} | "
                    f"{show(host['base'].get(n))} | {show(host['A2'].get(n))} | {show(host['A4'].get(n))} |")
    if not names:
        rows.append("| – | – | – | – | – | – | – |")
    return "\n".join(rows)


def a2_checks_section(ctx):
    rows = ["| rung | check | result | detail |", "|---|---|---|---|"]
    for rung in ("A2", "A4"):
        d = ctx.load(f"{rung}/a2_checks.json")
        if not d:
            continue
        for name, c in d["checks"].items():
            detail = ""
            if name == "clamp_counts":
                detail = "; ".join(f"{x['config']}: {x['count']} pixels, area {x['total']}" for x in c["configs"])
            elif name == "natural_fallback":
                f = c.get("fallback") or {}
                detail = f"free {f.get('free_mib')} MiB, needed {f.get('need_mib')} MiB, parity consistency {c.get('parity_consistency')}"
            rows.append(f"| {rung} | {name} | {'pass' if c['pass'] else 'FAIL'} | {detail or '; '.join(c.get('failures', []))} |")
    if len(rows) == 2:
        rows.append("| – | – | – | – |")
    return "\n".join(rows)


def sweep_section(ctx):
    d = ctx.load("round2/threshold_sweep.json")
    if not d:
        return "| – |\n|---|"
    rungs, summ = d["rungs"], d["summary"]
    names = cf.LADDER
    rows = ["| variant | " + " | ".join(f"{n} [s]" for n in names) + " |", "|---|" + "---|" * len(names)]
    for r in rungs:
        rows.append(f"| {r} | " + " | ".join(fmt(summ.get(r, {}).get(n, {}).get("median_t_shoot_cells")) for n in names) + " |")
    return "\n".join(rows)


def render(root):
    ctx = Ctx(root)
    parts = ["# IPM round 2: result tables", ""]
    ladder, env = kernel_and_wall_tables(ctx, "ladder", "final/timing_ladder.json", "round2/ladder_interleaved.json", cf.LADDER, "Ladder")
    parts += [f"All tables: {caption(env)}." if env else "Environment: –  (round2/ladder_interleaved.json missing).", ""]
    parts += ladder

    # full set: base kernel from round 1's full matrix, wall from round 2's base
    full_d, base_full = ctx.summary("R5/timing_full.json")
    _, base_ladder = ctx.summary("final/timing_ladder.json")
    inter = ctx.load("round2/full_interleaved.json")
    kb = values(base_full, "median_t_shoot_cells")
    if "hm100_f" in kb and "hm100_f" in base_ladder:
        kb["_ref_hm100_f"] = base_ladder["hm100_f"]["median_t_shoot_cells"]
    base_note = run_note(full_d["env"]) if full_d else "missing"
    r2_note = run_note(inter["env"]) if inter else "missing"
    parts += ["## Full set: median shooting time", "", f"Base from round 1 (R5/timing_full.json; {base_note}); other rungs from "
              f"round2/full_interleaved.json ({r2_note}).", "", full_table(ctx, "median_t_shoot_cells", "Full kernel", kb, inter), ""]
    if "_ref_hm100_f" in kb:
        parts += [HM100_NOTE.format(ladder=fmt(kb["_ref_hm100_f"])), ""]
    wb = values(inter["summary"].get("base", {}), "median_wall_s") if inter else {}
    parts += ["## Full set: median end-to-end wall time", "", f"All rungs from round2/full_interleaved.json ({r2_note}); base re-measured there.", "",
              full_table(ctx, "median_wall_s", "Full wall", wb, inter), ""]

    fb = ctx.load("round2/fallback_interleaved.json")
    fsum = fb["summary"] if fb else {}
    frungs = [r for r in (fb["rungs"] if fb else ["base", "A2", "A4"])]
    fnames = sorted({c for r in frungs for c in fsum.get(r, {})}) or ["fallback_f"]
    fk = {r: values(fsum.get(r, {}), "median_t_shoot_cells") for r in frungs}
    fw = {r: values(fsum.get(r, {}), "median_wall_s") for r in frungs}
    parts += ["## Fallback configuration (float parity map near the memory limit)", "",
              f"From round2/fallback_interleaved.json ({run_note(fb['env']) if fb else 'missing'}); the base kernel time is the one re-measured there, "
              "since round 1 has no figure for this configuration.", "",
              "Kernel:", "", rung_table(ctx, "Fallback kernel", fk, frungs, fnames, fk.get("base", {})), "",
              "Wall:", "", rung_table(ctx, "Fallback wall", fw, frungs, fnames, fw.get("base", {})), ""]

    parts += ["## Python wrapper: run() including the copy into numpy", "",
              "All rungs from their python.json of this job; slowdowns are judged against base/python.json, and base-repeat shows the noise.", "",
              python_section(ctx), ""]
    parts += ["## Peak memory", "",
              "Device: base from round 1 (R5/timing_memory.json, GPU 4, job 906277) for round 1's configurations and base/timing_memory.json for the rest; "
              "A2 and A4 from their own timing_memory.json. Host: peak_host_mem_mib of each rung's timing_memory.json.", "", memory_section(ctx), ""]
    for rung in ("A1", "A2", "A2-direct", "A4"):
        for set_name in ("ladder", "full"):
            if not (ctx.root / rung / f"regression_{set_name}.json").exists():
                ctx.missing.append(f"{rung}/regression_{set_name}.json")
    parts += ["## Regression results", "", regression_table(ctx.root, ["A1", "A2", "A2-direct", "A4"]), ""]
    parts += ["## A2 checks (clamp counts, small and non-square maps, natural fallback)", "", a2_checks_section(ctx), ""]
    parts += ["## A2 threshold sweep (kernel median)", "", sweep_section(ctx), ""]
    parts += ["## Flagged slowdowns (more than 3 % slower than base)", ""]
    parts += [f"- {f}" for f in ctx.flags] or ["None."]
    parts += ["", "## Missing inputs", ""]
    parts += [f"- {m}" for m in dict.fromkeys(ctx.missing)] or ["None."]
    return "\n".join(parts) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="default: $IPM_BENCH_ROOT/results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    text = render(a.results or str(bl.bench_path("results")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
