# IPM benchmark and regression harness

These tools build versions of the IPM code out of tree, run a fixed set of configurations with each build, compare the
maps pixel by pixel with reference maps, time the runs, and measure the throughput of batches of maps. They produced the
performance and accuracy figures of #33 (tiled shooting kernel), #34 (perimeter deposit, maps kept on the device) and
#36 (batches of IPM maps). Neither the package nor its tests need them. Each script's docstring and `--help` give the
details; this file summarises them.

## Environment

- **Software:** nvcc with a host compiler it accepts, Python 3 with numpy, and `nvidia-smi`. Host-compiler or linker
  flags that a build needs go in `NVCC_PREPEND_FLAGS`. The builds use the flags of `compile` (`sm_80`) plus
  `-Xptxas -v`. `profile_rung.sh` and `gpu_busy.py` need Nsight Systems.
- **`IPM_BENCH_ROOT`** is the directory that holds:
  - `builds/<name>/`: one build per version of the code, made by `build_rung.sh`. It holds `ipm_gpu`,
    `ipm_gpu_double`, `lib_ipm.so` and `lib_ipm_double.so`; the forced-direct-deposit executables `ipm_gpu_direct`
    and `ipm_gpu_double_direct` when the source has `IPM<T>::Deposit` (or a `tests/gpu/ipm_force_direct.cu` of its
    own); and `pyroot/`, a copy of the `microlensing` package with the build's libraries, for `PYTHONPATH`.
    `build.log`, `git_rev.txt`, `flags.txt` and `nvcc_version.txt` record how it was built.
  - `ref/`: the reference maps, `ref/<config>/run1` to `run3` (one run for the double-truth configurations) and, for
    the double configurations under regression, `shift1`. Another reference directory, such as `ref-U0`, is passed
    with `--ref-root`.
  - `results/`: `tolerances.json`, and a directory `results/<name>/` per build with the JSON results of the drivers:
    `regression_<set>.json`, `timing_<set>.json`, `python.json`, `a2_checks.json`.
- **`IPM_BENCH_TMP`** (default `/tmp/ipm_bench`) is the scratch directory for the output of each run. The drivers
  delete each run's output after the run; `regress.py` and `a2_checks.py` keep it with `--keep-outputs`.
- **Build names.** The tools call a build a *rung*, after the ladder of builds that each added one change: `R0` to
  `R5` (`bench_lib.RUNGS`), which `time_ladder.py` times by default and `report.py` tabulates. `R0` is also the
  default reference build of `make_reference.py`, and the results label comparisons with the references `vs_r0`. Any
  other name works; below, `U0` is the build of `development` (676ac9c), and `pr33`, `pr34` and `pr36` are the builds
  of the three PRs. `a2_checks.py` is named after the rung that introduced the perimeter deposit.
- **Configurations** are in `configs.py` (`python tests/gpu/configs.py --list`): the sets `ladder` (`nb_f`, `nb_d`,
  `rect_f`, `hm100_f`), `full` (17 configurations), `truth` (the double twins of float configurations), `memory`,
  `fallback` (54000² with parity maps) and `parity` (`-nry 10` with parity maps). All use seed 20261015.
  `configs.SHIFT` moves the source-plane centre by 10⁻⁷ for the shifted-centre comparisons of double maps.

### GPUs

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so that CUDA's device numbers are the indices `nvidia-smi` shows.
- Under Slurm, the drivers refuse a GPU outside the job's allocation (`SLURM_STEP_GPUS`, else `SLURM_JOB_GPUS`).
  `regress.py`, `time_rung.py`, `time_ladder.py`, `python_check.py` and `a2_checks.py` compare the first
  `CUDA_VISIBLE_DEVICES` entry with the first allocated GPU; `bench_batch.py` checks every entry.
  `IPM_BENCH_EXTRA_GPUS=i,j` counts further GPUs as allocated. Without those variables nothing is checked.
- `gpu_idle.py [index ...]` exits 0 when the GPUs have no compute process and at most 100 MiB in use. Run it
  immediately before and after each timing.
- `time_rung.py` and `time_ladder.py` list the compute processes on their GPU before and after every run. A run that
  shares the GPU with another process is repeated (`--foreign-retries`, default 3). Records that stay unclean are left
  out of the medians and listed in `unclean_records`. `bench_batch.py` picks the GPUs with the fewest foreign
  processes and records those it finds; its table marks such rows.

## Tools

In `tests/gpu`:

| tool | what it does |
|---|---|
| `build_rung.sh <src> <out> [nvcc flags]` | Builds a git checkout out of tree into `<out>` (see Environment). Extra flags such as `-DIPM_WARP_AGGREGATE=0` make variant builds. |
| `ipm_force_direct_member.cu` | The forced-direct driver: `src/main_ipm.cu` with `map_maker.deposit = Deposit::direct`. |
| `configs.py` | The configuration matrix. `--command <config> --build <dir> --ref-root <dir>` prints a run's command line. |
| `bench_lib.py` | Shared helpers: map and parameter-file readers, `run_ipm` (one run with its wall time, `t_shoot_cells` and optionally peak device and host memory), the GPU checks, environment and build records. |
| `compare_maps.py` | Map metrics and pass rules. On pixels at or above 10⁻³ of the reference mean: mean, p99.9, p99.99 and max of the relative difference; also the largest difference below that floor, relative to the mean (`offmask_max`), the total-area ratio (`sum_ratio`), finiteness and sign. `compare_maps.py <candidate.bin> <reference.bin>` prints them. |
| `make_reference.py` | Runs the reference build over the `full` and `truth` sets into `<ref-root>/<config>/run<k>`: three runs of each `full` configuration, one of each `truth` one. `--shift` runs the double regression configurations with the shifted centre into `shift1`. |
| `calibrate.py` | From the references: the noise floors (run2 and run3 against run1), the float build's error against the double truth, the double maps' shift sensitivity and the limits derived from it. Writes `tolerances.json`; thresholds edited by hand are kept. |
| `regress.py` | Runs a build on a set (`ladder`, `full`, `parity`) and judges each map. Float: its error against the double truth may exceed the reference build's own error by at most 1.5×, and its total area must agree with the truth's within the tolerance. Double: the shifted-centre run against `shift1` under `tol["double"]`, and the real run against `run1` under `tol["double_actual"]`. Parity maps: also total = minima + saddles. `--exe-suffix _direct` runs the forced-direct executables. |
| `parity_compare.py` | Compares the total, minima and saddle maps of a parity configuration between two builds, or a build and a reference run, against a maximum relative difference (default 10⁻⁴). `--shift` shifts the centre of both builds' runs. |
| `a2_checks.py` | Checks of the perimeter deposit that the regression does not cover: a 3000×2000 double map and 1×1 and 7×3 maps against the forced direct deposit, the pixels the row sums clamp on the ladder configurations, and the fallback to the direct deposit at 54000² with parity maps. Needs the `_direct` executables. |
| `time_rung.py` | Times one build on a set: N repeats and their medians. `--set memory` also samples the peak device memory (`nvidia-smi`) and the peak host RSS (VmHWM). |
| `time_ladder.py` | Interleaved timing of several builds: each repeat runs every build on every configuration. Resumable (`--out`). |
| `python_check.py` | Times and checks `IPM.run()` of a build's `pyroot` on `nb_f`, `nb_d`, `par_f`, `hm100_f` and `px40k_f`. An untimed warm-up precedes each configuration; the maps of the first timed run are judged by the rules of `regress.py`, and `save()` is checked on `nb_f`. Run it with `PYTHONPATH=<build>/pyroot`. |
| `report.py` | Renders the tables of the `R0`–`R5` ladder from `results/` as Markdown. |
| `ptxas_stats.py <build> ...` | Registers, spills, stack and shared memory of the shooting kernels, from `build.log`. |
| `profile_rung.sh <build> <name> [config ...]` | Nsight Systems profile of a build (default `nb_f hm100_f`), with CSV summaries in `results/<name>/nsys/`. |
| `check_commits.sh <checkout> <base> <name>` | For every commit of `<base>..HEAD`: a build into `builds/<name>-<n>`, the commit's C++ tests (with nvcc, and with `--host` when its runner has it), `unittest` discovery over its `tests/python` and `tests/batch`, and the ladder regression of `nb_f` and `nb_d` against `ref-U0` with `results/U0/tolerances.json`. One line per commit. |
| `gpu_idle.py`, `fake_ipm.py`, `test_bench.py` | The idle check (see GPUs), a stand-in for `ipm_gpu` in the unit tests, and the unit tests. |

In `tests/batch`:

| tool | what it does |
|---|---|
| `bench_batch.py` | Throughput of a loop of `IPM(...).run()` (and `save()`) on one GPU against `BatchedIPM` on 1 to g GPUs, in memory or save mode. Configurations `small_f`, `nb_f`, `stars1e6_f`, `px40k_f` and `mixed_f` (alternating `small_f` and `nb_f` maps). Each measurement runs in a fresh child process after an untimed warm-up; `px40k_f` runs max(`--maps`, 8 g) maps. Resumable (`--out`); prints a table of medians, gains and peak RSS. |
| `gpu_busy.py` | GPU busy and kernel-overlap fractions from a `nsys stats -r cuda_gpu_trace` CSV. |
| `measure_tree_bytes.py [device]` | Measures the tree bytes per star on four maps, the basis of `IPM<T>::tree_bytes_per_star`. |
| `test_bench_batch.py` | Unit tests of `bench_batch.py` and `gpu_busy.py`. |

## Unit tests

They need no GPU. Run them from a directory outside the checkout, so that the checkout's `microlensing` package and
its precompiled libraries are not imported ahead of a build's `pyroot`:

```bash
cd "$(mktemp -d)"
export PYTHONPATH=$IPM_BENCH_ROOT/builds/<name>/pyroot:<checkout>
python -m unittest discover -s <checkout>/tests/gpu -p test_bench.py -v
python -m unittest discover -s <checkout>/tests/batch -p test_bench_batch.py -v
```

`test_bench_batch.py` imports `microlensing.IPM`, so the libraries of the `pyroot` must load.

## How the figures were produced

Builds, references and tolerances. Run every GPU command with the GPU checks above.

```bash
H=<this checkout>; B=$IPM_BENCH_ROOT/builds; R=$IPM_BENCH_ROOT/results
bash $H/tests/gpu/build_rung.sh <checkout of development at 676ac9c> $B/U0
bash $H/tests/gpu/build_rung.sh <checkout of #33> $B/pr33          # likewise pr34 and pr36
# references of development, for the 15 configurations up to 8000²
python $H/tests/gpu/make_reference.py --build $B/U0 --rung U0 --ref-root $IPM_BENCH_ROOT/ref-U0 --results $R/U0
python $H/tests/gpu/make_reference.py --build $B/U0 --rung U0 --ref-root $IPM_BENCH_ROOT/ref-U0 --results $R/U0 --shift
python $H/tests/gpu/calibrate.py --ref-root $IPM_BENCH_ROOT/ref-U0 --out $R/U0/tolerances.json
# references of #33 in the default ref/ and results/tolerances.json, for the two 40000² configurations
python $H/tests/gpu/make_reference.py --build $B/pr33 --rung pr33 --results $R/pr33-ref
python $H/tests/gpu/make_reference.py --build $B/pr33 --rung pr33 --results $R/pr33-ref --shift
python $H/tests/gpu/calibrate.py
REF_U0="--ref-root $IPM_BENCH_ROOT/ref-U0 --tolerances $R/U0/tolerances.json"
FULL15="nb_f nb_d rect_f hm100_f circ_a0_f rect_a0_f rect_d rect_a0_d par_f nry10_f big_f hm1000_f hm100_d sf_f stars1e6_f"
```

**Accuracy (all three PRs).** For a build `X` (`pr33`, `pr34`, `pr36`):

```bash
python $H/tests/gpu/regress.py --build $B/X --rung X --set ladder $REF_U0
python $H/tests/gpu/regress.py --build $B/X --rung X --set full --only $FULL15 $REF_U0
python $H/tests/gpu/regress.py --build $B/X --rung X --set full --only px40k_f px40k_d --results $R/X/px40k   # #34, #36
```

- Float maps: the error against the double truth is `vs_truth` in `regression_full.json`; the error of the float
  build of `development` is `float_truth_error` in `$R/U0/tolerances.json`.
- Double maps: `vs_r0_shift` is the comparison with both centres shifted, `vs_r0` the one without. The shift
  sensitivity of `development` alone is `double_shift_sensitivity` in the tolerances. Run-to-run noise is `floors`.
- Parity maps (#33): `parity_compare.py --config parnry10_d --a $B/pr33 --b $B/U0 --shift`.
- Every commit (all three): `check_commits.sh <checkout> <base of the PR> <name>`.
- Direct deposit (#34): `regress.py --build $B/pr34 --rung pr34-direct --exe-suffix _direct --set ladder $REF_U0`, and
  `--set full --only par_f`. The checks of the perimeter deposit: `a2_checks.py --build $B/pr34 --rung pr34`.
- Maps of #36 against those of #34: run every configuration with both builds, and a second time with one of them as
  the noise baseline, keeping the maps (`regress.py ... --keep-outputs --scratch <dir>` leaves them in
  `<dir>/<config>/`), then `compare_maps.py <map of pr36> <map of pr34>` for each.

**Single-map timing.** Medians of 3 interleaved repeats, after a warm-up run:

```bash
python $H/tests/gpu/gpu_idle.py
python $H/tests/gpu/time_ladder.py --rungs U0 pr33 --set full --repeats 3 --ref-root $IPM_BENCH_ROOT/ref-U0 --out $R/pr33/time_full.json
python $H/tests/gpu/time_ladder.py --rungs pr33 pr34 --set full --repeats 3 --ref-root $IPM_BENCH_ROOT/ref-U0 --out $R/pr34/time_full.json
python $H/tests/gpu/time_ladder.py --rungs pr33 pr34 --set fallback --repeats 3 --out $R/pr34/time_fallback.json
python $H/tests/gpu/time_ladder.py --rungs pr36 --set full --repeats 3 --ref-root $IPM_BENCH_ROOT/ref-U0 --out $R/pr36/time_full.json
python $H/tests/gpu/gpu_idle.py
```

- *Shooting* is `t_shoot_cells` and *wall* is `wall_s` in the `summary` of each file. #36 is compared with the
  `pr34` medians of `$R/pr34/time_full.json`.
- The per-step table of #34 times the builds of its commits that `check_commits.sh <checkout> <head of #33> pr34c`
  leaves in `$B/pr34c-<n>`: `time_ladder.py --rungs pr33 pr34c-2 pr34c-3 pr34c-4 pr34 --set ladder`.
- Memory, one run of each build compared (`U0` and `pr33` for #33, `pr33` and `pr34` for #34):
  `time_rung.py --build $B/X --rung X --set memory --repeats 1`.
- Python `IPM.run()`, for each build (`U0` and `pr33` for #33; `pr33` and `pr34` for #34; `pr36`):
  `PYTHONPATH=$B/X/pyroot python $H/tests/gpu/python_check.py --build $B/X --rung X --only nb_f nb_d hm100_f par_f $REF_U0`,
  and `--only px40k_f --results $R/X/python_px40k` against the references of #33.

**Throughput (#36).** With `PYTHONPATH=$B/pr36/pyroot`, on idle GPUs:

```bash
python $H/tests/batch/bench_batch.py --configs small_f nb_f stars1e6_f --gpus 1 2 4 --modes memory save \
    --save-root <dir> --repeats 3 --maps 64 --out $R/pr36/throughput.json
python $H/tests/batch/bench_batch.py --configs px40k_f --gpus 1 2 4 --modes memory save \
    --save-root <dir> --repeats 3 --maps 8 --out $R/pr36/throughput_px40k.json   # 8 maps per GPU
python $H/tests/batch/bench_batch.py --configs mixed_f --gpus 1 2 4 --modes memory \
    --repeats 3 --maps 64 --out $R/pr36/throughput_mixed.json
```

- The gains are the batch's median maps/s over the serial loop's; the gain per GPU divides that by the number of
  GPUs. The peak RSS is the table's `batch peak RSS MiB`.
- The `max_result_bytes = 64 GiB` figure for `px40k_f` comes from `bench_batch.measure("px40k_f", "batch", "memory",
  4, 32, None)` run with `BatchedIPM`'s `max_result_bytes` raised to 64 GiB; the command line has no option for it.
- Timelines: run one measurement child under `nsys profile` with `BENCH_PROFILE_GAP=1` (the command is in
  `bench_batch.py --help`), export it with `nsys stats -r cuda_gpu_trace --format csv`, then
  `gpu_busy.py <trace.csv> --after-gap 0.5`.
