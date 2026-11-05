#!/usr/bin/env bash
# Builds and checks every commit of <base>..HEAD of a worktree, one line per commit.
# usage: check_commits.sh [--no-regress] <worktree> <base> <name> [ref-configs...]
#   <worktree>     checkout whose HEAD is checked; nothing is written to it
#   <base>         exclusive start of the range
#   <name>         builds go to $IPM_BENCH_ROOT/builds/<name>-<n> (n = 1-based commit index), results to
#                  $IPM_BENCH_ROOT/results/check/<name>-<n>
#   ref-configs    configurations of the ladder regression step (default: nb_f nb_d)
#   --no-regress   skip the regression step (quick runs)
# For each commit: out-of-tree build, tests/run_cuda_tests.sh on its tests/host/*.cu (also in --host mode when the
# commit's script has that flag) and tests/gpu/test_*.cu, unittest discovery over its tests/python and tests/batch,
# then regress.py against the references in $IPM_BENCH_ROOT/ref-U0 with the tolerances in
# $IPM_BENCH_ROOT/results/U0/tolerances.json (both made with the base build U0). A directory the commit lacks is
# skipped. The commit's own run_cuda_tests.sh is used when it has one, otherwise the copy of the harness checkout.
# Exits non-zero if any step of any commit fails.
# Needs nvcc on PATH (with the host-compiler flags in NVCC_PREPEND_FLAGS, as for build_rung.sh), IPM_BENCH_ROOT and a
# GPU allocated to this shell.
HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
regress=1
if [ "${1:-}" = --no-regress ]; then regress=0; shift; fi
if [ $# -lt 3 ]; then
	echo "usage: $0 [--no-regress] <worktree> <base> <name> [ref-configs...]" >&2
	exit 2
fi
src=$(cd "$1" && pwd) || exit 2
base=$2
name=$3
shift 3
cfgs=("$@")
[ ${#cfgs[@]} -gt 0 ] || cfgs=(nb_f nb_d)
: "${IPM_BENCH_ROOT:?set IPM_BENCH_ROOT to the directory that holds builds/, ref-U0/ and results/}"
B=$IPM_BENCH_ROOT/builds
R=$IPM_BENCH_ROOT/results
tmp_root=$(mktemp -d "${IPM_BENCH_TMP:-/tmp}/check_commits.XXXXXX") || exit 2
tmpwt=""
cleanup() {
	[ -n "$tmpwt" ] && git -C "$src" worktree remove --force "$tmpwt" >/dev/null 2>&1
	git -C "$src" worktree prune
	rm -rf "$tmp_root"
}
trap cleanup EXIT
commits=$(git -C "$src" rev-list --reverse "$base..HEAD") || exit 2
overall=0
n=0
for sha in $commits; do
	n=$((n + 1))
	tmpwt=$tmp_root/wt
	build_dir=$B/$name-$n
	log=$tmp_root/log-$n
	git -C "$src" worktree add --detach "$tmpwt" "$sha" >"$log" 2>&1 || { echo "cannot check out $sha" >&2; exit 2; }
	short=$(git -C "$src" rev-parse --short "$sha")
	build=FAIL cuda=skip py=skip reg=skip
	if bash "$HARNESS/tests/gpu/build_rung.sh" "$tmpwt" "$build_dir" >>"$log" 2>&1; then build=ok; else overall=1; fi
	if [ $build = ok ]; then
		runner=$tmpwt/tests/run_cuda_tests.sh
		[ -f "$runner" ] || { cp "$HARNESS/tests/run_cuda_tests.sh" "$runner"; }
		cuda=ok
		host=("$tmpwt"/tests/host/*.cu)
		[ -e "${host[0]}" ] || host=()
		gpu=("$tmpwt"/tests/gpu/test_*.cu)
		[ -e "${gpu[0]}" ] || gpu=()
		cuda_out=$build_dir/cuda_tests
		if [ ${#host[@]} -gt 0 ]; then
			bash "$runner" "$cuda_out/host" "${host[@]}" >>"$log" 2>&1 || cuda=FAIL
			if grep -q -e --host "$runner"; then
				bash "$runner" --host "$cuda_out/host-g++" "${host[@]}" >>"$log" 2>&1 || cuda=FAIL
			fi
		fi
		if [ ${#gpu[@]} -gt 0 ]; then
			bash "$runner" "$cuda_out/gpu" "${gpu[@]}" >>"$log" 2>&1 || cuda=FAIL
		fi
		[ $cuda = FAIL ] && overall=1
		py=ok
		# some tests start children with the checkout as cwd: give it this build's libraries instead of its tracked ones
		cp "$build_dir"/lib_ipm.so "$build_dir"/lib_ipm_double.so "$tmpwt/microlensing/lib/"
		pyenv="$build_dir/pyroot:$tmpwt"   # not run from $tmpwt: its tracked microlensing/lib/*.so would shadow the build's
		for d in tests/python tests/batch; do
			[ -d "$tmpwt/$d" ] || continue
			(cd "$tmp_root" && PYTHONPATH=$pyenv python -m unittest discover -s "$tmpwt/$d" >>"$log" 2>&1) || py=FAIL
		done
		[ $py = FAIL ] && overall=1
		if [ $regress = 1 ]; then
			reg=ok
			python "$HARNESS/tests/gpu/regress.py" --set ladder --only "${cfgs[@]}" \
				--ref-root "$IPM_BENCH_ROOT/ref-U0" --tolerances "$IPM_BENCH_ROOT/results/U0/tolerances.json" \
				--build "$build_dir" --rung "$name-$n" --results "$R/check/$name-$n" >>"$log" 2>&1 || reg=FAIL
			[ $reg = FAIL ] && overall=1
		fi
	fi
	mkdir -p "$R/check/$name-$n"
	cp "$log" "$R/check/$name-$n/check_commits.log"
	echo "$n $short build=$build cuda=$cuda py=$py regress=$reg"
	git -C "$src" worktree remove --force "$tmpwt" >/dev/null 2>&1
	tmpwt=""
done
exit $overall
