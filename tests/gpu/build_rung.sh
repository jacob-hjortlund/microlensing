#!/usr/bin/env bash
# Out-of-tree build of the four IPM targets with the flags of ./compile plus -Xptxas -v, and of the test drivers
# ipm_gpu_direct / ipm_gpu_double_direct (direct deposit forced): from the source's tests/gpu/ipm_force_direct.cu
# (static hook) when it has one, else from ipm_force_direct_member.cu next to this script when the source's
# include/ipm.cuh has the deposit member (enum class Deposit).
# usage: build_rung.sh <src_dir> <out_dir> [extra nvcc flags...]
# Needs nvcc on PATH. Extra host-compiler or linker flags can be passed through NVCC_PREPEND_FLAGS.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
shift 2
extra=("$@")
command -v nvcc >/dev/null || { echo "nvcc not on PATH" >&2; exit 1; }
flags=(-std=c++20 -arch=compute_80 -code="sm_80,compute_80" --use_fast_math -I "$src/include" -O3 -Xptxas -v)
# provenance first: a source that is not a git checkout fails here, before the long builds
rev=$(git -C "$src" rev-parse HEAD)
state=clean
if [ -n "$(git -C "$src" status --porcelain --untracked-files=no)" ]; then state=dirty; fi
echo "$rev $state" > "$out/git_rev.txt"
nvcc --version | tail -1 > "$out/nvcc_version.txt"
printf '%s\n' "${flags[@]}" ${extra[@]+"${extra[@]}"} "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}" > "$out/flags.txt"
# drop the outputs of earlier runs so that a failed rebuild cannot leave a stale mix
rm -rf "$out/pyroot" "$out/ipm_gpu" "$out/ipm_gpu_double" "$out/lib_ipm.so" "$out/lib_ipm_double.so" \
  "$out/ipm_gpu_direct" "$out/ipm_gpu_double_direct"
: > "$out/build.log"
built=()
build() {   # name define shared|exe source-path [more nvcc flags...]
  local name=$1 define=$2 kind=$3 source=$4
  shift 4
  local more=("$@")
  echo "=== $name ===" >> "$out/build.log"
  local shared=()
  if [ "$kind" = shared ]; then shared=(--compiler-options '-fPIC' -shared); fi
  if ! nvcc "${flags[@]}" ${extra[@]+"${extra[@]}"} ${more[@]+"${more[@]}"} ${shared[@]+"${shared[@]}"} -D "$define" \
      -o "$out/$name" "$source" 2>> "$out/build.log"; then
    echo "build of $name failed, see $out/build.log" >&2; tail -n 20 "$out/build.log" >&2; exit 1
  fi
  built+=("$name")
}
build ipm_gpu is_float exe "$src/src/main_ipm.cu"
build ipm_gpu_double is_double exe "$src/src/main_ipm.cu"
build lib_ipm.so is_float shared "$src/src/lib_ipm.cu"
build lib_ipm_double.so is_double shared "$src/src/lib_ipm.cu"
# the forced-direct driver: the source's own (static hook), else the harness's (deposit member), else none
direct_driver=
direct_flags=()
if [ -f "$src/tests/gpu/ipm_force_direct.cu" ]; then
  direct_driver=$src/tests/gpu/ipm_force_direct.cu
elif grep -q 'enum class Deposit' "$src/include/ipm.cuh" 2>/dev/null; then
  direct_driver=$here/ipm_force_direct_member.cu
  direct_flags=(-I "$src")   # it includes src/main_ipm.cu of the source
fi
if [ -n "$direct_driver" ]; then
  build ipm_gpu_direct is_float exe "$direct_driver" ${direct_flags[@]+"${direct_flags[@]}"}
  build ipm_gpu_double_direct is_double exe "$direct_driver" ${direct_flags[@]+"${direct_flags[@]}"}
fi
rm -rf "$out/pyroot"
mkdir -p "$out/pyroot"
cp -r "$src/microlensing" "$out/pyroot/microlensing"
find "$out/pyroot/microlensing" -name __pycache__ -type d -exec rm -rf {} +
rm -f "$out/pyroot/microlensing/lib/"*.so "$out/pyroot/microlensing/lib/"*.dll
cp "$out/lib_ipm.so" "$out/lib_ipm_double.so" "$out/pyroot/microlensing/lib/"
touch "$out/pyroot/microlensing/lib/__init__.py"
echo "built $out from $rev ($state)"
echo "targets: ${built[*]}"
echo "direct driver: ${direct_driver:-none}"
grep -c "Compiling entry function" "$out/build.log" | sed 's/^/kernels compiled: /'
