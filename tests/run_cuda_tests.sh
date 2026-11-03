#!/bin/bash
# usage: tests/run_cuda_tests.sh [--host] <out_dir> <file.cu>...
# Compiles each test program and runs it. Exits non-zero if any compile or run fails.
# By default the programs are compiled with nvcc and the project flags (sm_80).
# With --host they are compiled as plain C++ with $CXX (g++ if unset), with
# __host__, __device__ and __forceinline__ defined away; this suits the tests
# in tests/host and needs no CUDA. $EXTRA_CXXFLAGS is appended in that mode,
# for example EXTRA_CXXFLAGS=-fsanitize=address,undefined.
usage="usage: $0 [--host] <out_dir> <file.cu>..."
host=0
if [ "$1" = --host ]; then
	host=1; shift
fi
if [ $# -lt 2 ]; then
	echo "$usage" >&2
	exit 2
fi
out_dir=$1; shift
mkdir -p "$out_dir" || exit 2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
status=0
for src in "$@"; do
	name=$(basename "${src%.cu}")
	exe="$out_dir/$name"
	echo "== compile $src"
	if [ $host = 1 ]; then
		# shellcheck disable=SC2086 # EXTRA_CXXFLAGS may hold several flags
		"${CXX:-g++}" -std=c++20 -O2 -x c++ -D__host__= -D__device__= -D__forceinline__=inline -I "$root/include" \
			$EXTRA_CXXFLAGS -o "$exe" "$src"
	else
		nvcc -std=c++20 -arch=compute_80 -code=sm_80,compute_80 --use_fast_math -O3 -I "$root/include" -o "$exe" "$src"
	fi
	if [ $? -ne 0 ]; then
		echo "COMPILE FAILED: $src"; status=1; continue
	fi
	echo "== run $name"
	if ! "$exe"; then
		echo "RUN FAILED: $name"; status=1
	fi
done
exit $status
