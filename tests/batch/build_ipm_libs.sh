#!/bin/bash
# Build microlensing/lib/lib_ipm.so and lib_ipm_double.so in parallel with the flags of `compile` lines 3-4.
# Run after `source ~/microlensing_bench/bench_env.sh`.
if [ -z "$NVCC_PREPEND_FLAGS" ]; then
	echo "NVCC_PREPEND_FLAGS is unset: run 'source ~/microlensing_bench/bench_env.sh' first (it sets -ccbin and the RPATH flags)." >&2
	exit 1
fi
cd "$(dirname "$0")/../.." || exit 1

nvcc -std=c++20 -arch=compute_80 -code="sm_80,compute_80" --compiler-options '-fPIC' --use_fast_math -I ./include -O3 -D is_float -shared -o ./microlensing/lib/lib_ipm.so ./src/lib_ipm.cu &
pid_f=$!
nvcc -std=c++20 -arch=compute_80 -code="sm_80,compute_80" --compiler-options '-fPIC' --use_fast_math -I ./include -O3 -D is_double -shared -o ./microlensing/lib/lib_ipm_double.so ./src/lib_ipm.cu &
pid_d=$!

status=0
wait $pid_f || { echo "lib_ipm.so build failed" >&2; status=1; }
wait $pid_d || { echo "lib_ipm_double.so build failed" >&2; status=1; }
exit $status
