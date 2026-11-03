/******************************************************************************
Host test of ipm_step_buffers (include/ipm_step_buffers.hpp), the sizing of
the step buffer of the perimeter deposit: need_bytes adds the local memory the
runtime reserves for the larger kernel and the margin to the buffer's bytes,
and fits compares with the free memory. Plain C++; it compiles with nvcc or
through tests/run_cuda_tests.sh --host. One PASS/FAIL line per case; exit 0
only if all pass.
******************************************************************************/
#include "ipm_step_buffers.hpp"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>

static int n_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail)
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

static void need_includes_local_memory()
{
	const size_t got = ipm_step_buffers::need_bytes(100, 512, 2048, 108);
	const size_t expected = 100 + size_t(512) * 2048 * 108 + (size_t(64) << 20);
	report("need_includes_local_memory", got == expected, "got " + std::to_string(got) + ", expected " + std::to_string(expected));

	//the margin is a parameter, and no local memory adds nothing but the margin
	const size_t no_local = ipm_step_buffers::need_bytes(4000000, 0, 2048, 108, 1 << 20);
	report("need_without_local_memory", no_local == 4000000 + (size_t(1) << 20),
		"got " + std::to_string(no_local) + ", expected " + std::to_string(4000000 + (size_t(1) << 20)));
}

static void fits_at_boundary()
{
	const size_t n = ipm_step_buffers::need_bytes(8000000, 160, 2048, 108);
	const bool equal = ipm_step_buffers::fits(n, n);
	const bool one_short = ipm_step_buffers::fits(n + 1, n);
	report("fits_at_boundary", equal && !one_short,
		!equal ? "fits(n, n) is false" : "fits(n + 1, n) is true");
}

static void clamp_warning_threshold()
{
	//bound = 16 * eps * factor * rows = 16 * 1e-7 * 2 * 1000 = 3.2e-3
	const double eps = 1e-7, factor = 2, bound = 16 * eps * factor * 1000;
	const bool above = ipm_step_buffers::clamp_is_suspicious(5, -bound * 1.01, factor, 1000, eps);
	const bool below = ipm_step_buffers::clamp_is_suspicious(5, -bound * 0.99, factor, 1000, eps);
	const bool none = ipm_step_buffers::clamp_is_suspicious(0, -1e9, factor, 1000, eps);
	report("clamp_warning_threshold", above && !below && !none,
		!above ? "false just above the bound" : !below ? "true just below the bound" : "true for n_clamped == 0");
}

int main()
{
	need_includes_local_memory();
	fits_at_boundary();
	clamp_warning_threshold();
	std::printf("%s: %d failed\n", n_fail ? "FAILED" : "OK", n_fail);
	return n_fail ? 1 : 0;
}
