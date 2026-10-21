// Tests for running IPM<T> on a chosen device and its own stream (include/microlensing.cuh, include/ipm.cuh):
// a map made on the last visible device on a created stream must match one made by a default object, and an
// invalid device must make run() fail. Needs the GPU; with one visible GPU both objects share device 0.
// The invalid-device case prints an error message to stderr, which is expected output.
// float and double cannot share one program: the kernels of fmm.cuh declare their extern shared array with the
// same name for both types. This file therefore tests float; tests/gpu/test_ipm_batch_double.cu defines
// IPM_BATCH_TEST_DOUBLE and includes it to test double.
// Prints one "PASS <name>" or "FAIL <name>: <detail>" line per case and exits 1 if any case fails.
#include "ipm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = "")
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
	std::fflush(stdout);
}

// the lens of the brief: kappa 0.4, shear 0.4, kappa_star 0.2, half-lengths 10, 600x600 pixels, seed 7
template <typename T>
static void configure(IPM<T>& ipm)
{
	ipm.kappa_tot = static_cast<T>(0.4);
	ipm.shear = static_cast<T>(0.4);
	ipm.kappa_star = static_cast<T>(0.2);
	ipm.half_length_y = Complex<T>(10, 10);
	ipm.num_pixels_y = Complex<int>(600, 600);
	ipm.random_seed = 7;
	ipm.write_stars = 0;
}

template <typename T>
static void own_stream_on_last_device_matches_default(const char* name, double sum_tol, double diff_tol)
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	IPM<T> a;
	IPM<T> b;
	b.device = n_devices - 1;
	if (!b.create_stream()) { report(name, false, "create_stream failed"); return; }
	configure(a);
	configure(b);
	if (!a.run(0) || !b.run(0)) { report(name, false, "run failed"); return; }

	const size_t n = 600ULL * 600ULL;
	const int num_stars = a.get_num_stars();
	if (num_stars != b.get_num_stars()) { report(name, false, "different numbers of stars"); return; }
	// the order of the stars inside the tree is not reproducible even for two default objects (the star sort
	// places stars with atomics), so the star fields are compared as sets: sorted by position and mass, bit for bit
	auto sorted_stars = [&](IPM<T>& ipm)
	{
		std::vector<star<T>> v(ipm.get_stars(), ipm.get_stars() + num_stars);
		std::sort(v.begin(), v.end(), [](const star<T>& p, const star<T>& q)
		{
			if (p.position.re != q.position.re) return p.position.re < q.position.re;
			if (p.position.im != q.position.im) return p.position.im < q.position.im;
			return p.mass < q.mass;
		});
		return v;
	};
	const std::vector<star<T>> sa_stars = sorted_stars(a);
	const std::vector<star<T>> sb_stars = sorted_stars(b);
	if (std::memcmp(sa_stars.data(), sb_stars.data(), num_stars * sizeof(star<T>)) != 0)
	{
		report(name, false, "star fields differ");
		return;
	}

	std::vector<T> ma(n), mb(n);
	if (!a.copy_pixels(ma.data()) || !b.copy_pixels(mb.data())) { report(name, false, "copy_pixels failed"); return; }
	double sa = 0, sb = 0;
	for (size_t i = 0; i < n; i++) { sa += ma[i]; sb += mb[i]; }
	const double mean_a = sa / n;
	const double sum_ratio = std::abs(sb / sa - 1);
	size_t considered = 0, bad = 0;
	for (size_t i = 0; i < n; i++)
	{
		if (!(mb[i] >= 1e-3 * mean_a)) continue;
		considered++;
		if (!(std::abs(1.0 * mb[i] - ma[i]) <= diff_tol * std::abs(1.0 * ma[i]))) bad++;
	}
	const double fraction = considered ? 1.0 * bad / considered : 1.0;
	const bool ok = sum_ratio < sum_tol && fraction < 1e-3;
	report(name, ok, "sum ratio - 1 = " + std::to_string(sum_ratio) + ", fraction of differing pixels = " + std::to_string(fraction));
}

#ifndef IPM_BATCH_TEST_DOUBLE
static void set_cuda_devices_rejects_invalid_device()
{
	int n_devices = 0;
	cudaGetDeviceCount(&n_devices);
	IPM<float> c;
	configure(c);
	c.device = n_devices;
	const bool ran = c.run(0);
	IPM<float> d;
	configure(d);
	d.device = -1;
	const bool ran_negative = d.run(0);
	report("set_cuda_devices_rejects_invalid_device", !ran && !ran_negative);
}
#endif

int main()
{
#ifdef IPM_BATCH_TEST_DOUBLE
	own_stream_on_last_device_matches_default<double>("own_stream_on_last_device_matches_default_double", 1e-12, 1e-8);
#else
	own_stream_on_last_device_matches_default<float>("own_stream_on_last_device_matches_default_float", 1e-5, 1e-2);
	set_cuda_devices_rejects_invalid_device();
#endif
	return n_fail ? 1 : 0;
}
