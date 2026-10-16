/******************************************************************************
Host-only test of tri_cov (current) against tri_cov_ref (frozen copy of the
coverage code at e3168bc). One PASS/FAIL line per case; exit 0 only if all pass.
******************************************************************************/
#include "triangle_coverage.cuh"
#include "tri_cov_ref.cuh"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail)
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

template <typename P> P eps_tol();
template <> double eps_tol<double>() { return 1e-12; }
template <> float eps_tol<float>() { return 1e-5f; }

template <typename P>
static P inv2s_of(P slope)
{
	return slope != 0 ? P(1) / (2 * std::fabs(slope)) : P(0);
}

/******************************************************************************
1D integrals
******************************************************************************/
template <typename P, bool CLAMP>
static void test_integral(const char* name, const char* ptag)
{
	std::mt19937_64 gen(12345);
	std::uniform_real_distribution<double> da(-3, 3), dL(1e-9, 1), ds(-50, 50), d01(0, 1);
	double worst = 0; long bad = 0; std::string first;
	for (int i = 0; i < 1000000; i++)
	{
		P a = static_cast<P>(da(gen));
		P L = static_cast<P>(dL(gen));
		double sd = ds(gen);
		int k = i % 10;
		if (k == 0) sd = 0;
		else if (k == 1) sd = 1e-6;
		else if (k == 2) sd = -1e-6;
		P slope = static_cast<P>(sd);
		P b = a + L * slope;
		P ref, got;
		if (CLAMP)
		{
			ref = tri_cov_ref::int_clamp01<P>(a, b, L);
			got = tri_cov::int_clamp01<P>(a, b, L, inv2s_of(slope));
		}
		else
		{
			ref = tri_cov_ref::int_max0<P>(a, b, L);
			got = tri_cov::int_max0<P>(a, b, L, inv2s_of(slope));
		}
		double scale = std::fmax(1.0, std::fabs((double)ref));
		double err = std::fabs((double)got - (double)ref) / scale;
		if (!(err <= eps_tol<P>()))
		{
			if (bad == 0)
			{
				char buf[200];
				std::snprintf(buf, sizeof buf, "a=%.9g b=%.9g L=%.9g slope=%.9g ref=%.9g got=%.9g", (double)a, (double)b, (double)L, (double)slope, (double)ref, (double)got);
				first = buf;
			}
			bad++;
		}
		if (!(err <= worst)) worst = err;
	}
	char buf[400];
	std::snprintf(buf, sizeof buf, "%ld of 1e6 outside tolerance (worst %.3g); first: %s", bad, worst, first.c_str());
	report(std::string(name) + "_" + ptag, bad == 0, buf);
}

/******************************************************************************
direct path
******************************************************************************/
template <typename P>
struct Tri { Complex<P> p0, p1, p2; };

template <typename P>
static void test_direct(const char* ptag)
{
	const Complex<int> npix(64, 48);
	std::mt19937_64 gen(777);
	std::uniform_real_distribution<double> dx(-10, 74), dy(-10, 58), d01(0, 1), dsm(-1, 1);
	std::vector<Tri<P>> tris;
	auto push = [&](double x0, double y0, double x1, double y1, double x2, double y2)
	{
		tris.push_back({Complex<P>(P(x0), P(y0)), Complex<P>(P(x1), P(y1)), Complex<P>(P(x2), P(y2))});
	};
	//random
	for (int i = 0; i < 3000; i++) push(dx(gen), dy(gen), dx(gen), dy(gen), dx(gen), dy(gen));
	//random moderate-size triangles (a few pixels)
	for (int i = 0; i < 3000; i++)
	{
		double cx = 64 * d01(gen), cy = 48 * d01(gen);
		push(cx + 3 * dsm(gen), cy + 3 * dsm(gen), cx + 3 * dsm(gen), cy + 3 * dsm(gen), cx + 3 * dsm(gen), cy + 3 * dsm(gen));
	}
	//slivers, area ~1e-6
	for (int i = 0; i < 500; i++)
	{
		double x0 = 64 * d01(gen), y0 = 48 * d01(gen), x1 = 64 * d01(gen), y1 = 48 * d01(gen);
		double len = std::hypot(x1 - x0, y1 - y0);
		double t = d01(gen);
		double h = 2e-6 / len; //area = 0.5 * len * h = 1e-6
		double nx = -(y1 - y0) / len, ny = (x1 - x0) / len;
		push(x0, y0, x1, y1, x0 + t * (x1 - x0) + h * nx, y0 + t * (y1 - y0) + h * ny);
	}
	//axis-aligned right triangles, integer and half-integer vertices
	for (int i = 0; i < 1000; i++)
	{
		double x = std::floor(2 * 64 * d01(gen)) / 2, y = std::floor(2 * 48 * d01(gen)) / 2;
		double w = (1 + std::floor(2 * 20 * d01(gen))) / 2, h = (1 + std::floor(2 * 20 * d01(gen))) / 2;
		if (i % 4 == 0) push(x, y, x + w, y, x, y + h);
		else if (i % 4 == 1) push(x, y, x + w, y, x + w, y + h);
		else if (i % 4 == 2) push(x, y + h, x + w, y, x + w, y + h);
		else push(x, y, x + w, y + h, x, y + h);
	}
	//crossing each map edge
	push(-5, 10, 5, 5, 3, 20);
	push(60, 10, 70, 15, 62, 25);
	push(10, -5, 20, 3, 30, -2);
	push(10, 40, 20, 52, 30, 45);
	push(-5, -5, 5, 2, 2, 5);
	push(60, 44, 70, 55, 66, 40);
	push(-5, 20, 70, 25, 30, 30);
	//covering the whole map
	push(-100, -100, 300, -100, -100, 300);
	push(-1, -1, 200, 0, 0, 200);
	//both orientations
	const size_t n0 = tris.size();
	for (size_t i = 0; i < n0; i++) tris.push_back({tris[i].p0, tris[i].p2, tris[i].p1});

	const P factor = P(1.7);
	const size_t N = size_t(npix.re) * npix.im;
	double worst = 0; long bad = 0; std::string first;
	std::vector<P> got(N), ref(N);
	for (size_t t = 0; t < tris.size(); t++)
	{
		std::fill(got.begin(), got.end(), P(0));
		std::fill(ref.begin(), ref.end(), P(0));
		const Tri<P>& tr = tris[t];
		P area = P(0.5) * ((tr.p1.re - tr.p0.re) * (tr.p2.im - tr.p0.im) - (tr.p2.re - tr.p0.re) * (tr.p1.im - tr.p0.im));
		auto add_g = [&](long long i, P v) { got[i] += v; };
		auto add_r = [&](long long i, P v) { ref[i] += v; };
		tri_cov::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, area, factor, npix, add_g, add_g);
		tri_cov_ref::allocate_triangle<P, P>(tr.p0, tr.p1, tr.p2, area, factor, npix, add_r, add_r);
		double xmin = std::fmin(tr.p0.re, std::fmin(tr.p1.re, tr.p2.re)), xmax = std::fmax(tr.p0.re, std::fmax(tr.p1.re, tr.p2.re));
		double ymin = std::fmin(tr.p0.im, std::fmin(tr.p1.im, tr.p2.im)), ymax = std::fmax(tr.p0.im, std::fmax(tr.p1.im, tr.p2.im));
		double tol;
		if (sizeof(P) == 8) tol = 1e-12 * factor;
		else tol = 8.0 * FLT_EPSILON * factor * ((xmax - xmin) + (ymax - ymin));
		/******************************************************************************
		conditioning floor: per pixel the coverage is a sum of O(1) edge integrals that
		cancel down to ~area, then scaled by factor / area, so any change of rounding
		path moves a pixel by up to ~eps * factor * (bbox width + bbox height) / area.
		this exceeds the thresholds above only for slivers (area ~1e-6) and thin
		triangles; for well-conditioned triangles it is below them. measured worst
		ratio of the observed difference to this bound is ~1 for both precisions.
		******************************************************************************/
		{
			long double ex = (long double)tr.p1.re - tr.p0.re, ey = (long double)tr.p1.im - tr.p0.im;
			long double fx = (long double)tr.p2.re - tr.p0.re, fy = (long double)tr.p2.im - tr.p0.im;
			double ar = (double)std::fabs(0.5L * (ex * fy - fx * ey));
			double eps = sizeof(P) == 8 ? DBL_EPSILON : (double)FLT_EPSILON;
			double floor_tol = 4.0 * eps * factor * ((xmax - xmin) + (ymax - ymin)) / ar;
			if (floor_tol > tol) tol = floor_tol;
		}
		for (size_t i = 0; i < N; i++)
		{
			double err = std::fabs((double)got[i] - (double)ref[i]);
			if (!(err <= tol))
			{
				if (bad == 0)
				{
					char buf[300];
					std::snprintf(buf, sizeof buf, "tri %zu pixel %zu got=%.9g ref=%.9g tol=%.3g", t, i, (double)got[i], (double)ref[i], tol);
					first = buf;
				}
				bad++;
			}
			if (!(err / tol <= worst)) worst = err / tol;
		}
	}
	char buf[400];
	std::snprintf(buf, sizeof buf, "%ld pixels outside tolerance over %zu triangles (worst err/tol %.3g); first: %s", bad, tris.size(), worst, first.c_str());
	report(std::string("direct_path_matches_reference_") + ptag, bad == 0, buf);
}

int main()
{
	test_integral<double, false>("int_max0_matches_reference", "double");
	test_integral<float, false>("int_max0_matches_reference", "float");
	test_integral<double, true>("int_clamp01_matches_reference", "double");
	test_integral<float, true>("int_clamp01_matches_reference", "float");
	test_direct<double>("double");
	test_direct<float>("float");
	return n_fail == 0 ? 0 : 1;
}
