/******************************************************************************
Host test of tri_cov: int_max0 and int_clamp01 against their closed form in
long double, and allocate_triangle against exact clipping (exact_clip.h).
It compiles with nvcc, or as plain C++ through tests/run_cuda_tests.sh --host.
Each case runs for the two instantiations the IPM kernel uses: float vertices
with float pixel arithmetic, and double with double. One PASS/FAIL line per
case (INFO lines give, per triangle category, the worst ratio of error to
tolerance and the number of triangles whose tolerance checks nothing); exit 0
only if all pass.
******************************************************************************/
#ifdef __CUDACC__
//the callbacks are host lambdas handed to the __host__ __device__ allocate_triangle,
//which nvcc reports for every call; nothing here runs on the device
#pragma nv_diag_suppress 20013
#endif

#include "triangle_coverage.cuh"
#include "exact_clip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail)
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

/******************************************************************************
the 1D integrals against their closed form, evaluated in long double from the
same inputs: for u linear from a to b over an interval of length L,
∫ max(u, 0) = L * (max(a,0) + max(b,0))^2 / (2 * (|a| + |b|)), 0 if a = b = 0
******************************************************************************/
static long double closed_form_max0(long double a, long double b, long double L)
{
	const long double pa = a > 0 ? a : 0, pb = b > 0 ? b : 0;
	const long double den = std::fabs(a) + std::fabs(b);
	return den > 0 ? L * (pa + pb) * (pa + pb) / (2 * den) : 0;
}

/******************************************************************************
tolerance of an integral: 16 * eps(P) times the magnitude of its int_max0
terms, |int_max0(a, b)| for int_max0 and |int_max0(a, b)| + |int_max0(a - 1, b - 1)|
for int_clamp01, whose two terms can cancel. each term is a product of a few
rounded factors. where one endpoint is <= 0, the code multiplies by
1 / (2 |slope|) where the closed form divides by 2 (|a| + |b|) / L; the two
differ by the rounding of b = a + L * slope, which relative to |a| + |b| >= |b|
is a few eps. a term with no positive endpoint has magnitude 0, so the
tolerance is 0 and the result must be exactly 0
******************************************************************************/
template <typename P>
static double integral_tolerance(long double magnitude)
{
	return 16 * std::numeric_limits<P>::epsilon() * (double)magnitude;
}

//sign pattern of the endpoints of an int_max0 term: both positive (the trapezoid), one (the division-free form), none (0)
static int sign_pattern(long double a, long double b)
{
	return a > 0 && b > 0 ? 0 : (a > 0 || b > 0 ? 1 : 2);
}
static const char* const pattern_names[3] = {"both", "one", "none"};

/******************************************************************************
1e6 segments: a in [-3, 3], L in (1e-9, 1], slope in [-50, 50] (one in ten 0,
one in ten 1e-6, one in ten -1e-6) and b = a + L * slope in precision P. inv2s
is make_edge's, for the edge from (0, 0) to (slope, 1), whose slope is
exactly `slope`. the samples are counted by the sign patterns of their
int_max0 terms (for int_clamp01 the pairs of patterns of (a, b) and
(a - 1, b - 1) that can occur), and the case fails if a pattern has none.
without tolerance: a result whose terms have no positive endpoint is exactly 0
******************************************************************************/
template <typename P, bool CLAMP>
static void test_integral(const char* name, const char* ptag)
{
	std::mt19937_64 gen(12345);
	std::uniform_real_distribution<double> da(-3, 3), dL(1e-9, 1), ds(-50, 50);
	long count[9] = {}, bad = 0, bad_zero = 0;
	double worst = 0;
	std::string first;
	char buf[300];
	for (int i = 0; i < 1000000; i++)
	{
		const P a = static_cast<P>(da(gen));
		const P L = static_cast<P>(dL(gen));
		double sd = ds(gen);
		if (i % 10 == 0) sd = 0;
		else if (i % 10 == 1) sd = 1e-6;
		else if (i % 10 == 2) sd = -1e-6;
		const P slope = static_cast<P>(sd);
		const P b = a + L * slope;
		const P inv2s = tri_cov::make_edge<P>(P(0), P(0), slope, P(1)).inv2s;
		const P got = CLAMP ? tri_cov::int_clamp01<P>(a, b, L, inv2s) : tri_cov::int_max0<P>(a, b, L, inv2s);

		const long double am1 = (long double)a - 1, bm1 = (long double)b - 1;
		const long double t0 = closed_form_max0(a, b, L), t1 = CLAMP ? closed_form_max0(am1, bm1, L) : 0;
		const int p0 = sign_pattern(a, b), p1 = CLAMP ? sign_pattern(am1, bm1) : 2;
		count[3 * p0 + p1]++;
		const double tol = integral_tolerance<P>(std::fabs(t0) + std::fabs(t1));
		const double err = (double)std::fabs((long double)got - (t0 - t1));
		if (!(err <= tol))
		{
			if (bad == 0)
			{
				std::snprintf(buf, sizeof buf, "a=%.9g b=%.9g L=%.9g slope=%.9g closed form=%.12Lg got=%.12g tol=%.3g",
					(double)a, (double)b, (double)L, (double)slope, t0 - t1, (double)got, tol);
				first = buf;
			}
			bad++;
		}
		if (tol > 0 && !(err / tol <= worst)) worst = err / tol;
		if (p0 == 2 && p1 == 2 && got != 0) bad_zero++;
	}

	//the patterns that can occur: for int_max0 those of (a, b); for int_clamp01 the pairs with a - 1 < a
	static const int max0_patterns[] = {2, 5, 8};
	static const int clamp_patterns[] = {0, 1, 2, 4, 5, 8};
	const int* pats = CLAMP ? clamp_patterns : max0_patterns;
	const int npats = CLAMP ? 6 : 3;
	std::string info, missing;
	for (int k = 0; k < npats; k++)
	{
		const int pat = pats[k];
		if (CLAMP) std::snprintf(buf, sizeof buf, " %s/%s %ld", pattern_names[pat / 3], pattern_names[pat % 3], count[pat]);
		else std::snprintf(buf, sizeof buf, " %s %ld", pattern_names[pat / 3], count[pat]);
		info += buf;
		if (count[pat] == 0) missing += buf;
	}
	std::printf("INFO %s_%s samples per number of positive endpoints%s:%s; worst err/tol %.3g\n",
		name, ptag, CLAMP ? " of (a, b)/(a - 1, b - 1)" : "", info.c_str(), worst);
	std::snprintf(buf, sizeof buf, "%ld of 1e6 outside tolerance, %ld with no positive endpoint not exactly 0, patterns without samples:%s; first: ",
		bad, bad_zero, missing.empty() ? " none" : missing.c_str());
	report(std::string(name) + "_" + ptag, bad == 0 && bad_zero == 0 && missing.empty(), buf + first);
}

/******************************************************************************
an edge with a subnormal slope: 1 / (2 |slope|) overflows, so make_edge must
give a finite inv2s (0), and the integrals must not become NaN
******************************************************************************/
template <typename P>
static void test_subnormal_slope(const char* ptag)
{
	const P tiny = std::numeric_limits<P>::denorm_min();
	const tri_cov::Edge<P> e = tri_cov::make_edge<P>(P(0), P(0), tiny, P(1));
	const P both_nonpos = tri_cov::int_max0<P>(P(-0.25), P(-0.25) + e.slope, P(1), e.inv2s);
	const P mixed = tri_cov::int_max0<P>(P(0), e.slope, P(1), e.inv2s);
	const P clamped = tri_cov::int_clamp01<P>(P(0.5), P(0.5) + e.slope, P(1), e.inv2s);
	const bool ok = e.slope > 0 && std::isfinite(e.inv2s) && both_nonpos == 0
		&& std::isfinite(mixed) && std::fabs(mixed) <= tiny && clamped == P(0.5);
	char buf[300];
	std::snprintf(buf, sizeof buf, "slope=%.3g inv2s=%.3g int_max0(both<=0)=%.3g int_max0(mixed)=%.3g int_clamp01=%.9g",
		(double)e.slope, (double)e.inv2s, (double)both_nonpos, (double)mixed, (double)clamped);
	report(std::string("inv2s_finite_for_subnormal_slope_") + ptag, ok, buf);
}

/******************************************************************************
triangle sets
******************************************************************************/
enum { CAT_RANDOM, CAT_MODERATE, CAT_SLIVER, CAT_AXIS, CAT_CROSSING, CAT_COVERING, NCAT };
static const char* const cat_names[NCAT] = {"random", "moderate", "sliver", "axis-aligned", "crossing-edge", "covering"};

template <typename T>
struct Tri { Complex<T> p0, p1, p2; int cat; };

/******************************************************************************
triangles made for a 64 x 48 map, scaled by (sx, sy) to other maps; both
orientations of each
******************************************************************************/
template <typename T>
static std::vector<Tri<T>> triangle_set(double sx, double sy)
{
	std::mt19937_64 gen(777);
	std::uniform_real_distribution<double> dx(-10, 74), dy(-10, 58), d01(0, 1), dsm(-1, 1);
	std::vector<Tri<T>> tris;
	int cat = CAT_RANDOM;
	auto push = [&](double x0, double y0, double x1, double y1, double x2, double y2)
	{
		tris.push_back({Complex<T>(T(sx * x0), T(sy * y0)), Complex<T>(T(sx * x1), T(sy * y1)), Complex<T>(T(sx * x2), T(sy * y2)), cat});
	};
	//random
	cat = CAT_RANDOM;
	for (int i = 0; i < 3000; i++) push(dx(gen), dy(gen), dx(gen), dy(gen), dx(gen), dy(gen));
	//random moderate-size triangles (a few pixels)
	cat = CAT_MODERATE;
	for (int i = 0; i < 3000; i++)
	{
		double cx = 64 * d01(gen), cy = 48 * d01(gen);
		push(cx + 3 * dsm(gen), cy + 3 * dsm(gen), cx + 3 * dsm(gen), cy + 3 * dsm(gen), cx + 3 * dsm(gen), cy + 3 * dsm(gen));
	}
	//slivers, area ~1e-6
	cat = CAT_SLIVER;
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
	cat = CAT_AXIS;
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
	cat = CAT_CROSSING;
	push(-5, 10, 5, 5, 3, 20);
	push(60, 10, 70, 15, 62, 25);
	push(10, -5, 20, 3, 30, -2);
	push(10, 40, 20, 52, 30, 45);
	push(-5, -5, 5, 2, 2, 5);
	push(60, 44, 70, 55, 66, 40);
	push(-5, 20, 70, 25, 30, 30);
	//covering the whole map
	cat = CAT_COVERING;
	push(-100, -100, 300, -100, -100, 300);
	push(-1, -1, 200, 0, 0, 200);
	//both orientations
	const size_t n0 = tris.size();
	for (size_t i = 0; i < n0; i++) tris.push_back({tris[i].p0, tris[i].p2, tris[i].p1, tris[i].cat});
	return tris;
}

template <typename T>
static T shoelace(const Tri<T>& tr)
{
	return T(0.5) * ((tr.p1.re - tr.p0.re) * (tr.p2.im - tr.p0.im) - (tr.p2.re - tr.p0.re) * (tr.p1.im - tr.p0.im));
}

/******************************************************************************
the triangle in double, with its area (computed in long double, so exact to
far below the rounding of either precision), longest edge and bounding box
******************************************************************************/
struct Geom
{
	double x[3], y[3];
	double area, longest;
	double xmin, xmax, ymin, ymax;
};

template <typename T>
static Geom geom(const Tri<T>& tr)
{
	Geom g = {{tr.p0.re, tr.p1.re, tr.p2.re}, {tr.p0.im, tr.p1.im, tr.p2.im}, 0, 0, 0, 0, 0, 0};
	const long double ex = (long double)g.x[1] - g.x[0], ey = (long double)g.y[1] - g.y[0];
	const long double fx = (long double)g.x[2] - g.x[0], fy = (long double)g.y[2] - g.y[0];
	g.area = (double)std::fabs(0.5L * (ex * fy - fx * ey));
	for (int k = 0; k < 3; k++)
	{
		const int m = (k + 1) % 3;
		g.longest = std::fmax(g.longest, std::hypot(g.x[m] - g.x[k], g.y[m] - g.y[k]));
	}
	g.xmin = std::fmin(g.x[0], std::fmin(g.x[1], g.x[2]));
	g.xmax = std::fmax(g.x[0], std::fmax(g.x[1], g.x[2]));
	g.ymin = std::fmin(g.y[0], std::fmin(g.y[1], g.y[2]));
	g.ymax = std::fmax(g.y[0], std::fmax(g.y[1], g.y[2]));
	return g;
}

/******************************************************************************
tolerance per pixel: 64 * eps(P) * factor * c, where P is the precision of the
pixel arithmetic and c = max(1, L^2 / |area|) with L the longest edge. the
deposit is factor * coverage / area with the area recomputed in P from the
vertices, and that area carries a relative rounding error of order
eps(P) * L^2 / |area|; c is this conditioning, which is large for slivers.
the worst measured ratios of error to tolerance are about 1e-2 per pixel and
5e-2 for the sums (INFO lines).

where the tolerance reaches factor it checks nothing, because allocate_triangle
clamps every deposit to [0, factor]. that is the case for every float sliver
of the set (c ~ 1e9); the INFO lines count these triangles as unchecked by
tolerance, and no_deposit_away_from_triangle checks all triangles without one.

the coverage itself is rounded at about eps(P) * L pixels of area, because its
coordinates are of pixel size; c does not cover that for triangles much smaller
than a pixel that straddle a pixel border, where the ratio grows like 1/L and
passes 1 near L = 0.02 px. the set has no such triangles
******************************************************************************/
template <typename P>
static double pixel_tolerance(const Geom& g, double factor)
{
	const double c = std::fmax(1.0, g.longest * g.longest / g.area);
	return 64 * std::numeric_limits<P>::epsilon() * factor * c;
}

/******************************************************************************
deposit every triangle of the set on a 64 x 48 map and compare with exact
clipping:
- each pixel against factor * exact_overlap / |area|, within the pixel tolerance
- the sum of the deposits against factor * (area inside the map) / |area|,
  within the same tolerance. not n times it for n pixels: the main error is the
  rounding of the area recomputed in P, one relative error shared by every
  pixel of the triangle, so the pixel errors are correlated and sum to about
  eps(P) * c * factor. with n the sum would follow from the per-pixel check
- without any tolerance: a pixel at least half a pixel away from the triangle,
  i.e. whose box grown by half a pixel on each side does not overlap it,
  receives exactly 0
******************************************************************************/
template <typename P, typename T>
static void test_against_exact_clipping(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const long long N = 1LL * npix.re * npix.im;
	const std::vector<Tri<T>> tris = triangle_set<T>(1, 1);
	const P factor = P(1.7);

	std::vector<P> dep(N);
	double worst_px[NCAT] = {}, worst_sum[NCAT] = {};
	long n_unchecked[NCAT] = {};
	long bad_px = 0, bad_sum = 0, bad_away = 0, n_bad_index = 0, n_bad_value = 0, n_degenerate = 0;
	double largest_away = 0;
	std::string first_px, first_sum, first_away;
	char buf[300];
	for (size_t t = 0; t < tris.size(); t++)
	{
		const Tri<T>& tr = tris[t];
		const int c = tr.cat;
		std::fill(dep.begin(), dep.end(), P(0));
		auto add = [&](long long i, P v)
		{
			if (i < 0 || i >= N) { n_bad_index++; return; }
			if (!(v >= 0 && v <= factor)) n_bad_value++;
			dep[i] += v;
		};
		tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, add, add);

		const Geom g = geom(tr);
		//a float sliver can round to collinear vertices; its coverage is then undefined
		if (!(g.area > 0)) { n_degenerate++; continue; }
		const double tol = pixel_tolerance<P>(g, factor);
		if (tol >= factor) n_unchecked[c]++;

		//the exact overlap is zero outside the triangle's bounding box
		const int bx0 = (int)std::fmax(0.0, std::floor(g.xmin)), bx1 = (int)std::fmin(npix.re - 1.0, std::floor(g.xmax));
		const int by0 = (int)std::fmax(0.0, std::floor(g.ymin)), by1 = (int)std::fmin(npix.im - 1.0, std::floor(g.ymax));
		long double sum = 0;
		for (int iy = 0; iy < npix.im; iy++)
		{
			for (int ix = 0; ix < npix.re; ix++)
			{
				const long long k = 1LL * npix.re * iy + ix;
				const bool in_box = ix >= bx0 && ix <= bx1 && iy >= by0 && iy <= by1;
				const double ov = in_box ? exact_overlap(g.x, g.y, ix, iy) : 0;
				const double want = factor * ov / g.area;
				const double got = dep[k];
				sum += got;
				const double err = std::fabs(got - want);
				if (!(err <= tol))
				{
					if (bad_px == 0)
					{
						std::snprintf(buf, sizeof buf, "%s tri %zu pixel (%d, %d) got=%.9g want=%.9g tol=%.3g",
							cat_names[c], t, ix, iy, got, want, tol);
						first_px = buf;
					}
					bad_px++;
				}
				if (!(err / tol <= worst_px[c])) worst_px[c] = err / tol;

				if (got != 0 && exact_clip::rect_overlap(g.x, g.y, ix - 0.5, iy - 0.5, ix + 1.5, iy + 1.5) == 0)
				{
					if (bad_away == 0)
					{
						std::snprintf(buf, sizeof buf, "%s tri %zu pixel (%d, %d) got=%.9g", cat_names[c], t, ix, iy, got);
						first_away = buf;
					}
					bad_away++;
					largest_away = std::fmax(largest_away, std::fabs(got));
				}
			}
		}

		const double want_sum = factor * exact_clip::rect_overlap(g.x, g.y, 0, 0, npix.re, npix.im) / g.area;
		const double err_sum = (double)std::fabs(sum - (long double)want_sum);
		if (!(err_sum <= tol))
		{
			if (bad_sum == 0)
			{
				std::snprintf(buf, sizeof buf, "%s tri %zu sum=%.12g want=%.12g tol=%.3g",
					cat_names[c], t, (double)sum, want_sum, tol);
				first_sum = buf;
			}
			bad_sum++;
		}
		if (!(err_sum / tol <= worst_sum[c])) worst_sum[c] = err_sum / tol;
	}

	std::string info, unchecked;
	for (int c = 0; c < NCAT; c++)
	{
		std::snprintf(buf, sizeof buf, " %s %.3g/%.3g", cat_names[c], worst_px[c], worst_sum[c]);
		info += buf;
		std::snprintf(buf, sizeof buf, " %s %ld", cat_names[c], n_unchecked[c]);
		unchecked += buf;
	}
	std::printf("INFO exact_clipping_%s worst err/tol per pixel/sum:%s; %ld degenerate triangles skipped\n",
		ptag, info.c_str(), n_degenerate);
	std::printf("INFO exact_clipping_%s triangles unchecked by tolerance (tolerance >= factor):%s\n", ptag, unchecked.c_str());

	std::snprintf(buf, sizeof buf, "%ld pixels outside tolerance over %zu triangles, %ld out-of-map indices, %ld values outside [0, factor]; first: ",
		bad_px, tris.size(), n_bad_index, n_bad_value);
	report(std::string("direct_path_matches_exact_clipping_") + ptag, bad_px == 0 && n_bad_index == 0 && n_bad_value == 0, buf + first_px);
	std::snprintf(buf, sizeof buf, "%ld of %zu triangles outside tolerance; first: ", bad_sum, tris.size());
	report(std::string("deposits_sum_to_covered_area_") + ptag, bad_sum == 0, buf + first_sum);
	std::snprintf(buf, sizeof buf, "%ld pixels at least half a pixel from their triangle received a deposit (largest %.3g); first: ",
		bad_away, largest_away);
	report(std::string("no_deposit_away_from_triangle_") + ptag, bad_away == 0, buf + first_away);
}

/******************************************************************************
triangles entirely left of, right of, below or above the 64 x 48 map, each
overlapping the map's range in the other coordinate: a large one, one with an
edge on the map's border, and one inside a single pixel next to the map (its
index, if computed, would land in a neighbouring row or outside the map). both
orientations
******************************************************************************/
template <typename P, typename T>
static void test_off_map(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	static const double tris[][6] = {
		//left
		{-30.5, -5.25, -0.5, 20.75, -12.25, 60.5},
		{-5, 10, 0, 10, 0, 20},
		{-0.75, 5.25, -0.25, 5.5, -0.5, 5.75},
		//right
		{64.5, -5.25, 94.5, 20.75, 70.25, 60.5},
		{64, 10, 69, 10, 64, 20},
		{64.25, 5.25, 64.75, 5.5, 64.5, 5.75},
		//below
		{-5.25, -30.5, 70.75, -0.5, 20.5, -12.25},
		{10, -5, 20, 0, 10, 0},
		{5.25, -0.75, 5.5, -0.25, 5.75, -0.5},
		//above
		{-5.25, 48.5, 70.75, 78.5, 20.5, 60.25},
		{10, 48, 20, 48, 10, 53},
		{5.25, 48.25, 5.5, 48.75, 5.75, 48.5},
	};
	std::string detail;
	char buf[200];
	for (size_t t = 0; t < sizeof tris / sizeof tris[0]; t++)
	{
		for (int orient = 0; orient < 2; orient++)
		{
			const double* v = tris[t];
			Tri<T> tr = {Complex<T>(T(v[0]), T(v[1])), Complex<T>(T(v[2]), T(v[3])), Complex<T>(T(v[4]), T(v[5])), 0};
			if (orient) std::swap(tr.p1, tr.p2);
			long calls = 0;
			auto count = [&](long long, P) { calls++; };
			tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, count, count);
			if (calls != 0)
			{
				std::snprintf(buf, sizeof buf, "triangle %zu orientation %d: %ld deposits; ", t, orient, calls);
				detail += buf;
			}
		}
	}
	report(std::string("off_map_triangles_deposit_nothing_") + ptag, detail.empty(), detail);
}

/******************************************************************************
triangles reaching far beyond the map are dropped, so the bounding-box
conversions to int cannot overflow. a triangle just under the limit is still
allocated, and every pixel index stays inside the map. out-of-range indices
are counted rather than written
******************************************************************************/
template <typename P, typename T>
static void test_far_reaching(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	const long long total = 1LL * npix.re * npix.im;
	std::string detail;
	char buf[200];

	//far triangles: to +1e10, to -1e10 (mirror), both ends, and the same in y
	static const double far_tris[][6] = {
		{10, 16.25, 1e10, 16.75, 1e10, 24.5},
		{54, 16.25, -1e10, 16.75, -1e10, 24.5},
		{-1e10, 16.25, 1e10, 16.75, 1e10, 24.5},
		{20.25, 10, 20.75, 1e10, 24.5, 1e10},
		{20.25, 38, 20.75, -1e10, 24.5, -1e10},
		{20.25, -1e10, 20.75, 1e10, 24.5, 1e10},
	};
	for (size_t t = 0; t < sizeof far_tris / sizeof far_tris[0]; t++)
	{
		for (int orient = 0; orient < 2; orient++)
		{
			const double* v = far_tris[t];
			Tri<T> tr = {Complex<T>(T(v[0]), T(v[1])), Complex<T>(T(v[2]), T(v[3])), Complex<T>(T(v[4]), T(v[5])), 0};
			if (orient) std::swap(tr.p1, tr.p2);
			long calls = 0;
			long bad = 0;
			auto count = [&](long long idx, P) { calls++; if (idx < 0 || idx >= total) bad++; };
			tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, count, count);
			if (calls != 0 || bad != 0)
			{
				std::snprintf(buf, sizeof buf, "triangle %zu orientation %d: %ld deposits, %ld out of range; ",
					t, orient, calls, bad);
				detail += buf;
			}
		}
	}
	report(std::string("far_reaching_triangles_are_dropped_") + ptag, detail.empty(), detail);

	//the largest extent below the limit: still deposits, all inside the map
	detail.clear();
	const T ext = std::nextafter(T(tri_cov::MAX_EXTENT), T(0));
	const T h = ext / 2;
	for (int orient = 0; orient < 2; orient++)
	{
		Tri<T> tr = {Complex<T>(-h, T(16.25)), Complex<T>(h, T(16.75)), Complex<T>(h, T(24.5)), 0};
		if (orient) std::swap(tr.p1, tr.p2);
		long calls = 0;
		long bad = 0;
		auto count = [&](long long idx, P) { calls++; if (idx < 0 || idx >= total) bad++; };
		tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, count, count);
		if (calls == 0 || bad != 0)
		{
			std::snprintf(buf, sizeof buf, "orientation %d: %ld deposits, %ld out of range; ", orient, calls, bad);
			detail += buf;
		}
	}
	report(std::string("extent_just_below_limit_still_deposits_") + ptag, detail.empty(), detail);
}

int main()
{
	test_integral<double, false>("int_max0_matches_reference", "double");
	test_integral<float, false>("int_max0_matches_reference", "float");
	test_integral<double, true>("int_clamp01_matches_reference", "double");
	test_integral<float, true>("int_clamp01_matches_reference", "float");
	test_subnormal_slope<double>("double");
	test_subnormal_slope<float>("float");
	test_against_exact_clipping<double, double>("double");
	test_against_exact_clipping<float, float>("float");
	test_off_map<double, double>("double");
	test_off_map<float, float>("float");
	test_far_reaching<double, double>("double");
	test_far_reaching<float, float>("float");
	return n_fail == 0 ? 0 : 1;
}
