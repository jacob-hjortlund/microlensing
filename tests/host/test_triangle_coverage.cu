/******************************************************************************
Host test of tri_cov: int_max0 and int_clamp01 against their closed form in
long double, and allocate_triangle against exact clipping (exact_clip.h).
It compiles with nvcc, or as plain C++ through tests/run_cuda_tests.sh --host.
Each case runs for the two instantiations the IPM kernel uses: float vertices
with float pixel arithmetic, and double with double (the case of edges far
outside the map runs with double vertices and either pixel arithmetic). The
cases of the direct per-pixel path pass edge_min_bbox = LLONG_MAX, so no
triangle takes the perimeter path, and count the calls of the step callback,
which only that path makes, as failures. The cases of the perimeter path pass
a small edge_min_bbox, collect the steps in a buffer and add its row-wise
prefix sums to the map, as the IPM kernel does. One PASS/FAIL line per
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
#include <climits>
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
	long bad_px = 0, bad_sum = 0, bad_away = 0, n_bad_index = 0, n_bad_value = 0, n_degenerate = 0, n_steps = 0;
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
		auto no_step = [&](long long, P) { n_steps++; };
		tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, LLONG_MAX, add, add, no_step);

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

	std::snprintf(buf, sizeof buf, "%ld pixels outside tolerance over %zu triangles, %ld out-of-map indices, %ld values outside [0, factor], %ld steps; first: ",
		bad_px, tris.size(), n_bad_index, n_bad_value, n_steps);
	report(std::string("direct_path_matches_exact_clipping_") + ptag, bad_px == 0 && n_bad_index == 0 && n_bad_value == 0 && n_steps == 0, buf + first_px);
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
			long steps = 0;
			auto count = [&](long long, P) { calls++; };
			auto no_step = [&](long long, P) { steps++; };
			tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, LLONG_MAX, count, count, no_step);
			if (calls != 0 || steps != 0)
			{
				std::snprintf(buf, sizeof buf, "triangle %zu orientation %d: %ld deposits, %ld steps; ", t, orient, calls, steps);
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
			long steps = 0;
			auto count = [&](long long idx, P) { calls++; if (idx < 0 || idx >= total) bad++; };
			auto no_step = [&](long long, P) { steps++; };
			tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, LLONG_MAX, count, count, no_step);
			if (calls != 0 || bad != 0 || steps != 0)
			{
				std::snprintf(buf, sizeof buf, "triangle %zu orientation %d: %ld deposits, %ld out of range, %ld steps; ",
					t, orient, calls, bad, steps);
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
		long steps = 0;
		auto count = [&](long long idx, P) { calls++; if (idx < 0 || idx >= total) bad++; };
		auto no_step = [&](long long, P) { steps++; };
		tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, LLONG_MAX, count, count, no_step);
		if (calls == 0 || bad != 0 || steps != 0)
		{
			std::snprintf(buf, sizeof buf, "orientation %d: %ld deposits, %ld out of range, %ld steps; ", orient, calls, bad, steps);
			detail += buf;
		}
	}
	report(std::string("extent_just_below_limit_still_deposits_") + ptag, detail.empty(), detail);
}

/******************************************************************************
the perimeter path: allocate_triangle with the given edge_min_bbox, with the
deposits of both callbacks in map E and the steps in buffer S. out-of-map
indices are counted rather than written. add_row_prefix then adds the row-wise
inclusive prefix sums of S to E, as the IPM kernel does after shooting
******************************************************************************/
struct Calls { long many = 0, steps = 0, bad_index = 0; };

template <typename P, typename T>
static Calls deposit_perimeter(const Tri<T>& tr, P factor, Complex<int> npix, long long edge_min_bbox,
	std::vector<P>& E, std::vector<P>& S, std::vector<long long>* step_idx = nullptr)
{
	const long long N = 1LL * npix.re * npix.im;
	Calls n;
	auto put = [&](std::vector<P>& v, long long i, P x) { if (i < 0 || i >= N) n.bad_index++; else v[i] += x; };
	auto add_one = [&](long long i, P v) { put(E, i, v); };
	auto add_many = [&](long long i, P v) { n.many++; put(E, i, v); };
	auto add_step = [&](long long i, P v) { n.steps++; if (step_idx) step_idx->push_back(i); put(S, i, v); };
	tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, edge_min_bbox, add_one, add_many, add_step, 0.0);
	return n;
}

template <typename P>
static void add_row_prefix(std::vector<P>& E, const std::vector<P>& S, Complex<int> npix)
{
	for (int iy = 0; iy < npix.im; iy++)
	{
		P run = 0;
		for (int ix = 0; ix < npix.re; ix++)
		{
			const long long k = 1LL * npix.re * iy + ix;
			run += S[k];
			E[k] += run;
		}
	}
}

/******************************************************************************
step residue. in a row without a vertex two edges cross the whole row, L = 1
for both, and their steps cancel exactly; the two edges from the lowest or the
highest vertex have the same L in its row, and cancel exactly too. in a row
that holds the middle vertex (in y) one edge ends and another begins, their L
sum to the third's only in exact arithmetic, and the steps, each L times
factor / area, leave a rounding residue that the prefix sum carries to every
column right of the triangle. the steps are at most factor / area in size,
and a row has at most three, so the residue is at most a few eps(P) times
factor / area: the tolerance is 8 eps(P) * factor / area. it is not the pixel
tolerance, which is much larger for triangles of more than a pixel
******************************************************************************/
template <typename P>
static double step_residue_tolerance(const Geom& g, double factor)
{
	return 8 * std::numeric_limits<P>::epsilon() * factor / g.area;
}

//whether row iy holds the middle vertex (in y) of the triangle
static bool holds_middle_vertex(const Geom& g, int iy)
{
	double y[3] = {g.y[0], g.y[1], g.y[2]};
	std::sort(y, y + 3);
	return iy <= y[1] && y[1] <= iy + 1;
}

/******************************************************************************
deposit every triangle of the set with edge_min_bbox = 0, so that every
triangle that is not inside one pixel takes the perimeter path, on a 64 x 48
map and on maps of 7 x 3 and 1 x 1 pixels (that the triangles of the set
cross or cover), add the prefix sums of the steps, and compare:
- each pixel against factor * exact_overlap / |area|, and against the direct
  path (edge_min_bbox = LLONG_MAX), within the pixel tolerance of the direct
  path. the deposits on the perimeter path are partial values that are not
  clamped, so here a tolerance of factor or more still bounds the error; the
  INFO lines count these triangles all the same
- the sum of the deposits on the pixels within half a pixel of the triangle
  against factor * (area inside the map) / |area|, within the same tolerance
  (not scaled by the number of pixels, see test_against_exact_clipping)
- the other pixels, at least half a pixel from the triangle: in a row that
  holds its middle vertex and right of its leftmost point within the step
  residue tolerance (those right of the triangle can carry the residue), and
  everywhere else without tolerance: exactly 0
- a triangle that calls add_many takes the perimeter path, i.e. also adds steps
******************************************************************************/
template <typename P, typename T>
static void test_perimeter_against_exact_clipping(const char* ptag)
{
	const Complex<int> grids[3] = {Complex<int>(64, 48), Complex<int>(7, 3), Complex<int>(1, 1)};
	const std::vector<Tri<T>> tris = triangle_set<T>(1, 1);
	const P factor = P(1.7);

	double worst_px[NCAT] = {}, worst_direct[NCAT] = {}, worst_sum[NCAT] = {}, worst_residue[NCAT] = {};
	long n_unchecked[NCAT] = {}, n_unchecked_residue[NCAT] = {};
	long bad_px = 0, bad_direct = 0, bad_sum = 0, bad_away = 0, bad_residue = 0, n_bad_index = 0, n_degenerate = 0;
	long n_perimeter = 0, n_direct_on_perimeter = 0, n_direct_steps = 0, n_residue_px = 0, n_zero_px = 0, ntri = 0;
	std::string first_px, first_direct, first_sum, first_away, first_residue, first_direct_path;
	char buf[300];
	for (const Complex<int>& npix : grids)
	{
		const long long N = 1LL * npix.re * npix.im;
		std::vector<P> E(N), S(N), D(N);
		for (size_t t = 0; t < tris.size(); t++, ntri++)
		{
			const Tri<T>& tr = tris[t];
			const int c = tr.cat;
			std::fill(E.begin(), E.end(), P(0));
			std::fill(S.begin(), S.end(), P(0));
			std::fill(D.begin(), D.end(), P(0));
			const Calls n = deposit_perimeter(tr, factor, npix, 0, E, S);
			add_row_prefix(E, S, npix);
			n_bad_index += n.bad_index;
			if (n.steps > 0) n_perimeter++;
			if (n.many > 0 && n.steps == 0)
			{
				if (n_direct_on_perimeter == 0)
				{
					std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu", npix.re, npix.im, cat_names[c], t);
					first_direct_path = buf;
				}
				n_direct_on_perimeter++;
			}
			auto add = [&](long long i, P v) { D[i] += v; };
			auto no_step = [&](long long, P) { n_direct_steps++; };
			tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, LLONG_MAX, add, add, no_step);

			const Geom g = geom(tr);
			if (!(g.area > 0)) { n_degenerate++; continue; }
			const double tol = pixel_tolerance<P>(g, factor);
			const double tol_residue = step_residue_tolerance<P>(g, factor);
			if (tol >= factor) n_unchecked[c]++;
			if (tol_residue >= factor) n_unchecked_residue[c]++;

			//the triangle's bounding box grown by a pixel: pixels outside it are at least half a pixel from the triangle
			const double gx0 = std::floor(g.xmin) - 1, gx1 = std::floor(g.xmax) + 1;
			const double gy0 = std::floor(g.ymin) - 1, gy1 = std::floor(g.ymax) + 1;
			long double sum = 0;
			for (int iy = 0; iy < npix.im; iy++)
			{
				const bool middle_row = holds_middle_vertex(g, iy);
				for (int ix = 0; ix < npix.re; ix++)
				{
					const long long k = 1LL * npix.re * iy + ix;
					const bool near_box = ix >= gx0 && ix <= gx1 && iy >= gy0 && iy <= gy1;
					const bool away = !near_box || exact_clip::rect_overlap(g.x, g.y, ix - 0.5, iy - 0.5, ix + 1.5, iy + 1.5) == 0;
					const double ov = near_box ? exact_overlap(g.x, g.y, ix, iy) : 0;
					const double want = factor * ov / g.area;
					const double got = E[k];

					const double err = std::fabs(got - want);
					if (!(err <= tol))
					{
						if (bad_px == 0)
						{
							std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu pixel (%d, %d) got=%.9g want=%.9g tol=%.3g",
								npix.re, npix.im, cat_names[c], t, ix, iy, got, want, tol);
							first_px = buf;
						}
						bad_px++;
					}
					if (!(err / tol <= worst_px[c])) worst_px[c] = err / tol;

					const double err_direct = std::fabs(got - (double)D[k]);
					if (!(err_direct <= tol))
					{
						if (bad_direct == 0)
						{
							std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu pixel (%d, %d) perimeter=%.9g direct=%.9g tol=%.3g",
								npix.re, npix.im, cat_names[c], t, ix, iy, got, (double)D[k], tol);
							first_direct = buf;
						}
						bad_direct++;
					}
					if (!(err_direct / tol <= worst_direct[c])) worst_direct[c] = err_direct / tol;

					if (!away) sum += got;
					else if (middle_row && ix > g.xmin)
					{
						n_residue_px++;
						if (!(std::fabs(got) <= tol_residue))
						{
							if (bad_residue == 0)
							{
								std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu pixel (%d, %d) got=%.9g tol=%.3g",
									npix.re, npix.im, cat_names[c], t, ix, iy, got, tol_residue);
								first_residue = buf;
							}
							bad_residue++;
						}
						if (!(std::fabs(got) / tol_residue <= worst_residue[c])) worst_residue[c] = std::fabs(got) / tol_residue;
					}
					else
					{
						n_zero_px++;
						if (got != 0)
						{
							if (bad_away == 0)
							{
								std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu pixel (%d, %d) got=%.9g",
									npix.re, npix.im, cat_names[c], t, ix, iy, got);
								first_away = buf;
							}
							bad_away++;
						}
					}
				}
			}

			const double want_sum = factor * exact_clip::rect_overlap(g.x, g.y, 0, 0, npix.re, npix.im) / g.area;
			const double err_sum = (double)std::fabs(sum - (long double)want_sum);
			if (!(err_sum <= tol))
			{
				if (bad_sum == 0)
				{
					std::snprintf(buf, sizeof buf, "grid %dx%d %s tri %zu sum=%.12g want=%.12g tol=%.3g",
						npix.re, npix.im, cat_names[c], t, (double)sum, want_sum, tol);
					first_sum = buf;
				}
				bad_sum++;
			}
			if (!(err_sum / tol <= worst_sum[c])) worst_sum[c] = err_sum / tol;
		}
	}

	std::string info, unchecked;
	for (int c = 0; c < NCAT; c++)
	{
		std::snprintf(buf, sizeof buf, " %s %.3g/%.3g/%.3g/%.3g", cat_names[c], worst_px[c], worst_direct[c], worst_sum[c], worst_residue[c]);
		info += buf;
		std::snprintf(buf, sizeof buf, " %s %ld/%ld", cat_names[c], n_unchecked[c], n_unchecked_residue[c]);
		unchecked += buf;
	}
	std::printf("INFO perimeter_%s worst err/tol per pixel vs exact/vs direct/sum/step residue:%s; %ld degenerate triangles skipped\n",
		ptag, info.c_str(), n_degenerate);
	std::printf("INFO perimeter_%s %ld of %ld triangles took the perimeter path; away from their triangle %ld pixels checked for exactly 0, %ld against the step residue tolerance\n",
		ptag, n_perimeter, ntri, n_zero_px, n_residue_px);
	std::printf("INFO perimeter_%s triangles unchecked by tolerance (pixel/step residue tolerance >= factor):%s\n", ptag, unchecked.c_str());

	std::snprintf(buf, sizeof buf, "%ld pixels outside tolerance of exact clipping and %ld of the direct path over %ld triangles, %ld out-of-map indices, "
		"%ld triangles on the direct path (first: %s), %ld steps with edge_min_bbox = LLONG_MAX; first: ",
		bad_px, bad_direct, ntri, n_bad_index, n_direct_on_perimeter, first_direct_path.c_str(), n_direct_steps);
	report(std::string("perimeter_path_matches_direct_") + ptag,
		bad_px == 0 && bad_direct == 0 && n_bad_index == 0 && n_direct_on_perimeter == 0 && n_direct_steps == 0 && n_perimeter > 0,
		buf + (bad_px ? first_px : first_direct));
	std::snprintf(buf, sizeof buf, "%ld of %ld triangles outside tolerance; first: ", bad_sum, ntri);
	report(std::string("perimeter_path_conserves_area_") + ptag, bad_sum == 0, buf + first_sum);
	std::snprintf(buf, sizeof buf, "%ld pixels at least half a pixel from their triangle received a deposit, %ld of %ld in the middle vertex's row "
		"more than the step residue, over %ld exact-zero pixels (must be > 0); first: ", bad_away, bad_residue, n_residue_px, n_zero_px);
	report(std::string("perimeter_path_no_deposit_away_from_triangle_") + ptag, bad_away == 0 && bad_residue == 0 && n_zero_px > 0 && n_residue_px > 0,
		buf + (bad_away ? first_away : first_residue));
}

/******************************************************************************
threshold: the clamped bounding box of this triangle is 9 x 9 = 81 pixels
(columns 10..18, rows 20..28) and its area 36.7, which is over 4 per row (well above the sliver limit). with edge_min_bbox = 81 it takes the direct
path (deposits, no step), with 80 the perimeter path (steps). both orientations
******************************************************************************/
template <typename P, typename T>
static void test_threshold(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	const Tri<T> tri = {Complex<T>(T(10.25), T(20.25)), Complex<T>(T(18.95), T(20.5)), Complex<T>(T(12.5), T(28.75)), 0};
	const long long N = 1LL * npix.re * npix.im;
	std::vector<P> E(N), S(N);
	std::string detail;
	char buf[200];
	for (int orient = 0; orient < 2; orient++)
	{
		Tri<T> tr = tri;
		if (orient) std::swap(tr.p1, tr.p2);
		const Calls n81 = deposit_perimeter(tr, factor, npix, 81, E, S);
		const Calls n80 = deposit_perimeter(tr, factor, npix, 80, E, S);
		if (!(n81.many > 0 && n81.steps == 0 && n80.steps >= 2 && n81.bad_index == 0 && n80.bad_index == 0))
		{
			std::snprintf(buf, sizeof buf, "orientation %d: edge_min_bbox = 81 gave %ld deposits and %ld steps (want > 0 and 0), 80 gave %ld steps (want >= 2), %ld out-of-map indices; ",
				orient, n81.many, n81.steps, n80.steps, n81.bad_index + n80.bad_index);
			detail += buf;
		}
	}
	report(std::string("threshold_selects_path_") + ptag, detail.empty(), detail);
}

/******************************************************************************
slivers. a triangle of length 100 and area 1e-3 over a map holding a background
of 1, through allocate_triangle with the default thresholds (the bounding box
limit of the IPM kernel, 64 pixels, and the default sliver criterion). no pixel
right of the triangle's bounding box may change by more than 4 eps(P): the
perimeter path leaves a rounding residue of about eps(P) * factor / area on
every such pixel, which is why slivers go to the direct path. the sliver is
tilted so that it spans many rows; both orientations
******************************************************************************/
template <typename P, typename T>
static void test_slivers(const char* ptag)
{
	const Complex<int> npix(200, 80);
	const P factor = P(1);
	const long long N = 1LL * npix.re * npix.im;
	const double eps = std::numeric_limits<P>::epsilon();
	//200 slivers, each with its apex at a random point along the long edge, offset by h perpendicular to it;
	//area = 0.5 * 107.7 * h = 1e-3. a rounding residue is small and appears only in some of them
	const double dx = 100, dy = 40, len = std::sqrt(dx * dx + dy * dy);
	const double h = 2e-3 / len;
	std::mt19937 rng(7);
	std::uniform_real_distribution<double> uni(0, 1);
	std::vector<Tri<T>> slivers;
	for (int i = 0; i < 200; i++)
	{
		const double x0 = 10 + 20 * uni(rng), y0 = 10 + 20 * uni(rng), f = 0.2 + 0.6 * uni(rng);
		slivers.push_back({Complex<T>(T(x0), T(y0)), Complex<T>(T(x0 + dx), T(y0 + dy)),
			Complex<T>(T(x0 + f * dx - h * dy / len), T(y0 + f * dy + h * dx / len)), 0});
	}
	long n_checked = 0, n_changed_inside = 0, n_deposits = 0, n_steps = 0;
	double worst = 0;
	std::string detail;
	char buf[200];
	for (int orient = 0; orient < 2; orient++)
	for (const Tri<T>& tri : slivers)
	{
		Tri<T> tr = tri;
		if (orient) std::swap(tr.p1, tr.p2);
		std::vector<P> E(N, P(1)), S(N, P(0));
		long many = 0, steps = 0, bad_index = 0;
		auto put = [&](std::vector<P>& v, long long i, P x) { if (i < 0 || i >= N) bad_index++; else v[i] += x; };
		auto add_one = [&](long long i, P v) { many++; put(E, i, v); };
		auto add_many = [&](long long i, P v) { many++; put(E, i, v); };
		auto add_step = [&](long long i, P v) { steps++; put(S, i, v); };
		tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, 64, add_one, add_many, add_step);
		add_row_prefix(E, S, npix);
		n_deposits += many;
		n_steps += steps;
		const double xmax = std::max(tr.p0.re, std::max(tr.p1.re, tr.p2.re));
		long first_bad = -1;
		for (long long k = 0; k < N; k++)
		{
			const int ix = static_cast<int>(k % npix.re);
			const double change = std::fabs(double(E[k]) - 1);
			if (ix > std::floor(xmax) + 1)
			{
				n_checked++;
				if (!(change <= 4 * eps) && first_bad < 0) first_bad = k;
				if (!(change <= worst)) worst = change;
			}
			else if (change > 0) n_changed_inside++;
		}
		if ((first_bad >= 0 || bad_index != 0) && detail.empty())
		{
			std::snprintf(buf, sizeof buf, "orientation %d: pixel (%ld, %ld) right of the bounding box changed by %.3g (limit %.3g), %ld out-of-map indices; ",
				orient, first_bad % npix.re, first_bad / npix.re, first_bad >= 0 ? std::fabs(double(E[first_bad]) - 1) : 0.0, 4 * eps, bad_index);
			detail += buf;
		}
	}
	//the check must not pass over nothing: pixels were checked and the sliver did deposit something
	if (n_checked == 0 || n_deposits == 0 || n_changed_inside == 0)
	{
		std::snprintf(buf, sizeof buf, "vacuous: %ld pixels checked, %ld deposits, %ld changed pixels inside the box; ", n_checked, n_deposits, n_changed_inside);
		detail += buf;
	}
	std::printf("INFO slivers_%s %ld pixels right of the bounding box checked, worst change %.3g (limit %.3g), %ld deposits, %ld steps\n",
		ptag, n_checked, worst, 4 * eps, n_deposits, n_steps);
	report(std::string("slivers_leave_no_row_tail_") + ptag, detail.empty(), detail);
}

/******************************************************************************
steps outside the map. a triangle entirely left of column 0, overlapping the
map in y, calls nothing.

the left edge of the second triangle, from (-5.5, 10.25) to (3.25, 30.75), is
at x < 0 for y < 23.14, so in rows 10..23 an edge-row span starts at a
negative column and its step must land on column 0 of that row. the third
triangle hangs over the right edge of the map; its edge from (80.25, 12.5) to
(70.25, 30.75) lies entirely right of the map and gives no step, and no step
may wrap to column 0 of the next row (all its steps are at columns >= 60).
both orientations; after the prefix sums each pixel must match
factor * exact_overlap / |area| within the pixel tolerance
******************************************************************************/
template <typename P, typename T>
static void test_steps_outside(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	const long long N = 1LL * npix.re * npix.im;
	std::string detail;
	char buf[300];

	{
		const Tri<T> tr = {Complex<T>(T(-10.5), T(5.25)), Complex<T>(T(-0.5), T(9.75)), Complex<T>(T(-6.25), T(30.5)), 0};
		long calls = 0;
		auto count = [&](long long, P) { calls++; };
		tri_cov::allocate_triangle<P, T>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, 0, count, count, count, 0.0);
		if (calls != 0) { std::snprintf(buf, sizeof buf, "left of the map: %ld callback calls (want 0); ", calls); detail += buf; }
	}

	const Tri<T> left = {Complex<T>(T(-5.5), T(10.25)), Complex<T>(T(20.25), T(12.5)), Complex<T>(T(3.25), T(30.75)), 0};
	const Tri<T> right = {Complex<T>(T(60.5), T(10.25)), Complex<T>(T(80.25), T(12.5)), Complex<T>(T(70.25), T(30.75)), 0};
	for (int side = 0; side < 2; side++)
	{
		for (int orient = 0; orient < 2; orient++)
		{
			Tri<T> tr = side == 0 ? left : right;
			if (orient) std::swap(tr.p1, tr.p2);
			std::vector<long long> step_idx;
			std::vector<P> E(N, P(0)), S(N, P(0));
			const Calls n = deposit_perimeter(tr, factor, npix, 0, E, S, &step_idx);
			add_row_prefix(E, S, npix);

			if (n.bad_index != 0) { std::snprintf(buf, sizeof buf, "side %d orientation %d: %ld out-of-map indices; ", side, orient, n.bad_index); detail += buf; }
			if (side == 0)
			{
				for (int r = 10; r <= 23; r++)
				{
					if (std::find(step_idx.begin(), step_idx.end(), 1LL * npix.re * r) == step_idx.end())
					{
						std::snprintf(buf, sizeof buf, "left orientation %d: no step on column 0 of row %d; ", orient, r);
						detail += buf;
					}
				}
			}
			else
			{
				if (step_idx.empty()) { std::snprintf(buf, sizeof buf, "right orientation %d: no step; ", orient); detail += buf; }
				for (long long i : step_idx)
				{
					if (i % npix.re < 60 || i / npix.re < 10 || i / npix.re > 30)
					{
						std::snprintf(buf, sizeof buf, "right orientation %d: step at row %lld column %lld; ", orient, i / npix.re, i % npix.re);
						detail += buf;
					}
				}
			}

			const Geom g = geom(tr);
			const double tol = pixel_tolerance<P>(g, factor);
			double worst = 0;
			for (int iy = 0; iy < npix.im; iy++)
				for (int ix = 0; ix < npix.re; ix++)
					worst = std::fmax(worst, std::fabs(E[1LL * npix.re * iy + ix] - factor * exact_overlap(g.x, g.y, ix, iy) / g.area));
			if (!(worst <= tol))
			{
				std::snprintf(buf, sizeof buf, "side %d orientation %d: worst |perimeter - exact| %.3g > %.3g; ", side, orient, worst, tol);
				detail += buf;
			}
		}
	}
	report(std::string("steps_outside_map_are_dropped_") + ptag, detail.empty(), detail);
}

/******************************************************************************
edges far outside the map, on the perimeter path. each triangle has a vertex A
in row 16 of the map and an edge B-C far right (or far left) of it, in rows
16..24.

with B and C at x = +-2147483647 (beyond the int range) the triangle is wider
than MAX_EXTENT, and the extent guard drops it: no deposit and no step.

with B and C 2^30 - 1 pixels from A it is just narrower and is deposited. far
right, the edge B-C must deposit nothing and add no step, and each of the two
edges from A must deposit on all four map columns of its span in row 16, with
its step on the first of them; far left, B-C must put a step on column 0 of
every row it crosses. the span columns, about 2^30 from the map, must not
overflow the int conversions or the sums with ix0.

T = double, so that the coordinates are exact; P = double and P = float (in
which the shifted coordinates of B and C round to +-2^30)
******************************************************************************/
template <typename P>
static void test_far_outside(const char* ptag)
{
	const Complex<int> npix(64, 48);
	const P factor = P(1.7);
	const long long N = 1LL * npix.re * npix.im;
	std::string dropped, kept;
	char buf[300];
	for (int side = 0; side < 2; side++)
	{
		const bool right = side == 0;
		const char* where = right ? "right" : "left";
		const double ax = right ? 60.5 : 3.5;
		const int c0 = right ? 60 : 0, c1 = right ? 63 : 3; //map columns of the spans in row 16
		for (int orient = 0; orient < 2; orient++)
		{
			for (int beyond = 0; beyond < 2; beyond++)
			{
				const double bx = beyond ? (right ? 2147483647.0 : -2147483647.0) : (right ? ax + (0x1p30 - 1) : ax - (0x1p30 - 1));
				Tri<double> tr = {Complex<double>(ax, 16.25), Complex<double>(bx, 16.75), Complex<double>(bx, 24.5), 0};
				if (orient) std::swap(tr.p1, tr.p2);
				std::vector<long long> deps, steps;
				auto add = [&](long long i, P) { deps.push_back(i); };
				auto step = [&](long long i, P) { steps.push_back(i); };
				tri_cov::allocate_triangle<P, double>(tr.p0, tr.p1, tr.p2, shoelace(tr), factor, npix, 0, add, add, step, 0.0);

				if (beyond)
				{
					if (!deps.empty() || !steps.empty())
					{
						std::snprintf(buf, sizeof buf, "%s orientation %d: %zu deposits and %zu steps (want 0); ", where, orient, deps.size(), steps.size());
						dropped += buf;
					}
					continue;
				}
				for (long long i : deps)
				{
					if (i < 0 || i >= N || i / npix.re != 16 || i % npix.re < c0 || i % npix.re > c1)
					{
						std::snprintf(buf, sizeof buf, "%s orientation %d: deposit at index %lld (row %lld column %lld); ", where, orient, i, i / npix.re, i % npix.re);
						kept += buf;
					}
				}
				if (deps.size() != 8)
				{
					std::snprintf(buf, sizeof buf, "%s orientation %d: %zu deposits (want 8: columns %d..%d of row 16 for each edge from A); ", where, orient, deps.size(), c0, c1);
					kept += buf;
				}
				if (right)
				{
					//one step per edge from A, at its span's first column; none from B-C
					const bool two_at_a = steps.size() == 2 && steps[0] == 16LL * npix.re + 60 && steps[1] == 16LL * npix.re + 60;
					if (!two_at_a)
					{
						std::snprintf(buf, sizeof buf, "right orientation %d: %zu steps (want 2, both on row 16 column 60)", orient, steps.size());
						kept += buf;
						for (size_t k = 0; k < steps.size() && k < 4; k++)
						{
							std::snprintf(buf, sizeof buf, "%s%lld", k ? ", " : ": ", steps[k]);
							kept += buf;
						}
						kept += "; ";
					}
				}
				else
				{
					for (long long i : steps)
					{
						if (i < 0 || i >= N || i % npix.re != 0)
						{
							std::snprintf(buf, sizeof buf, "left orientation %d: step at index %lld (want column 0); ", orient, i);
							kept += buf;
						}
					}
					for (int r = 16; r <= 24; r++)
					{
						if (std::find(steps.begin(), steps.end(), 1LL * npix.re * r) == steps.end())
						{
							std::snprintf(buf, sizeof buf, "left orientation %d: no step on column 0 of row %d; ", orient, r);
							kept += buf;
						}
					}
				}
			}
		}
	}
	report(std::string("edges_far_outside_map_are_dropped_") + ptag, dropped.empty(), dropped);
	report(std::string("edges_far_outside_map_below_extent_limit_deposit_") + ptag, kept.empty(), kept);
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
	test_perimeter_against_exact_clipping<double, double>("double");
	test_perimeter_against_exact_clipping<float, float>("float");
	test_threshold<double, double>("double");
	test_threshold<float, float>("float");
	test_slivers<double, double>("double");
	test_slivers<float, float>("float");
	test_steps_outside<double, double>("double");
	test_steps_outside<float, float>("float");
	test_far_outside<double>("double");
	test_far_outside<float>("float_pixels_double_vertices");
	return n_fail == 0 ? 0 : 1;
}
