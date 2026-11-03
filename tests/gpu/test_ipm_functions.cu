/******************************************************************************
GPU test of the device helpers in ipm_functions.cuh. Needs a GPU; compile and
run it through tests/run_cuda_tests.sh. Each case runs for the two
instantiations the IPM kernel uses: float pixels with float arithmetic, and
double with double. One PASS/FAIL line per case; exit 0 only if all pass.
******************************************************************************/
#include "ipm_functions.cuh"
#include "lens_equations.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

static int n_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail)
{
	if (ok) std::printf("PASS %s\n", name.c_str());
	else { std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str()); n_fail++; }
}

/******************************************************************************
warp_aggregated_add

each kernel runs as a single warp. the destination array is chosen per lane by
a select and the call sits after a __syncwarp, so that all participating lanes
reach it together as one coalesced group, which is the case the aggregation
has to get right. the sums are small integers and therefore exact.
******************************************************************************/
constexpr int NPIX = 8; //entries per array

//lanes 0-15 add 1 to a[7], lanes 16-31 add 2 to b[7]
template <typename T>
__global__ void same_index_different_arrays_kernel(T* a, T* b)
{
	const int lane = threadIdx.x;
	T* base = lane < 16 ? a : b;
	const T v = lane < 16 ? T(1) : T(2);
	__syncwarp();
	ipm_functions::warp_aggregated_add<T, T>(base, 7, v);
}

//even lanes add 1 to a[3], odd lanes add 1 to b[3]
template <typename T>
__global__ void interleaved_lanes_kernel(T* a, T* b)
{
	const int lane = threadIdx.x;
	T* base = (lane & 1) ? b : a;
	__syncwarp();
	ipm_functions::warp_aggregated_add<T, T>(base, 3, T(1));
}

//lane k adds 1 to a[k % 5]
template <typename T>
__global__ void mixed_indices_one_array_kernel(T* a, T* b)
{
	const int lane = threadIdx.x;
	__syncwarp();
	ipm_functions::warp_aggregated_add<T, T>(a, lane % 5, T(1));
}

//only lanes 0-20 take part; each adds 1 to a[0] and then 1 to b[0]
template <typename T>
__global__ void partial_warp_kernel(T* a, T* b)
{
	const int lane = threadIdx.x;
	if (lane <= 20)
	{
		ipm_functions::warp_aggregated_add<T, T>(a, 0, T(1));
		ipm_functions::warp_aggregated_add<T, T>(b, 0, T(1));
	}
}

/******************************************************************************
run one kernel on zeroed arrays a and b of NPIX entries and compare them with
the expected contents
******************************************************************************/
template <typename T>
static void run_case(const std::string& name, void (*kernel)(T*, T*),
	const std::vector<T>& want_a, const std::vector<T>& want_b)
{
	T* d = nullptr;
	std::vector<T> got(2 * NPIX, T(-1));
	cudaError_t err = cudaMalloc(&d, 2 * NPIX * sizeof(T));
	if (err == cudaSuccess) err = cudaMemset(d, 0, 2 * NPIX * sizeof(T));
	if (err == cudaSuccess)
	{
		kernel<<<1, 32>>>(d, d + NPIX);
		err = cudaGetLastError();
	}
	if (err == cudaSuccess) err = cudaDeviceSynchronize();
	if (err == cudaSuccess) err = cudaMemcpy(got.data(), d, 2 * NPIX * sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(d);
	if (err != cudaSuccess)
	{
		report(name, false, std::string("CUDA error: ") + cudaGetErrorString(err));
		return;
	}

	std::string detail;
	char buf[128];
	for (int arr = 0; arr < 2; arr++)
	{
		const std::vector<T>& want = arr ? want_b : want_a;
		for (int i = 0; i < NPIX; i++)
		{
			const T g = got[arr * NPIX + i];
			if (g != want[i])
			{
				std::snprintf(buf, sizeof buf, "%c[%d] = %g, expected %g; ", arr ? 'b' : 'a', i,
					static_cast<double>(g), static_cast<double>(want[i]));
				detail += buf;
			}
		}
	}
	report(name, detail.empty(), detail);
}

template <typename T>
static void test_warp_aggregated_add(const char* ptag)
{
	const std::string tag = std::string("_") + ptag;
	const std::vector<T> zero(NPIX, T(0));
	std::vector<T> a, b;

	a = zero; b = zero;
	a[7] = 16; b[7] = 32;
	run_case<T>("same_index_different_arrays" + tag, same_index_different_arrays_kernel<T>, a, b);

	a = zero; b = zero;
	a[3] = 16; b[3] = 16;
	run_case<T>("interleaved_lanes" + tag, interleaved_lanes_kernel<T>, a, b);

	a = zero;
	a[0] = 7; a[1] = 7; a[2] = 6; a[3] = 6; a[4] = 6;
	run_case<T>("mixed_indices_one_array" + tag, mixed_indices_one_array_kernel<T>, a, zero);

	a = zero; b = zero;
	a[0] = 21; b[0] = 21;
	run_case<T>("partial_warp" + tag, partial_warp_kernel<T>, a, b);
}

/******************************************************************************
in_staged_leaf

a corner on the boundary of a leaf must not count as inside it, so that it is
mapped through the tree walk, which both tiles sharing the corner use. points
just inside, one ulp toward the centre, are inside.
******************************************************************************/
template <typename T>
void test_leaf_test_is_strict(const char* ptag)
{
	const T cx = T(0.5), cy = T(-0.25), hl = T(0.125);
	bool edges_out = true, inside_in = true;
	for (int dx = -1; dx <= 1; dx++)
	{
		for (int dy = -1; dy <= 1; dy++)
		{
			if (dx == 0 && dy == 0) continue;
			//the 8 edge midpoints and corners; any nonzero offset puts a coordinate on the boundary
			Complex<T> on(cx + dx * hl, cy + dy * hl);
			if (ipm_functions::in_staged_leaf<T>(on, cx, cy, hl)) edges_out = false;
			Complex<T> in(dx == 0 ? cx : std::nextafter(on.re, cx), dy == 0 ? cy : std::nextafter(on.im, cy));
			if (!ipm_functions::in_staged_leaf<T>(in, cx, cy, hl)) inside_in = false;
		}
	}
	Complex<T> centre(cx, cy);
	if (!ipm_functions::in_staged_leaf<T>(centre, cx, cy, hl)) inside_in = false;
	report(std::string("leaf_test_is_strict_") + ptag, edges_out && inside_in,
		std::string(edges_out ? "" : "boundary point counted as inside; ") + (inside_in ? "" : "interior point counted as outside"));
}

/******************************************************************************
lens equation agreement

ipm_functions::w_fast and w_compact are FMA-expanded copies of
microlensing::w. one hand-built node (no neighbours) with random stars and
random local coefficients, and random points inside it, are evaluated by all
three with the same data. the comparison is |diff| <= tol * (1 + |w|), over
rectangular x approx x taylor_smooth x with/without stars.
******************************************************************************/
template <typename T>
struct AgreeInput
{
	T kappa, gamma, theta, kappastar;
	Complex<T> corner;
	ipm_functions::SmoothConst<T> sc;
	int rectangular, approx, taylor_smooth;
	int npoints;
	const Complex<T>* z;
	star<T>* stars;
	TreeNode<T>* node;
	const ipm_leaf::StarXYM<T>* ds;
	int count;
	const T* cre;
	const T* cim;
	Complex<T>* out_w;
	Complex<T>* out_fast;
	Complex<T>* out_compact;
};

template <typename T>
__global__ void agree_kernel(AgreeInput<T> in)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= in.npoints) return;
	const ipm_functions::SmoothConst<T>& sc = in.sc;
	const TreeNode<T>* node = in.node;
	const T theta2 = in.theta * in.theta;
	in.out_w[i] = microlensing::w<T>(in.z[i], in.kappa, in.gamma, in.theta, in.stars, in.kappastar, in.node,
		in.rectangular, in.corner, in.approx, in.taylor_smooth);
	in.out_fast[i] = ipm_functions::w_fast<T>(in.z[i], in.kappa, in.gamma, theta2, in.stars, in.kappastar, node,
		in.rectangular, in.corner, in.approx, in.taylor_smooth, sc);
	in.out_compact[i] = ipm_functions::w_compact<T>(in.z[i], in.kappa, in.gamma, theta2, in.kappastar,
		node->center.re, node->center.im, node->half_length, node->expansion_order, in.cre, in.cim, in.ds, in.count,
		in.rectangular, in.corner, in.approx, in.taylor_smooth, sc);
}

template <typename T>
static void test_lens_equation_agreement(const char* ptag, double tol)
{
	constexpr int NPTS = 4096;
	constexpr int NSTARS = 20;
	constexpr int ORDER = 8;
	std::srand(12345);
	auto rnd = [] { return std::rand() / double(RAND_MAX) * 2 - 1; }; //uniform in [-1, 1]

	const T hl = T(1.5);
	Complex<T> center(T(0.25), T(-0.5));
	TreeNode<T>* node = nullptr;
	star<T>* stars = nullptr;
	Complex<T>* z = nullptr;
	Complex<T>* out = nullptr; //w, w_fast, w_compact in that order
	ipm_leaf::StarXYM<T>* ds = nullptr;
	T* coef = nullptr; //cre, then cim
	bool alloc_ok = cudaMallocManaged(&node, sizeof(TreeNode<T>)) == cudaSuccess
		&& cudaMallocManaged(&stars, NSTARS * sizeof(star<T>)) == cudaSuccess
		&& cudaMallocManaged(&z, NPTS * sizeof(Complex<T>)) == cudaSuccess
		&& cudaMallocManaged(&out, 3 * NPTS * sizeof(Complex<T>)) == cudaSuccess
		&& cudaMallocManaged(&ds, NSTARS * sizeof(ipm_leaf::StarXYM<T>)) == cudaSuccess
		&& cudaMallocManaged(&coef, 2 * (ORDER + 1) * sizeof(T)) == cudaSuccess;
	if (!alloc_ok)
	{
		report(std::string("lens_equation_agreement_") + ptag, false, "cudaMallocManaged failed");
		return;
	}

	new (node) TreeNode<T>(center, hl);
	node->num_neighbors = 0;
	node->expansion_order = ORDER;
	T* cre = coef;
	T* cim = coef + (ORDER + 1);
	for (int i = 0; i <= ORDER; i++)
	{
		cre[i] = T(0.3 * rnd() / (i + 1));
		cim[i] = T(0.3 * rnd() / (i + 1));
		node->local_coeffs[i] = Complex<T>(cre[i], cim[i]);
	}
	for (int i = 0; i < NSTARS; i++)
	{
		stars[i].position = Complex<T>(center.re + T(rnd() * 1.5), center.im + T(rnd() * 1.5));
		stars[i].mass = T(0.2 + 0.8 * (rnd() * 0.5 + 0.5));
		ds[i] = ipm_leaf::StarXYM<T>{stars[i].position.re, stars[i].position.im, stars[i].mass, T(0)};
	}
	node->stars = 0;
	for (int i = 0; i < NPTS; i++)
	{
		z[i] = Complex<T>(center.re + T(rnd() * 1.5), center.im + T(rnd() * 1.5));
	}

	//the corner is outside the node, so that the smooth-matter series converge
	const Complex<T> corner(T(5), T(4));
	double worst_all = 0;
	for (int rect = 0; rect <= 1; rect++)
	for (int approx = 0; approx <= 1; approx++)
	for (int ts : {1, 11, 51})
	for (int ns : {0, NSTARS})
	{
		node->numstars = ns;
		AgreeInput<T> in{T(0.4), T(0.3), T(1.3), T(0.2), corner, ipm_functions::make_smooth_const<T>(corner), rect, approx, ts, NPTS,
			z, stars, node, ds, ns, cre, cim, out, out + NPTS, out + 2 * NPTS};
		agree_kernel<T><<<NPTS / 128, 128>>>(in);
		cudaError_t err = cudaGetLastError();
		if (err == cudaSuccess) err = cudaDeviceSynchronize();
		char name[128];
		std::snprintf(name, sizeof name, "lens_equation_agreement_%s_rect%d_approx%d_taylor%d_stars%d", ptag, rect, approx, ts, ns);
		if (err != cudaSuccess)
		{
			report(name, false, std::string("CUDA error: ") + cudaGetErrorString(err));
			continue;
		}
		double worst_fast = 0, worst_compact = 0;
		for (int i = 0; i < NPTS; i++)
		{
			const Complex<T> a = out[i];
			const double scale = 1 + std::hypot(double(a.re), double(a.im));
			const Complex<T> b = out[NPTS + i];
			const Complex<T> c = out[2 * NPTS + i];
			const double df = std::hypot(double(a.re) - double(b.re), double(a.im) - double(b.im)) / scale;
			const double dc = std::hypot(double(a.re) - double(c.re), double(a.im) - double(c.im)) / scale;
			//written so that a NaN fails
			if (!(df <= worst_fast)) worst_fast = df;
			if (!(dc <= worst_compact)) worst_compact = dc;
		}
		char detail[160];
		std::snprintf(detail, sizeof detail, "worst |diff|/(1+|w|): w_fast %.3g, w_compact %.3g, tolerance %.3g",
			worst_fast, worst_compact, tol);
		if (worst_fast > worst_all) worst_all = worst_fast;
		if (worst_compact > worst_all) worst_all = worst_compact;
		report(name, worst_fast <= tol && worst_compact <= tol, detail);
	}
	std::printf("     lens_equation_agreement_%s: worst over all cases %.3g (tolerance %.3g)\n", ptag, worst_all, tol);
	cudaFree(node); cudaFree(stars); cudaFree(z); cudaFree(out); cudaFree(ds); cudaFree(coef);
}

int main()
{
	test_warp_aggregated_add<float>("float");
	test_warp_aggregated_add<double>("double");
	test_leaf_test_is_strict<float>("float");
	test_leaf_test_is_strict<double>("double");
	test_lens_equation_agreement<float>("float", 1e-4);
	test_lens_equation_agreement<double>("double", 1e-10);
	return n_fail == 0 ? 0 : 1;
}
