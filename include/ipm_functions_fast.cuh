#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "deflection_angles/alpha_smooth.cuh"
#include "polygon.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util/util.cuh"

#include <numbers>


namespace ipm_fast
{
	/******************************************************************************
	constants for the smooth-matter deflection that only depend on the star field
	geometry. computed once on the host instead of once per ray.
	******************************************************************************/
	template <typename T>
	struct SmoothConst
	{
		Complex<T> log_c;      //log(corner)
		Complex<T> log_cbar;   //log(conj(corner))
		Complex<T> log_mc;     //log(-corner)
		Complex<T> log_mcbar;  //log(-conj(corner))
		Complex<T> inv_c;      //1 / corner
		Complex<T> inv_cbar;   //1 / conj(corner)
		T rad2;                //|corner|^2 (circular field)
	};

	template <typename T>
	SmoothConst<T> make_smooth_const(Complex<T> corner)
	{
		SmoothConst<T> sc;
		sc.log_c = corner.log();
		sc.log_cbar = corner.conj().log();
		sc.log_mc = (-corner).log();
		sc.log_mcbar = (-corner).conj().log();
		sc.inv_c = Complex<T>(1) / corner;
		sc.inv_cbar = Complex<T>(1) / corner.conj();
		sc.rad2 = corner.abs() * corner.abs();
		return sc;
	}

	/******************************************************************************
	plain (uncompensated) complex arithmetic for the hot loops. the compensated
	Ksop/Smith versions in complex.cuh cost 3-5x as many instructions, and the
	shooting path only needs ~1e-5 relative accuracy (alpha_error is a tenth of
	a pixel).
	******************************************************************************/
	template <typename T>
	__device__ __forceinline__ Complex<T> cmul(Complex<T> a, Complex<T> b)
	{
		return Complex<T>(fma(a.re, b.re, -a.im * b.im), fma(a.re, b.im, a.im * b.re));
	}
	//a * b + c
	template <typename T>
	__device__ __forceinline__ Complex<T> cfma(Complex<T> a, Complex<T> b, Complex<T> c)
	{
		return Complex<T>(fma(a.re, b.re, fma(-a.im, b.im, c.re)), fma(a.re, b.im, fma(a.im, b.re, c.im)));
	}

	/******************************************************************************
	theta^2 * sum_i m_i (z - z_i) / |z - z_i|^2   (= conj of theta^2 sum m_i/(z-z_i))
	one real reciprocal per star instead of a Smith-formula complex division
	******************************************************************************/
	template <typename T>
	__device__ __forceinline__ void accumulate_stars(Complex<T> z, const star<T>* __restrict__ stars, int begin, int n, T& acc_re, T& acc_im)
	{
		for (int i = 0; i < n; i++)
		{
			star<T> s = stars[begin + i];
			T dre = z.re - s.position.re;
			T dim = z.im - s.position.im;
			T w = s.mass / fma(dre, dre, dim * dim);
			acc_re = fma(dre, w, acc_re);
			acc_im = fma(dim, w, acc_im);
		}
	}

	template <typename T>
	__device__ __forceinline__ Complex<T> alpha_star_fast(Complex<T> z, T theta2, const star<T>* __restrict__ stars, const TreeNode<T>* node)
	{
		T acc_re = 0, acc_im = 0;
		accumulate_stars(z, stars, node->stars, node->numstars, acc_re, acc_im);
		int nn = node->num_neighbors;
		for (int j = 0; j < nn; j++)
		{
			const TreeNode<T>* nb = node->neighbors[j];
			accumulate_stars(z, stars, nb->stars, nb->numstars, acc_re, acc_im);
		}
		return Complex<T>(acc_re * theta2, acc_im * theta2);
	}

	/******************************************************************************
	conj( theta^2 / hl * sum_{i>=1} i a_i ((z - z0)/hl)^(i-1) ), Horner with plain FMAs
	******************************************************************************/
	template <typename T>
	__device__ __forceinline__ Complex<T> alpha_local_fast(Complex<T> z, T theta2, const TreeNode<T>* node)
	{
		T inv_hl = T(1) / node->half_length;
		Complex<T> dz((z.re - node->center.re) * inv_hl, (z.im - node->center.im) * inv_hl);
		Complex<T> acc(0, 0);
		const Complex<T>* c = node->local_coeffs;
		for (int i = node->expansion_order; i >= 1; i--)
		{
			Complex<T> ci = c[i];
			acc = cfma(acc, dz, Complex<T>(ci.re * i, ci.im * i));
		}
		T f = theta2 * inv_hl;
		return Complex<T>(acc.re * f, -acc.im * f);
	}

	/******************************************************************************
	smooth-matter deflection. for the rectangular/approx case the four geometric
	series share the same 1/i coefficients and the ratios z_bar/corner etc. are
	loop invariant; the original recomputed four complex divisions and four
	scalar divisions per term, plus four complex logarithms, for every ray.
	******************************************************************************/
	template <typename T>
	__device__ __forceinline__ Complex<T> alpha_smooth_fast(Complex<T> z, T kappastar, int rectangular, Complex<T> corner,
		int approx, int taylor_smooth, const SmoothConst<T>& sc)
	{
		if (!rectangular)
		{
			if (approx || fma(z.re, z.re, z.im * z.im) <= sc.rad2)
			{
				return Complex<T>(-kappastar * z.re, -kappastar * z.im);
			}
			T w = -kappastar * sc.rad2 / fma(z.re, z.re, z.im * z.im);
			return Complex<T>(w * z.re, w * z.im);
		}
		if (!approx)
		{
			return microlensing::alpha_smooth<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);
		}

		Complex<T> zb(z.re, -z.im);
		Complex<T> r1 = cmul(zb, sc.inv_c);
		Complex<T> r2 = cmul(zb, sc.inv_cbar);
		Complex<T> r3(-r1.re, -r1.im);
		Complex<T> r4(-r2.re, -r2.im);
		Complex<T> s1(0, 0), s2(0, 0), s3(0, 0), s4(0, 0);
		for (int i = taylor_smooth; i >= 1; i--)
		{
			T inv_i = T(1) / i;
			s1.re += inv_i; s2.re += inv_i; s3.re += inv_i; s4.re += inv_i;
			s1 = cmul(s1, r1);
			s2 = cmul(s2, r2);
			s3 = cmul(s3, r3);
			s4 = cmul(s4, r4);
		}

		Complex<T> t1 = cmul(Complex<T>(corner.re - zb.re, corner.im - zb.im), Complex<T>(sc.log_c.re - s1.re, sc.log_c.im - s1.im));
		Complex<T> t2 = cmul(Complex<T>(corner.re - zb.re, -corner.im - zb.im), Complex<T>(sc.log_cbar.re - s2.re, sc.log_cbar.im - s2.im));
		Complex<T> t3 = cmul(Complex<T>(-corner.re - zb.re, -corner.im - zb.im), Complex<T>(sc.log_mc.re - s3.re, sc.log_mc.im - s3.im));
		Complex<T> t4 = cmul(Complex<T>(-corner.re - zb.re, corner.im - zb.im), Complex<T>(sc.log_mcbar.re - s4.re, sc.log_mcbar.im - s4.im));
		Complex<T> a(t1.re - t2.re + t3.re - t4.re, t1.im - t2.im + t3.im - t4.im);
		T k = -kappastar * std::numbers::inv_pi_v<T>;
		//multiply by i*k: (a.re + i a.im)(i k) = -k a.im + i k a.re
		Complex<T> res(-k * a.im, k * a.re);
		res.re -= kappastar * 2 * (corner.re + z.re);
		return res;
	}

	/******************************************************************************
	lens equation, w = z - alpha_macro - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	template <typename T>
	__device__ __forceinline__ Complex<T> w_fast(Complex<T> z, T kappa, T gamma, T theta2, const star<T>* __restrict__ stars, T kappastar,
		const TreeNode<T>* node, int rectangular, Complex<T> corner, int approx, int taylor_smooth, const SmoothConst<T>& sc)
	{
		//alpha_macro = kappa z - gamma conj(z) = ((kappa - gamma) z.re, (kappa + gamma) z.im)
		Complex<T> a_star = alpha_star_fast(z, theta2, stars, node);
		Complex<T> a_local = alpha_local_fast(z, theta2, node);
		Complex<T> a_smooth = alpha_smooth_fast(z, kappastar, rectangular, corner, approx, taylor_smooth, sc);
		return Complex<T>(z.re * (1 - kappa + gamma) - a_star.re - a_local.re - a_smooth.re,
						  z.im * (1 - kappa - gamma) - a_star.im - a_local.im - a_smooth.im);
	}

	template <typename T>
	__device__ __forceinline__ bool inside(Complex<T> z, const TreeNode<T>* node)
	{
		return fabs(z.re - node->center.re) <= node->half_length && fabs(z.im - node->center.im) <= node->half_length;
	}

	constexpr int TILE = 16; //cells per tile side; a block handles one TILE x TILE tile at a time
	constexpr int NCORNER = (TILE + 1) * (TILE + 1);
#ifndef IPM_MIN_BLOCKS_PER_SM
#define IPM_MIN_BLOCKS_PER_SM 0 //tuning knob: 0 = automatic (4 blocks/SM for float, 3 for double), else the value given
#endif
	/******************************************************************************
	resident blocks per SM that ptxas is asked to make room for (sm_80, 256 threads):
	float fits 4 blocks in 64 registers without spilling; double fits 3 in 80
	registers with a few hundred bytes of spills, which beats 2 blocks at 130 registers
	******************************************************************************/
	template <typename T>
	constexpr int min_blocks_per_sm() { return IPM_MIN_BLOCKS_PER_SM > 0 ? IPM_MIN_BLOCKS_PER_SM : (sizeof(T) == 4 ? 4 : 3); }

	/******************************************************************************
	shoot cells from the image plane to the source plane, one TILE x TILE tile of
	cells per block iteration.

	- the (TILE+1)^2 lattice corners of a tile are mapped once into shared memory
	  instead of 4x per corner
	- the leaf of the tile centre is looked up once and reused for every corner
	  that lies inside it
	- plain-FMA complex arithmetic and hoisted smooth-matter series (w_fast); the pixel allocation is the original Polygon one

	\tparam T -- precision of the lens mapping and of the pixel arrays
	******************************************************************************/
	template <typename T>
	__global__ void __launch_bounds__(TILE * TILE, min_blocks_per_sm<T>())
	shoot_cells_tiled_kernel(T kappa, T gamma, T theta, const star<T>* __restrict__ stars, T kappastar, TreeNode<T>* root,
		int rectangular, Complex<T> corner, int approx, int taylor_smooth, SmoothConst<T> sc,
		Complex<T> ray_half_sep, Complex<int> num_ray_threads, Complex<T> center_x, Complex<T> hlx,
		Complex<T> center_y, Complex<T> hly, T* pixmin, T* pixsad, T* pixels, Complex<int> npixels,
		unsigned long long int* percentage, int verbose)
	{
		//plain arrays: a class with a constructor cannot be a __shared__ variable
		__shared__ T s_cre[NCORNER];
		__shared__ T s_cim[NCORNER];
		__shared__ TreeNode<T>* s_node;

		const int tiles_x = (num_ray_threads.re + TILE - 1) / TILE;
		const int tiles_y = (num_ray_threads.im + TILE - 1) / TILE;
		const int num_tiles = tiles_x * tiles_y;
		const int tid = threadIdx.y * TILE + threadIdx.x;

		const T theta2 = theta * theta;

		const Complex<T> origin = center_x - hlx; //lattice corner (0, 0)
		const Complex<T> step = 2 * ray_half_sep;
		const T sx = npixels.re / (2 * hly.re); //pixels per unit length
		const T sy = npixels.im / (2 * hly.im);
		//area of one triangle (half a cell) in pixel units; this is what one triangle deposits in total
		const T image_plane_area = 2 * ray_half_sep.re * ray_half_sep.im * sx * sy;
		const T max_area = 10000 * image_plane_area;
		const bool parities = (pixmin != nullptr && pixsad != nullptr);

		for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
		{
			const int i0 = (tile % tiles_x) * TILE;
			const int j0 = (tile / tiles_x) * TILE;

			/******************************************************************************
			leaf of the tile centre
			******************************************************************************/
			if (tid == 0)
			{
				Complex<T> xc(origin.re + step.re * (i0 + TILE / 2), origin.im + step.im * (j0 + TILE / 2));
				TreeNode<T>* node = treenode::get_nearest_node(xc, root);
				//cache a leaf only: for a centre outside the root box the walk returns the root itself
				s_node = (node->num_children == 0) ? node : nullptr;
			}
			__syncthreads();
			TreeNode<T>* tile_node = s_node;

			/******************************************************************************
			map the lattice corners of this tile
			******************************************************************************/
			for (int k = tid; k < NCORNER; k += TILE * TILE)
			{
				const int a = k % (TILE + 1);
				const int b = k / (TILE + 1);
				if (i0 + a > num_ray_threads.re || j0 + b > num_ray_threads.im) continue;

				Complex<T> x(origin.re + step.re * (i0 + a), origin.im + step.im * (j0 + b));
				TreeNode<T>* node = (tile_node != nullptr && inside(x, tile_node)) ? tile_node : treenode::get_nearest_node(x, root);
				Complex<T> y = w_fast(x, kappa, gamma, theta2, stars, kappastar, node, rectangular, corner, approx, taylor_smooth, sc);
				/******************************************************************************
				pixel coordinates, y axis reversed so the array is in image orientation
				******************************************************************************/
				s_cre[k] = (y.re - center_y.re + hly.re) * sx;
				s_cim[k] = npixels.im - (y.im - center_y.im + hly.im) * sy;
			}
			__syncthreads();

			const int i = i0 + threadIdx.x;
			const int j = j0 + threadIdx.y;
			if (i < num_ray_threads.re && j < num_ray_threads.im)
			{
				const int k00 = threadIdx.y * (TILE + 1) + threadIdx.x;
				const int k01 = k00 + (TILE + 1);
				const Complex<T> y0(s_cre[k01 + 1], s_cim[k01 + 1]); //(i+1, j+1)
				const Complex<T> y1(s_cre[k01], s_cim[k01]);         //(i,   j+1)
				const Complex<T> y2(s_cre[k00], s_cim[k00]);         //(i,   j)
				const Complex<T> y3(s_cre[k00 + 1], s_cim[k00 + 1]); //(i+1, j)

				/******************************************************************************
				same two triangles, area test and parity rule as the original kernel
				******************************************************************************/
				auto deposit = [&](Complex<T> pa, Complex<T> pb, Complex<T> pc)
				{
					Polygon<T> poly;
					poly.points[0] = pa;
					poly.points[1] = pb;
					poly.points[2] = pc;
					poly.numsides = 3;
					T area = poly.area();
					//rejects NaN (ray on top of a star) and cells straddling a critical curve
					if (!(fabs(area) < max_area)) return;
					T* target = parities ? (area < 0 ? pixmin : pixsad) : pixels;
					poly.allocate_area_among_pixels(image_plane_area, target, npixels);
				};
				deposit(y0, y1, y2);
				deposit(y2, y3, y0);
			}

			if (tid == 0)
			{
				unsigned long long int p = atomicAdd(percentage, 1);
				if (p * 100 / num_tiles > (p - 1) * 100 / num_tiles)
				{
					print_progress(verbose, p, num_tiles);
				}
			}
			__syncthreads(); //the shared arrays are reused by the next tile
		}
	}
}
