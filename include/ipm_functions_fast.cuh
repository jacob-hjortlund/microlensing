#pragma once

#include "array_functions.cuh"
#include "complex.cuh"
#include "lens_equations.cuh"
#include "polygon.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util/util.cuh"


namespace ipm_fast
{
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
	- the lens mapping and the pixel allocation are the original ones
	  (microlensing::w and Polygon::allocate_area_among_pixels)

	\tparam T -- precision of the lens mapping and of the pixel arrays
	******************************************************************************/
	template <typename T>
	__global__ void __launch_bounds__(TILE * TILE, min_blocks_per_sm<T>())
	shoot_cells_tiled_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root,
		int rectangular, Complex<T> corner, int approx, int taylor_smooth,
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
				Complex<T> y = microlensing::w<T>(x, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
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
