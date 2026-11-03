#pragma once

#include <cmath> //for fabs
#include <cstddef> //for size_t


/******************************************************************************
sizing of the step buffer of the perimeter deposit (see IPM<T>::shoot_cells).
the buffer and the local memory of the kernels come out of the same free device
memory: when it launches a kernel that needs local memory, the runtime reserves
that kernel's local memory per thread for every thread the device can keep
resident, the maximum resident threads per SM times the number of SMs
******************************************************************************/
namespace ipm_step_buffers
{

	/******************************************************************************
	bytes of free device memory needed to allocate the step buffer(s) and still
	launch the kernels

	\param step_bytes -- bytes of the step buffer(s)
	\param local_bytes_per_thread -- the larger localSizeBytes of the kernels
	\param max_threads_per_sm -- maximum resident threads per SM
	\param num_sms -- number of SMs
	\param margin -- bytes kept free beyond these

	\return step_bytes + local_bytes_per_thread * max_threads_per_sm * num_sms + margin
	******************************************************************************/
	inline size_t need_bytes(size_t step_bytes, size_t local_bytes_per_thread, int max_threads_per_sm, int num_sms, size_t margin = 64 << 20)
	{
		return step_bytes + local_bytes_per_thread * static_cast<size_t>(max_threads_per_sm) * static_cast<size_t>(num_sms) + margin;
	}

	/******************************************************************************
	whether need bytes fit into free_bytes of free device memory
	******************************************************************************/
	inline bool fits(size_t need, size_t free_bytes)
	{
		return need <= free_bytes;
	}

	/******************************************************************************
	whether the pixels the row sums of the perimeter deposit clamped to 0 are more
	than rounding explains. each clamped pixel is a rounding residue; their sum
	stays below 16 * eps * factor * rows unless a deposit is wrong

	\param n_clamped -- number of clamped pixels
	\param clamped_total -- sum of their values before clamping
	\param factor -- what one triangle deposits in total (the scale of the step values)
	\param rows -- number of rows the row sums ran over
	\param eps -- machine epsilon of the type of the step buffer

	\return n_clamped > 0 && fabs(clamped_total) > 16 * eps * factor * rows
	******************************************************************************/
	inline bool clamp_is_suspicious(unsigned long long n_clamped, double clamped_total, double factor, long long rows, double eps)
	{
		return n_clamped > 0 && std::fabs(clamped_total) > 16 * eps * factor * static_cast<double>(rows);
	}

}
