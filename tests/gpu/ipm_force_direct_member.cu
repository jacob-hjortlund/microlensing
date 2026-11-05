/******************************************************************************
Test driver: the IPM executable of a source tree (its src/main_ipm.cu, same
options and outputs) with map_maker.deposit = Deposit::direct, so that every
triangle takes the direct deposit as when the step buffer does not fit. For
source trees whose include/ipm.cuh has IPM<T>::Deposit and no
tests/gpu/ipm_force_direct.cu. Built by tests/gpu/build_rung.sh with
-I <source tree> as ipm_gpu_direct (-D is_float) and ipm_gpu_double_direct
(-D is_double).
******************************************************************************/

#define main ipm_main
#include "src/main_ipm.cu"
#undef main

int main(int argc, char* argv[])
{
	map_maker.deposit = IPM<dtype>::Deposit::direct;
	return ipm_main(argc, argv);
}
