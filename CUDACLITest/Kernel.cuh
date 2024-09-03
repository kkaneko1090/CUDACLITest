#pragma
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// �x�N�g���a���v�Z����J�[�l���֐�
/// </summary>
/// <param name="vec_0">�x�N�g��0</param>
/// <param name="vec_1">�x�N�g��1</param>
/// <param name="result">�v�Z���ʂ̃x�N�g��</param>
/// <param name="length">�x�N�g���̒���</param>
/// <returns></returns>
__global__ void CudaAddKernel(float* vec_0, float* vec_1, float* result, int* length);