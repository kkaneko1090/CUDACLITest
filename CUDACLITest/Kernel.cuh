#pragma
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// ベクトル和を計算するカーネル関数
/// </summary>
/// <param name="vec_0">ベクトル0</param>
/// <param name="vec_1">ベクトル1</param>
/// <param name="result">計算結果のベクトル</param>
/// <param name="length">ベクトルの長さ</param>
/// <returns></returns>
__global__ void CudaAddKernel(float* vec_0, float* vec_1, float* result, int* length);