#pragma once
#include "Kernel.cuh"

/// <summary>
/// ベクトル和を計算するカーネル関数
/// </summary>
/// <param name="vec_0">ベクトル0</param>
/// <param name="vec_1">ベクトル1</param>
/// <param name="result">計算結果のベクトル</param>
/// <param name="length">ベクトルの長さ</param>
/// <returns></returns>
__global__ void CudaAddKernel(float* vec_0, float* vec_1, float* result, int* length) {
	//インデックス
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//インデックスが範囲内のとき
	if (index < *length) {
		//ベクトルの要素どうしを足し合わせる
		result[index] = vec_0[index] + vec_1[index];
	}
}