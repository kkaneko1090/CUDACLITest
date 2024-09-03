#pragma once
#include "Kernel.cuh"

/// <summary>
/// �x�N�g���a���v�Z����J�[�l���֐�
/// </summary>
/// <param name="vec_0">�x�N�g��0</param>
/// <param name="vec_1">�x�N�g��1</param>
/// <param name="result">�v�Z���ʂ̃x�N�g��</param>
/// <param name="length">�x�N�g���̒���</param>
/// <returns></returns>
__global__ void CudaAddKernel(float* vec_0, float* vec_1, float* result, int* length) {
	//�C���f�b�N�X
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//�C���f�b�N�X���͈͓��̂Ƃ�
	if (index < *length) {
		//�x�N�g���̗v�f�ǂ����𑫂����킹��
		result[index] = vec_0[index] + vec_1[index];
	}
}