#pragma once
#include "Calculator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Kernel.cuh"

namespace CUDA {
	/// <summary>
	/// ���������̃x�N�g���̘a
	/// </summary>
	/// <param name="vec_0">�x�N�g��0</param>
	/// <param name="vec_1">�x�N�g��1</param>
	/// <returns></returns>
	array<float>^ Calculator::Add(array<float>^ vec_0, array<float>^ vec_1) {
		
#pragma region �z�X�g�ϐ��̗p��
		//pin_ptr�Ŕz����Œ肵�A�|�C���^�[���擾
		pin_ptr<float> vec_0_pin_ptr = &vec_0[0];
		pin_ptr<float> vec_1_pin_ptr = &vec_1[0];
		//�A���}�l�[�W�h�z��̃z�X�g�ϐ���p��
		float* h_vec_0 = vec_0_pin_ptr; //�x�N�g��0�̃A���}�l�[�W�h�z��
		float* h_vec_1 = vec_1_pin_ptr; //�x�N�g��1�̃A���}�l�[�W�h�z��
		float* h_result = new float[vec_0->Length]; //�v�Z���ʂ̃A���}�l�[�W�h�z��
		int h_length = vec_0->Length; //�x�N�g���̒���
#pragma endregion

#pragma region �f�o�C�X�ϐ��̗p��
		//�z�X�g�ϐ��ɑΉ������C�f�o�C�X�ϐ���p��
		float* d_vec_0; //�x�N�g��0
		float* d_vec_1; //�x�N�g��0
		float* d_result; //�v�Z����
		int* d_length; //�x�N�g���̒���
		//�f�o�C�X���w��
		cudaError_t cuda_status = cudaSetDevice(0);
		//�f�o�C�X�ϐ��̃������m��
		cuda_status = cudaMalloc(&d_vec_0, h_length * sizeof(float));
		cuda_status = cudaMalloc(&d_vec_1, h_length * sizeof(float));
		cuda_status = cudaMalloc(&d_result, h_length * sizeof(float));
		cuda_status = cudaMalloc(&d_length, sizeof(int));
		//�z�X�g�ϐ��̒l���f�o�C�X�ϐ��ɃR�s�[
		cuda_status = cudaMemcpy(d_vec_0, h_vec_0, h_length * sizeof(float), cudaMemcpyHostToDevice);
		cuda_status = cudaMemcpy(d_vec_1, h_vec_1, h_length * sizeof(float), cudaMemcpyHostToDevice);
		cuda_status = cudaMemcpy(d_length, &h_length, sizeof(int), cudaMemcpyHostToDevice);
#pragma endregion

#pragma region �J�[�l���֐��̎��s
		//����v�Z����
		int max_thread_num = 256; //�ő�X���b�h��
		dim3 grid(h_length / max_thread_num + 1); //�O���b�h�̎���
		dim3 block(max_thread_num); //�u���b�N�̎���
		//����
		void* args[] = { &d_vec_0, &d_vec_1, &d_result, &d_length };
		//�J�[�l�����s
		cuda_status = cudaLaunchKernel((const void*)CudaAddKernel, grid, block, args);
		//�����҂�
		cuda_status = cudaDeviceSynchronize();
#pragma endregion

		//�v�Z���ʂ̃f�o�C�X�ϐ��̒l���C�z�X�g�ɃR�s�[
		cuda_status = cudaMemcpy(h_result, d_result, h_length * sizeof(float), cudaMemcpyDeviceToHost);
		//�A���}�l�[�W�h�z����C�}�l�[�W�h�z��ɕϊ�
		array<float>^ result_managed = gcnew array<float>(vec_0->Length);
		for (int i = 0; i < result_managed->Length; i++) result_managed[i] = h_result[i];

		//new�Ŋm�ۂ����������̊J��
		delete[] h_result;
		//�f�o�C�X�������̊J��
		cuda_status = cudaFree(d_vec_0);
		cuda_status = cudaFree(d_vec_1);
		cuda_status = cudaFree(d_result);
		cuda_status = cudaFree(d_length);

		return result_managed;
	}
}