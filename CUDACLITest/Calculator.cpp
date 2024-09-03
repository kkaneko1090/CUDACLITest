#pragma once
#include "Calculator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Kernel.cuh"

namespace CUDA {
	/// <summary>
	/// 同じ長さのベクトルの和
	/// </summary>
	/// <param name="vec_0">ベクトル0</param>
	/// <param name="vec_1">ベクトル1</param>
	/// <returns></returns>
	array<float>^ Calculator::Add(array<float>^ vec_0, array<float>^ vec_1) {
		
#pragma region ホスト変数の用意
		//pin_ptrで配列を固定し、ポインターを取得
		pin_ptr<float> vec_0_pin_ptr = &vec_0[0];
		pin_ptr<float> vec_1_pin_ptr = &vec_1[0];
		//アンマネージド配列のホスト変数を用意
		float* h_vec_0 = vec_0_pin_ptr; //ベクトル0のアンマネージド配列
		float* h_vec_1 = vec_1_pin_ptr; //ベクトル1のアンマネージド配列
		float* h_result = new float[vec_0->Length]; //計算結果のアンマネージド配列
		int h_length = vec_0->Length; //ベクトルの長さ
#pragma endregion

#pragma region デバイス変数の用意
		//ホスト変数に対応した，デバイス変数を用意
		float* d_vec_0; //ベクトル0
		float* d_vec_1; //ベクトル0
		float* d_result; //計算結果
		int* d_length; //ベクトルの長さ
		//デバイスを指定
		cudaError_t cuda_status = cudaSetDevice(0);
		//デバイス変数のメモリ確保
		cuda_status = cudaMalloc(&d_vec_0, h_length * sizeof(float));
		cuda_status = cudaMalloc(&d_vec_1, h_length * sizeof(float));
		cuda_status = cudaMalloc(&d_result, h_length * sizeof(float));
		cuda_status = cudaMalloc(&d_length, sizeof(int));
		//ホスト変数の値をデバイス変数にコピー
		cuda_status = cudaMemcpy(d_vec_0, h_vec_0, h_length * sizeof(float), cudaMemcpyHostToDevice);
		cuda_status = cudaMemcpy(d_vec_1, h_vec_1, h_length * sizeof(float), cudaMemcpyHostToDevice);
		cuda_status = cudaMemcpy(d_length, &h_length, sizeof(int), cudaMemcpyHostToDevice);
#pragma endregion

#pragma region カーネル関数の実行
		//並列計算条件
		int max_thread_num = 256; //最大スレッド数
		dim3 grid(h_length / max_thread_num + 1); //グリッドの次元
		dim3 block(max_thread_num); //ブロックの次元
		//引数
		void* args[] = { &d_vec_0, &d_vec_1, &d_result, &d_length };
		//カーネル実行
		cuda_status = cudaLaunchKernel((const void*)CudaAddKernel, grid, block, args);
		//処理待ち
		cuda_status = cudaDeviceSynchronize();
#pragma endregion

		//計算結果のデバイス変数の値を，ホストにコピー
		cuda_status = cudaMemcpy(h_result, d_result, h_length * sizeof(float), cudaMemcpyDeviceToHost);
		//アンマネージド配列を，マネージド配列に変換
		array<float>^ result_managed = gcnew array<float>(vec_0->Length);
		for (int i = 0; i < result_managed->Length; i++) result_managed[i] = h_result[i];

		//newで確保したメモリの開放
		delete[] h_result;
		//デバイスメモリの開放
		cuda_status = cudaFree(d_vec_0);
		cuda_status = cudaFree(d_vec_1);
		cuda_status = cudaFree(d_result);
		cuda_status = cudaFree(d_length);

		return result_managed;
	}
}