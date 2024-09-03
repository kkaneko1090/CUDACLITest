#pragma once


namespace CUDA {

	public ref class Calculator {

	public:
		/// <summary>
		/// ベクトルの和
		/// </summary>
		/// <param name="vec_0">ベクトル0</param>
		/// <param name="vec_1">ベクトル1</param>
		/// <returns></returns>
		static array<float>^ Add(array<float>^ vec_0, array<float>^ vec_1);		

	};
}