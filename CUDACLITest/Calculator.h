#pragma once


namespace CUDA {

	public ref class Calculator {

	public:
		/// <summary>
		/// �x�N�g���̘a
		/// </summary>
		/// <param name="vec_0">�x�N�g��0</param>
		/// <param name="vec_1">�x�N�g��1</param>
		/// <returns></returns>
		static array<float>^ Add(array<float>^ vec_0, array<float>^ vec_1);		

	};
}