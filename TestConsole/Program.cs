
using CUDA;
using System.Diagnostics;

//ベクトルの長さ
int vectorLength = 1000;

//2つのベクトルを用意
float[] vector0 = new float[vectorLength];
float[] vector1 = new float[vectorLength];

//ベクトルの中身を設定
for(int i = 0; i < vectorLength; i++)
{
    vector0[i] = i;
    vector1[i] = i * 1000;
}

//GPUでベクトル和を計算
float[] sum = CUDA.Calculator.Add(vector0, vector1);

//コンソール表示
foreach(float value in sum) Console.WriteLine(value); //計算結果