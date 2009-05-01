//CUDA includes
#include <cufft.h>
#include <cutil.h>
#include <cublas.h>

typedef unsigned int uint;
typedef cufftComplex Complex;
enum WindowType{HAMMING, HANNING, TUKEY};
#define BLOCK_DIM 16
#define BLOCK_SIZE 16

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

//***********************************************************************************
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

//***********************************************************************************
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

//***********************************************************************************
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

static __device__ __host__ inline Complex ComplexConj(Complex a)
{
	Complex b;
	b.x = a.x;
	b.y = -a.y;
	return b;
}

static __device__ __host__ inline Complex ComplexMulConj(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * (-b.y);
	c.y = a.x * (-b.y) + a.y * b.x;

	return c;
}

//***********************************************************************************
__global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);     
}

//***********************************************************************************
__global__ void ComplexVectorMultiplication(const Complex* a, const Complex* b, Complex* c, int size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = threadID; i < size; i += numThreads)
		c[i] = ComplexMul(a[i], b[i]);
}

//***********************************************************************************
__global__ void ComplexVectorMultiplication(Complex* a, const Complex* b, int size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = threadID; i < size; i += numThreads)
		a[i] = ComplexMul(a[i], b[i]);
}

//***********************************************************************************
__global__ void ComplexConjugate(Complex* a, int size)
{
	const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexConj(a[i]);
	}
}

//***********************************************************************************
__global__ void FIRFilter(const Complex* signal, uint signal_size, const Complex* coeff, uint coeff_size, Complex* output) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i = threadID; i < signal_size; i += numThreads)
	{
		//initialize the output to zero
		output[i].x = 0.0f;
		output[i].y = 0.0f;

		for(int k = 0; k < coeff_size; k++) {
			if(i-k > 0) {
			output[i].x = output[i].x + coeff[k].x * signal[i-k].x;
			output[i].y = output[i].y + coeff[k].y * signal[i-k].y;
			}
		}
	}
}

//***********************************************************************************
__global__ void DecimateSignal1D(const Complex* orig_signal, Complex* new_signal, int signal_size, int desired_size) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	float inc = (float)signal_size / (float)desired_size;
	float r = 0.0, portion;
	Complex result, v1, v2;

	if(threadID < desired_size) {
		//find the calculated position in the orignal array for this new value
		r = inc * threadID;
		portion = ceil(r) - r;

		//find the two closest real values
		v1 = orig_signal[(int)floor(r)];
		v2 = orig_signal[(int)ceil(r)];

		if(v1.x == v2.x && v1.y == v2.y) {
			new_signal[threadID] = v1;
		}
		else {
			//interpolate the desired value from the two closest real values
			result.x = v1.x * (1.0f - portion) + v2.x * portion;
			result.y = v1.y * (1.0f - portion) + v2.y * portion;
			//assign this value into the desired place in the array
			new_signal[threadID] = result;
		}
	}
}

//***********************************************************************************
__global__ void ThresholdSignal1D(const Complex* orig_signal, Complex* new_signal, int signal_size, float percent_of_max, Complex* threshold) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	//create a new signal that has been properly thresholded
	for(int i = threadID; i < signal_size; i += numThreads) {
		//initialize everything in the new signal to be 0
		new_signal[i].x = 0.0;
		new_signal[i].y = 0.0;

		//test if the matching value in the old signal is above or equal to the threshold
		if(orig_signal[i].x >= threshold->x && orig_signal[i].y >= threshold->y) {
			new_signal[i].x = orig_signal[i].x;
			new_signal[i].y = orig_signal[i].y;
		}
	}
}
//***********************************************************************************
__global__ void GenerateWindowHamming(Complex* coeff, uint coeff_size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = coeff_size;
	const float PI = 3.14159265358979; 

	//generate the desired number of Hamming coefficients
	for(int i = threadID; i < coeff_size; i += numThreads)
	{
		coeff[i].x = 0.53836 - 0.46164 * __cosf((2.0 * PI * i) /(M-1));
		coeff[i].y = 0.0;
	}
}

//***********************************************************************************
__global__ void GenerateWindowHanning(Complex* coeff, uint coeff_size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = coeff_size;
	const float PI = 3.14159265358979;

	//generate the desired number of Hanning coefficients
	for(int i = threadID; i < coeff_size; i += numThreads)
	{
		coeff[i].x = 0.5 * (1.0 - __cosf((2.0 * PI * i) / (M - 1)));
		coeff[i].y = 0.0;
	}
}

//***********************************************************************************
__global__ void GenerateWindowTukey(Complex* coeff, uint coeff_size, float alpha) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = coeff_size;
	const float PI = 3.14159265358979;

	//generate the desired number of Tukey coefficients
	for(int i = threadID; i < coeff_size; i += numThreads)
	{
		float numerator = i - (1.0 + alpha) * (M - 1.0)/2.0;
		float denominator = (1.0 - alpha) * (M - 1.0)/2.0;

		coeff[i].x = 0.5 * (1.0 + __cosf((numerator / denominator) * PI));
		coeff[i].y = 0.0;
	}
}
