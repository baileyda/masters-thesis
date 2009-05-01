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

static __device__ __host__ inline void ComplexMulConjVector(Complex* a, Complex* b, Complex* c, long size)
{
	for(long q = 0; q < size; q++) {
		c[q].x = a[q].x * b[q].x - a[q].y * (-b[q].y);
		c[q].y = a[q].x * (-b[q].y) + a[q].y * b[q].x;
	}
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

//***********************************************************************************
__global__ void FAM_MakeWindow(Complex* window, const Complex* coeff, int Np) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadID < Np) {
		//multiply the given window by the appropriate complex demodulate
		window[threadID * Np + threadID] = coeff[threadID];
	}
}

//***********************************************************************************
__global__ void FAM_DownConvert(Complex* complex_demod, int Np, int P, int L) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float PI = 3.1415926535897932;

	if(i < P && j < Np) {
		float m = (float)i;
		float k = j - Np/2.0;
		Complex temp;
		temp.x = 0;
		temp.y = -(2.0*PI*k*m*L)/Np;

		//compute the complex exponential value that will be multipled
		//against a specific element in the channelized signal matrix
		Complex result;
		result.x = exp(temp.x) * cos(temp.y);
		result.y = exp(temp.x) * sin(temp.y);

		//multiply the exponential value by the specific value in the signal matrix
		complex_demod[i * Np + j] = ComplexMul(complex_demod[i * Np + j], result);
	}
}

//***********************************************************************************

__global__ void FAM_ComputeProdSeq(Complex* prod_seq, Complex* complex_demod, long Np, long P) {
	long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	//long element = blockIdx.y * blockDim.y + threadIdx.y;

	if(threadID < (Np*Np - Np)/2){// && element < P) {
		long count = 0;
		long temp = threadID;
		do{
			count++;
			temp = temp - (Np-count);
		}while(temp >= 0);

		temp += (Np-count);

		long row = count-1;
		long col = temp + count;

		if(row < Np-1 && col < Np) {
			Complex* row1 = complex_demod + row*P;
			Complex* row2 = complex_demod + col*P;

			Complex* result1 = prod_seq + (row*Np+col)*P;
			Complex* result2 = prod_seq + (col*Np+row)*P;

			ComplexMulConjVector(row1, row2, result1, P);
			ComplexMulConjVector(row2, row1, result2, P);

			
			//for(long q = 0; q < P; q++) {
			//	result1[q] = ComplexMulConj(row1[q], row2[q]);
			//	result2[q] = ComplexMulConj(row2[q], row1[q]);
			//}
			

			//result1[element] = ComplexMulConj(row1[element], row2[element]);
			//result2[element] = ComplexMulConj(row2[element], row1[element]);
		}
	}
}


/*
__global__ void FAM_ComputeProdSeq(Complex* prod_seq, Complex* complex_demod, long Np, long P) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	long j = blockIdx.y * blockDim.y + threadIdx.y;
	long k = blockIdx.z * blockDim.z + threadIdx.z;

	if(i < Np && j < Np){// && k < P) {
		//Complex* row1 = complex_demod + i*P;
		//Complex* row2 = complex_demod + j*P;

		//Complex* result = prod_seq + (i*Np+j)*P;

		//result[k] = ComplexMulConj(row1[k], row2[k]);
		for(long k = 0; k < P; k++) {
		prod_seq[(i*Np+j)*P + k] = ComplexMulConj(complex_demod[i*P + k], complex_demod[j*P + k]);
		}
	}
}
*/

//***********************************************************************************
/*
__global__ void FAM_ComputeProdSeqDiag(Complex* prod_seq, Complex* complex_demod, long Np, long P) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < Np) {
		Complex* row = complex_demod + i*P;

		Complex* result = prod_seq + (i*Np+i) * P;

		ComplexMulConjVector(row, row, result, P);
		
		//for(long q = 0; q < P; q++) {			
		//	result[q] = ComplexMulConj(row[q], row[q]);
		//}
		
	}
}
*/


__global__ void FAM_ComputeProdSeqDiag(Complex* prod_seq, Complex* complex_demod, long Np, long P) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < Np && j < P) {
		Complex* row = complex_demod + i*P;

		Complex* result = prod_seq + (i*Np+i) * P;

		//for(long q = 0; q < P; q++) {
			result[j] = ComplexMulConj(row[j], row[j]);
		//}
	}
}


//***********************************************************************************
__global__ void FAM_FFTShift_Horizontal(Complex* complex_demod, long Np, long P) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < P/2 && j < Np) {
		Complex temp = complex_demod[i * Np + j];
		complex_demod[i * Np + j] = complex_demod[(i+P/2) * Np + j];
		complex_demod[(i+P/2) * Np + j] = temp;
	}
}

//***********************************************************************************
__global__ void FAM_FFTShift_Vertical(Complex* complex_demod, long Np, long P) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < P && j < Np/2) {
		Complex temp = complex_demod[i * Np + j];
		complex_demod[i * Np + j] = complex_demod[i * Np + j + Np/2];
		complex_demod[i * Np + j + Np/2] = temp;
	}
}

//***********************************************************************************
__global__ void FAM_ComputeSCDFunction(float* Sx, Complex* prod_seq, long N, long Np, long P, long L) {
	long k1 = blockIdx.x * blockDim.x + threadIdx.x;
	long k2 = blockIdx.y * blockDim.y + threadIdx.y;

	if(k1 < P/2 && k2 < (Np*Np)) {
		float l,k,p,alpha,f;
		long kk, ll;
		if(k2 % Np == 0)
			l = Np/2-1;
		else
			l = (k2%Np) - Np/2.0 - 1.0;

		k = ceil((float)k2/(float)Np) - Np/2.0 - 1.0;
		p = k1 - P/4 - 1.0;
		alpha = (k-l)/Np + (p-1)/L/P;
		f = (k+l) / 2.0 / Np;

		if(alpha >= -1 && alpha <= 1 && f >= -.5 && f <= .5) {
			kk = ceil(1 + Np * (f + .5));
			ll = 1 + N * (alpha + 1);
			
			Complex temp = prod_seq[k2*P + (k1+P/4)];

			Sx[ll*Np + kk] = sqrt(temp.x * temp.x + temp.y * temp.y);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void matrixMul(Complex* C, Complex* A, Complex* B, long wA, long wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    long aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    long aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    long aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    long bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    long bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    Complex Csub;
	Csub.x = 0;  Csub.y = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ Complex As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ Complex Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub = ComplexAdd(Csub, ComplexMul(AS(ty,k), BS(k,tx)));
            //Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void transpose(Complex *odata, Complex *idata, int width, int height)
{
	__shared__ Complex block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}