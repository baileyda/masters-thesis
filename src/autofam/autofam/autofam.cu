#include <stdlib.h>
#include <math.h>
#include <assert.h>

//CUDA includes
#include <cufft.h>
#include <cutil.h>
#include <cublas.h>
#include <cutil_inline.h>

#include "autofam_kernel.cu"

typedef unsigned int uint;

extern "C"
Complex* loadData(char* real, char* imaginary, uint size, Complex* pData);

extern "C"
Complex* loadFilter(char* filter, uint size, Complex* pData);

extern "C"
void writeData(Complex* pResult, char* output, uint size);

extern "C"
void writeMatrixData(float* pResult, char* output, int num, long Np, long P);

extern "C"
void writeMatrixDataComplex(Complex* pResult, char* output, int num, long Np, long P);

////////////////////////////Signal Processing functions/////////////////////////
float* ComputeFAM(Complex* signal, Complex* window, long N, long Np, long P, long L, long & x, long & y, float & totalTime);
Complex* ChannelizeSignal(Complex* signal, int signal_size, long Np, long P, long L);
Complex* ComputeProductSequences(Complex* complex_demod, long Np, long P);

//Resamples a given signal to the desired size
Complex* DecimateSignal1D(const Complex* signal, int signal_size, int desired_size);
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv ) 
{
	CUT_DEVICE_INIT(argc, argv);

	//size parameters for the FFT
	char* signalReal = NULL;
	char* signalImag = NULL;
	char* outFile = NULL;
	int signal_size;

	//process the command line arguments to extract necessary parameters
	for(int q = 1; q < argc; q++) {
		if(strcmp(argv[q], "-signalR") == 0) {
			assert(q+1 < argc);
			signalReal = argv[++q];
		}
		else if(strcmp(argv[q], "-signalI") == 0) {
			assert(q+1 < argc);
			signalImag = argv[++q];
		}
		else if(strcmp(argv[q], "-output") == 0) {
			assert(q+1 < argc);
			outFile = argv[++q];
		}
		else if(strcmp(argv[q], "-s") == 0) {
			assert(q+1 < argc);
			signal_size = atoi(argv[++q]);
		}
	}

	/*
	Complex* window;
	cudaMalloc((void**) &window, 1024 * sizeof(Complex));

	GenerateWindowHamming<<<1024/16, 16>>>(window, 1024);
	cudaThreadSynchronize();

	Complex* pWin;
	cudaMallocHost((void**) &pWin, 1024 * sizeof(Complex));

	cudaMemcpy(pWin, window, 1024 * sizeof(Complex), cudaMemcpyDeviceToHost);

	writeData(pWin, "window_hamming", 1024);
	*/

	
	//read in the signal, filter, and window data from the given files
	Complex* pSignal;
	cudaMallocHost((void**) &pSignal, signal_size * sizeof(Complex));
	loadData(signalReal, signalImag, signal_size, pSignal);

	unsigned int timer;
	float totalTime = 0;
	float temp;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	//allocate space for and transfer the signal data to GPU memory
	Complex* gpuSignal;
	CUDA_SAFE_CALL(cudaMalloc((void**) &gpuSignal, signal_size * sizeof(Complex)));
	//copy the signal data from host memory to gpu memory
	cudaMemcpy(gpuSignal, pSignal, signal_size * sizeof(Complex), cudaMemcpyHostToDevice);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tSignal data transfer to GPU memory:  %f msecs.\n", temp);

	/*

	cutResetTimer(timer);
	cutStartTimer(timer);

	//decimate the signal data down to a managable size
	int new_signal_size = signal_size / 8;
	Complex* gpuSignal_d = DecimateSignal1D(gpuSignal, signal_size, new_signal_size);
	//release the original signal data
	cudaFree(gpuSignal);


	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tSignal decimated:  %f msecs.\n", temp);
	*/
	printf("\tTotal time to load and decimate signal data: %f msec.\n\n\n", totalTime);
	cutDeleteTimer(timer);

	long sig_portion = 2048;
	long N = sig_portion;//signal_size;//new_signal_size;
	float fs = 1.0 / 16.0;
	long Np =(long)(N * fs);

	//float df = fs / Np;
	float dalpha = fs / N;

	long L = (long)(Np / 4.0);
	long P = (long)(fs / dalpha / L);

	assert(Np >= P);

	//generate an Np point hamming window that will be applied
	//to each channel pair region prior to applying an Np-point ffts
	Complex* hamming_win;
	CUDA_SAFE_CALL(cudaMalloc((void**)&hamming_win, Np * sizeof(Complex)));
	GenerateWindowHamming<<<Np/P, P>>>(hamming_win, Np);
	CUDA_SAFE_THREAD_SYNC();

	//turn the hamming window vector into a matrix with the
	//window coefficients on the diagonal
	Complex* window;
	CUDA_SAFE_CALL(cudaMalloc((void**) &window, Np*Np*sizeof(Complex)));
	cudaMemset(window, 0, Np*Np*sizeof(Complex));
	//make the window vector into a diagonal matrix
	FAM_MakeWindow<<<Np/16, 16>>>(window, hamming_win, Np);
	CUDA_SAFE_THREAD_SYNC();
	//release the coefficients vector, its no longer needed
	cudaFree(hamming_win);


	float time;
	totalTime = 0;
	int numIter = signal_size / sig_portion;
	for(int q = 0; q < numIter; q++) {

		Complex* tempSig;
		CUDA_SAFE_CALL(cudaMalloc((void**) &tempSig, sig_portion * sizeof(Complex)));
		//copy the signal data from host memory to gpu memory
		cudaMemcpy(tempSig, gpuSignal+(q*sig_portion), sig_portion * sizeof(Complex), cudaMemcpyDeviceToDevice);

		long x,y;
		//Compute the spectral correlation density function of the given signal
		float * gpuResult_f;
		float * pResult_f;
		gpuResult_f = ComputeFAM(tempSig, window, N, Np, P, L, x, y, time);
		totalTime += time;
		
		cudaFree(tempSig);

		if(gpuResult_f != NULL) {
			cudaMallocHost((void**) &pResult_f, x * y * sizeof(float));
			cudaMemcpy(pResult_f, gpuResult_f, x * y * sizeof(float), cudaMemcpyDeviceToHost);
	
			cudaFree(gpuResult_f);

			//writeMatrixData(pResult_f, outFile, q+1, x, y);

			cudaFreeHost(pResult_f);
		}
		printf("\n");
	}

	printf("Total time to process all segments of the signal = %f msecs\n\n", totalTime);

	//release the window and signal data
	cudaFree(window);
	cudaFree(gpuSignal);

/*
	if(gpuResult_f != NULL) {
		cudaMallocHost((void**) &pResult_f, x * y * sizeof(float));
		cudaMemcpy(pResult_f, gpuResult_f, x * y * sizeof(float), cudaMemcpyDeviceToHost);
	
		cudaFree(gpuResult_f);

		//writeMatrixData(pResult_f, outFile, x, y);
	}

	//free up the allocated host memory
	cudaFreeHost(pSignal);
	cudaFreeHost(pResult_f);
*/
	
	//terminate execution
	//CUT_EXIT(argc, argv);
	return 0;
}

/************************************************************************************
 * ComputeFAM()
 *	Performs cyclostationary spectral analysis using the FFT Accumulation Method.
 *
 * Arguments:
 *		Complex* signal:		pointer to array of signal data
 *		Complex* window:		pointer to matrix with Hamming window coefficents on diagonal
 *		long N:					size of the signal array
 *		long Np:				Number of input channels
 *		long P:					# of rows in the channelization matrix
 *		long L:					Offset between points in the same column at consecutive
								rows in the same channelization matrix
 *		long x:					returns the width of the resulting SCD matrix
 *		long y:					returns the height of the resulting SCD matrix
 *
 * Return:
 *		Complex*:				pointer to the resulting SCD matrix. Resides in GPU mem.
 ************************************************************************************/
float* ComputeFAM(Complex* signal, Complex* window, long N, long Np, long P, long L, long & x, long & y, float & totalTime) {
	unsigned int timer;
	totalTime = 0;
	float temp;
	cutCreateTimer(&timer);

	cublasStatus status;
	//initialize the cublas library and ensure it was done so succssfully
	status = cublasInit();

	//check if cublas failed to initialize properly
	if(status != CUBLAS_STATUS_SUCCESS)
		return NULL;

	cutStartTimer(timer);
	//break the signal up into channel pair regions
	Complex* complex_demod = ChannelizeSignal(signal, N, Np, P, L);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tChannelize input signal data:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);

	Complex* complex_demod_w;
	CUDA_SAFE_CALL(cudaMalloc((void**) &complex_demod_w, Np*P*sizeof(Complex)));
	
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Np / threads.x, P / threads.y);

	//multiply the complex demodulates by the window matrix
    matrixMul<<< grid, threads >>>(complex_demod_w, complex_demod, window, Np, Np);
	CUDA_SAFE_THREAD_SYNC();

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tApply Np-point window to each channel:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);
	
	//release the unwindowed channel data
	cudaFree(complex_demod);

	//create an fft plan to batch process all P Np-point FFTs in parallel
	cufftHandle Np_fft_plan;
	cufftSafeCall(cufftPlan1d(&Np_fft_plan, Np, CUFFT_C2C, P));
	cufftSafeCall(cufftExecC2C(Np_fft_plan, complex_demod_w, complex_demod_w, CUFFT_FORWARD));
	cufftSafeCall(cufftDestroy(Np_fft_plan));
	
	
	dim3 block3(16,16);
	dim3 grid3(P / block3.x, Np / block3.y);
	FAM_FFTShift_Vertical<<<grid3,block3>>>(complex_demod_w, Np, P);
	cutilCheckMsg("FAM_FFTShift_Vertical kernel failed");
	

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tApply Np-point fft to each channel:  %f msecs.\n", temp);


	cutResetTimer(timer);
	cutStartTimer(timer);

	//perform the down conversion of the channelized signal data
	dim3 dimBlock(16,16);
	dim3 dimGrid(P / dimBlock.x, Np / dimBlock.y);
	FAM_DownConvert<<<dimGrid, dimBlock>>>(complex_demod_w, Np, P, L);
	CUDA_SAFE_THREAD_SYNC();

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tDown convert each channel:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);
	
	//we now want to transpose the matrix of complex demodulates
	//to make it easier to compute the product sequences
	Complex* complex_demod_T;
	CUDA_SAFE_CALL(cudaMalloc((void**) &complex_demod_T, P * Np * sizeof(Complex)));


	//setup execution parameters for matrix transposition
	unsigned int size_x = Np + (BLOCK_DIM-(Np%BLOCK_DIM));
	unsigned int size_y = P + (BLOCK_DIM-(P%BLOCK_DIM));

	dim3 grid2(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
	dim3 threads2(BLOCK_DIM, BLOCK_DIM, 1);
	//perform the matrix transpose
	transpose<<<grid2,threads2>>>(complex_demod_T, complex_demod_w, Np, P);
	CUDA_SAFE_THREAD_SYNC();

	//release the untransposed data since we do not need it anymore
	cudaFree(complex_demod_w);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tTranspose the channel pair matrix:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);

	//compute the product sequences
	Complex* prod_seq = ComputeProductSequences(complex_demod_T, Np, P);

	//release the complex demodulates since they are no longer needed
	cudaFree(complex_demod_T);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tCompute the product sequences:  %f msecs.\n", temp);

	/*
	Complex* pProd_seq;
	cudaMallocHost((void**) &pProd_seq, Np * Np * P * sizeof(Complex));
	cudaMemcpy(pProd_seq, prod_seq, Np * Np * P * sizeof(Complex), cudaMemcpyDeviceToHost);

	writeMatrixDataComplex(pProd_seq, "prod_seq", P, Np*Np);

	cudaFreeHost(pProd_seq);
	*/

	cutResetTimer(timer);
	cutStartTimer(timer);

	//once we have the product sequences we can perform the second
	//P point fft on the data.
	cufftHandle P_fft_plan;
	cufftSafeCall(cufftPlan1d(&P_fft_plan, P, CUFFT_C2C, Np));
	cufftSafeCall(cufftExecC2C(P_fft_plan, prod_seq, prod_seq, CUFFT_FORWARD));
	cufftSafeCall(cufftDestroy(P_fft_plan));


	
	dim3 block4(16,16);
	dim3 grid4((Np*Np) / block4.x, P / block4.y);
	FAM_FFTShift_Vertical<<<grid4,block4>>>(prod_seq, P, Np*Np);
	cutilCheckMsg("FAM_FFTShift_Vertical kernel failed");
	

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tApply a P-point fft to each channel:  %f msecs.\n", temp);


	cutResetTimer(timer);
	cutStartTimer(timer);

	float* Sx;
	CUDA_SAFE_CALL(cudaMalloc((void**) &Sx, Np * 2 * N * sizeof(float)));
	cudaMemset(Sx, 0, Np*2*N*sizeof(float));

	dim3 threads6(16,16);
	dim3 blocks6((P/2)/threads.x, (Np*Np)/threads.y);
	FAM_ComputeSCDFunction<<<blocks6, threads6>>>(Sx, prod_seq, N, Np, P, L);
	cutilCheckMsg("FAM_ComputeSCDFunction kernel failed");

	cudaFree(prod_seq);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tCalculate SCD function:  %f msecs.\n", temp);

	//properly shutdown the cublas library
	status = cublasShutdown();

	cutDeleteTimer(timer);

	printf("\tTotal time to compute FAM:  %f msecs.\n", totalTime);

	//cutilSafeCall(cudaFree(prod_seq));

	x = Np;
	y = 2*N;

	return Sx;
}


Complex* ChannelizeSignal(Complex* signal, int signal_size, long Np, long P, long L) {
	Complex* padded_signal;
	long padded_signal_size = (P-1)*L+Np;
	//pad the end of the signal with zeros for when we channelize the signal
	CUDA_SAFE_CALL(cudaMalloc((void**) &padded_signal, padded_signal_size * sizeof(Complex)));

	//cudaMemset(padded_signal, 0, padded_signal_size * sizeof(Complex));
	cudaMemcpy(padded_signal, signal, signal_size * sizeof(Complex), cudaMemcpyDeviceToDevice);
	cudaMemset(padded_signal+signal_size, 0, (padded_signal_size - signal_size) * sizeof(Complex));

	//create the channelized signal matrix which will be NpxP
	//each row will represent one complex demodulate of size Np
	//and there will be P complex demodulates that need to be processed.
	Complex* channelized_signal;
	CUDA_SAFE_CALL(cudaMalloc((void**) &channelized_signal, Np * P * sizeof(Complex)));

	//split up the data of the original signal into the desired channels
	for(long q = 0; q < P; q++) {
		cudaMemcpy(channelized_signal + (Np*q), padded_signal + (L*q), Np * sizeof(Complex), cudaMemcpyDeviceToDevice);
	}

	//return the channelized data set
	cudaFree(padded_signal);
	return channelized_signal;
}

Complex* ComputeProductSequences(Complex* complex_demod, long Np, long P) {
	Complex* prod_seq;
	long Np2 = Np*Np;
	CUDA_SAFE_CALL(cudaMalloc((void**) &prod_seq, P * Np2 * sizeof(Complex)));
	cudaMemset(prod_seq, 0, P*Np2*sizeof(Complex));
	
	/*
	//dim3 threads1(16);
	//dim3 blocks1(Np/16);
	dim3 threads1(16, 16);
	dim3 blocks1(Np/16, P/16);
	FAM_ComputeProdSeqDiag<<<blocks1, threads1>>>(prod_seq, complex_demod, Np, P);
	CUDA_SAFE_THREAD_SYNC();
	
	
	long size = (Np*Np - Np)/2;
	//dim3 threads2((uint)ceil((float)size/(float)Np), 16);
	//dim3 blocks2(Np, P/16);
	dim3 threads2((uint)ceil((float)size/(float)(Np*2)));
	dim3 blocks2(Np*2);

	FAM_ComputeProdSeq<<<blocks2, threads2>>>(prod_seq, complex_demod, Np, P);
	CUDA_SAFE_THREAD_SYNC();
	*/

	dim3 threads(Np);
	dim3 blocks(Np);
	FAM_ComputeProdSeq<<<blocks, threads, P * sizeof(Complex)>>>(prod_seq, complex_demod, Np, P);
	CUDA_SAFE_THREAD_SYNC();

	return prod_seq;
}

/************************************************************************************
 * DecimateSignal1D()
 *	Performs signal resampling, or decimation, on a given 1D signal.  The signal data
 *	is assumed to reside in GPU memory prior to the call to this function.  If it does
 *	not then this will likely cause an error to occur.
 * 
 *	Arguments:
 *		Complex* signal:		pointer to the array of signal data
 *		uint signal_size:		size of the signal data array
 *		uint desired_size:		size to resample the signal data to.
 *
 *	Return:
 *		Complex*:				pointer to the resampled data set. Data resides in
 *								in GPU memory.
 *
 ************************************************************************************/
Complex* DecimateSignal1D(const Complex* signal, int signal_size, int desired_size) {	
	//allocate the necessary space for the new decimated signal
	Complex* new_signal;
	cudaMalloc((void**) &new_signal, desired_size * sizeof(Complex));

	dim3 threads(32);
    dim3 grid(desired_size / threads.x);

	//perform the decimation opertion on the gpu
	DecimateSignal1D<<<desired_size/32, 32>>>(signal, new_signal, signal_size, desired_size);
	cudaThreadSynchronize();

	CUT_CHECK_ERROR("ERROR: DecimateSignal1D kernel failed");

	//return the pointer to the new data set
	return new_signal;
}
