#include <stdlib.h>
#include <math.h>
#include <assert.h>

//CUDA includes
#include <cufft.h>
#include <cutil.h>
#include <cublas.h>
#include <cutil_inline.h>

#include "detect_spectrum_kernel.cu"

typedef unsigned int uint;

extern "C"
Complex* loadData(char* real, char* imaginary, uint size, Complex* pData);

extern "C"
Complex* loadFilter(char* filter, uint size, Complex* pData);

extern "C"
void writeData(Complex* pResult, char* output, uint size);

extern "C"
void writeMatrixData(float* pResult, char* output, long Np, long P);

////////////////////////////Signal Processing functions/////////////////////////
Complex* DetectSpectrum(Complex* signal, Complex* window, Complex* filter, int &signal_size, int window_size, int filter_size);
//GPU based function for applying a FIR filter toa  given input signal.
Complex* ComputeFirFilter(const Complex* signal, const Complex* window_coeff, uint signal_size, uint coeff_size, WindowType window);
//GPU based functions for computing a 1D, 2D, or 3D FFT
Complex* Compute1DFFT(Complex* pData, uint size);
//Takes a signal, finds the max magnitude, calculates a threshold, applies the
//threshold to the signal.  Returned signal has 0 for all values below threshold
Complex* ThresholdSignal1D(Complex* signal, int signal_size, float percent_of_max);
//GPU based function to compute a cross correlation between a signal and filter
Complex* ComputeCrossCorrelation1D(Complex* signal, int signal_size, Complex* filter, int filter_size, int & result_size);
//Resamples a given signal to the desired size
Complex* DecimateSignal1D(const Complex* signal, int signal_size, int desired_size);
//helper function for CalculateCrossCorrelation, pads signal and filter data for proper calculation
void PadData(const Complex* signal, Complex** d_signal, int &signal_size, 
			const Complex* filter, Complex** d_filter, int &filter_size);
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
	char* filterFile = NULL;
	char* windowFile = NULL;
	char* outFile = NULL;
	int signal_size, filter_size, window_size;

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
		else if(strcmp(argv[q], "-filter") == 0) {
			assert(q+1 < argc);
			filterFile = argv[++q];
		}
		else if(strcmp(argv[q], "-window") == 0) {
			assert(q+1 < argc);
			windowFile = argv[++q];
		}
		else if(strcmp(argv[q], "-output") == 0) {
			assert(q+1 < argc);
			outFile = argv[++q];
		}
		else if(strcmp(argv[q], "-s") == 0) {
			assert(q+1 < argc);
			signal_size = atoi(argv[++q]);
		}
		else if(strcmp(argv[q], "-f") == 0) {
			assert(q+1 < argc);
			filter_size = atoi(argv[++q]);
		}
		else if(strcmp(argv[q], "-w") == 0) {
			assert(q+1 < argc);
			window_size = atoi(argv[++q]);
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

	Complex* pFilter;
	cudaMallocHost((void**) &pFilter, filter_size * sizeof(Complex));
	loadFilter(filterFile, filter_size, pFilter);

	Complex* pWindow;
	cudaMallocHost((void**) &pWindow, window_size * sizeof(Complex));
	loadFilter(windowFile, window_size, pWindow);

	Complex* gpuResult = NULL;
	Complex* pResult = NULL;

	//perform spectrum detection using the given information
	gpuResult = DetectSpectrum(pSignal, pWindow, pFilter, signal_size, window_size, filter_size);

	if(gpuResult != NULL) {
		cudaMallocHost((void**) &pResult, signal_size * sizeof(Complex));
		cudaMemcpy(pResult, gpuResult, signal_size * sizeof(Complex), cudaMemcpyDeviceToHost);

		cudaFree(gpuResult);

		//write the results out to the desired output file
		writeData(pResult, outFile, signal_size);
	}

	//free up the allocated host memory
	cudaFreeHost(pWindow);
	cudaFreeHost(pSignal);
	cudaFreeHost(pFilter);
	cudaFreeHost(pResult);
	
	//terminate execution
	CUT_EXIT(argc, argv);
}


/************************************************************************************
 * DetectSpectrum()
 *	Performs spectrum sensing using the given input values.  The signal is first
 *	windowed using given FIR window coefficeints.  The signal is then cross
 *	correlated with the given set of filter coefficients.  After that the signal
 *	is transformed into the frequency domain using an FFT.  Finally the signal
 *	is thresholded to remove everything but the peaks remaining in the signal.
 * 
 *	Arguments:
 *		Complex* signal:		pointer to array of signal data
 *		Complex* window:		pointer to array of FIR window coefficients
 *		Complex* filter:		pointer to array of filter coefficients
 *		uint signal_size:		size of the signal array
 *		uint window_size:		size of the window coefficient array
 *		uint filter_size:		size of the filter coefficient array
 *
 *	Return:
 *		Complex*:				pointer to the resulting signal array. Resides in GPU mem.
 *
 ************************************************************************************/
Complex* DetectSpectrum(Complex* signal, Complex* window, Complex* filter, int &signal_size, int window_size, int filter_size) {
	unsigned int timer;
	float totalTime = 0;
	float temp;
	cutCreateTimer(&timer);
	
	cublasStatus status;
	//initialize the cublas library and ensure it was done so succssfully
	status = cublasInit();

	//check if cublas failed to initialize properly
	if(status != CUBLAS_STATUS_SUCCESS)
		return NULL;

	printf("Detecting spectrum availability...\n");

	//allocate space for and transfer the filter data to GPU memory
	Complex* gpuFilter;
	cudaMalloc((void**) &gpuFilter, filter_size * sizeof(Complex));
	cudaMemcpy(gpuFilter, filter, filter_size * sizeof(Complex), cudaMemcpyHostToDevice);
	//allocate space for and transfer the window data to GPU memory
	Complex* gpuWindow;
	cudaMalloc((void**) &gpuWindow, window_size * sizeof(Complex));
	cudaMemcpy(gpuWindow, window, window_size * sizeof(Complex), cudaMemcpyHostToDevice);

	cutStartTimer(timer);

	//allocate space for and transfer the signal data to GPU memory
	Complex* gpuSignal;
	cudaMalloc((void**) &gpuSignal, signal_size * sizeof(Complex));
	cudaMemcpy(gpuSignal, signal, signal_size * sizeof(Complex), cudaMemcpyHostToDevice);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tSignal data transfer to GPU memory:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);

	//allocate sufficient space for the resulting windowed signal
	Complex* wSig;
	cudaMalloc((void**) &wSig, signal_size * sizeof(Complex));
	//multiply the signal and window vectors together
	ComplexVectorMultiplication<<<32, 256>>>(gpuSignal, gpuWindow, wSig, signal_size);
	cudaThreadSynchronize();

	//can now release the orginal signal and the window coefficients
	cudaFree(gpuSignal);
	cudaFree(gpuWindow);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tWindow signal:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);

	//apply the given filter to the windowed signal
	int result_size;
	Complex* fSig = ComputeCrossCorrelation1D(wSig, signal_size, gpuFilter, filter_size, result_size);
	signal_size = result_size;
	cudaFree(wSig);
	cudaFree(gpuFilter);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tCross Correlate Signal and Filter:  %f msecs.\n", temp);

	cutResetTimer(timer);
	cutStartTimer(timer);

	//convert the signal from time domain to frequency domain
	Complex* freqSig = Compute1DFFT(fSig, signal_size);
	cudaFree(fSig);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tCompute %d point FFT on signal:  %f msecs.\n", signal_size, temp);
/*
	cutResetTimer(timer);
	cutStartTimer(timer);

	Complex* gpuResult = ThresholdSignal1D(freqSig, signal_size, .9);
	cudaFree(freqSig);

	cutStopTimer(timer);
	temp = cutGetTimerValue(timer);
	totalTime += temp;

	printf("\tThreshold signal:  %f msecs.\n", temp);
*/
	printf("Total time of execution:  %f msecs.\n", totalTime);

	//properly shutdown the cublas library
	status = cublasShutdown();

	cutDeleteTimer(timer);
	//return the final result
	return freqSig;
	//return gpuResult;
}

/************************************************************************************
 * ComputeFirFilter()
 *	Applies an FIR filter to the given input signal.  The coefficients of
 *	the filter are generated by windowing functions.  The type of window
 *	to use is defined by the user.  It is assumed that the signal data is
 *	already located in GPU memory.  The resulting data will also reside
 *	in the GPU memory.
 * 
 *	Arguments:
 *		Complex* signal:		pointer to array of signal data
 *		uint signal_size:		size of the signal array
 *		uint coeff_size:		size of the coefficent array to generate
 *		WindowType window:		windowing method to use in coefficient generation
 *
 *	Return:
 *		Complex*:				pointer to the resulting data array. Resides in GPU mem.
 *
 ************************************************************************************/
Complex* ComputeFirFilter(const Complex* signal, const Complex* window_coeff, uint signal_size, uint coeff_size, WindowType window) {

	//setup and start a timer
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	
	//allocate memory for the output vector on the GPU
	Complex* gpuOutput;
	cudaMalloc((void**) &gpuOutput, signal_size * sizeof(Complex));
	cudaMemset(gpuOutput, 0, signal_size * sizeof(Complex));

	//compute the FIR filter on the given signal with the desired coefficients
	FIRFilter<<<32, 256>>>(signal, signal_size, window_coeff, coeff_size, gpuOutput);

	cudaThreadSynchronize();

	cutStopTimer(timer);
	printf("ComputeFIRFilter Time: %f msecs.\n", cutGetTimerValue(timer));
	cutDeleteTimer(timer);

	//Transfer the results out of the GPU into main memory and return that pointer
	return gpuOutput;
}

/************************************************************************************
 * Compute1DFFT()
 *	Computes a one dimensional FFT on the given data set. It is assumed that the 
 *	data has already been transfer from main memory into GPU memory.
 * 
 *	Arguments:
 *		Complex* signal:	pointer to the array of data
 *		uint size:			size of the data array
 *
 *	Return:
 *		Complex*:			resulting array of data after the FFT
 ************************************************************************************/
Complex* Compute1DFFT(Complex* pData, uint size) {
	Complex* gpuResult;
	cudaMalloc((void**) &gpuResult, size * sizeof(Complex));

	//create and initialize a 1D FFT plan
	cufftHandle FFTplan;
	cufftPlan1d(&FFTplan, size, CUFFT_C2C, 1);

	//compute the 1D FFT of the given data in place
	cufftExecC2C(FFTplan, pData, gpuResult, CUFFT_FORWARD);

	//free those resource allocated on the GPU
	cufftDestroy(FFTplan);

	return gpuResult;
}

/************************************************************************************
 * ThresholdSignal1D()
 *	Performs a simple thresholding algorithm on the given signal data.  The maximum
 *	magnitude in the signal is first identified.  The threshold value is then determined
 *	by multiplying max by percent_of_max.  All signal values below the threshold value
 *	are set to zero and all those greater than or equal to are kept intact.  It is assumed
 *	that the signal data resides in GPU memory prior to calling this function.
 * 
 *	Arguments:
 *		Complex* signal:		pointer to the array of signal data
 *		uint signal_size:		size of the signal data array
 *		float percent_of_max:	value between 0 and 1 indicating what percentage
 *								below the maximum the threshold will be set at.
 *
 *	Return:
 *		Complex*:				resulting thresholded signal data the resides in main mem
 *
 ************************************************************************************/
Complex* ThresholdSignal1D(Complex* signal, int signal_size, float percent_of_max) {	
	//declare the new array for the thresholded signal
	Complex* thresholded_signal;
	cudaMalloc((void**) &thresholded_signal, signal_size * sizeof(Complex));

	Complex* threshold;
	cudaMalloc((void**) &threshold, sizeof(Complex));

	//find the minimum index of the maximum magnitude using cublas library
	int index = cublasIcamax(signal_size, signal, 1);
	//result is based off of 1-based indexing, so subtract off 1 from returned value
	index = index - 1;

	cudaMemcpy(threshold, signal+index, sizeof(Complex), cudaMemcpyDeviceToDevice);

	Complex* temp;
	cudaMallocHost((void**)&temp, sizeof(Complex));
	cudaMemcpy(temp, threshold, sizeof(Complex), cudaMemcpyDeviceToHost);

	printf("Max magnitude found at index %d\n", index);
	printf("Max magnitude = %f\n", temp->x);

	//perform the thresholding and generate a new signal
	ThresholdSignal1D<<<32, 256>>>(signal, thresholded_signal, signal_size, percent_of_max, threshold);

	cudaThreadSynchronize();

	cudaFree(threshold);

	//return the thresholded signal
	return thresholded_signal;
}

/************************************************************************************
 * ComputeCrossCorrelation1D()
 *	Computes the cross correlation between a given 1D signal and a 1D filter.  The
 *	result of the cross correlation computation is returned that the resulting size
 *	is returned through a pass by reference argument.  These computations are performed
 *	on the GPU.
 * 
 *	Arguments:
 *		Complex* signal:	pointer to the array of signal data
 *		uint signal_size:	size of the signal data array
 *		Complex* filter:	pointer to the array of filter data
 *		uint filter_size:	size of the filter data array
 *		int & result_size:	size of the resulting data array
 *
 *	Return:
 *		Complex*:			resulting data array which resides in main memory
 *
 ************************************************************************************/
Complex* ComputeCrossCorrelation1D(Complex* signal, int signal_size, 
										Complex* filter, int filter_size, int & result_size) {
	//define various local pointers
	Complex* d_signal = NULL;
	Complex* d_filter = NULL;

	//if the signal and filter are not the same size then one must be padded with zeros
	if(signal_size != filter_size) {
		printf("Signal size did not equal filter size\n");
		PadData(signal, &d_signal, signal_size, filter, &d_filter, filter_size);

		printf("signal_size = %d   filter_size = %d\n", signal_size, filter_size);
	}

	//create and initialize a 1D FFT plan
	cufftHandle FFTplan;
	cufftPlan1d(&FFTplan, signal_size, CUFFT_C2C, 1);

	//transform the signal and filter
	cufftExecC2C(FFTplan, d_signal, d_signal, CUFFT_FORWARD);
	cufftExecC2C(FFTplan, d_filter, d_filter, CUFFT_FORWARD);

	//get the complex conjugate of the filter prior to multiplication
	ComplexConjugate<<<32, 256>>>(d_filter, filter_size);
	//ensure we wait for all parallel cuda threads to finish processing before moving on
	cudaThreadSynchronize();

	//multiply the coefficients together and normalize the result
    ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter, signal_size, 1.0f / signal_size);
	//ensure we wait for all parallel cuda threads to finish processing before moving on
	cudaThreadSynchronize();

	//transform the signal back
	cufftExecC2C(FFTplan, d_signal, d_signal, CUFFT_INVERSE);

	//release the generated fft plan object
	cufftDestroy(FFTplan);

	result_size = signal_size;

	//return the resulting cross correlation
	return d_signal;
}

/************************************************************************************
 * PadData()
 *	Helper function for the cross correlation operation.  Pads the signal and filter
 *	data appropriately for proper computation on the GPU.  Data is assumed to reside
 *	in main memory for these operations.
 * 
 *	Arguments:
 *		Complex* signal:			pointer to the array of unpadded signal data
 *		Copmlex** padded_signal:	pointer to the array of padded signal data
 *		uint signal_size:			size of the signal data array
 *		Complex* filter:			pointer to the array of unpadded filter data
 *		Copmlex** padded_filter:	pointer to the array of padded filter data
 *		uint filter_size:			size of the filter data array
 *
 ************************************************************************************/
void PadData(const Complex* signal, Complex** d_signal, int &signal_size, 
			const Complex* filter, Complex** d_filter, int &filter_size) {

	if(signal_size > filter_size)
	{
		printf("Signal size > filter size\n");
		cudaMalloc((void**) d_signal, sizeof(Complex) * signal_size);
		cudaMemcpy(*d_signal, signal, signal_size * sizeof(Complex), cudaMemcpyDeviceToDevice);

		//Allocate space for the padded filter
		cudaMalloc((void**) d_filter, sizeof(Complex) * signal_size);
		//copy the original filter and then add the necessary padding
		cudaMemcpy(*d_filter, filter, filter_size * sizeof(Complex), cudaMemcpyDeviceToDevice);
		cudaMemset(*d_filter + filter_size, 0, (signal_size - filter_size) * sizeof(Complex));
		//set the given filter to the new size and data
		filter_size = signal_size;
	}
	else if(filter_size > signal_size)
	{
		printf("filter_size > signal_size\n");
		cudaMalloc((void**) d_filter, sizeof(Complex) * filter_size);
		cudaMemcpy(*d_filter, filter, filter_size * sizeof(Complex), cudaMemcpyDeviceToDevice);

		//Allocate space for the padded signal
		cudaMalloc((void**) d_signal, sizeof(Complex) * filter_size);
		//copy the original signal and then add the necessary padding
		cudaMemcpy(*d_signal, signal, signal_size * sizeof(Complex), cudaMemcpyDeviceToDevice);
		cudaMemset(*d_signal + signal_size, 0, (filter_size - signal_size) * sizeof(Complex));
		//set the given signal to the new size and data
		signal_size = filter_size;
	}
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
