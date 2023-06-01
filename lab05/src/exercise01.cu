#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

#define N 65536
#define THREADS_PER_BLOCK 128
#define PUMP_RATE 2

#define READ_BYTES N*(2*4)  //2 reads of 4 bytes (a and b)
#define WRITE_BYTES N*(4*1) //1 write of 4 bytes (to c)

void checkCUDAError(const char*);
void random_ints(int *a);

/*
Theoretical Bandwith

theoreticalBW = memoryClockRate ∗ memoryBusWidth

GPU Max Clock rate:      1590 MHz (1.59 GHz)  = 1.590.000 kilohertz = 1.59 x 1e9 hertz
Memory Clock rate:       5001 Mhz
Memory Bus Width:        256-bit = 256/8 bytes

theoreticalBW = 5001 / 1e3 x 256 / 8 Giga Bytes per second
*/

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void vectorAdd() {
	int i = blockIdx.x* blockDim.x + threadIdx.x;
	
	if (i < N)
		d_c[i] = d_a[i] + d_b[i];
}


int main(void) {
	int *a, *b, *c;			// host copies of a, b, c
	//int errors;
	unsigned int size = N * sizeof(int);
	int deviceCount = 0;
	double theoretical_BW;

	cudaGetDeviceCount(&deviceCount);
	if (deviceCount > 0)
	{
		cudaSetDevice(0);
		// cudaDeviceProp struct: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		theoretical_BW = deviceProp.memoryClockRate * PUMP_RATE * (deviceProp.memoryBusWidth / 8.0) / 1e6; //convert to GB/s
	}

	double theoreticalBW = 5001 / 1e3 * 256 / 8 * PUMP_RATE;
	assert(theoretical_BW ==  theoreticalBW);
	printf("Theoretical Bandwidth = %f GB/s \n", theoreticalBW);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpyToSymbol(d_a, a, size);
	cudaMemcpyToSymbol(d_b, b, size);
	checkCUDAError("CUDA memcpy");

	// Start recording time
	cudaEventRecord(start, 0);

	// Launch add() kernel on GPU
	vectorAdd <<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>();
	checkCUDAError("CUDA kernel");

	// stop recording and sync to finish all kernel operations
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time for kernel execution and print
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Kernel execution time = %.3f miliseconds \n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Compute measured bandwidth
	int bytesRead = sizeof(d_a) + sizeof(d_b) + sizeof(int); //bytes
	int bytesWritten = sizeof(d_c);
	float measuredBW = ((bytesRead + bytesWritten)/1e9)/(elapsedTime/1e3);
	printf("Measured bandwidth = %.2f GB/s \n", measuredBW);
	

	// Copy result back to host
	cudaMemcpyFromSymbol(c, d_c, size);
	checkCUDAError("CUDA memcpy");

	// Cleanup
	free(a); free(b); free(c);
	//cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}
