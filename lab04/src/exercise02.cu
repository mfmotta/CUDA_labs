#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__device__ unsigned int count = 0;


__global__ void vectorAdd(int *a, int *b, int *c, int max) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < max){
		c[i] = a[i] + b[i];
	}	
}

__host__ int vectorAddCPU(int *a, int *b, int *c_ref, int max) {

	int i;
	for (i = 0; i < max; i++){
		c_ref[i] = a[i] + b[i];
	}
	return *c_ref;
}

__global__ void validate(int *d_c, int *d_c_ref, int max){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < max && d_c[i] != d_c_ref[i]){
		printf("ERROR: %d != %d.\n", d_c[i], d_c_ref[i]);
		atomicAdd(&count, 1);
	}
}


int main(void) {
	int *a, *b, *c, *c_ref;			        // host copies of a, b, c
	int *d_a, *d_b, *d_c, *d_c_ref;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	//(int *)cudaMalloc(&d_input, size); would work, but type casting unnecessary, 
	//since cudaMalloc returns a pointer of type void *, which can be assigned to any pointer type without an explicit cast.
	cudaMalloc((void **)&d_a, size); 
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_c_ref, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch CPU function
	vectorAddCPU(a, b, c_ref, N);
	// Copy c_ref to device
	cudaMemcpy(d_c_ref, c_ref, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	dim3 blocksPerGrid((unsigned int)ceil(N / (double)THREADS_PER_BLOCK), 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_c, N);
	checkCUDAError("CUDA kernel");

	
	// Launch validate() kernel on GPU
	validate << <blocksPerGrid, threadsPerBlock >> >(d_c, d_c_ref, N);
	checkCUDAError("CUDA kernel");

	// Print counts of mismatch between vectorAddCPU and vectorAdd
	unsigned int total;
  	cudaMemcpyFromSymbol(&total, count, sizeof(unsigned int));
  	printf("Total threads counted: %d \n", total);
	

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
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
