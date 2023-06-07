#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

#define N 1024
#define A_WIDTH N
#define A_HEIGHT N
#define B_WIDTH N
#define B_HEIGHT N
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

void checkCUDAError(const char *msg);
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);
int requiredSM(int ThreadsPerBlock);

__constant__ int BLOCK_SIZE;


__global__ void matrixMulCUDASharedMemory()
{    
	// Block index
	int blockCol = blockIdx.x;
	int blockRow = blockIdx.y;
	int threadCol = threadIdx.x;
	int threadRow = threadIdx.y;
	
	extern __shared__ int sm[];  //dynamic shared memory
	int *sm_pointer = sm;
	float *As = (float*)&sm_pointer[0];
	float *Bs = (float*)&As[BLOCK_SIZE*BLOCK_SIZE];

    //Running sum of product of A and B matrices
    float Csub = 0;
 
	//iterate through the number of sub matrices of A and B
	for (int i = 0; i < NUM_SUBS; i++){
		//Indices of A and B matrix required to load the shared block of memory
		int a_col = threadCol + i*BLOCK_SIZE; 
		int a_row = threadRow + blockRow*BLOCK_SIZE;
		int b_col = threadCol + blockCol*BLOCK_SIZE;
		int b_row = threadRow + i*BLOCK_SIZE; 
		        
        //Each thread should load a single element of sub block of matrices A an B into shared memory
		//global indices along blocks 
		int global_i = threadRow*BLOCK_SIZE + threadCol;
		As[global_i] = d_A[a_row][a_col];
		Bs[global_i] = d_B[b_row][b_col];

        // Sync to ensure sub matrix is fully loaded
		__syncthreads();
		        
        // Sum products of A and B sub matrices
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[threadRow*BLOCK_SIZE + k] * Bs[k*BLOCK_SIZE + threadCol];
		}
        
        // Sync to prevent run ahead (blocks loading new SM values before others have completed)
		__syncthreads();    
	}

    //Calculate the indices of sub matrix C
	int c_col = threadCol + blockCol*BLOCK_SIZE; //==b_col;
	int c_row = threadRow + blockRow*BLOCK_SIZE; //==a_row;
    
	// Store the product value of C matrix
	d_C[c_row][c_col] = Csub;
}


int main(int argc, char **argv)

{
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	unsigned int col, row, errors;
	int maxActiveBlocks;
	float msec, occupancy;
	int nDevice;
	int count;
	cudaDeviceProp prop;


	cudaGetDeviceCount(&nDevice);
	for (count = 0; count < nDevice; count++){
		cudaGetDeviceProperties (&prop, count);
		if (count == 0){
			printf("Properties of Device %d \n", count);
			printf("maxBlocksPerMultiProcessor = %d \n", prop.maxBlocksPerMultiProcessor);
			printf("maxThreadsPerMultiProcessor = %d \n", prop.maxThreadsPerMultiProcessor);
			printf("multiProcessorCount = %d \n", prop.multiProcessorCount);
		}
	}

	cudaEvent_t start, stop; 

	if (A_WIDTH != B_HEIGHT){
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	mem_size_A = sizeof(float)* A_WIDTH* A_HEIGHT;
	mem_size_B = sizeof(float)* B_WIDTH* B_HEIGHT;
	mem_size_C = sizeof(float)* C_WIDTH* C_HEIGHT;

	// Initialise A
	for (row = 0; row < A_HEIGHT; row++)
	for (col = 0; col < A_WIDTH; col++)
		h_A[row][col] = (float)rand() / RAND_MAX;
	// Initialise B
	for (row = 0; row < B_HEIGHT; row++)
	for (col = 0; col < B_WIDTH; col++)
		h_B[row][col] = (float)rand() / RAND_MAX;

	// copy host memory to device
	cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
	cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
	checkCUDAError("CUDA memcpy");

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCUDAError("CUDA event creation");

	// Optimizing grid size for maximum occupancy
	int minGridSize; 		 //Returned minimum grid size needed to achieve the best potential occupancy	
	int ThreadsPerBlock;     //Per-block dynamic shared memory usage intended, in bytes
	int blockSize;		
	//int blockSizeLimit = 0;  //The maximum block size func is designed to work with. 0 means no limit.
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &ThreadsPerBlock, matrixMulCUDASharedMemory, requiredSM, 0);
	
	ThreadsPerBlock = (int) pow(4, floor(log(ThreadsPerBlock) / log(4))); //round to nearest square power 2
	blockSize = (int) sqrt(ThreadsPerBlock);
	printf("For optimal occupancy: block size = %d and minimum grid size = %d \n", blockSize, minGridSize);
	
	minGridSize = (int) pow(4, floor(log(minGridSize) / log(4))); //round to nearest square power 2
	//assert( C_WIDTH / blockSize== minGridSize);
	//printf("used grid size %d \n", C_WIDTH / blockSize);
	
	//Copy to constant memory
	cudaMemcpyToSymbol(BLOCK_SIZE, &blockSize, sizeof(int));
	checkCUDAError("CUDA memcpy results");

	// Setup execution parameters
	dim3 threads(blockSize, blockSize); 
	dim3 grid(C_WIDTH / blockSize, C_HEIGHT / blockSize);	
	cudaEventRecord(start);
	
    // Kernel with dynamically allocated Shared Memory
    matrixMulCUDASharedMemory << < grid, threads, requiredSM(blockSize*blockSize) >> >();
   
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	checkCUDAError("CUDA kernel execution and timing");

	cudaEventElapsedTime(&msec, start, stop);
	cudaDeviceSynchronize();
	checkCUDAError("CUDA timing");

	// Compute the ocupancy
	assert(threads.x * threads.y * threads.z == blockSize*blockSize);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, matrixMulCUDASharedMemory, blockSize*blockSize, 0);
	occupancy = maxActiveBlocks * (blockSize*blockSize) / (float)(prop.maxThreadsPerMultiProcessor);

	// Copy result from device to host
	cudaMemcpyFromSymbol(h_C, d_C, mem_size_C);
	checkCUDAError("CUDA memcpy results");

	// Compute reference CPU version
	matrixMulCPU(h_A, h_B, h_C_ref);

	// Check for errors
	errors = matrixMulTest(h_C, h_C_ref);
	if (errors)
		printf("%d total errors\n", errors);
	else
		printf("Test passed successfully\n");

	printf("Kernel time was %f with occupancy %f \n", msec, occupancy);

}

void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH])
{
	int col, row, k;
	for (row = 0; row < C_HEIGHT; row++){
		for (col = 0; col < C_WIDTH; col++){
			C[row][col] = 0;
			for (k = 0; k < A_WIDTH; k++){
				C[row][col] += A[row][k] * B[k][col];
			}
		}
	}

}

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH])
{
	int errors = 0;
	int row, col;
	float epsilon = 1e+3; //accuracy difference between compilers. 
	//epsilon = 1e+3 for -fmad=false, see https://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf

	for (row = 0; row < C_HEIGHT; row++){
		for (col = 0; col < C_WIDTH; col++){
			if (round(C[row][col]*epsilon) != round(Cref[row][col]*epsilon)){
				errors++;
				printf("Device item c[%d][%d] = %f does not match host result %f\n", row, col, C[row][col], Cref[row][col]);
			}
		}
	}
	return errors;
}

int requiredSM(int ThreadsPerBlock){
	//A function to dynamically calculate shared memory usage based on a given block size.
	//dynamically assign shared memory at runtime, can be used when the amount of shared memory is not known at compile time.
	return 2*ThreadsPerBlock*sizeof(float);
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
