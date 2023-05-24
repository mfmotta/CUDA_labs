#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define H 1024 //16 //1024 //height --> number of rows, x<H 
#define N 2048 //12 //2048 //width  --> number of cols, y<N

#define THREADS_PER_BLOCK 256 //(unsigned int)ceil((double) N * H / 256)
#define BLOCK_HEIGHT 16 // (unsigned int)ceil((double) H / 24)
#define BLOCK_WIDTH (unsigned int)ceil((double) THREADS_PER_BLOCK/ BLOCK_HEIGHT)
#define GRID_WIDTH  (unsigned int)ceil((double)N / BLOCK_WIDTH)
#define GRID_HEIGHT (unsigned int)ceil((double)H / BLOCK_HEIGHT)

/*	
int x = blockIdx.x * blockDim.x + threadIdx.x; //height = number of rows, x < H
int y = blockIdx.y * blockDim.y + threadIdx.y; //width = number of cols, y < N

coordinates x,y correspond to the global coordinates of the thread in the grid, as if the
grid were a matrix of dimensions = (grid_height*block_height, grid_width*block_width) = (H, N)	
*/

/*
In my case, compute capabilities of a single GPU are:
	(040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
	Warp size:                                     32
	Maximum number of threads per multiprocessor:  1024
	Maximum number of threads per block:           1024
	Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
	Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)

When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to 
multiprocessors with available execution capacity. 
The threads of a thread block execute concurrently on one multiprocessor, 
and multiple thread blocks can execute concurrently on one multiprocessor. 
As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads --warps.
The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, 
increasing thread IDs with the first warp containing thread 0.
*/

void checkCUDAError(const char*);
void random_ints(int *a);


__global__ void matrixAdd(int* a, int* b, int *c) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < H && y < N){
		int i = x * N + y; 
		c[i] = a[i] + b[i];
		//if ((x<2 && y<2) || (x>=H-2 && y>=N-2)) //check some results. this type of branching potentially affecst performance
			//printf("(%d, %d) grid-mapped threads maps %d of linear matrix \n", x, y, i);
    }
}

__host__ int matrixAddCPU(int* a, int* b, int* c_ref) {

	int x, y;
	for (x = 0; x < H; x++){
        for (y = 0; y < N; y++){
			int i = x * N + y; 
			c_ref[i] = a[i] + b[i];
		}		    
	}
	return *c_ref;
}

__device__ unsigned int count = 0;

__global__ void validate(int *c, int *c_ref){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < H && y < N){
        int i = x * N + y;
		if (c[i] != c_ref[i]){
			printf("*ERROR: %d != %d.\n",c[i], c_ref[i]);
			atomicAdd(&count, 1);
		}		
	}
}

int main(void) {
	int *a, *b, *c, *c_ref;					// host copies of a, b, c
	int *d_a, *d_b, *d_c, *d_c_ref;			// device copies of a, b, c
	int errors;
	unsigned int size = N * H * sizeof(int);

	// Checking compatibilty with compute capabilities
	if (BLOCK_HEIGHT * BLOCK_WIDTH > 1024)
		fprintf(stderr, "ERROR: block dimensions too large \n");

	if (H > 2147483647 || N > 65535 || THREADS_PER_BLOCK > 1024)
		fprintf(stderr, "CUDA ERROR: %s: .\n", "invalid");

	
	// Alloc space for device copies of d_a, d_b, d_c
	cudaMalloc((void**)&d_a, size); 
	cudaMalloc((void**)&d_b, size); 
    cudaMalloc((void**)&d_c, size); 
    cudaMalloc((void**)&d_c_ref, size); 
	checkCUDAError("CUDA malloc");


	// Alloc space for host copies of a, b, c and setup input values
	//// Allocate memory for the rows
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);
	c_ref = (int*)malloc(size);

	random_ints(a);
	random_ints(b);
	
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");
	
	// Launch CPU function
	matrixAddCPU(a, b, c_ref);

	// Copy c_ref to device
	cudaMemcpy(d_c_ref, c_ref, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU	

	unsigned int block_width = BLOCK_WIDTH;
	unsigned int block_height = BLOCK_HEIGHT;
	unsigned int grid_width = GRID_WIDTH;
	unsigned int grid_height = GRID_HEIGHT;
	//printf("grid_width= %d \n", grid_width);
	//printf("grid_height= %d \n ", grid_height);
	//printf("block_width= %d \n ", block_width);
	//printf("block_height= %d \n ", block_height);
	

	dim3 dimGrid(grid_height, grid_width, 1);
	dim3 dimBlock(block_height, block_width, 1);
	matrixAdd <<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
	checkCUDAError("CUDA kernel");

	// Launch validate() kernel on GPU
	validate <<<dimGrid, dimBlock >>>(d_c, d_c_ref);
	checkCUDAError("CUDA kernel");
	
	//Print counts of mismatch between matrixAddCPU and matrixAdd
  	cudaMemcpyFromSymbol(&errors, count, sizeof(unsigned int));
  	printf("Total thread errors counted: %d \n", errors);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	// Cleanup
	free(a); free(b); free(c); free(c_ref);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_c_ref);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	int j = 0;
	for (unsigned int x = 0; x < H; x++){	
        for (unsigned int y = 0; y < N; y++){
			int i = x * N + y;
		    a[i] = j; //rand() % 100; //the matrix is constructed as a 1D list
			j++;
			//if ((x<2 && y<2) || (x>=H-2 && y>=N-2)) //check some results
			//	printf(" (%d, %d) maps to %d \n", x, y, a[i]);
		}
	}
}