# Shared Memory and Profiling

These exercises explore the concept of Shared Memory:

- It is an on-chip memory, therefore much faster than local and global memory (> 1 TB/s bandwidth).
- It is allocated on a threa-block level, i.e. all threads in the block have access to the same SM.
- Threads can access data that has been loaded from global memory by other threads in the same thread-block.
- Cache is user-configurable at thread-block level.

- SM can lead to race conditions (e.g. between threads in different warps), so it is important to perform thread synchronization (all threads within a thread block must call *__syncthreads()* at the same point).

# Profiling


For profiling, run:

`ncu --target-processes all exercise01 > profile01.txt`

## Occupancy

Occupancy is the ratio of active warps with respect the maximum number of warps per Streaming Multiprocessor (SM). Each multiprocessor on the device has a set of N registers available for use by CUDA program threads. These registers are a shared resource that are allocated among the thread blocks executing on a multiprocessor.

In exercise01, we manually compute the occupancy using [cudaGetDeviceProperties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0) to obtain the maximum number of threads per SM.

In exercise02, we find the block and grid sizes that optimize occupancy via [cudaOccupancyMaxPotentialBlockSizeVariableSMem](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g77b3bfb154b86e215a5bc01509ce8ea6) and calculate it with [cudaOccupancyMaxActiveBlocksPerMultiprocessor](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c).

We find that the optimal block size is 32, which is compatible with expectations, given that the SM launches threads in groups of 32 (warps) which execute a single instruction, so optimal block sizes should be multiples of 32.


## Scheduler Statistics

- Summary of the activity of the schedulers issuing instructions. 
- Each scheduler maintains a pool of warps that it can issue instructions for. 
- The upper bound of warps in the pool (Theoretical Warps) is limited by the launch configuration. 
- On every cycle each scheduler checks the state of the allocated warps in the pool (Active Warps). 
- Active warps that are not stalled (Eligible Warps) are ready to issue their next instruction. 
- From the set of eligible warps, the scheduler selects a single warp from which to issue one or more instructions (Issued Warp). 
- On cycles with no eligible warps, the issue slot is skipped and no instruction is issued. 
- Having many skipped issue slots indicates poor latency hiding.

Source: [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)


# Optimizations

-[Fundamental Optimizations in CUDA](https://developer.download.nvidia.com/GTC/PDF/1083_Wang.pdf)
