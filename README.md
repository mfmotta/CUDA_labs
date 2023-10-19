# Parallel Computing with GPUs

This repository contains my solutions to the CUDA laboratories of the course [Parallel Computing with GPUs](http://paulrichmond.shef.ac.uk/teaching/COM4521/) and some notes on the course content.

The provided solutions contain up-to-date versions (as of June 2023) of deprecated primitives and functionalities referred to in the labs descriptions, e.g. we use [cudaOccupancyMaxActiveBlocksPerMultiprocessor](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c) instead of the deprecated [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/).

We also use the profiling tool [Nsight Compute Cli](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). Notice we use the light-weight (not-interactive) Nsight Compute Cli and not [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).
