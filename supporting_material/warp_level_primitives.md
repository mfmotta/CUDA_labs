# CUDA Warp-Level Primitives

</br>
</br>


NVIDIA GPUs and the CUDA programming model employ an execution model called SIMT (Single Instruction, Multiple Thread). It's important to distinguish between SIMT and SIMD (Single Instruction, Multiple Data) architectures. In SIMD a single instruction is executed in parallel on many data elements, while in SIMT, a single instruction is executed in parallel by many threads on arbitrary parts of the data. 

NVIDIA GPUs execute warps of 32 parallel threads using SIMT, which enables each thread to access its own registers, to load and store from divergent addresses, and to follow divergent control flow paths.

Warp-level primitives are operations that take advantage of SIMT at warp level. They perform data exchange between registers and don't need to use shared memory --which would require a load, a store and an extra register to hold the address.

### Thread divergence

If threads belonging to a warp are required to exectute different instructions, this will cause branching --threads which take different branches lose concurrency until they reconverge, i.e. branching introduces latency, as some threads will have to wait for the execution of others to finish before they can start. Threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data.

In earlier architectures shuch as Pascal, algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from [[1](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)].


On Volta and later GPU architectues, data exchange primitives can be used in thread-divergent branches without great loss. These architectures introduce predication of PTX instructions (add link) which allow threads to diverge without causing branching. and the scheduler will issue a different warp to hid the latency of the thread-divergent warp.












#
Sources: 

https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
