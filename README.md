# two-lh-split-processes

# Rough Notes
Kernel Loader should be able to load two or more lower halves. Once it's done loading all lower halves, it should then load the upper-half and pass the control to the upper half.
These lower halves should be dynamically linked and their linked libraries should have constructors.

We would also want to control where to put each lower half. So, none of the halves collide with each other.

We'll need to build stub libraries for each lower half.
For a simple test program, we'll need a wrapper lib for MPI (MPI_Init, MPI_Comm_size, MPI_Comm_rank) and CUDA (cudaMalloc, cudaFree, cudaMemcpy, and probably constructor APIs). We can probably handle device functions and Cuda constructors after cudaMalloc and cudaFree).

## Lower-half
Currently, kernel-loader is the lower-half that can load the upper half. Now, we'll need to separate the loading part and the actual lower-half.

### Cuda_lh.exe
This should have dlsym and function pointers of CUDA.

### MPI_lh.exe
This should have dlsym and function pointers of MPI.

There can be two designs for the lower-half:
1. Kernel loader forks two child processes (one CUDA and other MPI lower half) and copy-the-bits both of the child into the upper-half. While this approach works with MPI but it may fail with CUDA as cuda's UVM region can not be shared between two processes.
2. Make Kernel loader linked with CUDA libraries then fork a child process (lh_proxy) and bring mpi_proxy to the process address space via copy-the-bits).

## Information transfer between upper and lower half (lhInfo)
Earlier, in CRAC, we were using external file to pass lhInfo to the upper-half. It was fine when we were handling only single process applications. However, for multiple lower-halves, it's better to use an in-memory solution. This time we can use more robust solutions like environment variable.
Env. var can point to the memory region that contains lower-half related information and the upper-half can use this region to put its information (for lower-half to use at restart). This env. var i.e., LHINFO_PTR, needs to be initialized with 0xffffffffffffffff passed to the kernel-loader. So, we can update once the lower-half initialized.
