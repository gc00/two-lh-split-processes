# two-lh-split-processes

# Rough Notes
Kernel Loader should be able to load two or more lower halves. Once it's done loading all lower halve's, it should then load upper-half and pass the control to the upper half.
These lower halves should be dynamically linked and their linked libraries should have constructors.

We would also want to control where to put each lower half. So, none of the half's collide with each other.

We'll need to build to stub libraries for each lower half.
For a simple test program, we'll need wrapper for MPI (MPI_Init, MPI_Comm_size, MPI_Comm_rank) and CUDA (cudaMalloc, cudaFree, cudaMemcpy and probably constructor APIs). We can probably handle device functions and cuda constructors after cudaMalloc and cudaFree).

Currently, kernel-loader is the lower-half that has capability to load the upper half. Now, we'll need to separate the loading part and the actual lower-half.

## Cuda_lh.exe
This should have dlsym and function pointers of CUDA.

## MPI_lh.exe
This should have dlsym and function pointers of MPI.

