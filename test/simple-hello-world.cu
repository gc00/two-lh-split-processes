#include <stdio.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c)
{
	*c = a+b;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int myrank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int * cuda_c = NULL;
  int c, num=1;
  cudaMalloc(&cuda_c, sizeof(int));
  add<<<1,1>>>(myrank, num, cuda_c);
  cudaError_t ret = cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("My rank is %d and my sum is %d\n", myrank, c);

  cudaFree(cuda_c);
  MPI_Finalize();
  return EXIT_SUCCESS;
}