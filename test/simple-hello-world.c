#include <stdio.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int myrank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  void * cuda_c = NULL;
  cudaMalloc(&cuda_c, sizeof(int));
  printf("My rank is %d \n", myrank);

  cudaFree(cuda_c);
  MPI_Finalize();
  return EXIT_SUCCESS;
}