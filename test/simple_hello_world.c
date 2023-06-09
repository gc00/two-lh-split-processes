#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int myrank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  printf("My rank is %d \n", myrank);
  int i;
  for(i=0;i<1000;i++) {
    printf("printing %d \n", i);
    sleep(1);
  }

  MPI_Finalize();
  return 0;
}
