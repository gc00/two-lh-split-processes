#include <mpi.h>
#include "switch_context.h"
#include "common.h"

#define REAL_FNC(fnc) \
  ({ fnc##_t fnc##Fnc = (fnc##_t) -1; \
  if (fnc##Fnc == (fnc##_t) -1) { \
    LhDlsymMPI_t dlsymFptr = (LhDlsymMPI_t)lhInfo.lhDlsymMPI; \
    fnc##Fnc = (fnc##_t)dlsymFptr(MPI_Fnc_##fnc); \
  } \
  fnc##Fnc; })


extern "C" int MPI_Comm_size(MPI_Comm comm, int *world_size) __attribute__((weak));
#define MPI_Comm_size(comm, world_size) (MPI_Comm_size ? MPI_Comm_size(comm, world_size) : 0)

extern "C" int MPI_Comm_rank(MPI_Group group, int *world_rank) __attribute__((weak));
#define MPI_Comm_rank(group, world_rank) (MPI_Comm_rank ? MPI_Comm_rank(group, world_rank) : 0)

extern "C" int MPI_Init(int *argc, char ***argv) __attribute__((weak));
#define MPI_Init(argc, argv) (MPI_Init ? MPI_Init(argc, argv) : 0)


#undef MPI_Comm_size
extern "C" int MPI_Comm_size(MPI_Comm comm, int *world_size)
{
  int retval = -1;
  typedef int (*MPI_Comm_size_t)(MPI_Comm comm, int *world_size);
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  retval = REAL_FNC(MPI_Comm_size)(comm, world_size);
  RETURN_TO_UPPER_HALF();
  return retval;
}

#undef MPI_Comm_rank
extern "C" int MPI_Comm_rank(MPI_Group group, int *world_rank)
{
  int retval = -1;
  typedef int (*MPI_Comm_rank_t)(MPI_Group group, int *world_rank);
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  retval = REAL_FNC(MPI_Comm_rank)(group, world_rank);
  RETURN_TO_UPPER_HALF();
  return retval;
}

#undef MPI_Init
extern "C" int MPI_Init(int *argc, char ***argv)
{
  int retval = -1;
  typedef int (*MPI_Init_t)(int *argc, char ***argv);
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  retval = REAL_FNC(MPI_Init)(NULL, NULL);
  RETURN_TO_UPPER_HALF();
  return retval;
}