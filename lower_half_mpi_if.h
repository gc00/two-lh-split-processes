#ifndef LOWER_HALF_MPI_IF_H
#define LOWER_HALF_MPI_IF_H


#define FOREACH_MPI_FNC(MACRO) \
 MACRO(MPI_Init) ,\
 MACRO(MPI_Comm_size) ,\
 MACRO(MPI_Comm_rank) ,

#define GENERATE_MPI_ENUM(ENUM) MPI_Fnc_##ENUM

#define GENERATE_FNC_PTR(FNC) ((void*)&FNC)

typedef enum __MPI_Fncs {
  MPI_Fnc_NULL,
  FOREACH_MPI_FNC(GENERATE_MPI_ENUM)
  MPI_Fnc_Invalid,
} MPI_Fncs_t;

static const char *MPI_Fnc_to_str[]  __attribute__((used)) =
{
  "MPI_Fnc_NULL",
  "MPI_Init",
  "MPI_Comm_size",
  "MPI_Comm_rank",
  "MPI_Fnc_Invalid"
};
#endif // LOWER_HALF_MPI_IF_H
