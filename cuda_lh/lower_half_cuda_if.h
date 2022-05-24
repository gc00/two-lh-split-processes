#ifndef LOWER_HALF_CUDA_IF_H
#define LOWER_HALF_CUDA_IF_H


#define FOREACH_CUDA_FNC(MACRO) \
MACRO(cudaMalloc) ,\
MACRO(cudaFree) ,

#define GENERATE_CUDA_ENUM(ENUM) Cuda_Fnc_##ENUM

#define GENERATE_FNC_PTR(FNC) ((void*)&FNC)

typedef enum __Cuda_Fncs {
  Cuda_Fnc_NULL,
  FOREACH_CUDA_FNC(GENERATE_CUDA_ENUM)
  Cuda_Fnc_Invalid,
} Cuda_Fncs_t;

static const char *cuda_Fnc_to_str[]  __attribute__((used)) =
{
  "Cuda_Fnc_NULL",
  "cudaMalloc",
  "cudaFree",
  "Cuda_Fnc_Invalid"
};
#endif // LOWER_HALF_CUDA_IF_H
