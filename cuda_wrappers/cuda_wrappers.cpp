#include <cuda_runtime.h>
#include "common/switch_context.h"
#include "common/common.h"

#define REAL_FNC(fnc) \
  ({ fnc##_t fnc##Fnc = (fnc##_t) -1; \
  if (fnc##Fnc == (fnc##_t) -1) { \
    LhDlsymCuda_t dlsymFptr = (LhDlsymCuda_t)lhInfo.lhDlsymCuda; \
    fnc##Fnc = (fnc##_t)dlsymFptr(Cuda_Fnc_##fnc); \
  } \
  fnc##Fnc; })

extern "C" cudaError_t cudaMalloc(void ** pointer, size_t size) __attribute__((weak));
#define cudaMalloc(pointer, size) (cudaMalloc ? cudaMalloc(pointer, size) : 0)

extern "C" cudaError_t cudaFree(void * pointer) __attribute__((weak));
#define cudaFree(pointer) (cudaFree ? cudaFree(pointer) : 0)

#undef cudaMalloc
extern "C" cudaError_t cudaMalloc(void ** pointer, size_t size) {
  typedef cudaError_t (*cudaMalloc_t)(void ** pointer, size_t size);
  cudaError_t ret_val = cudaSuccess;
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  ret_val = REAL_FNC(cudaMalloc)(pointer, size);
  RETURN_TO_UPPER_HALF();
  return ret_val;
}


#undef cudaFree
extern "C" cudaError_t cudaFree (void * pointer) {
  typedef cudaError_t (*cudaFree_t)(void * pointer);
  cudaError_t ret_val = cudaSuccess;
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  ret_val = REAL_FNC(cudaFree)(pointer);
  RETURN_TO_UPPER_HALF();
  return ret_val;
}