#ifndef CNN_CUDA_H
#define CNN_CUDA_H
#if HAVE_CUDA

#include <cassert>
#include <utility>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include "cnn/except.h"
#include <curand.h>

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw cnn::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = stmt;                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "CUBLAS failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw cnn::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

namespace cnn {

inline std::pair<int,int> SizeToBlockThreadPair(int n) {
  assert(n);
/*
the following commented out for windows. need to figure out support for both windows and linux
-  int logn;
-  asm("\tbsr %1, %0\n"
-      : "=r"(logn)
-      : "r" (n-1));
 */
  int logn = (int) log(n);
  logn = logn > 9 ? 9 : (logn < 4 ? 4 : logn);
  ++logn;
  int threads = 1 << logn;
  int blocks = (n + threads - 1) >> logn;
  blocks = blocks > 128 ? 128 : blocks;
  return std::make_pair(blocks, threads);
}

void Free_GPU();
void Initialize_GPU(int& argc, char**& argv, unsigned random_seed, int device_id);

#define CHECK_CUDNN(status) if (status != CUDNN_STATUS_SUCCESS) { cuda_exception("status = " + status); }
#define CHECK_CURND(status) if (status != CURAND_STATUS_SUCCESS) { cuda_exception("status = " + status); }

extern cudnnDataType_t cudnnDataType;
extern cublasHandle_t cublas_handle;
extern cudnnHandle_t cudnn_handle;
extern curandGenerator_t curndGeneratorHandle;

} // namespace cnn

#endif
#endif
