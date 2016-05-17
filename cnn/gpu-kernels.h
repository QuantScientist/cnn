#ifndef CNN_GPU_KERNELS_H
#define CNN_GPU_KERNELS_H

#include "cnn/cuda.h"
#include "macros.h"

namespace cnn {
    namespace gpu {

template<typename Func>
__global__ void unaryExprKernel(int n, const cnn::real* x, cnn::real* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accUnaryExprKernel(int n, const cnn::real* x, cnn::real* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void binaryExprKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x0[i], x1[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accBinaryExprKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y, Func func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        y[i] += func(x0[i], x1[i]);
        i += gridDim.x * blockDim.x;
    }
}

template<typename Func>
__global__ void accTripletExprKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real *x2, cnn::real* y, Func func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        y[i] += func(x0[i], x1[i], x2[i]);
        i += gridDim.x * blockDim.x;
    }
}

__global__ void ker_gradient_scaling(int n, const cnn::real *dense_param_grad_norm,
    int m, const cnn::real *sparse_param_grad_norm,
    cnn::real clip_threshold, int samples,
    cnn::real* gscale);

template<typename Func>
__global__ void accTripletWithOneGlbVariableExprKernel(int n, const cnn::real* r, const cnn::real* x, const cnn::real* g, cnn::real *v, cnn::real* y, Func func) {
    __shared__ cnn::real sr[1];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        sr[0] = *r;
        __syncthreads();
        y[i] += func(sr[0], x[i], g[i], v[i]);
        i += gridDim.x * blockDim.x;
    }
}

template<typename Func>
__global__ void slowReduceKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y, Func func) {
  cnn::real ty = 0;
  // THIS IS BAD - FIX THIS TO MAKE IT FAST
  for (int i = 0; i < n; ++i)
    ty += func(x0[i], x1[i]);
  y[0] = ty;
}

// adapted from NVIDIA example
__global__ void ker_l2_norm_reducer(int n, const cnn::real *x0, cnn::real* res, bool sq, bool acc);

// A kernel to calculate the dot product between two arrays
__global__ void ker_dotproduct(int n, const cnn::real* x, const cnn::real* y, cnn::real* z);

// adapted from NVIDIA example
__global__ void ker_sqeucdist(int n, const cnn::real *x0, const cnn::real *x1, cnn::real* res);

__global__ void ker_mem_cpy(int n, cnn::real *target, const cnn::real* src);

} // namespace gpu
} // namespace cnn

#endif
