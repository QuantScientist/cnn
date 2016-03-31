#ifndef CNN_GPU_KERNELS_H
#define CNN_GPU_KERNELS_H

#include "cnn/gpu-kernels.h"
#include "cnn/cuda.h"
#include "macros.h"

namespace cnn {
    namespace gpu {
        // adapted from NVIDIA example
        __global__ void ker_l2_norm_reducer(int n, const cnn::real *x0, cnn::real* res, bool sq, bool acc) {
            __shared__ cnn::real buf[256];
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                cnn::real sum = 0;
                for (int pos = i; pos < n; pos += 256) {
                    const cnn::real d = x0[pos];
                    sum += sq ? d * d : d;
                }
                buf[i] = sum;
            }
            for (int stride = 128; stride > 0; stride >>= 1) {
                __syncthreads();
                for (int i = threadIdx.x; i < stride; i += blockDim.x)
                    buf[i] += buf[stride + i];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                if (acc) res[0] += buf[0]; else res[0] = buf[0];
            }
        }

        // A kernel to calculate the dot product between two arrays
        __global__ void ker_dotproduct(int n, const cnn::real* x, const cnn::real* y, cnn::real* z) {
            __shared__ cnn::real buf[256];
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                cnn::real sum = 0;
                for (int pos = i; pos < n; pos += 256)
                    sum += x[pos] * y[pos];
                buf[i] = sum;
            }
            for (int stride = 128; stride > 0; stride >>= 1) {
                __syncthreads();
                for (int i = threadIdx.x; i < stride; i += blockDim.x)
                    buf[i] += buf[stride + i];
            }
            __syncthreads();
            if (threadIdx.x == 0)
                z[0] = buf[0];
        }

        // adapted from NVIDIA example
        __global__ void ker_sqeucdist(int n, const cnn::real *x0, const cnn::real *x1, cnn::real* res) {
            __shared__ cnn::real buf[256];
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                cnn::real sum = 0;
                for (int pos = i; pos < n; pos += 256) {
                    const cnn::real d = x0[pos] - x1[pos];
                    sum += d * d;
                }
                buf[i] = sum;
            }
            for (int stride = 128; stride > 0; stride >>= 1) {
                __syncthreads();
                for (int i = threadIdx.x; i < stride; i += blockDim.x)
                    buf[i] += buf[stride + i];
            }
            __syncthreads();
            if (threadIdx.x == 0) res[0] = buf[0];
        }

        /// compute gradient clipping
        /// c = rho * b + (1-rho)*a
        __global__ void ker_gradient_scaling(int n, const cnn::real *dense_param_grad_norm,
                                             int m, const cnn::real *sparse_param_grad_norm,
                                             cnn::real clip_threshold, int samples,
                                             cnn::real* gscale) 
        {
            __shared__ cnn::real buf[256];
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                cnn::real sum = 0;
                for (int pos = i; pos < n; pos += 256) {
                    sum += dense_param_grad_norm[pos];
                }
                for (int pos = i; pos < m; pos += 256) {
                    sum += sparse_param_grad_norm[pos];
                }
                buf[i] = sum;
            }
            for (int stride = 128; stride > 0; stride >>= 1) {
                __syncthreads();
                for (int i = threadIdx.x; i < stride; i += blockDim.x)
                    buf[i] += buf[stride + i];
            }
            __syncthreads();

            if (threadIdx.x == 0){
                *gscale = 1.0;
                buf[0] = (sizeof(cnn::real) == sizeof(float)) ? sqrtf(buf[0]) : sqrt(buf[0]);
                if (buf[0] > clip_threshold * samples) {
                    *gscale = (clip_threshold * samples) / buf[0];
                }
            }
        }
    } // namespace gpu
} // namespace cnn

#endif
