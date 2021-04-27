/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <ctime>
#include <sys/time.h>

// hip header file
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define NUM_LOOPS	1000

#define GPU_WARP_SIZE	64
#define ORT_ENFORCE(x)	assert(x)

//#include "layer_norm_impl.h"

using namespace std;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl(value, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_xor(value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_down(value, delta, width);
}

template <typename U, bool simplified>
__device__ void cuWelfordOnlineSum(
    const U curr,
    U& mu,
    U& sigma2,
    U& count) {
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  if (simplified) {
    sigma2 = sigma2 + curr * curr;
  } else {
    U delta2 = curr - lmean;
    sigma2 = sigma2 + delta * delta2;
  }
}

template <typename U, bool simplified>
__device__ void cuChanOnlineSum(
    const U muB,
    const U sigma2B,
    const U countB,
    U& mu,
    U& sigma2,
    U& count) {
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    if (simplified) {
      sigma2 = sigma2 + sigma2B;
    } else {
      sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
    }
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename T, typename U, bool simplified>
__device__ void cuWelfordMuSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    U& mu,
    U& sigma2,
    U* buf) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        cuWelfordOnlineSum<U, simplified>(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U, simplified>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    #pragma unroll
    for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
      U muB = WARP_SHFL_DOWN(mu, stride);
      U countB = WARP_SHFL_DOWN(count, stride);
      U sigma2B = WARP_SHFL_DOWN(sigma2, stride);
      cuChanOnlineSum<U, simplified>(muB, sigma2B, countB, mu, sigma2, count);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper __half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower __half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2 * threadIdx.y];
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U, simplified>(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / U(n2), 0);
    }
  }
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    HIP_DYNAMIC_SHARED( float, s_float)
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    HIP_DYNAMIC_SHARED( double, s_double)
    return s_double;
  }
};
}  // namespace

template <typename U>
__device__ U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template <>
__device__ float rsqrt(float v) {
  return rsqrtf(v);
}
template <>
__device__ double rsqrt(double v) {
  return rsqrt(v);
}

template <typename T, typename U, bool simplified>
__global__ void cuApplyLayerNorm(
    T* __restrict__ output_vals,
    U* __restrict__ mean,
    U* __restrict__ invvar,
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const U epsilon,
    const T* __restrict__ gamma,
    const T* __restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == GPU_WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2<T, U, simplified>(vals, n1, n2, i1, mu, sigma2, buf);
    const T* lvals = vals + i1 * n2;
    T* ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = thrx; i < n2; i += numx) {
      U curr = static_cast<U>(lvals[i]);
      T gamma_i = (gamma != NULL) ? gamma[i]: (T)1;
      T beta_i = (beta != NULL) ? beta[i] : (T) 0;
      if (simplified) {
        ovals[i] = gamma_i * static_cast<T>(c_invvar * curr);
      } else {
        ovals[i] = gamma_i * static_cast<T>(c_invvar * (curr - mu)) + beta_i;
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (mean != nullptr) mean[i1] = mu;
      if (invvar != nullptr) invvar[i1] = c_invvar;
    }
  }
}

template <typename T, typename U, bool simplified>
void HostApplyLayerNorm(
    const hipDeviceProp_t& prop,
    hipStream_t stream,
    T* output,
    U* mean,
    U* invvar,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const T* gamma,
    const T* beta) {
  const int maxGridY = prop.maxGridSize[1];
  const int warp_size = prop.warpSize;
  assert(warp_size == GPU_WARP_SIZE);

  const dim3 threads(warp_size, 4, 1);
  const dim3 blocks(1, std::min<unsigned int>(n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cuApplyLayerNorm<T, U, simplified>), dim3(blocks), dim3(threads), nshared, stream, 
      output,
      mean,
      invvar,
      input,
      n1, n2,
      U(epsilon),
      gamma, beta);
}

int main()
{
    __half *input, *d_input;
    __half *d_output;
    __half *d_gamma;
    __half *d_beta;
    float *d_mean;
    float *d_invvar;
	double epsilon = 0.00000001;
    struct timeval tpstart,tpend;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << "Device name " << devProp.name << endl;

	hipStream_t stream;
	hipStreamCreate(&stream);

    int i;
	int n1 = 128 * 128;
	int n2 = 1024;
	int input_size = n1 * n2;
    input = (__half*)malloc(input_size * sizeof(*input));

    // initialize the input data
	srand(time(NULL));
	#define RAND_N	999
    for (i = 0; i < input_size; i++) {
		input[i] = rand() % (RAND_N + 1) / (float)(RAND_N + 1);
    }

    // allocate the memory on the device side
    hipMalloc((void**)&d_input, input_size * sizeof(*d_input));
    hipMalloc((void**)&d_output, input_size * sizeof(*d_output));
    hipMalloc((void**)&d_gamma, n2 * sizeof(*d_gamma));
    hipMalloc((void**)&d_beta, n2 * sizeof(*d_beta));
    hipMalloc((void**)&d_mean, n1 * sizeof(*d_mean));
    hipMalloc((void**)&d_invvar, n1 * sizeof(*d_invvar));

    // Memory transfer from host to device
    hipMemcpy(d_input, input, input_size * sizeof(*input), hipMemcpyHostToDevice);

    gettimeofday(&tpstart, NULL);
	for (int i = 0; i < NUM_LOOPS; i++) {
		HostApplyLayerNorm<__half, float, false>(devProp, stream,
										d_output, d_mean, d_invvar, d_input, n1, n2, epsilon, d_gamma, d_beta);
	}
	hipDeviceSynchronize();
    gettimeofday(&tpend, NULL);
    long elapsed = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
	cout << "launch kernel done: " << NUM_LOOPS << " loops, total " << elapsed << " us, avg " << elapsed / NUM_LOOPS << " us" << endl;

    // free the resources on device side
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_gamma);
    hipFree(d_beta);
    hipFree(d_mean);
    hipFree(d_invvar);

    // free the resources on host side
    free(input);

    return 0;
}
