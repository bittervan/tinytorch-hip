#include <hip/hip_runtime.h>
#include <cstddef>
#include <stdexcept>

template<typename T>
__global__ void add_kernel(const T* __restrict__ x,
                           const T* __restrict__ y,
                           T* __restrict__ out,
                           long n) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i];
}

extern "C" void launch_add_f32(const float* x, const float* y, float* out, long n, hipStream_t stream) {
  const int bs = 256;  // 64 的倍数，友好于 wavefront=64
  const int gs = static_cast<int>((n + bs - 1) / bs);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<float>), dim3(gs), dim3(bs), 0, stream, x, y, out, n);
}

extern "C" void add_arrays_f32_host(const float* hx,
                                    const float* hy,
                                    float* ho,
                                    std::size_t n) {
  const std::size_t bytes = n * sizeof(float);
  float *dx = nullptr, *dy = nullptr, *do_ = nullptr;

  hipError_t e = hipMalloc(&dx, bytes);
  if (e != hipSuccess) throw std::runtime_error("hipMalloc dx failed");

  e = hipMalloc(&dy, bytes);
  if (e != hipSuccess) {
    hipFree(dx);
    throw std::runtime_error("hipMalloc dy failed");
  }

  e = hipMalloc(&do_, bytes);
  if (e != hipSuccess) {
    hipFree(dx);
    hipFree(dy);
    throw std::runtime_error("hipMalloc do_ failed");
  }

  hipStream_t stream = nullptr;  // 默认流；需要异步可传 py 侧 stream 句柄进来

  e = hipMemcpyAsync(dx, hx, bytes, hipMemcpyHostToDevice, stream);
  if (e != hipSuccess) {
    hipFree(dx);
    hipFree(dy);
    hipFree(do_);
    throw std::runtime_error("hipMemcpyAsync dx failed");
  }

  e = hipMemcpyAsync(dy, hy, bytes, hipMemcpyHostToDevice, stream);
  if (e != hipSuccess) {
    hipFree(dx);
    hipFree(dy);
    hipFree(do_);
    throw std::runtime_error("hipMemcpyAsync dy failed");
  }

  launch_add_f32(dx, dy, do_, static_cast<long>(n), stream);

  e = hipMemcpyAsync(ho, do_, bytes, hipMemcpyDeviceToHost, stream);
  if (e != hipSuccess) {
    hipFree(dx);
    hipFree(dy);
    hipFree(do_);
    throw std::runtime_error("hipMemcpyAsync ho failed");
  }

  e = hipStreamSynchronize(stream);
  if (e != hipSuccess) {
    hipFree(dx);
    hipFree(dy);
    hipFree(do_);
    throw std::runtime_error("hipStreamSynchronize failed");
  }

  hipFree(dx);
  hipFree(dy);
  hipFree(do_);
}
