#include <unistd.h>
#include <utils.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

__global__ void initialize(size_t n, float *a)
{
  size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid < n)
    a[xid] = xid % 1024;
}

__global__ void Copy(size_t n, float *odata, float *idata)
{
  size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid < n)
    odata[xid] = idata[xid];
}

__global__ void offsetCopy(size_t n, float *odata, float *idata, int offset)
{
  int wid = threadIdx.x / 32;
  size_t xid = blockIdx.x * 4 * blockDim.x + 4 * 32 * wid + threadIdx.x + offset;
  xid = xid % n;
  odata[xid] = idata[xid];
}

__global__ void stridedCopy(size_t n, float *odata, float *idata, int stride)
{
  size_t xid = stride * (blockIdx.x * blockDim.x + threadIdx.x);
  xid = xid % n;
  odata[xid] = idata[xid];
}

__global__ void randomCopy(size_t n, float *odata, float *idata, int *addr)
{
  size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid < n && addr[xid] < n)
    odata[xid] = idata[addr[xid]];
}

int main(void)
{
  const size_t one_G = 1 << 30;
  const size_t n = one_G;
  const int n_thread = 256;
  float *d_a, *d_b;

  /* Allocate memory */
  checkCudaErrors(cudaMalloc(&d_a, sizeof(float) * n));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(float) * n));

  printf("Number of GB allocated: %lu GB\n", 2 * sizeof(float) * n / one_G);

  size_t n_blocks = n / n_thread;
  assert(n_thread * n_blocks == n);

  printf(
      "Matrix size: %lu; number of threads per block: %d; number of blocks: "
      "%lu\n",
      n, n_thread, n_blocks);

  initialize<<<n_blocks, n_thread>>>(n, d_a);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  GpuTimer timer;

  printf("Units of time: msec; units of bandwidth: GB/sec\n");

  // Benchmarks
  const int n_runs = 4;
  for (int offset = 0; offset < 65; ++offset)
  {
    timer.start();
    for (int num_runs = 0; num_runs < n_runs; ++num_runs)
    {
      offsetCopy<<<n_blocks, n_thread>>>(n, d_b, d_a, offset);
    }
    timer.stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    float elapsed = timer.elapsed() / n_runs;
    printf("offset %3d %10.4f %e\n", offset, elapsed,
           double(n * 8) / (1e-3 * elapsed) / 1e9);
  }

  printf("\n\nStride\n");

  const int stride_max = 128;
  for (int stride = 1; stride <= stride_max; ++stride)
  {
    timer.start();
    for (int num_runs = 0; num_runs < n_runs; ++num_runs)
    {
      stridedCopy<<<n_blocks, n_thread>>>(n, d_b, d_a, stride);
    }
    timer.stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    float elapsed = timer.elapsed() / n_runs;
    printf("stride %3d %10.4f %e\n", stride, elapsed,
           double(n * 8) / (1e-3 * elapsed) / 1e9);
  }

  return 0;
}
