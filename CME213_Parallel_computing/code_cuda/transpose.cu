#include <cstdio>
#include <cstdlib>
#include <vector>
#include "utils.h"

#include "gtest/gtest.h"

const unsigned warp_size = 32;

class TransposeFixture : public ::testing::Test
{
protected:
  TransposeFixture()
  {
    n = (1 << 15);

    printf("Number of MB to transpose: %ld\n\n", sizeof(int) * n * n / 1024 / 1024);

    h_in.resize(n * n);
    h_out.resize(n * n);

    for (size_t i = 0; i < n * n; ++i)
    {
      h_in[i] = random() % 100;
    }
    checkCudaErrors(cudaMalloc(&d_in, sizeof(int) * n * n));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * n * n));

    checkCudaErrors(cudaMemcpy(d_in, &h_in[0], sizeof(int) * n * n,
                               cudaMemcpyHostToDevice));
  }

  ~TransposeFixture()
  {
    cudaFree(d_in);
    cudaFree(d_out);
  }

  int n;
  int *d_in, *d_out;
  std::vector<int> h_in;
  std::vector<int> h_out;
  GpuTimer timer;
};

__global__ void simpleTranspose(int *array_in, int *array_out, size_t n_rows, size_t n_cols)
{
  const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  size_t col = tid % n_cols;
  size_t row = tid / n_cols;

  if (col < n_cols && row < n_rows)
  {
    array_out[col * n_rows + row] = array_in[row * n_cols + col];
  }
}

__global__ void simpleTranspose2D(int *array_in, int *array_out, size_t n_rows, size_t n_cols)
{
  const size_t col = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t row = threadIdx.y + blockDim.y * blockIdx.y;

  if (col < n_cols && row < n_rows)
  {
    array_out[col * n_rows + row] = array_in[row * n_cols + col];
  }
}

template <int num_warps>
__global__ void fastTranspose(int *array_in, int *array_out, size_t n_rows, size_t n_cols)
{
  const int warp_id = threadIdx.y;
  const int lane = threadIdx.x;

  __shared__ int block[warp_size][warp_size + 1];

  const int bc = blockIdx.x;
  const int br = blockIdx.y;

  // Load 32x32 block into shared memory
  size_t gc = bc * warp_size + lane; // Global column index
  size_t gr;
  for (int i = 0; i < warp_size / num_warps; ++i)
  {
    gr = br * warp_size + i * num_warps + warp_id; // Global row index
    block[i * num_warps + warp_id][lane] = array_in[gr * n_cols + gc];
  }
  __syncthreads();

  // Now we switch to each warp outputting a row, which will read
  // from a column in the shared memory. This way everything remains
  // coalesced.

  gr = br * warp_size + lane;
  for (int i = 0; i < warp_size / num_warps; ++i)
  {
    gc = bc * warp_size + i * num_warps + warp_id;
    array_out[gc * n_rows + gr] = block[lane][i * num_warps + warp_id];
  }
}

void isTranspose(const std::vector<int> &A,
                 const std::vector<int> &B,
                 size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      ASSERT_EQ(A[n * i + j], B[n * j + i]);
    }
  }
}

void print_out(int n_iter, size_t n, float elapsed)
{
  printf("GPU took %g ms\n", elapsed / n_iter);
  printf("Effective bandwidth is %g GB/s\n",
         (2 * sizeof(int) * n * n * n_iter) / (1e9 * 1e-3 * elapsed));
}

#define MEMCOPY_ITERATIONS 10

TEST_F(TransposeFixture, memcopy)
{
  timer.start();
  for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
  {
    checkCudaErrors(cudaMemcpy(d_out, d_in, sizeof(int) * n * n,
                               cudaMemcpyDeviceToDevice));
  }
  timer.stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  printf("Bandwidth bench\n");
  print_out(MEMCOPY_ITERATIONS, n, timer.elapsed());
}

TEST_F(TransposeFixture, simple)
{
  int num_threads, num_blocks;

  num_threads = 256;
  num_blocks = (n * n + num_threads - 1) / num_threads;

  timer.start();
  for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
  {
    simpleTranspose<<<num_blocks, num_threads>>>(d_in, d_out, n, n);
  }
  timer.stop();

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  for (size_t i = 0; i < n * n; ++i)
  {
    h_out[i] = -1;
  }
  checkCudaErrors(cudaMemcpy(&h_out[0], d_out, sizeof(int) * n * n,
                             cudaMemcpyDeviceToHost));

  isTranspose(h_in, h_out, n);

  printf("simpleTranspose\n");
  print_out(MEMCOPY_ITERATIONS, n, timer.elapsed());
}

TEST_F(TransposeFixture, transpose2D)
{
  dim3 block_dim(8, 32);
  dim3 grid_dim(n / 8, n / 32);
  simpleTranspose2D<<<grid_dim, block_dim>>>(d_in, d_out, n, n);

  timer.start();
  for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
  {
    simpleTranspose2D<<<grid_dim, block_dim>>>(d_in, d_out, n, n);
  }
  timer.stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  for (size_t i = 0; i < n * n; ++i)
  {
    h_out[i] = -1;
  }
  checkCudaErrors(cudaMemcpy(&h_out[0], d_out, sizeof(int) * n * n,
                             cudaMemcpyDeviceToHost));

  isTranspose(h_in, h_out, n);

  printf("simpleTranspose2D\n");
  print_out(MEMCOPY_ITERATIONS, n, timer.elapsed());
}

TEST_F(TransposeFixture, fastTranspose)
{
  const int num_warps_per_block = 256 / 32;
  ASSERT_EQ(warp_size % num_warps_per_block, 0);

  dim3 block_dim(warp_size, num_warps_per_block);
  dim3 grid_dim(n / warp_size, n / warp_size);

  timer.start();
  for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
  {
    fastTranspose<num_warps_per_block><<<grid_dim, block_dim>>>(d_in, d_out, n,
                                                                n);
  }
  timer.stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  for (size_t i = 0; i < n * n; ++i)
  {
    h_out[i] = -1;
  }
  checkCudaErrors(cudaMemcpy(&h_out[0], d_out, sizeof(int) * n * n,
                             cudaMemcpyDeviceToHost));

  isTranspose(h_in, h_out, n);

  printf("fastTranspose\n");
  print_out(MEMCOPY_ITERATIONS, n, timer.elapsed());
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}