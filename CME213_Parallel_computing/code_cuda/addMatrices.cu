#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

#include "gtest/gtest.h"
#include "utils.h"

using std::vector;

int n = 1024;
int n_thread = 512;

__global__ void Initialize(int n, int *a, int *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        a[n * i + j] = j;
        b[n * i + j] = i - 2 * j;
    }
}

__global__ void Add(int n, int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n)
    {
        c[n * i + j] = a[n * i + j] + b[n * i + j];
    }
}

TEST(CUDA, add_matrices)
{
    printf("Dimensions of matrix: %5d x %5d\n", n, n);

    int *d_a, *d_b, *d_c;

    /* Allocate memory */
    checkCudaErrors(cudaMalloc(&d_a, sizeof(int) * n * n));
    checkCudaErrors(cudaMalloc(&d_b, sizeof(int) * n * n));
    checkCudaErrors(cudaMalloc(&d_c, sizeof(int) * n * n));

    ASSERT_GT(n_thread, 0) << "The number of threads should be a positive number";
    ASSERT_EQ(n_thread % 32, 0) << "The number of threads should be a multiple of 32";
    ASSERT_LE(n_thread, 1024) << "The number of threads should be smaller than 1024";

    dim3 th_block(32, n_thread / 32);

    ASSERT_LE(th_block.x * th_block.y, 1024);
    ASSERT_EQ(th_block.x, 32);
    ASSERT_GT(th_block.y, 0);

    int blocks_per_grid_x = (n + th_block.x - 1) / th_block.x;
    int blocks_per_grid_y = (n + th_block.y - 1) / th_block.y;
    /* This formula is needed to make sure we process all entries in matrix */
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    printf("Dimension of thread block: %2d x %2d\n", th_block.x, th_block.y);
    printf("Dimension of grid: %3d x %3d\n", num_blocks.x, num_blocks.y);

    /* Run calculation on GPU */
    Initialize<<<num_blocks, th_block>>>(n, d_a, d_b);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    Add<<<num_blocks, th_block>>>(n, d_a, d_b, d_c);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    /* Note that kernels execute asynchronously.
       They will fail without any error message!
       This can be confusing when debugging.
       The output arrays will be left uninitialized with no warning.
       */

    vector<int> h_c(n * n);
    /* Copy the result back */
    checkCudaErrors(cudaMemcpy(&h_c[0], d_c, sizeof(int) * n * n,
                               cudaMemcpyDeviceToHost));

    /* Test result */
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            ASSERT_EQ(h_c[n * i + j], i - j);
        }
    }
}

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, argv, "t"))
    {
        n_thread = getCmdLineArgumentInt(argc, argv, "t");
        printf("Using %d threads = %d warps\n", n_thread, (n_thread + 31) / 32);
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
