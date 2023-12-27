#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include "gtest/gtest.h"
#include "utils.h"

using std::vector;

__device__ __host__ int f(int i)
{
    return i * i;
}

__global__ void kernel(int *out)
{
    // blockIdx.x is the block index within a grid
    // blockDim.x is the number of threads in a block
    // threadIdx.x is the thread index within a block
    // gridDim.x is the number of blocks in a grid
    out[threadIdx.x] = f(threadIdx.x); // threadIdx.x is the thread index within a block
}

int N = 32;

TEST(CUDA, assign)
{
    int *d_output;
    checkCudaErrors(cudaMalloc(&d_output, N * sizeof(int)));

    kernel<<<1, N>>>(d_output); // 1 block, N threads. N / 32 warps 

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    vector<int> h_output(N);
    checkCudaErrors(cudaMemcpy(&h_output[0], d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // print can only happen on the host
    for (int i = 0; i < N; i++)
    {
        if (i == 0 || i == N - 1 || i % (N / 10) == 0)
            printf("Entry %10d, written by thread %5d\n", h_output[i], i);
        ASSERT_EQ(h_output[i],f(i)); // f is callabled on both host and device
    }

    checkCudaErrors(cudaFree(d_output));
}

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, argv, "N"))
    {
        N = getCmdLineArgumentInt(argc, argv, "N");
        printf("Using %d threads = %d warps\n", N, N / 32);
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

