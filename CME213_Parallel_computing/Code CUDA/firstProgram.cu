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
    out[threadIdx.x] = f(threadIdx.x);
}

int N = 32;

TEST(CUDA, assign)
{
    int *d_output;

    /* checkCudaErrors:
       A wrapper function we wrote to test whether an error occurred
       when launching a kernel.
       cudaMalloc:
       Allocated memory on device
       */
    checkCudaErrors(cudaMalloc(&d_output, sizeof(int) * N));

    /* This is like a parallel for loop.
       kernel is the function above.
       d_output is the input variable.
       This call will execute the function kernel using N threads.
       Each thread gets a different threadIdx.x value.
       */
    kernel<<<1, N>>>(d_output);

    /* This is just to check that the kernel executed as expected. */
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    vector<int> h_output(N);
    /* This function copies the data back from GPU to CPU.
       See cudaMemcpyDeviceToHost
       You also have
       cudaMemcpyHostToDevice
       */
    checkCudaErrors(cudaMemcpy(&h_output[0], d_output, sizeof(int) * N,
                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i)
    {
        if (i == 0 || i == N - 1 || i % (N / 10) == 0)
            printf("Entry %10d, written by thread %5d\n", h_output[i], i);
        ASSERT_EQ(h_output[i], f(i));
    }

    /* Free memory on the device. */
    checkCudaErrors(cudaFree(d_output));
}

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, argv, "N"))
    {
        N = getCmdLineArgumentInt(argc, argv, "N");
        printf("Using %d threads = %d warps\n", N, (N + 31) / 32);
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}