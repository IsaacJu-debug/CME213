#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <cstdlib>

#include <omp.h>

#include "gtest/gtest.h"

using namespace std::chrono;
using std::vector;

const int size = 512;
// Output matrix C
vector<float> mat_c(size *size);

/* A(i,j) */
float MatA(int i, int j)
{
    if (i % 2)
    {
        return 1;
    }

    return -1;
}

/* B(i,j) */
float MatB(int i, int j)
{
    if ((i + j) % 2)
    {
        return i;
    }

    return j;
}

TEST(matrix_prod, parallel_for)
{
    const int n_thread = 2;

    printf("Size of matrix = %d\n", size);
    printf("Number of threads to create = %d\n", n_thread);

    // Setting the number of threads to use.
    // By default, openMP selects the largest possible number of threads given the processor.
    omp_set_num_threads(n_thread);

#pragma omp parallel for
    // Same as
    // #pragma omp parallel
    // #pragma omp for
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            float c_ij = 0;
            for (int k = 0; k < size; ++k)
            {
                c_ij += MatA(i, k) * MatB(k, j);
            }
            mat_c[i * size + j] = c_ij;
        }

    /* Check result */
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            float d_ij = 0.;
            for (int k = 0; k < size; ++k)
            {
                d_ij += MatA(i, k) * MatB(k, j);
            }
            ASSERT_EQ(d_ij, mat_c[i * size + j]);
        }
}

void bench(int n_thread)
{
    omp_set_num_threads(n_thread);

    high_resolution_clock::time_point time_begin = high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            float c_ij = 0;
            for (int k = 0; k < size; ++k)
            {
                c_ij += MatA(i, k) * MatB(k, j);
            }
            mat_c[i * size + j] = c_ij;
        }
    high_resolution_clock::time_point time_end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(time_end - time_begin).count();
    printf("Number of threads: %2d; elapsed time [millisec]: %4d\n", n_thread, static_cast<int>(duration));
}

TEST(matrix_prod, nthreads)
{
    bench(1);
    bench(2);
    bench(4);
}