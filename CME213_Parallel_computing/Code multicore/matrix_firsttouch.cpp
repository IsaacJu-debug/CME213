#include <vector>

#include <omp.h>

#include "gtest/gtest.h"

using std::vector;

const int n_thread = 16;

const int size = 32;
// Input matrices A and B
vector<float> mat_a(size *size);
vector<float> mat_b(size *size);
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

TEST(matrix_element_wise, parallel_for)
{
    printf("Size of matrix = %d\n", size);
    printf("Number of threads to create = %d\n", n_thread);

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            mat_a[i * size + j] = MatA(i, j);
            mat_b[i * size + j] = MatB(i, j);
        }

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            mat_c[i * size + j] = mat_a[i * size + j] * mat_b[i * size + j];
        }

    /* Check result */
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            ASSERT_EQ(mat_c[i * size + j], MatA(i, j) * MatB(i, j));
        }
}