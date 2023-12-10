#include <omp.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

using std::vector;

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

TEST(matrix_prod, simple)
{
  const int size = 512;
  const int n_thread = 2;

  printf("Size of matrix = %d\n", size);
  printf("Number of threads to create = %d\n", n_thread);

  // Setting the number of threads to use.
  // By default, openMP selects the largest possible number of threads given the
  // processor.
  omp_set_num_threads(n_thread);

  // Output matrix C
  vector<float> mat_c(size * size);

  /* Begin */
#pragma omp parallel
  {
    const long tid = omp_get_thread_num();
    const int n_threads = omp_get_num_threads();
    for (int i = tid; i < size; i += n_threads)
      for (int j = 0; j < size; ++j)
      {
        float c_ij = 0;
        for (int k = 0; k < size; ++k)
        {
          c_ij += MatA(i, k) * MatB(k, j);
        }
        mat_c[i * size + j] = c_ij;
      }
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
