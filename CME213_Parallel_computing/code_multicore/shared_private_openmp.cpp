//%%file shared_private.cpp

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "gtest/gtest.h"

const int n_thread = 4;

TEST(sharing_clause, shared)
{
  omp_set_num_threads(n_thread);

  int shared_int = -1;

#pragma omp parallel
  {
    // shared_int is shared
    int tid = omp_get_thread_num();
    printf("Thread ID %2d | shared_int = %d\n", tid, shared_int);
    assert(shared_int == -1);
  }
}

TEST(sharing_clause, private)
{
  int is_private = -2;

#pragma omp parallel private(is_private)
  {
    int tid = omp_get_thread_num();
    int rand_tid = rand();
    is_private = rand_tid;
    printf("Thread ID %2d | is_private = %d\n", tid, is_private);
    assert(is_private == rand_tid);
  }

  printf("Main thread | is_private = %d\n", is_private);
  ASSERT_EQ(is_private, -2);
}
