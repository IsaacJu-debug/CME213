#include <omp.h>

#include <cstdio>
#include <cstdlib>

int main(void)
{
// Fork a team of threads giving them their own copies of variables
#pragma omp parallel
  {
    // Obtain thread number
    int tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);

    // Only the main thread does this
    if (tid == 0)
    {
      int nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  } // All threads join master thread and disband

  return 0;
}