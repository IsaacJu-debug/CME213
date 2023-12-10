#include <cstdio>
#include <omp.h>

#include "mpi.h"

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int numprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    printf("host %s, rank %1d/%1d, thread ID %d/%d\n", hostname, rank, numprocs - 1, tid, nthreads - 1);
  }

  MPI_Finalize();
  return 0;
}
