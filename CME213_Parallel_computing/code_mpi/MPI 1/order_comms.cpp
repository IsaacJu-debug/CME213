#include "mpi.h"

int main(int argc, char *argv[])
{
  /* Obtain number of tasks and task ID */
  MPI_Init(&argc, &argv);

  int rank, count;
  float *sendbuf1, *sendbuf2, *recvbuf1, *recvbuf2;
  int tag;
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // example to demonstrate the order of receive operations
  if (rank == 0)
  {
    MPI_Send(sendbuf1, count, MPI_INT, 2, tag, MPI_COMM_WORLD);
    MPI_Send(sendbuf2, count, MPI_INT, 1, tag, MPI_COMM_WORLD);
  }
  else if (rank == 1)
  {
    MPI_Recv(recvbuf1, count, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    MPI_Send(recvbuf1, count, MPI_INT, 2, tag, MPI_COMM_WORLD);
  }
  else if (rank == 2)
  {
    MPI_Recv(recvbuf1, count, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD,
             &status);
    MPI_Recv(recvbuf2, count, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD,
             &status);
  }

  MPI_Finalize();
  return 0;
}