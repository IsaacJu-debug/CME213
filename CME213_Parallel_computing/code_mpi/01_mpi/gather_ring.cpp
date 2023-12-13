//%%file gather_ring.cpp

#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
using std::vector;

int main(int argc, char **argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Find out the rank and size
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  srand(rank + 2021);  // Initialize the random number generator
  const int number_send = rand() % 100;

  const int rank_receiver = rank == nproc - 1 ? 0 : rank + 1;
  const int rank_sender = rank == 0 ? nproc - 1 : rank - 1;

  // Array needed to store result
  vector<int> numbers(nproc);

  for (int i = 0; i < nproc; ++i) {
    numbers[i] = -1;
  }

  numbers[rank] = number_send;
  printf("Number for process %d: %2d\n", rank, numbers[rank]);

  // Vector to store the status of the non-blocking sends
  vector<MPI_Request> send_req(nproc - 1);
  for (int i = 0; i < nproc - 1; ++i) {
    // Send to the right: Isend
    int i_send = (rank - i + nproc) % nproc;
    assert(i_send >= 0 && i_send < nproc);
    int *p_send = &numbers[i_send];
    printf("MPI_Isend %2d to   %2d: %2d\n", rank, rank_receiver, i);
    MPI_Isend(p_send, 1, MPI_INT, rank_receiver, 0, MPI_COMM_WORLD,
              &send_req[i]);
    // We can proceed; no need to wait now.
    // Receive from the left: Recv
    int i_recv = (rank - i - 1 + nproc) % nproc;
    assert(i_recv >= 0 && i_recv < nproc);
    int *p_recv = &numbers[i_recv];
    printf("MPI_Recv  %2d from %2d: %2d\n", rank, rank_sender, i);
    MPI_Recv(p_recv, 1, MPI_INT, rank_sender, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // We need to wait; we cannot move forward until we have that data.
  }

  if (rank == 0) {
    for (int i = 0; i < nproc; ++i) {
      printf("Number gathered at root node from process %2d: %2d\n", i,
             numbers[i]);
    }
  }

  for (int i = 0; i < nproc - 1; ++i) {
    // Wait for communications to complete
    MPI_Status status;
    MPI_Wait(&send_req[i], &status);
  }

  MPI_Finalize();
}
