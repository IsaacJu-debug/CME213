#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define NPROCS 8

int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if(nprocs != NPROCS) {
    if(rank == 0) {
      printf("The number of processes must be %d. Terminating.\n",NPROCS);
    }

    MPI_Finalize(); // Don't forget to finalize!
    exit(0);
  }

  /* Moving code here to shorten the rest of the code for slide */
  MPI_Group world_group;
  MPI_Comm sub_group_comm;
  MPI_Group sub_group[2];
  int group_rank;

  int sendbuf = rank;
  int recvbuf;

  /* Extract the original group handle */
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  /* Divide tasks into two distinct groups based upon rank */
  int mygroup = 0;
  if(rank >= NPROCS/2) {
    mygroup = 1;
  }

  int ranks1[4]= {0,1,2,3}, ranks2[4]= {4,5,6,7};
  /* These arrays specify the rank to be used
   * to create 2 separate process groups.
   */
  MPI_Group_incl(world_group, NPROCS/2, ranks1, &sub_group[0]);
  MPI_Group_incl(world_group, NPROCS/2, ranks2, &sub_group[1]);

  /* Create new new communicator and then perform collective communications */
  MPI_Comm_create(MPI_COMM_WORLD, sub_group[mygroup], &sub_group_comm);
  // Summing up the value of the rank for all processes in my group
  MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, sub_group_comm);

  MPI_Group_rank(sub_group[mygroup], &group_rank);
  printf("Rank= %d; Group rank= %d; recvbuf= %d\n",rank,group_rank,recvbuf);

  MPI_Finalize();
}
