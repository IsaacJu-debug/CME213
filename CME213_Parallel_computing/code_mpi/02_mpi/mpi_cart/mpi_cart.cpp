#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  /* Get information about the communicator */
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm comm_cart;

  int ndims = 2; // 3x2 2D grid
  int dims[2];
  dims[0] = 3; // rows
  dims[1] = 2; // columns
  assert(nprocs >= dims[0] * dims[1]);
  int periods[2];
  periods[0] = 1;
  periods[1] = 1;
  int reorder = 1;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims,
                  periods, reorder, &comm_cart);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank < dims[0] * dims[1])
  {
    /* Get my rank in the new topology */
    int my2drank;
    MPI_Comm_rank(comm_cart, &my2drank);

    /* Get my coordinates */
    int mycoords[2];
    MPI_Cart_coords(comm_cart, my2drank, 2, mycoords);

    /* Get coordinates of process below me */
    int rank_down, coords[2];
    coords[0] = mycoords[0] + 1; // i coordinate (one row below in matrix)
    coords[1] = mycoords[1];
    MPI_Cart_rank(comm_cart, coords, &rank_down);

    /* Get coordinates of process to my right */
    int rank_right;
    coords[0] = mycoords[0];
    coords[1] = mycoords[1] + 1; // j coordinate (to the right in matrix)
    MPI_Cart_rank(comm_cart, coords, &rank_right);

    printf("Process rank = (WR) %d (GR) %d coords=(%d,%d); rank down %d rank right %d\n",
           world_rank, my2drank, mycoords[0], mycoords[1], rank_down, rank_right);
  }
  else
  {
    printf("World rank = %d not part of the group\n", world_rank);
  }

  MPI_Finalize();
}
