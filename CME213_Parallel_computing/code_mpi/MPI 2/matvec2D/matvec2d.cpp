#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>
#include <unistd.h>

using std::vector;
using std::max;
using std::abs;

// Matrix A
float AEntry(unsigned i, unsigned j, unsigned n) {
  return i*n + j;
}

// Vector b
float BEntry(unsigned i, unsigned n) {
  return (i+1)*n;
}

int main(int argc, char* argv[]) {

  const unsigned n = 64; // size of matrix

  const unsigned ROW=0, COL=1; /* To improve readability */

  MPI_Init(&argc,&argv);

  /* Get information about the communicator */
  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);

  printf("Task %d starting on node on %s!\n", myrank, hostname);

  /* Compute the size of the square grid.
   * We assume that nprocs is a square and that the matrix size
   * is a multiple of sqrt(nprocs).
   */
  int dims[2];
  dims[ROW] = dims[COL] = int(sqrt(nprocs));
  assert(dims[ROW] * dims[COL] == nprocs); // Test that nprocs is a square
  assert(n % dims[ROW] == 0); // Must divide exactly.
  const unsigned nlocal = n/dims[ROW];

  /* Set up the Cartesian topology and get the rank &
  * coordinates of the process in this topology
  */
  int periods[2];
  periods[ROW] = periods[COL] = 1;
  /* We will use wrap-around connections. */

  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

  /* Get my rank in the new topology */
  int my2drank;
  MPI_Comm_rank(comm_2d, &my2drank);

  /* Get my coordinates */
  int mycoords[2];
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  /* Global index offsets */
  const unsigned i_offset = mycoords[ROW]*nlocal;
  const unsigned j_offset = mycoords[COL]*nlocal;

  /* Initialize matrix A */
  vector<float> a(nlocal*nlocal);

  for(unsigned i=0; i<nlocal; ++i) {
    unsigned i_global = i+i_offset; // Global row index

    for(unsigned j=0; j<nlocal; ++j) {
      unsigned j_global = j+j_offset; // Global column index
      a.at(i*nlocal+j) = AEntry(i_global,j_global,n);
    }
  }

  /* Initialize vector b */
  vector<float> b(nlocal);

  if(mycoords[COL] == 0) {  // First column processes only
    for(unsigned i=0; i<nlocal; ++i) {
      const unsigned i_global = i+i_offset; // Global row index
      b.at(i) = BEntry(i_global,n);
    }
  }

  /* Distribute the b vector. */
  /* Step 1. The processes along the 0th column
   send their data to the diagonal process. */

  // Send to diagonal block
  if(mycoords[COL] == 0 && mycoords[ROW] != 0) {
    /* I'm in the first column */
    int drank;
    int coords[2];
    coords[ROW] = mycoords[ROW];
    coords[COL] = mycoords[ROW]; // coordinates of diagonal block
    MPI_Cart_rank(comm_2d, coords, &drank); // 2D communicator
    /* Send data to the diagonal block */
    MPI_Send(&b[0], nlocal, MPI_FLOAT, drank, 1, comm_2d);
  }

  // Receive from column 0
  if(mycoords[ROW] == mycoords[COL] && mycoords[ROW] != 0) {
    /* I am a diagonal block */
    int col0rank;
    int coords[2];
    coords[ROW] = mycoords[ROW];
    coords[COL] = 0; // Receiving from column 0
    MPI_Cart_rank(comm_2d, coords, &col0rank); // 2D communicator
    MPI_Recv(&b[0], nlocal, MPI_FLOAT, col0rank, 1, comm_2d,
             MPI_STATUS_IGNORE);
  }

  if(mycoords[ROW] == mycoords[COL] && mycoords[ROW] != 0) {
    for(unsigned i=0; i<nlocal; ++i) {
      const unsigned i_global = i+i_offset; // Global row index
      assert(b.at(i) == BEntry(i_global,n));
    }
  }

  /* Step 2. The diagonal processes perform a
   * column-wise broadcast
   */
  {
    /* Create the column-based sub-topology */
    MPI_Comm comm_col;
    int keep_dims[2];
    keep_dims[ROW] = 1;
    keep_dims[COL] = 0;
    MPI_Cart_sub(comm_2d, keep_dims, &comm_col);

    /* Broadcast inside column */
    int drank;
    int coord = mycoords[COL]; // Coordinate in 1D column topology
    MPI_Cart_rank(comm_col, &coord, &drank);
    MPI_Bcast(&b[0], nlocal, MPI_FLOAT, drank, comm_col);
  }

  /* Get into the main computational loop: A*b */
  vector<float> px(nlocal);

  for(unsigned i=0; i<nlocal; i++) {
    float p = 0.0;

    for(unsigned j=0; j<nlocal; j++) {
      p += a.at(i*nlocal+j) * b.at(j);
    }

    px.at(i) = p;
  }

  /* Perform the sum-reduction along the rows to add up
   * the partial dot-products; result is stored in column 0.
   */
  vector<float> x(nlocal);
  {
    /* Create the row-based sub-topology */
    MPI_Comm comm_row;
    int keep_dims[2];
    keep_dims[ROW] = 0;
    keep_dims[COL] = 1;
    MPI_Cart_sub(comm_2d, keep_dims, &comm_row);

    // Row-wise reduction
    int col0rank;
    int coord = 0; // Coordinate in 1D row topology
    MPI_Cart_rank(comm_row, &coord, &col0rank);
    MPI_Reduce(&px[0], &x[0], nlocal, MPI_FLOAT, MPI_SUM, col0rank, comm_row);
  }

  /* Test */
  if(mycoords[COL] == 0) {
    float emax = 0.;
    float refval = 0.;

    for(unsigned i=0; i<nlocal; i++) {
      float x0 = 0.0;
      const unsigned i_global = i+i_offset; // Global row index

      for(unsigned j=0; j<n; j++) {
        const float a0 = AEntry(i_global, j, n);
        const float b0 = BEntry(j, n);
        x0 += a0*b0;
      }

      emax = max(emax, abs(x0-x.at(i)));
      refval = max(refval, abs(x0));
    }

    printf("Row block = %d, error = %8.1e\n", mycoords[ROW], emax/refval);
    assert(emax == 0);
  }

  MPI_Barrier(MPI_COMM_WORLD); // Checking whether all tests have passed

  if(myrank == 0) {
    printf("All tests successfully passed\n");
  }

  MPI_Finalize();

  return 0;
}
