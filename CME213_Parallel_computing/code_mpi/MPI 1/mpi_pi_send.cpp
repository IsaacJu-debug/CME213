#include <cstdio>
#include <cstdlib>

#include "mpi.h"

#define DARTS 500000 /* number of throws at dartboard */
#define ROUNDS 10    /* number of times "darts" is iterated */
#define MAIN 0       /* task ID of MAIN task */

#define sqr(x) ((x) * (x))

/*
  Explanation of constants and variables used in this function:
  darts       = number of throws at dartboard
  score       = number of darts that hit circle
  n           = index variable
  r           = random number scaled between 0 and 1
  x_coord     = x coordinate, between -1 and 1
  x_sqr       = square of x coordinate
  y_coord     = y coordinate, between -1 and 1
  y_sqr       = square of y coordinate
  pi          = computed value of pi
 */
double DartBoard(int darts) {
  int score = 0;

  /* "throw darts at board" */
  for (int n = 1; n <= darts; n++) {
    /* generate random numbers for x and y coordinates */
    double r = (double)rand() / (double)(RAND_MAX);
    double x_coord = (2.0 * r) - 1.0;
    r = (double)rand() / (double)(RAND_MAX);
    double y_coord = (2.0 * r) - 1.0;

    /* if dart lands in circle, increment score */
    if ((sqr(x_coord) + sqr(y_coord)) <= 1.0) {
      score++;
    }
  }

  /* estimate pi */
  return 4.0 * (double)score / (double)darts;
}

int main(int argc, char *argv[]) {
  /* Obtain number of tasks and task ID */
  MPI_Init(&argc, &argv);
  int rank,     /* also used as seed number */
      numprocs, /* number of tasks */
      len;      /* length of hostname (no. of chars) */
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);
  printf("MPI process %2d has started on %s [total number of processors %d]\n",
         rank, hostname, numprocs);

  /* Set seed for random number generator equal to task ID */
  srandom(2022 + (rank << 4));

  double avepi = 0.;
  double pirecv;
  MPI_Status status;

  for (int i = 0; i < ROUNDS; i++) {
    /* All tasks calculate pi using the dartboard algorithm */
    double my_pi = DartBoard(DARTS);

    int tag = i;         // Message tag is set to the iteration count
    if (rank != MAIN) {  // Workers send my_pi to MAIN
      // Message tag is set to the iteration count
      int rc = MPI_Send(&my_pi, 1, MPI_DOUBLE, MAIN, tag, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) printf("%d: send error | %d\n", rank, i);
    } else {
      // MAIN receives messages from all workers
      double pisum = 0;
      for (int n = 1; n < numprocs; n++) {
        /* Message source is set to the wildcard MPI_ANY_SOURCE: */
        int rc = MPI_Recv(&pirecv, 1, MPI_DOUBLE, MPI_ANY_SOURCE, tag,
                          MPI_COMM_WORLD, &status);
        if (rc != MPI_SUCCESS)
          printf("%d: rcv error | %d; msg %d\n", rank, i, n);

        /* Running total of pi */
        pisum += pirecv;
      }
      /* MAIN calculates the average value of pi for this iteration */
      double pi = (pisum + my_pi) / numprocs;
      /* MAIN calculates the average value of pi over all iterations */
      avepi = ((avepi * i) + pi) / (i + 1);
      printf("   After %8d throws, average value of pi = %10.8f\n",
             (DARTS * (i + 1) * numprocs), avepi);
    }
  }

  if (rank == MAIN) printf("\nExact value of pi: 3.1415926535897 \n");

  MPI_Finalize();

  return 0;
}