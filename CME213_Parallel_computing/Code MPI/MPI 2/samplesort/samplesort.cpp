#include "mpi.h"
#include <unistd.h>
#include <iostream>
#include <iterator>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <ctime>
#include <climits>

using std::cout;
using std::endl;
using std::vector;

typedef std::vector<int> vint;

// #define VERBOSE

// Print a std::vector to an output stream
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v)
{
  if (!v.empty())
  {
    out << '[';
    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(out, ", "));
    out << v.back() << "]";
  }

  return out;
}

int SampleSort(int n, vint &elements, vint &sorted_elements)
{
  int npes, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  const int nlocal = n / npes;

  assert(int(elements.size()) == nlocal);

  std::sort(elements.begin(), elements.end());

  /* Select local npes-1 equally spaced elements */
  vint splitters(npes);

  for (int i = 1; i < npes; i++)
  {
    splitters[i - 1] = elements[(i * nlocal) / npes];
  }

#ifdef VERBOSE

  for (int i = 0; i < npes; ++i)
  {
    if (myrank == i)
    {
      cout << "Rank: " << myrank << " splitters " << splitters << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

#endif

  /* Gather the samples in the processors */
  vint allpicks(npes * (npes - 1)); // Splitter candidates
  MPI_Allgather(&splitters[0], npes - 1, MPI_INT,
                &allpicks[0], npes - 1, MPI_INT,
                MPI_COMM_WORLD);

  /* sort these samples */
  std::sort(allpicks.begin(), allpicks.end());

#ifdef VERBOSE

  if (myrank == 0)
  {
    cout << "Rank: " << myrank << " allpicks " << allpicks << endl;
  }

#endif

  /* Select splitters */
  for (int i = 1; i < npes; i++)
  {
    splitters[i - 1] = allpicks[i * (npes - 1)];
  }

  splitters[npes - 1] = INT_MAX;

#ifdef VERBOSE

  if (myrank == 0)
  {
    cout << "Rank: " << myrank << " splitters " << splitters << endl;
  }

#endif

  /* Compute the number of elements that belong to each bucket */
  vint scounts(npes, 0);

  int j = 0;

  /* Count the number of entries in each bucket
   * The elements are sorted so this process is very fast.
   */
  for (auto e : elements)
  {
    while (e >= splitters[j])
    {
      ++j; // Find the proper bucket
    }

    scounts[j]++; // Increment bucket count
  }

  /* Determine the starting location of each bucket's element in the elements array */
  vint sdispls(npes);
  sdispls[0] = 0;
  assert(sdispls.size() == scounts.size());
  std::partial_sum(scounts.begin(), scounts.end() - 1, sdispls.begin() + 1);

  /* Perform an all-to-all to inform the corresponding processes of the number of
   * elements they are going to receive. This information is stored in rcounts array
   */
  vint rcounts(npes);
  MPI_Alltoall(&scounts[0], 1, MPI_INT,
               &rcounts[0], 1, MPI_INT,
               MPI_COMM_WORLD);

  /* Based on rcounts, determine where in the local array the data from each
   * processor will be stored. This array will store the received elements as well as the
   * final sorted sequence.
   */
  vint rdispls(npes);
  rdispls[0] = 0;
  assert(rdispls.size() == rcounts.size());
  std::partial_sum(rcounts.begin(), rcounts.end() - 1, rdispls.begin() + 1);

  const int nsorted = rdispls.back() + rcounts.back();
  sorted_elements.resize(nsorted);

  /* Each process sends and receives the corresponding elements, using the
   * MPI_Alltoallv operation.
   * The arrays scounts and sdispls are used to specify the number of
   * elements to be sent and where these elements are stored, respectively.
   * The arrays rcounts and rdispls are used to specify the number of
   * elements to be received, and where these elements will be stored,
   * respectively.
   */
  MPI_Alltoallv(&elements[0],
                &scounts[0], &sdispls[0], MPI_INT,
                &sorted_elements[0],
                &rcounts[0], &rdispls[0], MPI_INT,
                MPI_COMM_WORLD);

  /* Perform the final local sort */
  std::sort(sorted_elements.begin(), sorted_elements.end());

#ifdef VERBOSE

  for (int i = 0; i < npes; ++i)
  {
    if (myrank == i)
    {
      cout << "Rank: " << myrank << " " << sorted_elements << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

#endif

  return nsorted;
}

void ProcessOpt(int argc, char **argv, int *n)
{
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1)
    switch (c)
    {
    case 'n':
      *n = atoi(optarg);
      break;

    case 'h':
      printf(
          "Options:\n-n SIZE\t\tInput vector size\n");
      exit(2);

    case '?':
      fprintf(stderr, "Unrecognized option: -%c\n", optopt);
      exit(2);
    }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  // My rank
  int npes, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int n = 2 * npes;
  ProcessOpt(argc, argv, &n);
  assert(n >= 1);

  if (myrank == 0)
  {
    cout << "Size of input vector: " << n << endl;
  }

  // Length of array sorted by this process
  const int nlocal = n / npes;

  assert(npes * nlocal == n);

  // Generate random data
  vector<std::uint32_t> seeds(npes); // Random seeds
  {
    std::seed_seq seq{time(0)};
    seq.generate(seeds.begin(), seeds.end());
  }
  std::default_random_engine generator(seeds[myrank]);
  // Uniform random integers
  //    std::uniform_int_distribution<int> distribution(0,INT_MAX);
  std::uniform_int_distribution<int> distribution(0, 9);

  // Initialize local array
  vint elements(nlocal);

  for (int &e : elements)
  {
    e = distribution(generator);
  }

  //    auto it = elements.begin();
  //    for (int i=myrank; i<n; i += npes, ++it) *it = i;
  //    for (int i=0; i<nlocal; ++i, ++it) *it = i + nlocal*myrank;

#ifdef VERBOSE

  for (int i = 0; i < npes; ++i)
  {
    if (myrank == i)
    {
      cout << "Rank: " << myrank << " " << elements << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

#endif

  // For debugging purposes gather the original sequences at the root.
  vint elements_all(n);
  MPI_Gather(&elements[0], nlocal, MPI_INT, &elements_all[0],
             nlocal, MPI_INT, 0, MPI_COMM_WORLD);

  // Will contain the sorted array
  vint sorted_elements;

  // Sort!

  double startTime = MPI_Wtime();
  SampleSort(n, elements, sorted_elements);
  const double elapsed = MPI_Wtime() - startTime;

  // Debugging!

  // Check the parallel sorted sequences.
  // We first need to know the length of each sorted sequence on each node
  int ssize = sorted_elements.size();
  vint rsize(npes);
  MPI_Gather(&ssize, 1, MPI_INT, &rsize[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
  // Now gather data from all nodes
  vint displs(npes);
  displs[0] = 0;
  std::partial_sum(rsize.begin(), rsize.end() - 1, displs.begin() + 1);
  vint elements_sorted_all(n);
  MPI_Gatherv(&sorted_elements[0], ssize, MPI_INT,
              &elements_sorted_all[0],
              &rsize[0], &displs[0], MPI_INT, 0,
              MPI_COMM_WORLD);

  if (myrank == 0)
  {
    cout << npes << " processes --- checking the result of parallel samplesort\n";
    startTime = MPI_Wtime();
    std::sort(elements_all.begin(), elements_all.end()); // Sort original sequence
    const double elapsed_seq = MPI_Wtime() - startTime;
#ifdef VERBOSE
    cout << "Original sequence sorted: " << elements_all << endl;
    cout << "Result of parallel sort:  " << elements_sorted_all << endl;
#endif
    assert(elements_sorted_all == elements_all);
    cout << "Test passed\n";
    cout << "Efficiency: " << elapsed_seq / (npes * elapsed)
         << "; runtime seq: " << elapsed_seq << ", par: " << elapsed << endl;
  }

  MPI_Finalize();

  return 0;
}
