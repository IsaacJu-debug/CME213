#include <omp.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "gtest/gtest.h"

using namespace std;

#define NUM_THREADS 2
#define SLEEP_THREAD 1
#define NUM_LOOPS 2

enum Types
{
    ThreadPrivate,
    Private,
    FirstPrivate,
    LastPrivate,
    Shared,
    MAX_TYPES
};

int nSave[NUM_THREADS][MAX_TYPES][NUM_LOOPS] = {{0}};
int nThreadPrivate;

// Each thread creates its own private copy of that variable.
#pragma omp threadprivate(nThreadPrivate)

TEST(openmp, sharing_clause)
{
    int nPrivate = -1;
    int nFirstPrivate = 5;
    int nLastPrivate = -3;
    int nShared = -4;
    nThreadPrivate = -5;

    printf(
        "These are the variables before entry "
        "into the parallel region.\n");
    printf("nThreadPrivate = %4d\n", nThreadPrivate);
    printf("      nPrivate = %4d\n", nPrivate);
    printf(" nFirstPrivate = %4d\n", nFirstPrivate);
    printf("  nLastPrivate = %4d\n", nLastPrivate);
    printf("       nShared = %4d\n\n", nShared);
    omp_set_num_threads(NUM_THREADS);

    // threadprivate specifies that a variable is private to a thread.
    // The copyin clause causes the listed variables to be copied from the
    // primary thread to all other threads in the team immediately after the
    // threads have been created. The variables in the list must be
    // threadprivate.
    // The firstprivate clause specifies that each thread should have its own
    // instance of a variable, and that the variable should be initialized with
    // the value of the variable before the parallel construct.
    // The lastprivate clause specifies that the enclosing context's version of
    // the variable is set equal to the private version of whichever thread
    // executes the final iteration (for-loop construct) or last section
    // (#pragma sections).

    // https://learn.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-clauses?view=msvc-170
#pragma omp parallel copyin(nThreadPrivate) private(nPrivate) shared(nShared) \
    firstprivate(nFirstPrivate)
    {
#pragma omp for schedule(static, 1) lastprivate(nLastPrivate)
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            for (int j = 0; j < NUM_LOOPS; ++j)
            {
                int nThread = omp_get_thread_num();
                assert(nThread < NUM_THREADS);

                if (nThread == SLEEP_THREAD)
                    this_thread::sleep_for(chrono::milliseconds(100));
                nSave[nThread][ThreadPrivate][j] = nThreadPrivate;
                nSave[nThread][Private][j] = nPrivate;
                nSave[nThread][Shared][j] = nShared;
                nSave[nThread][FirstPrivate][j] = nFirstPrivate;
                nSave[nThread][LastPrivate][j] = nLastPrivate;
                nThreadPrivate = 10 + nThread;
                nPrivate = 20 + nThread;
                nShared = 30 + nThread;
                nLastPrivate = 40 + i;
                nFirstPrivate -= 2 + i;
            }
        }
    }

    for (int i = 0; i < NUM_LOOPS; ++i)
    {
        for (int j = 0; j < NUM_THREADS; ++j)
        {
            printf(
                "These are the variables at entry of "
                "loop %d of thread %d.\n",
                i, j);
            printf("nThreadPrivate = %12d (copyin)\n", nSave[j][ThreadPrivate][i]);
            printf("      nPrivate = %12d (private)\n", nSave[j][Private][i]);
            printf(" nFirstPrivate = %12d (firstprivate)\n", nSave[j][FirstPrivate][i]);
            printf("  nLastPrivate = %12d (lastprivate)\n", nSave[j][LastPrivate][i]);
            printf("       nShared = %12d (shared)\n\n", nSave[j][Shared][i]);
        }
    }

    printf(
        "These are the variables after exit from "
        "the parallel region.\n");
    printf(
        "nThreadPrivate = %4d (The last value in the "
        "main thread)\n",
        nThreadPrivate);
    EXPECT_EQ(nThreadPrivate, 10);
    printf(
        "      nPrivate = %4d (The value prior to "
        "entering parallel region)\n",
        nPrivate);
    EXPECT_EQ(nPrivate, -1);
    printf(
        " nFirstPrivate = %4d (The value prior to "
        "entering parallel region)\n",
        nFirstPrivate);
    EXPECT_EQ(nFirstPrivate, 5);
    printf(
        "  nLastPrivate = %4d (The value from the "
        "last iteration of the loop)\n",
        nLastPrivate);
    EXPECT_EQ(nLastPrivate, 41);
    printf(
        "       nShared = %4d (The value assigned, "
        "from the delayed thread, %d)\n\n",
        nShared, SLEEP_THREAD);
    EXPECT_EQ(nShared, 31);
}