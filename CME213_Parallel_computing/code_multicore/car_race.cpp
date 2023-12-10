#include <omp.h>

#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

int main(void)
{
    string begin = "A ";
    string s0 = "race ";
    string s1 = "car ";
    string end = "is fun to watch\n";

    omp_set_num_threads(4);

    cout << "Run 1\n";

#pragma omp parallel
    for (int i = 0; i < 2; ++i)
    {
        cout << begin;
        cout << s0;
        cout << s1;
        cout << end;
    }

    cout << "\nRun 2\n";

#pragma omp parallel for
    for (int i = 0; i < 2; ++i)
    {
        cout << begin;
        cout << s0;
        cout << s1;
        cout << end;
    }

    cout << "\nRun 3\n";

    for (int i = 0; i < 2; ++i)
#pragma omp parallel
    {
        cout << begin;
        cout << s0;
        cout << s1;
        cout << end;
    }

    cout << "\nRun 4\n";

    for (int i = 0; i < 2; ++i)
#pragma omp parallel
#pragma omp single
    {
        cout << begin;
#pragma omp task
        cout << s0;
#pragma omp task
        cout << s1;
        cout << end;
    }

    cout << "\nRun 5\n";

#pragma omp parallel
#pragma omp single
    {
        cout << begin;
#pragma omp task
        cout << s0;
#pragma omp task
        cout << s1;
#pragma omp taskwait
        cout << end;
    }

    return 0;
}