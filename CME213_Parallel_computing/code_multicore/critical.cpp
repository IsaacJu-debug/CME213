#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>

#include "gtest/gtest.h"

using namespace std;

bool is_prime_test(int n)
{
    if (n <= 3)
        return n > 1;
    else if ((n % 2 == 0) || (n % 3 == 0))
        return false;

    for (int i = 5; i * i <= n; i += 6)
        if ((n % i == 0) || (n % (i + 2) == 0))
            return false; // Not a prime

    return true; // Must be prime
}

TEST(critical_prime, main)
{
    const int n = 104729;
    // Calculate all prime numbers smaller than n

    set<int> m;
#pragma omp parallel for
    for (int i = 2; i <= n; ++i)
    {
        bool is_prime = is_prime_test(i);

#pragma omp critical
        if (is_prime)
            m.insert(i);
        // Save this prime; this requires a critical clause.
        // Only a single thread can execute this line at a time.
    }

    printf("Number of prime numbers smaller than %d: %ld\n", n, m.size());

    ASSERT_EQ(m.size(), 10000);

    // Check
    {
        // Read primes from a file to test m
        auto it = m.begin();
        int count = 0; /* Read only the first 10,000 primes */

        ifstream prime_file("10000.txt");
        while (it != m.end() && count < 10000)
        {
            int next_prime;
            prime_file >> next_prime;   // Read from file
            ASSERT_EQ(*it, next_prime); // Test
            ++it;
            ++count;
        }
        prime_file.close();
    }
}
