#include <iostream>
#include <thread>

#include "gtest/gtest.h"

using namespace std;

void f1() { cout << "Hello World!\n"; }

void f2(int m) { cout << "Hello World with m = " << m << endl; }

void f3(int &k) {
  cout << "Hello World; k was passed by reference; k = " << k << endl;
  k += 3;
}

void f4(int m, int &k) {
  cout << "f4() called with m = " << m << " and k = " << k << endl;
  k += m;
}

TEST(cpp_threads, basic) {
  // Demonstrate using thread constructor
  thread t1(f1);
  /* wait for t1 to finish */
  t1.join();
}

TEST(cpp_threads, with_argument) {
  int m = 5;
  // With an argument
  // The argument is added after f2.
  thread t2(f2, m);

  /* wait for t2 to finish */
  t2.join();
}

TEST(cpp_threads, with_reference) {
  int k = 7;
  // With a reference
  thread t3(f3, ref(k));
  // The arguments to the thread function are moved or copied by value. If a
  // reference argument needs to be passed to the thread function, it has to be
  // wrapped (e.g., with std::ref or std::cref).

  // Wait for t3 to complete
  t3.join();
  // The variable k should now be updated

  cout << "k is now equal to " << k << endl;
  ASSERT_EQ(k, 10);
}

// Exercise
// f4(m,k) should compute k += m
// Before f4(m,k):
// m = 5
// k = 7
// After:
// k = 12
// Use join at the end
TEST(cpp_threads, exercise) {
  int m = 5;
  int k = 7;

  // Solution
  thread t4(f4, m, ref(k));
  t4.join();

  cout << "k is now equal to " << k << endl;
  ASSERT_EQ(k, 12);
}