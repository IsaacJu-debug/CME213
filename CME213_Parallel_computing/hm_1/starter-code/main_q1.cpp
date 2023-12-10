#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(testMatrix, sampleTest) {
  ASSERT_EQ(1000, 1000)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(2000, 2000)
      << "This does not fail, hence this message is not printed.";
  // If uncommented, the following line will make this test fail.
  // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
  // will be printed.";
}

/*
TODO:

For both the MatrixDiagonal and the MatrixSymmetric classes, do the following:

Write at least the following tests to get full credit here:
1. Declare an empty matrix with the default constructor for MatrixSymmetric.
Assert that the NormL0 and size functions return appropriate values for these.
2. Using the second constructor that takes size as argument, create a matrix of
size zero. Repeat the assertions from (1).
3. Provide a negative argument to the second constructor and assert that the
constructor throws an exception.
4. Create and initialize a matrix of some size, and verify that the NormL0
function returns the correct value.
5. Create a matrix, initialize some or all of its elements, then retrieve and
check that they are what you initialized them to.
6. Create a matrix of some size. Make an out-of-bounds access into it and check
that an exception is thrown.
7. Test the stream operator using std::stringstream and using the "<<" operator.

*/
