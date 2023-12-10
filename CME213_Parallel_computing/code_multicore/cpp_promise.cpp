#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <valarray>
#include <vector>

#include "gtest/gtest.h"

using namespace std;

typedef valarray<int> vint;

vint vec0 = {1, 2, 3, 4, 5, 6};
const int exact_value = vec0.sum();

int accumulate(vint &v) { return v.sum(); }

int global_sum;

void accumulate_global(vint &v) { global_sum = accumulate(v); }

void accumulate_ref(vint &v, int &sum) { sum = accumulate(v); }

void accumulate_promise(vint &v, promise<int> accumulate_promise) {
  int sum = accumulate(v);
  accumulate_promise.set_value(sum);  // Notify future
  printf("result of accumulate_promise [21 expected] = %d\n", sum);
}

void get_max(vector<int>::iterator first, vector<int>::iterator last,
             promise<int> max_promise) {
  int sum = *first;
  auto it = first;
  for (; it != last; ++it) sum = (*it > sum ? *it : sum);
  max_promise.set_value(sum);
  cout << "result of max_future [5 expected] = " << sum << '\n';
}

TEST(promise, global_var) {
  // Demonstrate using promise<int> to return a value
  thread th0(accumulate_global, ref(vec0));
  th0.join();
  printf("result of accumulate_global [21 expected] = %d\n", global_sum);
  ASSERT_EQ(global_sum, exact_value);
}

TEST(promise, reference_local) {
  int sum0;
  thread th1(accumulate_ref, ref(vec0), ref(sum0));
  th1.join();
  printf("result of accumulate_ref [21 expected] = %d\n", sum0);
  ASSERT_EQ(sum0, exact_value);
}

TEST(promise, using_promise) {
  promise<int> sum_promise;  // Will store the int
  future<int> sum_future = sum_promise.get_future();
  // Used to retrieve the value asynchronously, at a later time

  thread th2(accumulate_promise, ref(vec0), move(sum_promise));
  // move() will "move" the resources

  // future::get() waits until the future has a valid result and retrieves it
  int sum1 = sum_future.get();
  ASSERT_EQ(sum1, exact_value);
  th2.join();
}

TEST(promise, using_lambda_fn) {
  // Same mechanism but using a different syntax with a lambda expression
  promise<int> sum_promise;  // Will store the int
  future<int> sum_future = sum_promise.get_future();

  std::thread([&sum_promise] {
    int sum = accumulate(vec0);
    sum_promise.set_value(sum);
    printf("accumulate_promise with lambda function [21 expected] = %d\n", sum);
  }).detach();
  // detach(): separates the thread of execution from the thread object,
  // allowing execution to continue independently.

  int sum1 = sum_future.get();
  ASSERT_EQ(sum1, exact_value);
}

TEST(promise, exercise) {
  vector<int> vec_2 = {1, -2, 4, -10, 5, 4};
  promise<int> max_promise;
  future<int> max_future = max_promise.get_future();

  // TODO: use a thread and a promise to calculate the maximum of vec_2

  thread t6(get_max, vec_2.begin(), vec_2.end(), move(max_promise));

  const int max_result = max_future.get();
  ASSERT_EQ(max_result, 5);
  t6.join();
}