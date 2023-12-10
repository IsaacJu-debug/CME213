#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"

class Thing {
 public:
  Thing() : m_N(0) {}

  int value() { return m_N; }

  // TODO: modify Add using a lock_guard to protect access to m_N.
  void Add(int i) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    auto temp = m_N;
    m_N = 42;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    temp += i;
    m_N = temp;
  }

 private:
  mutable std::mutex m_Mutex;
  int m_N;
};

void munge(Thing& t) {
  for (int i = 0; i < 100; i++) {
    t.Add(1);
  }
}

TEST(lock_guard, exercise) {
  // Instantiate a single Thing instance which will be modified
  // at the same time by three threads
  Thing thing;

  // Start the three threads to modify our single Thing instance
  std::thread thread1(munge, std::ref(thing));
  std::thread thread2(munge, std::ref(thing));
  std::thread thread3(munge, std::ref(thing));

  // Wait for them all to complete
  thread1.join();
  thread2.join();
  thread3.join();

  // The correct result should be 300.
  // Without a lock_guard, the assert fails.

  ASSERT_EQ(thing.value(), 300);
}