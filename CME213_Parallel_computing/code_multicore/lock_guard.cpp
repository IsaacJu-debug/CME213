#include <iostream>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"

using namespace std;

int g_i = 10;
mutex g_i_mutex;  // protects g_i

void safe_increment() {
  const lock_guard<mutex> lock(g_i_mutex);
  g_i += 3;  // Increment the shared variable

  cout << "g_i: " << g_i << "; in thread #" << this_thread::get_id() << '\n';

  // g_i_mutex is automatically released when lock
  // goes out of scope
}

TEST(lock_guard, test) {
  cout << "g_i: " << g_i << "; in main()\n";

  // lock_guard avoids the race condition on g_i
  thread t1(safe_increment);
  thread t2(safe_increment);

  t1.join();
  t2.join();

  cout << "g_i: " << g_i << "; in main()\n";
  ASSERT_EQ(g_i, 16);
}

class Thing {
 public:
  Thing() : m_N(0) {}
  int value() { return m_N; }

  void Add(int i) {
    auto temp = m_N;
    m_N = 42;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    temp += i;
    m_N = temp;
  }

 private:
  int m_N;
};

void munge(Thing& t) {
  for (int i = 0; i < 100; i++) {
    t.Add(1);
  }
}

TEST(lock_guard, fail) {
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

  // This test should fail because we did not protect the access to m_N.
  ASSERT_EQ(thing.value(), 300);
}