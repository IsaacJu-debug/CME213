#include <atomic>
#include <cstdio>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

using namespace std;

mutex g_mutex;
queue<int> g_task_queue;

const int nthreads = 4;
const int norders = 16;

typedef vector<atomic<bool>> vec;

vec v_orders(norders);

void InsertNewOrder(const int order) {
  g_task_queue.push(order);
  v_orders[order].store(false);
  printf("Client no %3d\n", order);
}

void Delivery(const int order) {
  EXPECT_FALSE(v_orders[order].load());
  // Pretend I am really busy doing something here
  // Nice job, isn't it? The pay is good too.
  this_thread::sleep_for(chrono::milliseconds(20));
  v_orders[order].store(true);
}

void PizzaDeliveryPronto(int thread_id) {
  g_mutex.lock();

  while (!g_task_queue.empty()) {
    const int order_ = g_task_queue.front();
    printf("Thread %d: %3d\n", thread_id, order_);
    g_task_queue.pop();

    /* At this point, the thread delivers the pizza. */
    /* The mutex is unlocked so that other threads can work. */
    g_mutex.unlock();

    Delivery(order_);

    g_mutex.lock();
  }

  g_mutex.unlock();
  return;
}

TEST(mutex, test) {
  /* Insert some tasks */
  for (int i = 0; i < norders; ++i) {
    /* Got a phone call for pizza delivery. */
    v_orders[i] = false;
    InsertNewOrder(i);
  }

  /* Have threads consume tasks */
  queue<thread> thread_pool;
  for (int th = 0; th < nthreads; ++th)
    thread_pool.push(thread(PizzaDeliveryPronto, th));
  // push() or emplace()

  for (int th = 0; th < nthreads; ++th) {
    thread_pool.front().join();
    thread_pool.pop();
  }

  for (int i = 0; i < norders; ++i) {
    EXPECT_TRUE(v_orders[i].load()) << "order no " << i << " has value false";
  }
}