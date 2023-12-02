#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "gtest/gtest.h"

using namespace std;

struct Node
{
    int data;
    Node *next;
    Node() : next(NULL) {}
};

void Wait()
{
    // this_thread::sleep_for(chrono::milliseconds(100));
}

void Visit(Node *curr_node)
{
    /* Add 1 to data */
    ++(curr_node->data);
}

void IncrementListItems(Node *head)
{
#pragma omp parallel
#pragma omp single
    {
        Node *curr_node = head;
        while (curr_node)
        {
            printf("Main thread. %p\n", (void *)curr_node);
#pragma omp task
            {
                // curr_node is firstprivate by default
                Wait();
                int tid = omp_get_thread_num();
                Visit(curr_node);
                printf("Task @%2d: node %p data %d\n", tid, (void *)curr_node,
                       curr_node->data);
            }
            curr_node = curr_node->next;
        }
    }
}

TEST(list_omp, main)
{
    Node *root = new Node;

    int size = 10;
    int i;

    // Fill the list
    Node *head = root;
    root->data = 0;
    for (i = 0; i < size - 1; i++)
    {
        head = head->next = new Node;
        head->data = i + 1;
    }

    head = root;
    while (head != NULL)
    {
        cout << "data = " << head->data << endl;
        head = head->next;
    }

    omp_set_num_threads(4);
#pragma omp parallel
#pragma omp single
    cout << "Number of threads = " << omp_get_num_threads() << endl;

    IncrementListItems(root);

    cout << "Done incrementing data\n";

    head = root;
    i = 0;
    while (head != NULL)
    {
        ASSERT_EQ(head->data, ++i);
        head = head->next;
    }
}