#include <omp.h>

#include <cstdio>
#include <cstdlib>

bool InsertCond() { return float(rand()) / RAND_MAX < 0.9; }

struct Node
{
    int data;
    Node *left, *right;
    Node() : left(NULL), right(NULL) {}
};

void FillTree(int max_level, int level, Node *curr_node)
{
    if (level < max_level)
    {
        curr_node->left = new Node;
        curr_node->right = new Node;

        if (InsertCond())
            FillTree(max_level, level + 1, curr_node->left);

        if (InsertCond())
            FillTree(max_level, level + 1, curr_node->right);
    }
}

void Visit(Node *curr_node)
{
    /* do work here */
    curr_node->data = rand() % 100;
    printf("node %p data = %d\n", (void *)curr_node, curr_node->data);
}

// Pre-order traversal
void Traverse(struct Node *curr_node)
{
    // Pre-order = visit then call Traverse()
    Visit(curr_node);

    if (curr_node->left)
#pragma omp task
        Traverse(curr_node->left);

    if (curr_node->right)
#pragma omp task
        Traverse(curr_node->right);
}

int main()
{
    Node *root = new Node;

    int n_level = 5; // Maximum number of levels in the tree
    int level = 1;

    // Create a random tree
    srand(2022);
    FillTree(n_level, level, root);

#pragma omp parallel
#pragma omp single
    // Only a single thread should execute this
    Traverse(root);
}
