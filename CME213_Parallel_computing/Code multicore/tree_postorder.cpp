#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define P .8

bool InsertCond()
{
    return float(rand()) / RAND_MAX < P;
}

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
    // printf("node %p number of descendants = %d\n", (void *)curr_node,
    //       curr_node->data);
}

// Sequential code
int PostOrderTraverseSequential(struct Node *curr_node)
{
    int left = 0, right = 0;

    if (curr_node->left)
        left = PostOrderTraverseSequential(curr_node->left);

    if (curr_node->right)
        right = PostOrderTraverseSequential(curr_node->right);

    curr_node->data = left + right; // Number of children nodes
    return 1 + left + right;
}

// Parallel post-order traversal
int PostOrderTraverse(struct Node *curr_node)
{
    int left = 0, right = 0;

    if (curr_node->left)
#pragma omp task shared(left)
        left = PostOrderTraverse(curr_node->left);
    // Default attribute for task constructs is firstprivate
    if (curr_node->right)
#pragma omp task shared(right)
        right = PostOrderTraverse(curr_node->right);

#pragma omp taskwait
    curr_node->data = left + right; // Number of children nodes

    Visit(curr_node);
    return 1 + left + right;
}

int main()
{
    int n_level = 10; // Maximum number of levels in the tree
    int level = 1;

    Node *root = new Node;

    // Create a random tree
    srand(2022);
    FillTree(n_level, level, root);

#pragma omp parallel
#pragma omp single // Only a single thread should execute this
    printf("Number of nodes in the tree:      %d\n", PostOrderTraverse(root));

    printf("Result of sequential calculation: %d\n",
           PostOrderTraverseSequential(root));

    printf("(Expected number of nodes: %g)\n",
           (pow(2. * P, n_level) - 1) / (2. * P - 1.));
}
