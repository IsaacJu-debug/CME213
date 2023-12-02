#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

int myGEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    myGEMM_GPU<<<blocks, threads>>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

int myMULT_ABT(nn_real *A, nn_real *B, nn_real *C, int M, int N, int K) {
    //calculate C = AB^T
        //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    myMULT_ABT_GPU<<<blocks, threads>>>(A, B, C, M, N, K);
    return 0;
}

__global__
void myMULT_ABT_GPU(nn_real *A, nn_real *B, nn_real *C, int M, int N, int K)
{
    // each thread computes an element in C_ij
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    //calculate C[i][j]
    if (i < M && j < N) {
        nn_real d_ij = 0;
        for (int k = 0; k < K; k++) {
            // A[i][k]
            //row i column k
            nn_real a_ik = A[i + k*M];
            // nn_real a_ik = A[i + k*M];
            // B[j][k]
            nn_real b_jk = B[j + k*N];
            d_ij = d_ij + a_ik*b_jk;
        }
        C[i + j*M] = d_ij;
    }
}

int myMULT_ATB(nn_real *A, nn_real *B, nn_real *C, int M, int N, int K) {
    //calculate C = AB^T
        //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    myMULT_ATB_GPU<<<blocks, threads>>>(A, B, C, M, N, K);
    return 0;
}

__global__
void myMULT_ATB_GPU(nn_real *A, nn_real *B, nn_real *C, int M, int N, int K)
{
    // each thread computes an element in C_ij
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    //calculate C[i][j]
    if (i < M && j < N) {
        nn_real d_ij = 0;
        for (int k = 0; k < K; k++) {
            // A[i][k]
            //row i column k
            nn_real a_ki = A[k + i*M];
            // nn_real a_ik = A[i + k*M];
            // B[j][k]
            nn_real b_kj = B[k + j*N];
            d_ij = d_ij + a_ki*b_kj;
        }
        C[i + j*M] = d_ij;
    }
}

int myCOMPGRADZ1(nn_real *A, nn_real *B, nn_real *C, int M, int N) 
{
    //calculate C = AB^T
        //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    myCOMPGRADZ1_GPU<<<blocks, threads>>>(A, B, C, M, N);
    return 0;
}

__global__
void myCOMPGRADZ1_GPU(nn_real *A, nn_real *B, nn_real *C, int M, int N)
{
    // each thread computes an element in C_ij
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    //calculate C[i][j]
    if (i < M && j < N) {
        C[i + j*M] = A[i + j*M]*B[i + j*M]*(1 - B[i+j*M]);
    }
}

int myGRADIENT_UPDATE(nn_real *A, nn_real *B, nn_real learning_rate, int M, int N) 
{
    //calculate C = AB^T
        //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    myGRADIENT_UPDATE_GPU<<<blocks, threads>>>(A, B, learning_rate, M, N);
    return 0;
}

__global__
void myGRADIENT_UPDATE_GPU(nn_real *A,  nn_real *B, nn_real learning_rate, int M, int N)
{
    // each thread computes an element in C_ij
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    //calculate C[i][j]
    if (i < M && j < N) {
        B[i + j*M] = B[i + j*M] - learning_rate*A[i + j*M];
    }
}






/* Helper functions for neural networks */
// TODO
__global__ 
void myGEMM_GPU(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // each thread computes an element in C_ij
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    //calculate C[i][j]
    if (i < M && j < N) {
        nn_real d_ij = 0;
        for (int k = 0; k < K; k++) {
            // A[i][k]
            //row i column k
            nn_real a_ik = A[i + k*M];
            // B[k][j]
            nn_real b_kj = B[k + j*K];
            d_ij = d_ij + alpha*a_ik*b_kj;
        }
        C[i + j*M] = d_ij + beta*C[i + j*M];
    }
}


int mySIGMOID(nn_real *X_1, nn_real *X, int M, int N) {
    //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    mySIGMOID_GPU<<<blocks, threads>>>(X_1, X, M, N);
    return 0;
}

int myEXP( nn_real *X, int M, int N) {
    //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    // call GPU function
    myEXP_GPU<<<blocks, threads>>>(X, M, N);
    return 0;
}

__global__
void mySIGMOID_GPU(nn_real *X_1, nn_real *X, int M, int N) {
    // Every thread calculates sigmoid of one element in X
    // X[i][j]
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    // use of exp?
    if (i < M && j < N) {
        X_1[i + j*M] =  1.0 / (1.0 + std::exp(-X[i+j*M]));
    }

}

int mySOFTMAX(nn_real *X_1, nn_real *X, nn_real *rowsum_exp_X, int M, int N) {
    //input of softmax is a function
    //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);

    myEXP_GPU<<<blocks, threads>>>(X, M, N);
    myDIVISION_GPU<<<blocks, threads>>>(X_1, X, rowsum_exp_X, M, N);

    return 0;
}

int mySUMROW(nn_real*X, int M, int N) {
    //
    std::cout<<"mySUMROW"<<std::endl;
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    int threads = 1024;
    int blocks = (int) (N/threads) + 1;
    
    mySUMROW_GPU<<<blocks, threads>>>(X, M, N);
    return 0;
}

int myDIFFY_AVG_COL(nn_real*result, nn_real*y, nn_real*y_hat, int M, int N) {
    //input of softmax is a function
    //block size is less than 1024
    int num_blockDim_x = 32;
    int num_blockDim_y = 32;

    // calculate grid dimension based on the block size and matrix size
    // num_gridDim_x 
    int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
    int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

    dim3 blocks(num_gridDim_x, num_gridDim_y);
    dim3 threads(num_blockDim_x, num_blockDim_y);
    myDIFFY_AVG_COL_GPU<<<blocks, threads>>>(result, y, y_hat, M, N);
    return 0;
}

__global__
void myDIFFY_AVG_COL_GPU(nn_real *result, nn_real*y, nn_real *y_hat, int M, int N) {
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    // use of exp?
    if (i < M && j < N) {
        result[i + j*M] =  1/N*(y_hat[i + j*M] - y[i + j*M]);
    }

}

__global__
void myEXP_GPU(nn_real *X, int M, int N) {
    // Every thread calculates sigmoid of one element in X
    // Calculate exponent of each element in X
    // X[i][j]
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    // use of exp?
    if (i < M && j < N) {
        X[i + j*M] =  std::exp(X[i+j*M]);
    }
}

__global__
void mySOFTMAX_GPU(nn_real *X, nn_real *vec_sumrow, int M, int N) {
    // Every thread calculates sigmoid of one element in X
    // Calculate exponent of each element in X
    // X[i][j]
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    // use of exp?
    if (i < M && j < N) {
        X[i + j*M] =  X[i + j*M]/vec_sumrow[j];
    }
}

__global__
void mySUMROW_GPU(nn_real *X, int M, int N) {
    // Every thread calculates sigmoid of one element in X
    // Calculate exponent of each element in X
    // X[i][j]
    //calculate row as i, col as j from blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y
    int i = 0;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    // use of exp?
    nn_real sum = 0;
    if (i < M && j < N) {
        for (int k = 0; k < M; i++){
            sum = sum + X[k + j*M];
        }
    }
    X[i + j*M] = sum;
}

__global__
//calcualte A/B and save 
void myDIVISION_GPU(nn_real *A, nn_real *B, nn_real *C, int M, int N) {
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        A[i + j * M] =  B[i + j * M]/C[i + j * M];
    }
}




int feedforward_parallel(nn_real *W_1, nn_real *W_2, nn_real*b_1, nn_real*b_2, nn_real*z_1, nn_real*z_2, nn_real*a_1, nn_real*a_2, nn_real*X, nn_real*rowsum_X, nn_real*y, int row_W_1, int row_W_2, int row_X, int col_X, int row_y) {
    // multiplication my GEMM_GPU
    //1.calculate z_1 = W_1*X+z_1, y will be covered by new y
    // alpha = 1, beta = 1
    myGEMM(W_1, X, z_1,
           1, 1,
           row_W_1, col_X, row_X);
    //2.calculate sigmoid a_1 = sigmoid(z_1)
    mySIGMOID(a_1, z_1, row_W_1, col_X);
    //3.calculate z_2 = W_2*X+z_2
    myGEMM(W_2, a_1, z_2,
           1, 1,
           row_W_2, col_X, row_W_1);
    //4.calculate softmax a_2 = sofrmax(z_2)
    mySOFTMAX(a_2, z_2, rowsum_X, row_W_2, col_X);
    return 0;
}

// int backprop_parallel(nn_real *W_1, nn_real *W_2, nn_real*b_1, nn_real*b_2, nn_real*z_1, nn_real*z_2, nn_real*a_1, nn_real*a_2, nn_real*X, nn_real*rowsum_X, nn_real*y, int row_W_1, int row_W_2, int row_X, int col_X, int row_y) {
//     myDIFFY_AVG(nn_real*y_hat, nn_real*y, int M, int N)

// }

// // memory allocation
// void memory_alloc(NeuralNetwork &nn, const arma::Mat<nn_real> &X, const arma::Mat<nn_real> &y) {
// // 
    
// }

