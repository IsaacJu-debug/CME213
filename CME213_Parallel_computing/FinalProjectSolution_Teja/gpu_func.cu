#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
const int BLOCK_SIZE = 32;
const int NBx = 16;
const int NBy = 4;
int myGEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((N + NBx - 1) / NBx, (M + NBy*NBx - 1) / (NBy*NBx));
    gpuGEMMult<<< grid_dim, block_dim >>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

__global__ void gpuGEMMult(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int N, int K)
{
    // int blockRow = blockIdx.y;
    // int blockCol = blockIdx.x;

    nn_real Cvalue[NBx] = {0.0};
    
    int row = threadIdx.y;
    int col = threadIdx.x;

    int stride_row = NBy*NBx*blockIdx.y;
    int stride_col = NBx*blockIdx.x;
    int id_a, id_b;
    int nby_m;

    __shared__ nn_real Bs[NBy][NBx];
    nn_real As[NBy];
    id_a = row + NBy * col;
    for (int m = 0; m < ((K + NBy - 1)/NBy); ++m) {
        nby_m = NBy*m;
        id_b = nby_m + row + K*(stride_col + col);
        
        for (int ii = 0; ii < NBy; ++ii) {
            
            if (nby_m + ii < K && stride_row + id_a < M) {
                As[ii] = A[stride_row + id_a + M*(nby_m + ii)];
            } else {
                As[ii] = 0;
            }
            
        }
        if (stride_col + col < N && nby_m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < NBy; ++k) {
            for (int jj = 0; jj < NBx; ++jj) {
                Cvalue[jj] += As[k] * Bs[k][jj];
            }
        }
        __syncthreads();
    }
    // printf("  Cval %0.2f,\n", Cvalue[col]);
    
    for (int ii = 0; ii < NBx; ++ii) {
        if (stride_row + id_a < M && stride_col + ii < N) {
            int id_c = stride_row + id_a + M*(stride_col + ii);
            C[id_c] = beta*C[id_c] + alpha*Cvalue[ii];
        }
    }
}

int myGEMM_naive(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (M*N + block_size - 1)/block_size;
    gpuGEMMult_naive<<<grid_size, block_size>>>(A, B, C, alpha, beta, M, K);
    return 0;
}

__global__ void gpuGEMMult_naive(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int K)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    size_t col = tid / (M);
    size_t row = tid % (M);

    nn_real mul = 0;
    for (size_t kk = 0; kk < K; ++kk) {
        mul += A[row + M*kk] * B[kk + K*col];
    }

    C[row + M*col] = beta*C[row + M*col] + alpha*mul;
}

int myGEMMshared(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    // int block_size, grid_size;
    int ns = 32;
    dim3 block_dim(ns, ns);
    dim3 grid_dim((N + ns - 1) / ns, (M + ns - 1) / ns);
    // printf("  block dim %d, %d, grid dim %d, %d detected\n", ns,
    //        ns, grid_dim.x, grid_dim.y);
    gpuGEMMultShar<<< grid_dim, block_dim >>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

/* Helper functions for neural networks */
// TODO
__global__ void gpuGEMMultShar(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int N, int K)
{
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    nn_real Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;
    int id_a, id_b;
    int id_c = BLOCK_SIZE*blockRow + BLOCK_SIZE*M*blockCol + row + M*col;
    __shared__ nn_real As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ nn_real Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[row][col] = (nn_real)0.0;
    Bs[row][col] = (nn_real)0.0;
    for (int m = 0; m < ((K + BLOCK_SIZE - 1)/BLOCK_SIZE); ++m) {
        id_a = BLOCK_SIZE*blockRow + BLOCK_SIZE*M*m + row + M*col;
        id_b = BLOCK_SIZE*m + BLOCK_SIZE*K*blockCol + row + K*col;
        if (BLOCK_SIZE*blockRow + row < M && BLOCK_SIZE*m + col < K) {
            As[row][col] = A[id_a];
        } else {
            As[row][col] = 0;
        }

        if (BLOCK_SIZE*blockCol + col < N && BLOCK_SIZE*m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[row][k] * Bs[k][col];
        }
        
        __syncthreads();
        
    }
    if (BLOCK_SIZE*blockRow + row < M && BLOCK_SIZE*blockCol + col < N) {
        C[id_c] = beta*C[id_c] + alpha*Cvalue;
    }
}                

int myGEMMsharedB(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((N + NBx - 1) / NBx, (M + NBy*NBx - 1) / (NBy*NBx));
    // printf("  block dim %d, %d, grid dim %d, %d detected\n", ns,
    //        ns, grid_dim.x, grid_dim.y);
    gpuGEMMultSharB<<< grid_dim, block_dim >>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

__global__ void gpuGEMMultSharB(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int N, int K)
{
    // int blockRow = blockIdx.y;
    // int blockCol = blockIdx.x;

    nn_real Cvalue[NBx] = {0.0};
    
    int row = threadIdx.y;
    int col = threadIdx.x;

    int stride_row = NBy*NBx*blockIdx.y;
    int stride_col = NBx*blockIdx.x;
    int id_a, id_b;
    int nby_m;
    // const size_t tid = row + NBy * col;
    __shared__ nn_real Bs[NBy][NBx];
    nn_real As[NBy];
    id_a = row + NBy * col;
    for (int m = 0; m < ((K + NBy - 1)/NBy); ++m) {
        nby_m = NBy*m;
        id_b = nby_m + row + K*(stride_col + col);
        
        for (int ii = 0; ii < NBy; ++ii) {
            
            if (nby_m + ii < K && stride_row + id_a < M) {
                As[ii] = A[stride_row + id_a + M*(nby_m + ii)];
            } else {
                As[ii] = 0;
            }
            
        }
        if (stride_col + col < N && nby_m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < NBy; ++k) {
            for (int jj = 0; jj < NBx; ++jj) {
                Cvalue[jj] += As[k] * Bs[k][jj];
            }
        }
        __syncthreads();
    }
    // printf("  Cval %0.2f,\n", Cvalue[col]);
    
    for (int ii = 0; ii < NBx; ++ii) {
        if (stride_row + id_a < M && stride_col + ii < N) {
            int id_c = stride_row + id_a + M*(stride_col + ii);
            C[id_c] = beta*C[id_c] + alpha*Cvalue[ii];
        }
    }
    
}

int myGemNunet(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *__restrict__ D,
           int M, int N, int K)
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (M*N + block_size - 1)/block_size;
    gpuGemVMulNunet<<< grid_size, block_size >>>(A, B, C, D, M, N, K);
    return 0;
}

/* Helper functions for neural networks */
// TODO

__global__ void gpuGemVMulNunet(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                int M, int N, int K)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M*N) {
        size_t col = tid / (M);
        size_t row = tid % (M);

        nn_real mul = 0;
        for (size_t kk = 0; kk < K; ++kk) {
            mul += A[row + M*kk] * B[kk + K*col];
        }

        D[row + M*col] = C[row] + mul;
    }
}
int myGemNunetShared(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *__restrict__ D,
           int M, int N, int K)
{
    int ns = 32;
    dim3 block_dim(ns, ns);
    dim3 grid_dim((N + ns - 1) / ns, (M + ns - 1) / ns);
    gpuGemVMulNunetSh<<< grid_dim, block_dim >>>(A, B, C, D, M, N, K);
    return 0;
}
__global__ void gpuGemVMulNunetSh(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                int M, int N, int K)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    nn_real Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;
    int id_a, id_b;
    int id_c = BLOCK_SIZE*blockRow + BLOCK_SIZE*M*blockCol + row + M*col;
    __shared__ nn_real As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ nn_real Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[row][col] = (nn_real)0.0;
    Bs[row][col] = (nn_real)0.0;
    for (int m = 0; m < ((K + BLOCK_SIZE - 1)/BLOCK_SIZE); ++m) {
        id_a = BLOCK_SIZE*blockRow + BLOCK_SIZE*M*m + row + M*col;
        id_b = BLOCK_SIZE*m + BLOCK_SIZE*K*blockCol + row + K*col;
        if (BLOCK_SIZE*blockRow + row < M && BLOCK_SIZE*m + col < K) {
            As[row][col] = A[id_a];
        } else {
            As[row][col] = 0;
        }

        if (BLOCK_SIZE*blockCol + col < N && BLOCK_SIZE*m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[row][k] * Bs[k][col];
        }
        
        __syncthreads();
        
    }
    if (BLOCK_SIZE*blockRow + row < M && BLOCK_SIZE*blockCol + col < N) {
        D[id_c] = C[BLOCK_SIZE*blockRow + row] + Cvalue;
    }
}

int myGemNunetSharedB(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *__restrict__ D,
           int M, int N, int K)
{
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((N + NBx - 1) / NBx, (M + NBy*NBx - 1) / (NBy*NBx));
    gpuGemVMulNunetShB<<< grid_dim, block_dim >>>(A, B, C, D, M, N, K);
    return 0;
}

__global__ void gpuGemVMulNunetShB(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                int M, int N, int K)
{
    // int blockRow = blockIdx.y;
    // int blockCol = blockIdx.x;

    nn_real Cvalue[NBx] = {0.0};
    
    int row = threadIdx.y;
    int col = threadIdx.x;

    int stride_row = NBy*NBx*blockIdx.y;
    int stride_col = NBx*blockIdx.x;
    int id_a, id_b;
    int nby_m;
    // const size_t tid = row + NBy * col;
    __shared__ nn_real Bs[NBy][NBx];
    nn_real As[NBy];
    id_a = row + NBy * col;
    for (int m = 0; m < ((K + NBy - 1)/NBy); ++m) {
        nby_m = NBy*m;
        id_b = nby_m + row + K*(stride_col + col);
        
        for (int ii = 0; ii < NBy; ++ii) {
            
            if (nby_m + ii < K && stride_row + id_a < M) {
                As[ii] = A[stride_row + id_a + M*(nby_m + ii)];
            } else {
                As[ii] = 0;
            }
            
        }
        if (stride_col + col < N && nby_m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < NBy; ++k) {
            for (int jj = 0; jj < NBx; ++jj) {
                Cvalue[jj] += As[k] * Bs[k][jj];
            }
        }
        __syncthreads();
    }
    // printf("  Cval %0.2f,\n", Cvalue[col]);
    
    for (int ii = 0; ii < NBx; ++ii) {
        if (stride_row + id_a < M && stride_col + ii < N) {
            int id_c = stride_row + id_a + M*(stride_col + ii);
            D[id_c] = C[stride_row + id_a] + Cvalue[ii];
        }
    }
}

__global__ void gpuSigmoid(nn_real *__restrict__ mat, 
                            nn_real *__restrict__ mat2, int M, int N)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M*N) {
        size_t col = tid / (M);
        size_t row = tid % (M);

        mat2[row + M*col] = 1 / (1 + exp(-mat[row + M*col])); 
    }
}

int mySigmoid(nn_real *__restrict__ mat, 
                nn_real *__restrict__ mat2, int M, int N)
{
    int block_size, grid_size;
    block_size = 1024;
    grid_size = (M*N + block_size - 1)/block_size;

    gpuSigmoid<<<grid_size, block_size>>>(mat, mat2, M, N);
    return 0;
}

__global__ void gpuExpKernel(nn_real *__restrict__ mat, 
                            nn_real *__restrict__ mat2, int M, int N)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M*N) {
        size_t col = tid / (M);
        size_t row = tid % (M);

        mat2[row + M*col] = exp(mat[row + M*col]);
    }
}

__global__ void gpuSumExpKernel(nn_real *__restrict__ mat, 
                nn_real *__restrict__ mat2, int M, int N)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        nn_real su = 0;
        for (size_t i = 0; i < M; ++i) {
            su += mat[i + M*tid];
        }
        mat2[tid] = su;
    }
}

__global__ void gpuRepSum(nn_real *__restrict__ mat, nn_real *__restrict__ mat1,
                nn_real *__restrict__ mat2, int M, int N)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t col;
    nn_real buff;
    if (tid < M*N) {
        col = tid / (M);
        buff =  mat[tid] / mat2[col];
        mat[tid] = buff;
        mat1[tid] = buff;
    }
}
// int gpuSoftmax(struct cacheG cache, nn_real *__restrict__ mat, 
//                 nn_real *__restrict__ mat2, int M, int N) 
int gpuSoftmax(struct cacheG cache, int M, int N) 
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (M*N + block_size - 1)/block_size;

    gpuExpKernel<<< grid_size, block_size >>>(cache.z2, cache.a2, M, N);
    if (N < 512) {
        block_size = N;
    }
    grid_size = (N + block_size - 1)/block_size;

    // nn_real *prox;
    // cudaMalloc((void **)&prox, sizeof(nn_real) * N);
    gpuSumExpKernel<<< grid_size, block_size >>>(cache.a2, cache.prox, M, N);

    // nn_real *B;
    // B = (nn_real *)malloc(N * sizeof(nn_real));
    // cudaMemcpy(B, prox, sizeof(nn_real) * N, cudaMemcpyDeviceToHost);
    // for (int i; i < N; ++i) {
    // std::cout << "prox " << B[i]<< std::endl;
    // }
    // free(B);

    grid_size = (M*N + block_size - 1)/block_size;
    gpuRepSum<<<grid_size, block_size>>>(cache.a2, cache.yc, cache.prox, M, N);
    // cudaFree(prox);
    // cache.yc = cache.a2;
    // cudaMemcpy(cache.yc, cache.a2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToDevice);
    return 0;
}

void allocCpyMemGPU(nn_real *__restrict__ input, nn_real *out, int size)
{
    cudaMalloc((void **)&out, sizeof(nn_real) * size);
    cudaMemcpy(out, input, sizeof(nn_real) * size, cudaMemcpyHostToDevice);
}

int myBpStepW1(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int cur_batch, int bs, nn_real *y_bc)
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (cur_batch*cl + block_size - 1)/block_size;
    
    gpuDiffMat<<< grid_size, block_size >>>(cache.yc, 
                                y_bc, bpgrads.diff, 
                                        cur_batch, cl, bs);
    
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((num_nr + NBx - 1) / NBx, (cl + NBy*NBx - 1) / (NBy*NBx));
    gpuBpStepW1Shared<<< grid_dim, block_dim >>>(bpgrads.diff, 
                    cache.a1, nnP.w2, bpgrads.dw2, reg, cl, num_nr, cur_batch);
    return 0;
}
int myBpStepW1_3(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int cur_batch, int bs)
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (cur_batch*cl + block_size - 1)/block_size;
    
    gpuDiffMat<<< grid_size, block_size >>>(cache.yc, 
                                nnP.y_batch, bpgrads.diff, 
                                        cur_batch, cl, bs);
    
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((num_nr + NBx - 1) / NBx, (cl + NBy*NBx - 1) / (NBy*NBx));
    gpuBpStepW1Shared<<< grid_dim, block_dim >>>(bpgrads.diff, 
                    cache.a1, nnP.w2, bpgrads.dw2, reg, cl, num_nr, cur_batch);
    return 0;
}

__global__ void gpuDiffMat(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int cur_batch, int M, int bs)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M*cur_batch) {
        size_t col = tid / (M);
        size_t row = tid % (M);

        C[row + M*col] = (A[row + M*col] - B[row + M*col]) / bs;
    }
    
    // C[tid] = (A[tid] - B[tid]) / cur_batch;
}                            

__global__ void gpuBpStepW1(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                nn_real reg, int M, int N, int K)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M*N) {
        size_t col = tid / (M);
        size_t row = tid % (M);

        nn_real mul = 0;
        for (size_t kk = 0; kk < K; ++kk) {
            mul += A[row + M*kk] * B[col + N*kk]; // B[kk + K*col]
        }

        D[row + M*col] = reg * C[row + M*col] + mul;
    }
}

__global__ void gpuBpStepW1Shared(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                nn_real reg, int M, int N, int K)
{
    // int blockRow = blockIdx.y;
    // int blockCol = blockIdx.x;

    nn_real Cvalue[NBx] = {0.0};
    
    int row = threadIdx.y;
    int col = threadIdx.x;

    int stride_row = NBy*NBx*blockIdx.y;
    int stride_col = NBx*blockIdx.x;
    int id_a, id_b;
    int nby_m;
    // const size_t tid = row + NBy * col;
    __shared__ nn_real Bs[NBy][NBx];
    nn_real As[NBy];
    id_a = row + NBy * col;
    for (int m = 0; m < ((K + NBy - 1)/NBy); ++m) {
        nby_m = NBy*m;
        id_b = stride_col + col + N*(nby_m + row);
        
        for (int ii = 0; ii < NBy; ++ii) {
            
            if (nby_m + ii < K && stride_row + id_a < M) {
                As[ii] = A[stride_row + id_a + M*(nby_m + ii)];
            } else {
                As[ii] = 0;
            }
            
        }
        if (stride_col + col < N && nby_m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < NBy; ++k) {
            for (int jj = 0; jj < NBx; ++jj) {
                Cvalue[jj] += As[k] * Bs[k][jj];
            }
        }
        __syncthreads();
    }
    // printf("  Cval %0.2f,\n", Cvalue[col]);
    
    for (int ii = 0; ii < NBx; ++ii) {
        if (stride_row + id_a < M && stride_col + ii < N) {
            int id_c = stride_row + id_a + M*(stride_col + ii);
            D[id_c] = reg * C[id_c] + Cvalue[ii];
        }
    }
}


int myBpStepb1(struct gradsG &bpgrads, 
                int cl, int cur_batch)
{
    int block_size, grid_size;
    block_size = cl;
    grid_size = (cl + block_size - 1)/block_size;
    gpuBpStepb1<<< grid_size, block_size >>>(bpgrads.diff, bpgrads.db2, cl, cur_batch);
    
    return 0;
}

__global__ void gpuBpStepb1(nn_real *__restrict__ A, 
                        nn_real *__restrict__ B, int M, int K)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M) {
        nn_real ads = 0;
        for (size_t kk = 0; kk < K; ++kk) {
            ads += A[tid + M*kk];
        }
        B[tid] = ads;
    }
}

int myBpStepW0(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int N1, int cur_batch, nn_real *X_bc)
{
    int block_size, grid_size;
    block_size = 1024;
    grid_size = (cur_batch*num_nr + block_size - 1)/block_size;
    // gpuTranspMM<<< grid_size, block_size >>>(nnP.w2, bpgrads.diff, 
    //                                         cache.da1, num_nr, cl);
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((cur_batch + NBx - 1) / NBx, (num_nr + NBy*NBx - 1) / (NBy*NBx));
    gpuTranspMMShared<<< grid_dim, block_dim >>>(nnP.w2, bpgrads.diff, 
                                            cache.da1, num_nr, cur_batch, cl);

    gpuBpHadamand<<< grid_size, block_size >>>(cache.da1, cache.a1, 
                                            cache.dz1, num_nr);

    // grid_size = (N1*num_nr + block_size - 1)/block_size;
    // gpuBpStepW1<<< grid_size, block_size >>>(cache.dz1, 
    //                 nnP.x_batch, nnP.w1, bpgrads.dw1, 
    //                 reg, num_nr, N1, cur_batch);
    // dim3 block_dim(NBx, NBy);
    dim3 grid_dim2((N1 + NBx - 1) / NBx, (num_nr + NBy*NBx - 1) / (NBy*NBx));
    gpuBpStepW1Shared<<< grid_dim2, block_dim >>>(cache.dz1, 
                    X_bc, nnP.w1, bpgrads.dw1, 
                    reg, num_nr, N1, cur_batch);
    
    return 0;
}
int myBpStepW0_3(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int N1, int cur_batch)
{
    int block_size, grid_size;
    block_size = 1024;
    grid_size = (cur_batch*num_nr + block_size - 1)/block_size;
    // gpuTranspMM<<< grid_size, block_size >>>(nnP.w2, bpgrads.diff, 
    //                                         cache.da1, num_nr, cl);
    dim3 block_dim(NBx, NBy);
    dim3 grid_dim((cur_batch + NBx - 1) / NBx, (num_nr + NBy*NBx - 1) / (NBy*NBx));
    gpuTranspMMShared<<< grid_dim, block_dim >>>(nnP.w2, bpgrads.diff, 
                                            cache.da1, num_nr, cur_batch, cl);

    gpuBpHadamand<<< grid_size, block_size >>>(cache.da1, cache.a1, 
                                            cache.dz1, num_nr);

    // grid_size = (N1*num_nr + block_size - 1)/block_size;
    // gpuBpStepW1<<< grid_size, block_size >>>(cache.dz1, 
    //                 nnP.x_batch, nnP.w1, bpgrads.dw1, 
    //                 reg, num_nr, N1, cur_batch);
    // dim3 block_dim(NBx, NBy);
    dim3 grid_dim2((N1 + NBx - 1) / NBx, (num_nr + NBy*NBx - 1) / (NBy*NBx));
    gpuBpStepW1Shared<<< grid_dim2, block_dim >>>(cache.dz1, 
                    nnP.x_batch, nnP.w1, bpgrads.dw1, 
                    reg, num_nr, N1, cur_batch);
    
    return 0;
}
__global__ void gpuTranspMM(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int M, int K)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    size_t col = tid / (M);
    size_t row = tid % (M);

    nn_real mul = 0;
    for (size_t kk = 0; kk < K; ++kk) {
        mul += A[kk + K*row] * B[kk + K*col];
    }
    C[row + M*col] = mul;
}

__global__ void gpuTranspMMShared(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int M, int N, int K)
{
    // int blockRow = blockIdx.y;
    // int blockCol = blockIdx.x;

    nn_real Cvalue[NBx] = {0.0};
    
    int row = threadIdx.y;
    int col = threadIdx.x;

    int stride_row = NBy*NBx*blockIdx.y;
    int stride_col = NBx*blockIdx.x;
    int id_a, id_b;
    int nby_m;
    // const size_t tid = row + NBy * col;
    __shared__ nn_real Bs[NBy][NBx];
    nn_real As[NBy];
    id_a = row + NBy * col;
    for (int m = 0; m < ((K + NBy - 1)/NBy); ++m) {
        nby_m = NBy*m;
        id_b = nby_m + row + K*(stride_col + col);
        
        for (int ii = 0; ii < NBy; ++ii) {
            
            if (nby_m + ii < K && stride_row + id_a < M) {
                As[ii] = A[nby_m + ii + K*(stride_row + id_a)];
            } else {
                As[ii] = 0;
            }
            
        }
        if (stride_col + col < N && nby_m + row < K) {
            Bs[row][col] = B[id_b];
        } else {
            Bs[row][col] = 0;
        }
        __syncthreads();
        for (int k = 0; k < NBy; ++k) {
            for (int jj = 0; jj < NBx; ++jj) {
                Cvalue[jj] += As[k] * Bs[k][jj];
            }
        }
        __syncthreads();
    }
    // printf("  Cval %0.2f,\n", Cvalue[col]);
    
    for (int ii = 0; ii < NBx; ++ii) {
        if (stride_row + id_a < M && stride_col + ii < N) {
            int id_c = stride_row + id_a + M*(stride_col + ii);
            C[id_c] = Cvalue[ii];
        }
    }
}

__global__ void gpuBpHadamand(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int M)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    size_t col = tid / (M);
    size_t row = tid % (M);

    C[row + M*col] = A[row + M*col] * B[row + M*col] * (1 - B[row + M*col]);
}

int myBpStepb0(struct cacheG &cache, struct gradsG &bpgrads, 
                int num_nr, int cur_batch)
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (num_nr + block_size - 1)/block_size;
    gpuBpStepb0<<< grid_size, block_size >>>(cache.dz1, bpgrads.db1, num_nr, cur_batch);
    // nn_real *v3;
    // v3 = (nn_real *)malloc(10 * cur_batch * sizeof(nn_real));
    // cudaMemcpy(v3, bpgrads.diff, sizeof(nn_real) * 10 * cur_batch, cudaMemcpyDeviceToHost);
    // for (int i=0; i<cur_batch*10; ++i){
    //     std::cout << "diff after  " << v3[i]<< std::endl;
    // }
    return 0;
}
__global__ void gpuBpStepb0(nn_real *__restrict__ A, 
                        nn_real *__restrict__ B, int M, int K)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    nn_real ads = 0;
    for (size_t kk = 0; kk < K; ++kk) {
        ads += A[tid + M*kk];
    }
    if (tid < M){B[tid] = ads;}
    
}

int myWeightUpdate(struct NunetP &nnP, struct gradsG &bpgrads, 
                nn_real learning_rate, int num_nr, int N, int cl)
{
    int block_size, grid_size;
    block_size = 1024;
    
    grid_size = (num_nr*N + block_size - 1)/block_size;
    gpuWeiUpdate<<< grid_size, block_size >>>(bpgrads.dw1, nnP.w1, 
                                        learning_rate, num_nr, N);
    grid_size = (num_nr*cl + block_size - 1)/block_size;
    gpuWeiUpdate<<< grid_size, block_size >>>(bpgrads.dw2, nnP.w2, 
                                        learning_rate, cl, num_nr);
    // nn_real *v1;
    // v1 = (nn_real *)malloc(num_nr * sizeof(nn_real));
    // cudaMemcpy(v1, nnP.b1, sizeof(nn_real) * num_nr, cudaMemcpyDeviceToHost);
    // nn_real *v2;
    // v2 = (nn_real *)malloc(num_nr * sizeof(nn_real));
    // cudaMemcpy(v2, bpgrads.db1, sizeof(nn_real) * num_nr, cudaMemcpyDeviceToHost);
    // for (int i=num_nr-2; i<num_nr; ++i){
    //     std::cout << "b1  " << v1[i]<< std::endl;
    //     std::cout << "db1  " << v2[i]<< std::endl;
    //     v2[i] = (-learning_rate*v1[i] + v2[i]);
    //     // std::cout << "upd  " << v2[i]<< std::endl;
    // }                                    
    return 0;                                    
}
__global__ void gpuWeiUpdate(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real learning_rate, int M, int N)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    size_t col = tid / (M);
    size_t row = tid % (M);

    if (tid < M*N) {B[row + M*col] = B[row + M*col] - learning_rate * A[row + M*col];}
}

int myBiasUpdate(struct NunetP &nnP, struct gradsG &bpgrads, 
                nn_real learning_rate, int num_nr, int cl)
{
    int block_size, grid_size;
    block_size = 512;
    grid_size = (num_nr + block_size - 1)/block_size;

    gpuBiaUpdateS0<<< grid_size, block_size >>>(bpgrads.db1, nnP.b1, 
                            learning_rate, num_nr);
    block_size = cl; 
    grid_size = (cl + block_size - 1)/block_size;
    gpuBiaUpdateS0<<< grid_size, block_size >>>(bpgrads.db2, nnP.b2, 
                            learning_rate, cl);
    return 0;                                                     
}
// __global__ void gpuBiaUpdateS0(nn_real *__restrict__ A, nn_real *__restrict__ B, 
//                             nn_real learning_rate, int M, int batch)
// {
//     const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
// //1;//
//     if (tid < M) {
//         if (batch == 0) {
//             for (int ii=0; ii<M; ++ii) {
//                 B[ii] = - learning_rate * A[ii];
//             }
//         } else {
//             for (int ii=0; ii<M; ++ii) {
//                 B[ii] = B[ii] - learning_rate * A[ii];
//             }
//         }
//     } //B[tid] 
// }
__global__ void gpuBiaUpdateS0(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real learning_rate, int M)
{
    const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < M) {
        B[tid] = B[tid] - learning_rate * A[tid];
        } //B[tid] 
}

