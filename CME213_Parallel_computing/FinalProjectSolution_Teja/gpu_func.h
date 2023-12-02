#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(nn_real *A, nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
           int M, int N, int K);

// TODO
// Add additional function declarations
struct cacheG
{
  // nn_real* X;
  nn_real* z1;
  nn_real* z2;
  nn_real* a1;
  nn_real* a2;
  nn_real* yc;
  nn_real* dummy1;
  nn_real* da1;
  nn_real* dz1;
  nn_real* prox;
};

struct NunetP
{
  nn_real* w1;
  nn_real* w2;
  nn_real* b1;
  nn_real* b2;
  nn_real* x_batch;
  nn_real* y_batch;
  nn_real* X_all;
  nn_real* y_all;
};

struct mpiptrs
{
  nn_real* dW1;
  nn_real* dW2;
  nn_real* db1;
  nn_real* db2;
  nn_real* dW1_total;
  nn_real* dW2_total;
  nn_real* db1_total;
  nn_real* db2_total;
  nn_real* X_mini;
  nn_real* y_mini;
};

struct gradsG
{
  nn_real* dw1;
  nn_real* dw2;
  nn_real* db1;
  nn_real* db2;
  nn_real* diff;
};
__global__ void gpuGEMMult(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int N, int K);
__global__ void gpuSigmoid(nn_real *__restrict__ mat, 
                            nn_real *__restrict__ mat2, int M, int N);
int mySigmoid(nn_real *__restrict__ mat, 
                nn_real *__restrict__ mat2, int M, int N);
__global__ void gpuExpKernel(nn_real *__restrict__ mat, 
                            nn_real *__restrict__ mat2, int M, int N);
__global__ void gpuSumExpKernel(nn_real *__restrict__ mat, 
                nn_real *__restrict__ mat2, int M, int N);
__global__ void gpuRepSum(nn_real *__restrict__ mat, nn_real *__restrict__ mat1,
                nn_real *__restrict__ mat2, int M, int N);
// int gpuSoftmax(nn_real *__restrict__ mat, 
//                 nn_real *__restrict__ mat2, int M, int N); 
int gpuSoftmax(struct cacheG cache, int M, int N);

int myGemNunet(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *__restrict__ D,
           int M, int N, int K);

__global__ void gpuGemVMulNunet(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                int M, int N, int K);


int myBpStepW1(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int cur_batch, int bs, nn_real *y_bc);
__global__ void gpuDiffMat(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int cur_batch, int M, int bs);
__global__ void gpuBpStepW1(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                nn_real reg, int M, int N, int K);  

int myBpStepb1(struct gradsG &bpgrads, 
                int cl, int cur_batch);
__global__ void gpuBpStepb1(nn_real *__restrict__ A, 
                        nn_real *__restrict__ B, int M, int K); 

int myBpStepW0(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int N1, int cur_batch, nn_real *X_bc);
__global__ void gpuTranspMM(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int M, int K);
__global__ void gpuBpHadamand(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int M);

int myBpStepb0(struct cacheG &cache, struct gradsG &bpgrads, 
                int num_nr, int cur_batch);
__global__ void gpuBpStepb0(nn_real *__restrict__ A, 
                        nn_real *__restrict__ B, int M, int K);
int myWeightUpdate(struct NunetP &nnP, struct gradsG &bpgrads, 
                nn_real learning_rate, int num_nr, int N, int cl);
__global__ void gpuWeiUpdate(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real learning_rate, int M, int N);
int myBiasUpdate(struct NunetP &nnP, struct gradsG &bpgrads, 
                nn_real learning_rate, int num_nr, int cl);
__global__ void gpuBiaUpdateS0(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real learning_rate, int M);    
void parallelFeedforward(struct NunetP &nnP, struct cacheG &cache, 
                    int num_nr, int N, int cl, int bs, nn_real *X_bc);
void parallelBackProp(struct NunetP &nnP, struct cacheG &cache, 
                    struct gradsG &bpgrads, nn_real reg, int num_nr,
                    int N, int cl, int cur_batch, int bs, nn_real *X_bc, nn_real *y_bc);
void parallelGradientDes(struct NunetP &nnP, struct gradsG &bpgrads, 
                        nn_real learning_rate, int num_nr, int N, int cl, int batch);                    
int compareMatrices(nn_real *myC, nn_real *refC, int NI, int NJ);         


int myGEMMshared(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K);
__global__ void gpuGEMMultShar(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int N, int K);

int myGEMMsharedB(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K);

__global__ void gpuGEMMultSharB(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int N, int K);

int myGemNunetShared(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *__restrict__ D,
           int M, int N, int K);
__global__ void gpuGemVMulNunetSh(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                int M, int N, int K);

int myGemNunetSharedB(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real *__restrict__ D,
           int M, int N, int K);
__global__ void gpuGemVMulNunetShB(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                int M, int N, int K);
__global__ void gpuBpStepW1Shared(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real *__restrict__ D, 
                nn_real reg, int M, int N, int K);
__global__ void gpuTranspMMShared(nn_real *__restrict__ A, nn_real *__restrict__ B, 
                            nn_real *__restrict__ C, int M, int N, int K);

void parallelFeedforward3(struct NunetP &nnP, struct cacheG &cache, 
                    int num_nr, int N, int cl, int bs);
void parallelBackProp3(struct NunetP &nnP, struct cacheG &cache, 
                    struct gradsG &bpgrads, nn_real reg, int num_nr,
                    int N, int cl, int cur_batch, int bs);
int myBpStepW0_3(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int N1, int cur_batch);
int myBpStepW1_3(struct NunetP &nnP, struct cacheG &cache,
           struct gradsG &bpgrads, nn_real reg, 
           int cl, int num_nr, int cur_batch, int bs);           

int myGEMM_naive(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K);
__global__ void gpuGEMMult_naive(nn_real *__restrict__ A, nn_real *__restrict__ B,
                nn_real *__restrict__ C, nn_real alpha, nn_real beta, 
                int M, int K);           
void mallocMPI(struct mpiptrs &mpiptr, int num_nr, int cl, int N1);
#endif
