#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"
#define TOL 2e-6 // Tolerance for tests
#define ROOT 0
#define MPI_SAFE_CALL(call)                                                  \
  do                                                                         \
  {                                                                          \
    int err = call;                                                          \
    if (err != MPI_SUCCESS)                                                  \
    {                                                                        \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)
const int BAT_debug = 0;
const int proc_lim = 1;
int get_num_batches(int N, int batch_size)
{
  return (N + batch_size - 1) / batch_size;
}

int get_batch_size(int N, int batch_size, int batch)
{
  int num_batches = get_num_batches(N, batch_size);
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

int get_mini_batch_size(int batch_size, int num_procs, int rank)
{
  int mini_batch_size = batch_size / num_procs;
  return rank < batch_size % num_procs ? mini_batch_size + 1 : mini_batch_size;
}

nn_real norms(NeuralNetwork &nn)
{
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
  // for (int i; i < mat.n_rows*mat.n_cols; ++i) {
  //   std::cout << "mat2 " << mat2(i)<< std::endl;
    
  // }
}

// feedforward pass
void feedforward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                 struct cache &cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  // bpgrads.dif2 = diff;
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg)
{
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}
void myPrintMat(arma::Mat<nn_real> mat, string ss) {
  int NI = mat.n_rows;
  int NJ = mat.n_cols;
  for (int ii=NI*NJ-2; ii <NI*NJ; ++ii) {
    std::cout << ss << mat(ii)<< std::endl;
  }
}
/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label)
{
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i)
  {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}


/*
 * Train the neural network &nn
 */
void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug)
{
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;
  
  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, batch_size);

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      int last_col = batch_start + get_batch_size(N, batch_size, batch);
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      if (batch == num_batches - 1)
      {
        assert(last_col == X.n_cols);
      }
      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);
      
      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);
      
      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);
      
      if (print_every > 0 && iter % print_every == 0)
      {
        if (grad_check)
        {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i)
      {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i)
      {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }
      

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to the "cpu_save_dir" folder.
         In the later runs (with same parameters), you can use just the debug
         flag to output diff b/w CPU and GPU without running the CPU version
         version. */
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag)
      {
        save_cpu_data(nn, iter);
      }

      batch_start = last_col;
      iter++;
    }
  }
  
}
void createGPuMem(NeuralNetwork &nn, struct NunetP &nnP, struct cacheG &cache, 
                  struct gradsG &bpgrads, int num_nr, int bs, int cl, int N1)
{
  cudaMalloc((void **)&nnP.w1, sizeof(nn_real) * num_nr * N1);
  cudaMalloc((void **)&nnP.w2, sizeof(nn_real) * cl * num_nr);
  cudaMalloc((void **)&nnP.b1, sizeof(nn_real) * num_nr);
  cudaMalloc((void **)&nnP.b2, sizeof(nn_real) * cl);
  cudaMalloc((void **)&nnP.x_batch, sizeof(nn_real) * N1 * bs);
  cudaMalloc((void **)&nnP.y_batch, sizeof(nn_real) * cl * bs);
  
  cudaMalloc((void **)&cache.z1, sizeof(nn_real) * num_nr * bs);
  cudaMalloc((void **)&cache.a1, sizeof(nn_real) * num_nr * bs);
  cudaMalloc((void **)&cache.z2, sizeof(nn_real) * cl * bs);
  cudaMalloc((void **)&cache.a2, sizeof(nn_real) * cl * bs);
  cudaMalloc((void **)&cache.yc, sizeof(nn_real) * cl * bs);
  cudaMalloc((void **)&cache.dz1, sizeof(nn_real) * num_nr * bs);
  cudaMalloc((void **)&cache.da1, sizeof(nn_real) * num_nr * bs);
  cudaMalloc((void **)&cache.prox, sizeof(nn_real) * bs);
  
  cudaMalloc((void **)&bpgrads.dw1, sizeof(nn_real) * num_nr * N1);
  cudaMalloc((void **)&bpgrads.dw2, sizeof(nn_real) * num_nr * cl);
  cudaMalloc((void **)&bpgrads.db1, sizeof(nn_real) * num_nr);
  cudaMalloc((void **)&bpgrads.db2, sizeof(nn_real) * cl);
  cudaMalloc((void **)&bpgrads.diff, sizeof(nn_real) * cl * bs);
  
  // cudaMemcpy(nnP.w1, nn.W[0].memptr(), sizeof(nn_real) * cl * N_all, cudaMemcpyHostToDevice);  
  cudaMemcpy(nnP.w1, nn.W[0].memptr(), sizeof(nn_real) * num_nr * N1, cudaMemcpyHostToDevice);
  cudaMemcpy(nnP.w2, nn.W[1].memptr(), sizeof(nn_real) * cl * num_nr, cudaMemcpyHostToDevice);
  cudaMemcpy(nnP.b1, nn.b[0].memptr(), sizeof(nn_real) * num_nr, cudaMemcpyHostToDevice);
  cudaMemcpy(nnP.b2, nn.b[1].memptr(), sizeof(nn_real) * cl, cudaMemcpyHostToDevice);
}
void createGPuXy(const arma::Mat<nn_real> &X, const arma::Mat<nn_real> &y, 
                struct NunetP &nnP, int cl, int N1, int N_all)
{
  cudaMalloc((void **)&nnP.X_all, sizeof(nn_real) * N1 * N_all);
  cudaMalloc((void **)&nnP.y_all, sizeof(nn_real) * cl * N_all);
  cudaMemcpy(nnP.X_all, X.memptr(), sizeof(nn_real) * N1 * N_all, cudaMemcpyHostToDevice);
  cudaMemcpy(nnP.y_all, y.memptr(), sizeof(nn_real) * cl * N_all, cudaMemcpyHostToDevice);
}
void destroyGPuMem(struct NunetP &nnP, 
          struct cacheG &cache, struct gradsG &bpgrads)
{
  cudaFree(nnP.w1);
  cudaFree(nnP.w2);
  cudaFree(nnP.b1);
  cudaFree(nnP.b2);
  cudaFree(nnP.x_batch);
  cudaFree(nnP.y_batch);

  cudaFree(cache.z1);
  cudaFree(cache.z2);
  cudaFree(cache.a1);
  cudaFree(cache.a2);
  cudaFree(cache.da1);
  cudaFree(cache.dz1);
  cudaFree(cache.yc);
  cudaFree(cache.prox);

  cudaFree(bpgrads.dw1);
  cudaFree(bpgrads.dw2);
  cudaFree(bpgrads.db1);
  cudaFree(bpgrads.db2);
  cudaFree(bpgrads.diff);
}
void destroyGPuXy(struct NunetP &nnP)
{
  cudaFree(nnP.X_all);
  cudaFree(nnP.y_all);
} 
void destroyCPuMem(struct mpiptrs &mpiptr) //, nn_real *X_bc, nn_real *y_bc, const int num_batches
{
  free(mpiptr.dW1);
  free(mpiptr.dW2);
  free(mpiptr.db1);
  free(mpiptr.db2);
  free(mpiptr.dW1_total);
  free(mpiptr.dW2_total);
  free(mpiptr.db1_total);
  free(mpiptr.db2_total);

  free(mpiptr.X_mini);
  free(mpiptr.y_mini);

  
}
void mallocMPI(struct mpiptrs &mpiptr, int num_nr, int cl, int N1)
{
  mpiptr.dW1 = (nn_real *)malloc(num_nr * N1 * sizeof(nn_real));
  mpiptr.dW2 = (nn_real *)malloc(cl * num_nr * sizeof(nn_real));
  mpiptr.db1 = (nn_real *)malloc(num_nr * sizeof(nn_real));
  mpiptr.db2 = (nn_real *)malloc(cl * sizeof(nn_real));

  mpiptr.dW1_total = (nn_real *)malloc(num_nr * N1 * sizeof(nn_real));
  mpiptr.dW2_total = (nn_real *)malloc(cl * num_nr * sizeof(nn_real));
  mpiptr.db1_total = (nn_real *)malloc(num_nr * sizeof(nn_real));
  mpiptr.db2_total = (nn_real *)malloc(cl * sizeof(nn_real));
}
void interMemCpy(struct gradsG &bpgrads, struct mpiptrs &mpiptr, int num_nr, int cl, int N1) 
{
  cudaMemcpy(mpiptr.dW1, bpgrads.dw1, sizeof(nn_real) * num_nr * N1, cudaMemcpyDeviceToHost);
  cudaMemcpy(mpiptr.dW2, bpgrads.dw2, sizeof(nn_real) * cl * num_nr, cudaMemcpyDeviceToHost);
  cudaMemcpy(mpiptr.db1, bpgrads.db1, sizeof(nn_real) * num_nr, cudaMemcpyDeviceToHost);
  cudaMemcpy(mpiptr.db2, bpgrads.db2, sizeof(nn_real) * cl, cudaMemcpyDeviceToHost);
}

void myReduceMPIandCPY(struct mpiptrs &mpiptr, struct gradsG &bpgrads, int num_nr, int cl, int N1)
{
  MPI_SAFE_CALL(MPI_Allreduce(mpiptr.dW1, mpiptr.dW1_total, num_nr * N1, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
  MPI_SAFE_CALL(MPI_Allreduce(mpiptr.dW2, mpiptr.dW2_total, cl * num_nr, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
  MPI_SAFE_CALL(MPI_Allreduce(mpiptr.db1, mpiptr.db1_total, num_nr, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
  MPI_SAFE_CALL(MPI_Allreduce(mpiptr.db2, mpiptr.db2_total, cl, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

  cudaMemcpy(bpgrads.dw1, mpiptr.dW1_total, sizeof(nn_real) * num_nr * N1, cudaMemcpyHostToDevice);
  cudaMemcpy(bpgrads.dw2, mpiptr.dW2_total, sizeof(nn_real) * cl * num_nr, cudaMemcpyHostToDevice);
  cudaMemcpy(bpgrads.db1, mpiptr.db1_total, sizeof(nn_real) * num_nr, cudaMemcpyHostToDevice);
  cudaMemcpy(bpgrads.db2, mpiptr.db2_total, sizeof(nn_real) * cl, cudaMemcpyHostToDevice);
}

void parallelFeedforward(struct NunetP &nnP, struct cacheG &cache, 
                    int num_nr, int N, int cl, int bs, nn_real *X_bc)
{
    //// Using shared memory algorithm with 16x4 block size
    ////changing for MPI opt
    int err = myGemNunetSharedB(nnP.w1, X_bc, nnP.b1, cache.z1, num_nr, bs, N);

    int err2 = mySigmoid(cache.z1, cache.a1, num_nr, bs);

    int err3 = myGemNunetSharedB(nnP.w2, cache.a1, nnP.b2, cache.z2, cl, bs, num_nr);
    // int err4 = gpuSoftmax(cache.z2, cache.a2, cl, bs);
    int err4 = gpuSoftmax(cache, cl, bs);
    
    // cache.yc = cache.a2;

}
void parallelFeedforward3(struct NunetP &nnP, struct cacheG &cache, 
                    int num_nr, int N, int cl, int bs)
{
    //// Using shared memory algorithm with 16x4 block size
    int err = myGemNunetSharedB(nnP.w1, nnP.x_batch, nnP.b1, cache.z1, num_nr, bs, N);

    int err2 = mySigmoid(cache.z1, cache.a1, num_nr, bs);

    int err3 = myGemNunetSharedB(nnP.w2, cache.a1, nnP.b2, cache.z2, cl, bs, num_nr);
    // int err4 = gpuSoftmax(cache.z2, cache.a2, cl, bs);
    int err4 = gpuSoftmax(cache, cl, bs);
    
    // cache.yc = cache.a2;

}
void parallelBackProp(struct NunetP &nnP, struct cacheG &cache, 
                    struct gradsG &bpgrads, nn_real reg, int num_nr,
                    int N, int cl, int cur_batch, int bs, nn_real *X_bc, nn_real *y_bc)
{
  int err = myBpStepW1(nnP, cache, bpgrads, reg, cl, num_nr, cur_batch, bs, y_bc);

  int err2 = myBpStepb1(bpgrads, cl, cur_batch);

  int err3 = myBpStepW0(nnP, cache, bpgrads, reg, cl, num_nr, N, cur_batch, X_bc);

  int err4 = myBpStepb0(cache, bpgrads, num_nr, cur_batch);
}
void parallelBackProp3(struct NunetP &nnP, struct cacheG &cache, 
                    struct gradsG &bpgrads, nn_real reg, int num_nr,
                    int N, int cl, int cur_batch, int bs)
{
  int err = myBpStepW1_3(nnP, cache, bpgrads, reg, cl, num_nr, cur_batch, bs);

  int err2 = myBpStepb1(bpgrads, cl, cur_batch);

  int err3 = myBpStepW0_3(nnP, cache, bpgrads, reg, cl, num_nr, N, cur_batch);

  int err4 = myBpStepb0(cache, bpgrads, num_nr, cur_batch);
}


void parallelGradientDes(struct NunetP &nnP, struct gradsG &bpgrads, 
                        nn_real learning_rate, int num_nr, int N, int cl)
{
  int err = myWeightUpdate(nnP, bpgrads, learning_rate, num_nr, N, cl);
  
  int err2 = myBiasUpdate(nnP, bpgrads, learning_rate, num_nr, cl);
}
void calcSendCt(int* sendcnX, int* sendcny, int* dispcnX, 
        int* dispcny, int bs, int num_procs, int N1, int cl)
{
  int buf;
  buf = get_mini_batch_size(bs, num_procs, 0);
  sendcnX[0] = N1*buf;
  sendcny[0] = cl*buf;
  dispcnX[0] = 0;
  dispcny[0] = 0;
  for (int ii=1; ii<num_procs; ++ii) {
    buf = get_mini_batch_size(bs, num_procs, ii);
    sendcnX[ii] = N1*buf;
    sendcny[ii] = cl*buf;
    dispcnX[ii] = sendcnX[ii-1] + dispcnX[ii-1];
    dispcny[ii] = sendcny[ii-1] + dispcny[ii-1];
  }
}
/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                    const arma::Mat<nn_real> &y, nn_real learning_rate,
                    std::ofstream &error_file, nn_real reg, const int epochs,
                    const int batch_size, int print_every, int debug)
{
  assert(learning_rate > 0);
  assert(reg >= 0);
  assert(epochs >= 0);
  assert(batch_size > 0);

  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0)
  {
    assert(X.n_cols > 0);
    assert(X.n_rows == IMAGE_SIZE);
    assert(y.n_cols == X.n_cols);
    assert(y.n_rows == NUM_CLASSES);
    assert(nn.H[0] == IMAGE_SIZE);
    assert(nn.H[2] == NUM_CLASSES);
  }

  int N = (rank == 0) ? X.n_cols : 0;

  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  assert(N > 0);

  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way using memptr().
     Or you can allocate your own array memory space and store the elements in a
     row major way. Remember to update the Armadillo matrices in NeuralNetwork
     &nn of rank 0 before returning from the function. */

  /* allocate memory before the iterations */
  // Data sets
  const int num_batches = get_num_batches(N, batch_size);
  int mini_batch_size_alloc;
  {
    const int max_batch_size = batch_size;
    mini_batch_size_alloc = max_batch_size / num_procs + 1;
  }
  // TODO

  int num_nr = nn.W[0].n_rows;
  int bs = batch_size;
  int cl = nn.W[1].n_rows;
  int N1 = nn.W[0].n_cols;
  struct NunetP nnP;
  struct cacheG bpcache;
  struct gradsG bpgrads;
  nn_real* X_b_all[num_batches];
  nn_real* y_b_all[num_batches];
  nn_real* X_cuda[num_batches];
  nn_real* y_cuda[num_batches];

  int mini_bs = get_mini_batch_size(batch_size, num_procs, rank);
  createGPuMem(nn, nnP, bpcache, bpgrads, num_nr, mini_batch_size_alloc, cl, N1);
  struct mpiptrs mpiptr;
  if (num_procs > proc_lim) {
    mpiptr.X_mini = (nn_real *)malloc(N1 * mini_bs * sizeof(nn_real));
    mpiptr.y_mini = (nn_real *)malloc(cl * mini_bs * sizeof(nn_real));
    mallocMPI(mpiptr, num_nr, cl, N1);
  } else {
    createGPuXy(X, y, nnP, cl, N1, N);
  }
  int sendcnX[num_procs]= { };
  int sendcny[num_procs]= { };
  int dispcnX[num_procs]= { };
  int dispcny[num_procs]= { };
  // for (int ii = 0; ii < num_procs; ++ii) {
  //   std::cout << "sendcnX " << sendcnX[ii]<< std::endl;
  //   std::cout << "sendcny " << sendcny[ii]<< std::endl;
  //   std::cout << "dispcnX " << dispcnX[ii]<< std::endl;
  //   std::cout << "dispcny " << dispcny[ii]<< std::endl;

  // }
  
  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;
  arma::Mat<nn_real> X_batch;
  arma::Mat<nn_real> y_batch;
  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;

    if (num_procs == 3) {calcSendCt(sendcnX, sendcny, dispcnX, dispcny, bs, num_procs, N1, cl);}

    for (int batch = 0; batch < num_batches; ++batch)
    {
      /*
       * Possible implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */
      
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                          FEED FORWARD                            //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      /* z1 = W1 x + b1 */
      // 99999999999999
      int cur_batch = get_batch_size(N, batch_size, batch);
      int last_col = batch_start + cur_batch;
      int sendcnt = get_mini_batch_size(cur_batch, num_procs, rank);
      
      if (num_procs == 3 && batch == num_batches - 1) {calcSendCt(sendcnX, sendcny, dispcnX, dispcny, cur_batch, num_procs, N1, cl);}
      
      if ((num_procs == 2 || num_procs==4) && rank==0 && epoch==0) {
        X_batch = X.cols(batch_start, last_col - 1);
        y_batch = y.cols(batch_start, last_col - 1);
      } else if (num_procs == 3 && rank==0) {
        X_batch = X.cols(batch_start, last_col - 1);
        y_batch = y.cols(batch_start, last_col - 1);
      }
      if (num_procs > proc_lim && num_procs != 3) {
        if (epoch==0) {
          X_b_all[batch] = (nn_real *)malloc(N1 * sendcnt * sizeof(nn_real));
          y_b_all[batch] = (nn_real *)malloc(cl * sendcnt * sizeof(nn_real));
          MPI_SAFE_CALL(MPI_Scatter(X_batch.memptr(), N1 * sendcnt, MPI_FP, 
                          X_b_all[batch], N1 * sendcnt, MPI_FP, ROOT, MPI_COMM_WORLD));
          MPI_SAFE_CALL(MPI_Scatter(y_batch.memptr(), cl * sendcnt, MPI_FP, 
                            y_b_all[batch], cl * sendcnt, MPI_FP, ROOT, MPI_COMM_WORLD));
          cudaMalloc((void **)&X_cuda[batch], sizeof(nn_real) * N1 * sendcnt);
          cudaMalloc((void **)&y_cuda[batch], sizeof(nn_real) * cl * sendcnt);
          cudaMemcpy(X_cuda[batch], X_b_all[batch], sizeof(nn_real) *  N1 * sendcnt, cudaMemcpyHostToDevice);
          cudaMemcpy(y_cuda[batch], y_b_all[batch], sizeof(nn_real) * cl * sendcnt, cudaMemcpyHostToDevice);
        } 
        
      } else if (num_procs == 3) {
        
        MPI_SAFE_CALL(MPI_Scatterv(X_batch.memptr(), sendcnX, dispcnX, MPI_FP, 
                          mpiptr.X_mini, N1 * sendcnt, MPI_FP, ROOT, MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Scatterv(y_batch.memptr(), sendcny, dispcny, MPI_FP, 
                          mpiptr.y_mini, cl * sendcnt, MPI_FP, ROOT, MPI_COMM_WORLD));

        cudaMemcpy(nnP.x_batch, mpiptr.X_mini, sizeof(nn_real) *  N1 * sendcnt, cudaMemcpyHostToDevice);
        cudaMemcpy(nnP.y_batch, mpiptr.y_mini, sizeof(nn_real) * cl * sendcnt, cudaMemcpyHostToDevice);
      } else {
        nnP.x_batch = nnP.X_all + batch*N1*batch_size;
        nnP.y_batch = nnP.y_all + batch*cl*batch_size;
      }
      
      // std::cout << "Before Parallel feed forward" << std::endl;
      if (num_procs == 2 || num_procs==4) {
        parallelFeedforward(nnP, bpcache, num_nr, N1, cl, sendcnt, X_cuda[batch]);
      } else {
        parallelFeedforward3(nnP, bpcache, num_nr, N1, cl, sendcnt);
      }
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         BACK PROPAGATE                           //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      /* db2 = y_hat - y */
      // std::cout << "Send cnt "<< sendcnt << std::endl;
      // std::cout << "cur_batch "<< cur_batch << std::endl;
      // std::cout << "num_procs "<< num_procs << std::endl;
      if (num_procs == 2 || num_procs==4) {
        parallelBackProp(nnP, bpcache, bpgrads, reg/num_procs, 
              num_nr, N1, cl, sendcnt, cur_batch, X_cuda[batch], y_cuda[batch]);
      } else {
        parallelBackProp3(nnP, bpcache, bpgrads, reg/num_procs, 
              num_nr, N1, cl, sendcnt, cur_batch);
      }
      
      if (num_procs > proc_lim) {
        interMemCpy(bpgrads, mpiptr, num_nr, cl, N1);
        myReduceMPIandCPY(mpiptr, bpgrads, num_nr, cl, N1);
      }
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      parallelGradientDes(nnP, bpgrads, learning_rate, num_nr, N1, cl);
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      /* Following debug routine assumes that you have already updated the arma
         matrices in the NeuralNetwork nn.  */
      // if (debug && rank == 0 && print_flag)
      // {
      //   // TODO
      //   // Copy data back to the CPU
        
      //   /* The following debug routine assumes that you have already updated the
      //    arma matrices in the NeuralNetwork nn.  */

      //   save_gpu_error(nn, iter, error_file);
      // }
      batch_start = last_col;
      iter++;
    }
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  cudaMemcpy(nn.W[0].memptr(), nnP.w1, sizeof(nn_real) *  num_nr * N1, cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), nnP.w2, sizeof(nn_real) *  cl * num_nr, cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[0].memptr(), nnP.b1, sizeof(nn_real) *  num_nr, cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), nnP.b2, sizeof(nn_real) *  cl, cudaMemcpyDeviceToHost);
  // myPrintMat(nn.W[0], "Weight 0 ");
  // myPrintMat(nn.W[1], "Weight 1 ");
  // myPrintMat(nn.b[0], "Bias 0 ");
  // myPrintMat(nn.b[1], "Bias 1 ");
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  destroyGPuMem(nnP, bpcache, bpgrads);
  destroyGPuXy(nnP);
  if (num_procs > proc_lim) destroyCPuMem(mpiptr); //, X_b_all, y_b_all, num_batches
  if (num_procs == 2 || num_procs==4) {
    for (int ii=0; ii<num_batches; ++ii) { 
      free(X_b_all[ii]);
      free(y_b_all[ii]); 
      cudaFree(X_cuda[ii]);
      cudaFree(y_cuda[ii]);
    }
  }
  
}

int compareMatrices(nn_real *myC, nn_real *refC, int NI, int NJ)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(myC, NI, NJ, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(refC, NI, NJ, false);

  for (i = mysol.n_rows*mysol.n_cols-2; i < mysol.n_rows*mysol.n_cols; ++i) { //mysol.n_rows*mysol.n_cols - 2
    std::cout << "refC " << refsol(i)<< std::endl;
    std::cout << "myC " << mysol(i)<< std::endl;
  }
  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My Vec output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "Vec matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}

