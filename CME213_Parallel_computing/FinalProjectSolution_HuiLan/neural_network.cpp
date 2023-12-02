#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

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

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */

struct GPU_cache{
 nn_real *W1, *W2, *b1, *b2;
 nn_real *a1, *y_pred;
  /*
  M:number of featrues,
  H:number of neurons in hidden layer,
  C:number of classes, 10
  */
 GPU_cache(int M, int H, int C, int batch_size){
   cudaMalloc((void**)&W1,sizeof(nn_real)*H*M);
   cudaMalloc((void**)&W2,sizeof(nn_real)*C*H);
   cudaMalloc((void**)&b1,sizeof(nn_real)*H);
   cudaMalloc((void**)&b2,sizeof(nn_real)*C);
   cudaMalloc((void**)&a1,sizeof(nn_real)*H*batch_size);
   cudaMalloc((void**)&y_pred,sizeof(nn_real)*C*batch_size);
 }

 ~GPU_cache(){
   cudaFree(W1);
   cudaFree(W2);
   cudaFree(b1);
   cudaFree(b2);
   cudaFree(a1);
   cudaFree(y_pred);
 }
};

struct GPU_grad{
 nn_real *dW1, *dW2, *db1, *db2;
 nn_real *da1, *dz1, *diff;
  /*
  M:number of featrues,
  H:number of neurons in hidden layer,
  C:number of classes, 10
  */
 GPU_grad(int M, int H, int C, int batch_size){
   cudaMalloc((void**)&diff,sizeof(nn_real)*C*batch_size);
   cudaMalloc((void**)&dW1,sizeof(nn_real)*H*M);
   cudaMalloc((void**)&dW2,sizeof(nn_real)*C*H);
   cudaMalloc((void**)&db1,sizeof(nn_real)*H);
   cudaMalloc((void**)&db2,sizeof(nn_real)*C);
   cudaMalloc((void**)&da1,sizeof(nn_real)*H*batch_size);
   cudaMalloc((void**)&dz1,sizeof(nn_real)*H*batch_size);
 }

 ~GPU_grad(){
   cudaFree(diff);
   cudaFree(dW1);
   cudaFree(dW2);
   cudaFree(db1);
   cudaFree(db2);
   cudaFree(da1);
   cudaFree(dz1);
 }
};

struct CPU_grad{
 nn_real *dW1, *dW2, *db1, *db2;

 CPU_grad(int M, int H, int C){
  dW1 = (nn_real *)malloc(H * M * sizeof(nn_real));
  dW2 = (nn_real *)malloc(C * H * sizeof(nn_real));
  db1 = (nn_real *)malloc(H * sizeof(nn_real));
  db2 = (nn_real *)malloc(C * sizeof(nn_real));
 }

 ~CPU_grad(){
  free(dW1);
  free(dW2);
  free(db1);
  free(db2);
 }

};

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
  int M = nn.H[0];
  int H = nn.H[1];
  int C = nn.H[2];

  std::vector<nn_real *> X_batches(num_batches);
  std::vector<nn_real *> Y_batches(num_batches);

  for (int batch = 0; batch < num_batches; ++batch) {
      int this_batch_size = std::min((batch + 1) * batch_size - 1, N - 1) - batch*batch_size + 1;
      int nsample_per_proc = (this_batch_size + num_procs -1) / num_procs;

      int counts_X[num_procs], counts_Y[num_procs], displs_X[num_procs], displs_Y[num_procs];

      for(int i = 0; i < num_procs; i++){
        counts_X[i] = M*std::min(nsample_per_proc,this_batch_size - i*nsample_per_proc);
        counts_Y[i] = C*std::min(nsample_per_proc,this_batch_size - i*nsample_per_proc);
        displs_X[i] = i*M*nsample_per_proc;
        displs_Y[i] = i*C*nsample_per_proc;
      }

      arma::Mat<nn_real> X_batch(M,counts_X[rank]/M);
      MPI_SAFE_CALL(MPI_Scatterv(X.colptr(batch*batch_size),counts_X,displs_X,MPI_FP,X_batch.memptr(),
      counts_X[rank],MPI_FP,0,MPI_COMM_WORLD));
      
      arma::Mat<nn_real> Y_batch(C,counts_Y[rank]/C);
      MPI_SAFE_CALL(MPI_Scatterv(y.colptr(batch*batch_size),counts_Y,displs_Y,MPI_FP,Y_batch.memptr(),
      counts_Y[rank],MPI_FP,0,MPI_COMM_WORLD));

      cudaMalloc((void **)&X_batches[batch], counts_X[rank] * sizeof(nn_real));
      cudaMalloc((void **)&Y_batches[batch], counts_Y[rank] * sizeof(nn_real));
      cudaMemcpy(X_batches[batch], X_batch.memptr(), counts_X[rank]* sizeof(nn_real), cudaMemcpyHostToDevice);
      cudaMemcpy(Y_batches[batch], Y_batch.memptr(), counts_Y[rank] * sizeof(nn_real), cudaMemcpyHostToDevice); 
  }


  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  GPU_cache gpu_cache(M,H,C,batch_size);
  GPU_grad gpu_grad(M,H,C,batch_size);
  CPU_grad cpu_grad(M,H,C);
  cudaMemcpy(gpu_cache.W1, nn.W[0].memptr(),H*M*sizeof(nn_real),cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_cache.W2, nn.W[1].memptr(),C*H*sizeof(nn_real),cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_cache.b1, nn.b[0].memptr(),H*sizeof(nn_real),cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_cache.b2, nn.b[1].memptr(),C*sizeof(nn_real),cudaMemcpyHostToDevice);

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
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
      int start_col = batch*batch_size;
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      int this_batch_size = last_col - start_col + 1;
      int mini_batch_size = (this_batch_size + num_procs -1) / num_procs;
      int this_mini_batch = std::min(mini_batch_size,this_batch_size-rank*mini_batch_size);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                          FEED FORWARD                            //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      nn_real alpha = 1.0, beta = 1.0;
      gpu_repmat(gpu_cache.b1, gpu_cache.a1, H, this_mini_batch);
      myGEMM(gpu_cache.W1, X_batches[batch] , gpu_cache.a1, alpha, beta, H, this_mini_batch, M );
      gpu_sigmoid(gpu_cache.a1,H,this_mini_batch);
      gpu_repmat(gpu_cache.b2, gpu_cache.y_pred, C, this_mini_batch);
      myGEMM(gpu_cache.W2, gpu_cache.a1, gpu_cache.y_pred, alpha, beta, C, this_mini_batch, H);
      gpu_softmax(gpu_cache.y_pred, C, this_mini_batch);


      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         BACK PROPAGATE                           //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      nn_real ratio = 1.0/(nn_real)this_batch_size;
      reg = reg /num_procs;
      gpu_addmat(gpu_cache.y_pred,Y_batches[batch],gpu_grad.diff, ratio, -ratio, C, this_mini_batch);
      //gpu_fc_backprop(gpu_cache.W2, gpu_grad.dW2, gpu_grad.db2, gpu_grad.diff, gpu_cache.a1, reg, C, H, this_mini_batch);
      gpu_copy(gpu_grad.dW2,gpu_cache.W2,C,H);
      myGEMMT(gpu_grad.diff,gpu_cache.a1,gpu_grad.dW2,alpha,reg,C,H,this_mini_batch,false,true);
      gpu_row_sum(gpu_grad.diff, gpu_grad.db2, C, this_mini_batch);

      beta = 0.0;
      myGEMMT(gpu_cache.W2,gpu_grad.diff,gpu_grad.da1,alpha,beta,H,this_mini_batch,C,true,false);
      gpu_sigmoid_backprop(gpu_grad.da1,gpu_cache.a1,gpu_grad.dz1,H,this_mini_batch);
      //gpu_fc_backprop(gpu_cache.W1, gpu_grad.dW1,gpu_grad.db1, gpu_grad.dz1,X_batches[batch],reg, H, M, this_mini_batch);
      gpu_copy(gpu_grad.dW1,gpu_cache.W1,H,M);
      myGEMMT(gpu_grad.dz1, X_batches[batch], gpu_grad.dW1, alpha, reg, H, M, this_mini_batch,false,true);
      gpu_row_sum(gpu_grad.dz1, gpu_grad.db1, H, this_mini_batch);


      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      cudaMemcpy( cpu_grad.dW1, gpu_grad.dW1, H * M * sizeof(nn_real), cudaMemcpyDeviceToHost);
      cudaMemcpy( cpu_grad.dW2, gpu_grad.dW2, C * H * sizeof(nn_real), cudaMemcpyDeviceToHost);
      cudaMemcpy( cpu_grad.db1, gpu_grad.db1, H * sizeof(nn_real), cudaMemcpyDeviceToHost);
      cudaMemcpy( cpu_grad.db2, gpu_grad.db2, C * sizeof(nn_real), cudaMemcpyDeviceToHost);

      // reduce
      // arma::Mat<nn_real> dW1(size(nn.W[0]), arma::fill::zeros);
      // MPI_SAFE_CALL(MPI_Allreduce( cpu_grad.dW1, dW1.memptr(), H * M, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      // arma::Mat<nn_real> dW2(size(nn.W[1]), arma::fill::zeros);
      // MPI_SAFE_CALL(MPI_Allreduce( cpu_grad.dW2, dW2.memptr(), C * H, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      // arma::Col<nn_real> db1(size(nn.b[0]), arma::fill::zeros);
      // MPI_SAFE_CALL(MPI_Allreduce( cpu_grad.db1, db1.memptr(), H, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      // arma::Col<nn_real> db2(size(nn.b[1]), arma::fill::zeros);
      // MPI_SAFE_CALL(MPI_Allreduce( cpu_grad.db2, db2.memptr(), C, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_grad.dW1, H * M, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_grad.dW2, C * H, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_grad.db1, H, MPI_FP, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_grad.db2, C, MPI_FP, MPI_SUM, MPI_COMM_WORLD));

      // cudaMemcpy(gpu_grad.dW1, dW1.memptr(), H * M * sizeof(nn_real), cudaMemcpyHostToDevice);
      // cudaMemcpy(gpu_grad.dW2, dW2.memptr(), C * H * sizeof(nn_real), cudaMemcpyHostToDevice);
      // cudaMemcpy(gpu_grad.db1, db1.memptr(), H * sizeof(nn_real), cudaMemcpyHostToDevice);
      // cudaMemcpy(gpu_grad.db2, db2.memptr(), C * sizeof(nn_real), cudaMemcpyHostToDevice);   
      cudaMemcpy(gpu_grad.dW1, cpu_grad.dW1, H * M * sizeof(nn_real), cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_grad.dW2, cpu_grad.dW2, C * H * sizeof(nn_real), cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_grad.db1, cpu_grad.db1, H * sizeof(nn_real), cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_grad.db2, cpu_grad.db2, C * sizeof(nn_real), cudaMemcpyHostToDevice);       
      
      gpu_addmat(gpu_cache.W1,gpu_grad.dW1,gpu_cache.W1,1.0,-learning_rate,H,M);
      gpu_addmat(gpu_cache.W2,gpu_grad.dW2,gpu_cache.W2,1.0,-learning_rate,C,H);
      gpu_addmat(gpu_cache.b1,gpu_grad.db1,gpu_cache.b1,1.0,-learning_rate,H,1);
      gpu_addmat(gpu_cache.b2,gpu_grad.db2,gpu_cache.b2,1.0,-learning_rate,C,1);

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
      if (debug && rank == 0 && print_flag)
      {
        // TODO
        // Copy data back to the CPU
        cudaMemcpy(nn.W[0].memptr(), gpu_cache.W1, H * M * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), gpu_cache.W2, C * H * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), gpu_cache.b1, H * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), gpu_cache.b2, C * sizeof(nn_real), cudaMemcpyDeviceToHost);

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */

        save_gpu_error(nn, iter, error_file);
      }

      iter++;
    }
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  cudaMemcpy(nn.W[0].memptr(), gpu_cache.W1, H * M * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), gpu_cache.W2, C * H * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[0].memptr(), gpu_cache.b1, H * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), gpu_cache.b2, C * sizeof(nn_real), cudaMemcpyDeviceToHost);

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  for(int batch = 0; batch < num_batches; ++batch) {
    cudaFree(X_batches[batch]);
    cudaFree(Y_batches[batch]);
  }

}
