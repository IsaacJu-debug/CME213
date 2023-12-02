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

  //check soft max compare results of CPU and GPU
    std::cout<<"test print in softmax"<<std::endl;
  // mat is original 
  int M = mat.n_rows;
  int N = mat.n_cols;
  //host mat2 test
  nn_real *hmat2;
  //device mat2
  nn_real *dmat2;
  nn_real *mat2_ref;

  hmat2 = (nn_real *)malloc(M * N * sizeof(nn_real));
  //initialize hmat2
  for (int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      hmat2[i + j*M] = mat[i + j*M];
    }
  }
  //memory allocation
  cudaMalloc((void **)&dmat2, sizeof(nn_real) * M * N);
  //memory copy
  cudaMemcpy(dmat2, hmat2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  // //
  //block size is less than 1024
  int num_blockDim_x = 32;
  int num_blockDim_y = 32;

  // calculate grid dimension based on the block size and matrix size
  // num_gridDim_x 
  int num_gridDim_x = (int) (N/num_blockDim_x) + 1;
  int num_gridDim_y = (int) (M/num_blockDim_y) + 1;

  dim3 blocks(num_gridDim_x, num_gridDim_y);
  dim3 threads(num_blockDim_x, num_blockDim_y);

  // int result_mySOFTMAX = mySOFTMAX(dmat2, M, N);
  // cudaMemcpy(hmat2, dmat2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);
  // std::cout<<"softmax kernel"<<std::endl;
  // //compare hmat2(GPU) and mat2(serial)
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //     std::cout<<"row"<<i<<"col"<<j<<std::endl;
  //     // //compare element by element
  //     nn_real mat_ij = mat(i, j);
  //     std::cout<<"mat[i][j]="<<mat_ij<<std::endl;
  //     nn_real mat2_ij = mat2(i, j);
  //     std::cout<<"cpu mat2[i][j]="<<mat2_ij<<std::endl;
  //     nn_real dmat2_ij = hmat2[i + j*M];
  //     std::cout<<"gpu dmat2[i][j]="<<dmat2_ij<<std::endl;
  //     // std::cout<<"row"<<i<<"col"<<j<<std::endl;
  //     // std::cout<<"hmat2_ij="<<hmat2_ij<<", "<<"dmat2_ij="<<dmat2_ij<<std::endl;
  //     std::cout<<"Absolute difference: "<<abs(dmat2_ij - mat2_ij)<<std::endl;
  //   }
  // }
  free(hmat2);
  cudaFree(dmat2);
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

void backprop_add(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads, nn_real *diff_save)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  int M = y.n_rows;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      diff_save[i + j*M] = diff(i, j);
    }
  }
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

      // if (print_every > 0 && iter % print_every == 0)
      // {
      //   if (grad_check)
      //   {
      //     struct grads numgrads;
      //     numgrad(nn, X_batch, y_batch, reg, numgrads);
      //     assert(gradcheck(numgrads, bpgrads));
      //   }

      //   std::cout << "Loss at iteration " << iter << " of epoch " << epoch
      //             << "/" << epochs << " = "
      //             << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      // }

      // // Gradient descent step
      // for (int i = 0; i < nn.W.size(); ++i)
      // {
      //   nn.W[i] -= learning_rate * bpgrads.dW[i];
      // }

      // for (int i = 0; i < nn.b.size(); ++i)
      // {
      //   nn.b[i] -= learning_rate * bpgrads.db[i];
      // }

      // /* Debug routine runs only when debug flag is set. If print_every is zero,
      //    it saves for the first batch of each epoch to avoid saving too many
      //    large files. Note that for the first time, you have to run debug and
      //    serial modes together. This will run the following function and write
      //    out files to the "cpu_save_dir" folder.
      //    In the later runs (with same parameters), you can use just the debug
      //    flag to output diff b/w CPU and GPU without running the CPU version
      //    version. */
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


//add
void rowsum_CPU(nn_real* X, int M, int N) {
  int i = 0;
  nn_real* vec_rowsum_X;
  vec_rowsum_X = (nn_real *)malloc(N * sizeof(nn_real));

  for (int j = 0; j < N; j++) {
    vec_rowsum_X[j] = 0;
  }

  for(int j = 0; j < N; j++) {
    nn_real sum_col_j = 0;
    for (int i = 0; i < M; i++) {
      sum_col_j = sum_col_j + X[i + j*M];
    }
    vec_rowsum_X[j] = sum_col_j;
  }

  for(int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      X[i + j*M] = vec_rowsum_X[j];
    } 
  }
}

void expand_b_CPU(nn_real* b_expand, arma::Col<nn_real> b, int N) {
  int len_b = b.n_rows;
  for (int i = 0; i < len_b; i++) {
    for (int j = 0; j < N; j++) {
      b_expand[i + j*len_b] = b(i);
    }
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

  //add 

  int mini_batch_size = get_mini_batch_size(batch_size, num_procs, rank);

  std::cout<<"batch_size="<<batch_size<<std::endl;
  std::cout<<"mini_batch_size="<<mini_batch_size<<std::endl;

  // mini_batch_size_alloc is enough?
  // mini_batch_size_alloc for X.n_cols and y.n_cols

  printf("Test memory allocation starts.\n");

  // weight of first layer
  arma::Mat<nn_real> W_1 = nn.W[0];
  std::cout<<"row of W_1:"<<W_1.n_rows;
  std::cout<<"col of W_1:"<<W_1.n_cols<<std::endl;
  int col_W_1 = nn.H[0];
  int row_W_1 = nn.H[1];

  //weight of the second layer
  arma::Mat<nn_real> W_2 = nn.W[1];
  std::cout<<"row of W_2:"<<W_2.n_rows;
  std::cout<<"col of W_2:"<<W_2.n_cols<<std::endl;
  int col_W_2 = nn.H[1];
  int row_W_2 = nn.H[2];

  // first layer b
  arma::Col<nn_real> b_1 = nn.b[0];
  int len_b_1 = nn.H[1];



  // second layer b
  arma::Col<nn_real> b_2 = nn.b[1];
  int len_b_2 = nn.H[2];

  // size of X
  int row_X = nn.H[0];
  int col_X = mini_batch_size;

  // size of y
  int row_y = y.n_rows;
  int col_y = mini_batch_size;


  // size of z_1: row size: row_W_1, col size: mini_batch_size_alloc
  // create space for z_1
  nn_real* X_copy;
  X_copy = (nn_real *)malloc(X.n_rows * mini_batch_size* sizeof(nn_real));


  nn_real* z_1;
  int row_z_1 = row_W_1;
  int col_z_1 = mini_batch_size;
  z_1 = (nn_real *)malloc(row_z_1 * col_z_1* sizeof(nn_real));


  // a_1 z_1 size row size: row_W_1, col size: mini_batch_size_alloc
  nn_real* a_1;
  int row_a_1 = row_W_1;
  int col_a_1 = mini_batch_size;
  a_1 = (nn_real *)malloc(row_a_1 * col_a_1* sizeof(nn_real));

  // z_2
  //size of z_2: row size: row_W_2, col size: mini_batch_size_alloc
  nn_real* z_2;
  int row_z_2 = row_W_2;
  int col_z_2 = mini_batch_size;
  z_2 = (nn_real *)malloc(row_z_2 * col_z_2* sizeof(nn_real));

  // z_2 intermediate
  //size of z_2: row size: row_W_2, col size: mini_batch_size_alloc
  nn_real* z_2_inter;
  int row_z_2_inter = row_W_2;
  int col_z_2_inter = mini_batch_size;
  z_2_inter = (nn_real *)malloc(row_z_2_inter * col_z_2_inter* sizeof(nn_real));

  // a_2
  nn_real* a_2;
  int row_a_2 = row_W_2;
  int col_a_2 = mini_batch_size;
  a_2 = (nn_real *)malloc(row_a_2 * col_a_2* sizeof(nn_real));

  //back propagation
  // y_hat
  nn_real* y_hat;
  int row_y_hat = row_W_2;
  int col_y_hat = mini_batch_size;
  y_hat = (nn_real *)malloc(row_y_hat * col_y_hat* sizeof(nn_real));

  // grad_W_1
  nn_real* grad_W_1;
  int row_grad_W_1 = row_W_1;
  int col_grad_W_1 = col_y_hat;
  grad_W_1 = (nn_real *)malloc(row_a_2 * col_a_2* sizeof(nn_real));

  // grad_W_1
  nn_real* grad_a_1;
  int row_grad_a_1 = row_W_1;
  int col_grad_a_1 = col_y_hat;
  grad_a_1 = (nn_real *)malloc(row_grad_a_1 * col_grad_a_1* sizeof(nn_real));

  // grad_z_1
  nn_real* grad_z_1;
  int row_grad_z_1 = row_W_1;
  int col_grad_z_1 = col_grad_a_1;
  grad_z_1 = (nn_real *)malloc(row_grad_z_1 * col_grad_z_1* sizeof(nn_real));

  // grad_z_1
  nn_real* grad_b_1_matrix;
  int row_grad_b_1_matrix = row_W_1;
  int col_grad_b_1_matrix = col_grad_a_1;
  grad_b_1_matrix = (nn_real *)malloc(row_grad_b_1_matrix * col_grad_b_1_matrix* sizeof(nn_real));


  // grad_b_1
  nn_real* grad_b_1;
  int row_grad_b_1 = row_W_1;
  int col_grad_b_1 = 1;
  grad_b_1 = (nn_real *)malloc(row_grad_b_1 * col_grad_b_1* sizeof(nn_real));

  // grad_W_2
  nn_real* grad_W_2;
  int row_grad_W_2 = row_W_2;
  int col_grad_W_2 = col_y_hat;
  grad_W_2 = (nn_real *)malloc(row_grad_W_2 * col_grad_W_2* sizeof(nn_real));

  // grad_b_2_matrix
  nn_real* grad_b_2_matrix;
  int row_grad_b_2_matrix = row_W_2;
  int col_grad_b_2_matrix = col_y_hat;
  grad_b_2_matrix = (nn_real *)malloc(row_grad_b_2_matrix * col_grad_b_2_matrix* sizeof(nn_real));
  //initialization



  // grad_b_2
  nn_real* grad_b_2;
  int row_grad_b_2 = row_W_2;
  int col_grad_b_2 = 1;
  grad_b_2 = (nn_real *)malloc(row_grad_b_2 * col_grad_b_2* sizeof(nn_real));





  //rowsum_X
  nn_real* rowsum_exp_z_2;
  int row_rowsum_exp_z_2 = row_W_2;
  int col_rowsum_exp_z_2 = mini_batch_size;
  rowsum_exp_z_2 = (nn_real *)malloc(row_rowsum_exp_z_2 * col_rowsum_exp_z_2* sizeof(nn_real));

  //b_1_matrix = b_1*ones(1, mini)batch size)
  nn_real* b_1_matrix;
  int row_b_1_matrix = len_b_1;
  int col_b_1_matrix = mini_batch_size;
  b_1_matrix = (nn_real *)malloc(row_b_1_matrix * col_b_1_matrix* sizeof(nn_real));
  // assign value to b_1_matrix
  for (int i = 0; i < row_b_1_matrix; i++) {
    for (int j = 0; j < col_b_1_matrix; j++) {
      b_1_matrix[i+j*row_b_1_matrix] = b_1(i);
    }
  }


  //b_2_matrix
  nn_real* b_2_matrix;
  int row_b_2_matrix = len_b_2;
  int col_b_2_matrix = mini_batch_size;
  b_2_matrix = (nn_real *)malloc(row_b_2_matrix * col_b_2_matrix* sizeof(nn_real));
  // assign value to b_2_matrix
  for (int i = 0; i < row_b_2_matrix; i++) {
    for (int j = 0; j < col_b_2_matrix; j++) {
      b_2_matrix[i+j*row_b_2_matrix] = b_2(i);
    }
  }

  nn_real *X_batch_CPU;
  X_batch_CPU = (nn_real *)malloc(X.n_rows * mini_batch_size* sizeof(nn_real));
  nn_real *y_batch_CPU;
  y_batch_CPU = (nn_real *)malloc(y.n_rows * mini_batch_size* sizeof(nn_real));


  // create pointer for device memory
  nn_real *dW_1;
  nn_real *dW_2;
  nn_real *db_1;
  nn_real *db_2;
  nn_real *dX;
  nn_real *dy;
  //back propagation
  nn_real *dy_hat;
  nn_real *dgrad_W_1;
  nn_real *dgrad_z_1;
  nn_real *dgrad_a_1;
  nn_real *dgrad_b_1;
  nn_real *dgrad_b_1_matrix;
  nn_real *dgrad_W_2;
  nn_real *dgrad_b_2;
  nn_real *dgrad_b_2_matrix;


  
  //
  nn_real *dz_1;
  nn_real *da_1;
  nn_real *dz_2;
  nn_real *dz_2_inter;
  nn_real *da_2;
  nn_real *drowsum_exp_z_2;

  //cuda memory allocation on device
  cudaMalloc((void **)&dW_1, sizeof(nn_real) * col_W_1* row_W_1);
  cudaMalloc((void **)&dW_2, sizeof(nn_real) * col_W_2* row_W_2);
  cudaMalloc((void **)&db_1, sizeof(nn_real) * len_b_1);
  cudaMalloc((void **)&db_2, sizeof(nn_real) * len_b_2);
  cudaMalloc((void **)&dX, sizeof(nn_real) * col_X * row_X);
  cudaMalloc((void **)&dy, sizeof(nn_real) * col_y * row_y);
  cudaMalloc((void **)&dz_1, sizeof(nn_real) * col_z_1 * row_z_1);
  cudaMalloc((void **)&da_1, sizeof(nn_real) * col_a_1 * row_a_1);
  cudaMalloc((void **)&dz_2, sizeof(nn_real) * col_z_2 * row_z_2);
  cudaMalloc((void **)&dz_2_inter, sizeof(nn_real) * col_z_2 * row_z_2);
  cudaMalloc((void **)&da_2, sizeof(nn_real) * col_a_2 * row_a_2); 
  cudaMalloc((void **)&drowsum_exp_z_2, sizeof(nn_real) * col_rowsum_exp_z_2 * row_rowsum_exp_z_2);
   

  cudaMemcpy(dW_1, &W_1[0], sizeof(nn_real) * col_W_1* row_W_1, cudaMemcpyHostToDevice);
  cudaMemcpy(dW_2, &W_2[0], sizeof(nn_real) * col_W_2* row_W_2, cudaMemcpyHostToDevice);
  cudaMemcpy(db_1, &b_1[0], sizeof(nn_real) * len_b_1, cudaMemcpyHostToDevice);
  cudaMemcpy(db_2, &b_2[0], sizeof(nn_real) * len_b_2, cudaMemcpyHostToDevice);

  cudaMemcpy(dz_1, &b_1_matrix[0], sizeof(nn_real) * col_z_1 * row_z_1, cudaMemcpyHostToDevice);
  cudaMemcpy(da_1, &a_1[0], sizeof(nn_real) * col_a_1 * row_a_1, cudaMemcpyHostToDevice);
  cudaMemcpy(dz_2, &b_2_matrix[0], sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyHostToDevice);
  cudaMemcpy(da_2, &a_2[0], sizeof(nn_real) * col_a_2 * row_a_2, cudaMemcpyHostToDevice);
  
  // TODO
  // memory back propagation
    //cuda memory allocation on device
  cudaMalloc((void **)&dgrad_W_1, sizeof(nn_real) * col_grad_W_1* row_grad_W_1);
  cudaMalloc((void **)&dgrad_z_1, sizeof(nn_real) * col_grad_z_1* row_grad_z_1);
  cudaMalloc((void **)&dgrad_a_1, sizeof(nn_real) * col_grad_a_1* row_grad_a_1);
  cudaMalloc((void **)&dgrad_W_2, sizeof(nn_real) * col_grad_W_2* row_grad_W_2);
  cudaMalloc((void **)&dgrad_b_1_matrix, sizeof(nn_real) * col_grad_b_1_matrix* row_grad_b_1_matrix);
  cudaMalloc((void **)&dgrad_b_1, sizeof(nn_real) * col_grad_b_1* row_grad_b_1);
  cudaMalloc((void **)&dgrad_b_2_matrix, sizeof(nn_real) * col_grad_b_2_matrix* row_grad_b_2_matrix);
  cudaMalloc((void **)&dgrad_b_2, sizeof(nn_real) * col_grad_b_2* row_grad_b_2);
  cudaMalloc((void **)&dy_hat, sizeof(nn_real) * col_y_hat* row_y_hat);

  cudaMemcpy(dgrad_W_1, &grad_W_1[0], sizeof(nn_real) * col_grad_W_1* row_grad_W_1, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_z_1, &grad_z_1[0], sizeof(nn_real) * col_grad_z_1* row_grad_z_1, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_a_1, &grad_a_1[0], sizeof(nn_real) * col_grad_a_1* row_grad_a_1, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_W_2, &grad_W_2[0], sizeof(nn_real) * col_grad_W_2* row_grad_W_2, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_b_1, &grad_b_1[0], sizeof(nn_real) * col_grad_b_1* row_grad_b_1, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_b_2, &grad_b_2[0], sizeof(nn_real) * col_grad_b_2* row_grad_b_2, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_b_1_matrix, &grad_b_1_matrix[0], sizeof(nn_real) * col_grad_b_1_matrix* row_grad_b_1_matrix, cudaMemcpyHostToDevice);
  cudaMemcpy(dgrad_b_2_matrix, &grad_b_2_matrix[0], sizeof(nn_real) * col_grad_b_2_matrix* row_grad_b_2_matrix, cudaMemcpyHostToDevice);
  cudaMemcpy(dy_hat, &y_hat[0], sizeof(nn_real) * col_y_hat* row_y_hat, cudaMemcpyHostToDevice);
  printf("Test memory allocation ends.\n");

  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;
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
      int last_col = batch_start + mini_batch_size;
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      if (batch == num_batches - 1)
      {
        assert(last_col == X.n_cols);
      }
      //
      std::cout<<"Initializes X_batch"<<std::endl;
      std::cout<<"batch_start="<<batch_start<<std::endl;
      std::cout<<"last_col-1="<<last_col-1<<std::endl;
      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);
      
      // arma::Mat<nn_real> X_batch = X.colptr(batch_start*mini_batch_size);


      //head of X and y
      std::cout<<"row_X="<<row_X<<std::endl;
      std::cout<<"col_X="<<col_X<<std::endl;
      //X_batch_CPU
      memcpy(X_batch_CPU, X.colptr(batch_start*mini_batch_size), sizeof(nn_real) * col_X * row_X);
      memcpy(y_batch_CPU, y.colptr(batch_start*mini_batch_size), sizeof(nn_real) * col_y * row_y);
      // copy from CPU to GPU dX
      cudaMemcpy(dX, X_batch_CPU, sizeof(nn_real) * col_X * row_X, cudaMemcpyHostToDevice);
      cudaMemcpy(dy, y_batch_CPU, sizeof(nn_real) * col_y * row_y, cudaMemcpyHostToDevice);
      
      // // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      // //                          FEED FORWARD                            //
      // // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      /* z1 = W1 x + b1 */
      printf("Test feedforward_parallel starts.\n");

      // 1. z_1 = W_1X+ b_1
      myGEMM(dW_1, dX, dz_1,
           1, 1,
           row_W_1, col_X, row_X);
      // 2. a_1 = sigmoid(z_1)
      mySIGMOID(da_1, dz_1, row_W_1, col_X);
      // 3. z_2 = W_2a_1 + b_2
      myGEMM(dW_2, da_1, dz_2,
           1, 1,
           row_W_2, col_X, row_W_1);
      // 4. a_2 = softmax(z_2)
      // calculate rowsum_exp_X in CPU
      cudaMemcpy(z_2, dz_2, sizeof(nn_real) * col_z_2_inter * row_z_2_inter, cudaMemcpyDeviceToHost);
      //dz_2 to dz_2_inter
      for (int i = 0; i < row_z_2; i++) {
        for (int j = 0; j < col_z_2; j++) {
          z_2_inter[i + j*row_z_2] = z_2[i + j*row_z_2];
        }
      }
      // initialize matrix rowsum_exp_X
      for (int m = 0; m < row_z_2; m++) {
        for (int n = 0; n < col_z_2; n++) {
          rowsum_exp_z_2[m+n*row_z_2] = z_2_inter[m + n*row_z_2];
        }
      }
      cudaMemcpy(dz_2_inter, z_2_inter, sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyHostToDevice);
      cudaMemcpy(drowsum_exp_z_2, &rowsum_exp_z_2[0], sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyHostToDevice);
      // calculate exponent of X in GPU
      int val = myEXP(drowsum_exp_z_2, row_z_2_inter, col_z_2_inter);
      // copy back in CPU
      cudaMemcpy(rowsum_exp_z_2, drowsum_exp_z_2, sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyDeviceToHost);
      // calculate row sum i CPU
      // rowsum_CPU(rowsum_exp_z_2, row_z_2, col_z_2);
      nn_real* vec_rowsum_X;
      vec_rowsum_X = (nn_real *)malloc(col_z_2 * sizeof(nn_real));
      for (int j = 0; j < col_z_2; j++) {
        vec_rowsum_X[j] = 0;
      }

      for(int j = 0; j < col_z_2; j++) {
        nn_real sum_col_j = 0;
        for (int i = 0; i < row_z_2; i++) {
          sum_col_j = sum_col_j + rowsum_exp_z_2[i + j*row_z_2];
        }
        vec_rowsum_X[j] = sum_col_j;
      }

      for(int i = 0; i < row_z_2; i++) {
        for (int j = 0; j < col_z_2; j++) {
          rowsum_exp_z_2[i + j*row_z_2] = vec_rowsum_X[j];
        } 
      }

      cudaMemcpy(drowsum_exp_z_2, &rowsum_exp_z_2[0], sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyHostToDevice);
      // mySOFTMAX(da_2, dz_2_inter, drowsum_exp_z_2, row_W_2, col_X);
      mySOFTMAX(da_2, dz_2, drowsum_exp_z_2, row_W_2, col_X);
      //change dz_2
      for (int i = 0; i < row_z_2; i++) {
        for (int j = 0; j < col_z_2; j++) {
          z_2[i + j*row_z_2] = z_2_inter[i + j*row_z_2];
        }
      }
      cudaMemcpy(dz_2, z_2, sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyHostToDevice);
      printf("Test feedforward_parallel ends.\n");

      // //Test feed forward compare results with CPU
      printf("Run feedforward funciton in CPU.\n");
      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);
      printf("Train funciton in CPU ends.\n");

      //Compare results in GPU and CPU

      //Compare results in GPU and CPU
      //copy from GPU to CPU
      std::cout<<"start test feedforward"<<std::endl;
      cudaMemcpy(z_1, dz_1, sizeof(nn_real) * col_z_1 * row_z_1, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> z_1_GPU = arma::Mat<nn_real>(z_1, row_z_1, col_z_1);
      arma::Mat<nn_real> z_1_CPU = bpcache.z[0];
      nn_real diff = norm(z_1_GPU - z_1_CPU);
      std::cout<<"norm of the difference z_1="<<diff<<std::endl;
      // std::cout<<"first element of z_1_CPU"<<z_1_CPU(0, 0)<<std::endl;
      // std::cout<<"first element of z_1_GPU"<<z_1_GPU(0, 0)<<std::endl;

      //copy from GPU to CPU
      cudaMemcpy(a_1, da_1, sizeof(nn_real) * col_a_1 * row_a_1, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> a_1_GPU = arma::Mat<nn_real>(z_1, row_z_1, col_z_1);
      arma::Mat<nn_real> a_1_CPU = bpcache.z[0];
      diff = norm(a_1_GPU - a_1_CPU);
      std::cout<<"norm of the difference z_1="<<diff<<std::endl;
      // std::cout<<"first element of a_1_CPU"<<a_1_CPU(0, 0)<<std::endl;
      // std::cout<<"first element of a_1_GPU"<<a_1_GPU(0, 0)<<std::endl;

      cudaMemcpy(z_2, dz_2, sizeof(nn_real) * col_z_2 * row_z_2, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> z_2_GPU = arma::Mat<nn_real>(z_2, row_z_2, col_z_2);
      arma::Mat<nn_real> z_2_CPU = bpcache.z[1];
      diff = norm(z_2_GPU - z_2_CPU);
      std::cout<<"norm of the difference z_2="<<diff<<std::endl;
      // std::cout<<"first element of z_2_CPU"<<z_2_CPU(0, 0)<<std::endl;
      // std::cout<<"first element of z_2_GPU"<<z_2_GPU(0, 0)<<std::endl;

      cudaMemcpy(a_2, da_2, sizeof(nn_real) * col_a_2 * row_a_2, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> a_2_GPU = arma::Mat<nn_real>(a_2, row_a_2, col_a_2);
      arma::Mat<nn_real> a_2_CPU = bpcache.a[1];
      diff = norm(a_2_GPU - a_2_CPU);
      std::cout<<"norm of the difference a_2="<<diff<<std::endl;
      // std::cout<<"first element of a_2_CPU"<<a_2_CPU(0, 0)<<std::endl;
      // std::cout<<"first element of a_2_GPU"<<a_2_GPU(0, 0)<<std::endl;

      printf("Finish compare results in GPU and CPU.\n");
      std::cout<<"end test feedforward"<<std::endl;


      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         BACK PROPAGATE                           //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      printf("Run back propagation funciton in CPU.\n");
      nn_real *diff_save;
      diff_save = (nn_real *)malloc(row_y_hat * col_y_hat * sizeof(nn_real));
      struct grads bpgrads;
      backprop_add(nn, y_batch, reg, bpcache, bpgrads, diff_save);
      printf("Back propagation funciton in CPU ends.\n");


      /* grad_b_2 = y_hat - y */
      // db
      std::cout<<"start test back propagation"<<std::endl;
      //y_hat = a_2
      for (int i = 0; i < row_z_2; i++) {
        for (int j = 0; j < col_z_2; j++) {
          y_hat[i + j*row_z_2] = a_2[i + j*row_z_2];
        }
      }
      // arma::Mat<nn_real> y_hat_CPU = bpcache.yc;
      // arma::Mat<nn_real> y_hat_GPU = arma::Mat<nn_real>(y_hat, row_y_hat, col_y_hat);
      // diff = norm(y_hat_CPU - y_hat_GPU);
      // std::cout<<"norm of the difference y_hat in CPU and GPU"<<diff<<std::endl;

      // arma::Mat<nn_real> diff_y_hat_y_GPU = (y_hat_GPU - y_batch)/col_y_hat;
      // arma::Mat<nn_real> diff_y_hat_y_CPU = arma::Mat<nn_real>(diff_save, row_y_hat, col_y_hat);
      // diff = norm(diff_y_hat_y_GPU - diff_y_hat_y_CPU);
      // std::cout<<"norm of the difference between y_hat and y CPU and GPU"<<diff<<std::endl;

      //copy y_hat from CPU to GPU
      cudaMemcpy(dy_hat, y_hat, sizeof(nn_real) * col_y_hat * row_y_hat, cudaMemcpyHostToDevice);
      // cudaMemcpy(dgrad_b_2_matrix, y_hat, sizeof(nn_real) * col_y_hat * row_y_hat, cudaMemcpyHostToDevice);

      // cudaMemcpy(grad_b_2_matrix, dgrad_b_2_matrix, sizeof(nn_real) * col_y_hat * row_y_hat, cudaMemcpyDeviceToHost);
      // std::cout<<"first element of grad_b_2_matrix to GPU"<<grad_b_2_matrix[5]<<std::endl;
      // std::cout<<"first element of y_hat"<<y_hat[5]<<std::endl;
      //
      //calculate matrix y_hat - y GPU
      mySUBTRACT(dgrad_b_2_matrix, dy_hat, dy, row_y_hat, col_y_hat);
      //copy matrix grad_b_2_matrix from GPU to CPU
      cudaMemcpy(grad_b_2_matrix, dgrad_b_2_matrix, sizeof(nn_real) * col_y_hat * row_y_hat, cudaMemcpyDeviceToHost);
      //calculate row sum grad_b_2
      for (int i = 0; i < row_grad_b_2_matrix; i++) {
        nn_real sum = 0;
        for (int j = 0; j < col_grad_b_2_matrix; j++) {
          sum = sum + grad_b_2_matrix[i + j*row_grad_b_2_matrix];
        }
        grad_b_2[i] = sum/col_grad_b_2_matrix;
      }
      //copy grad_b_2 from CPU to GPU
      cudaMemcpy(dgrad_b_2, grad_b_2, sizeof(nn_real) * col_grad_b_2 * row_grad_b_2, cudaMemcpyHostToDevice);

      arma::Col<nn_real> grad_b_2_GPU = arma::Col<nn_real>(grad_b_2, row_grad_b_2);
      arma::Col<nn_real> grad_b_2_CPU = bpgrads.db[1];
      diff = norm(grad_b_2_GPU - grad_b_2_CPU);
      std::cout<<"norm of the difference grad_b_2="<<diff<<std::endl;

      // for (int i = 0; i < len_b_2; i++) {
      //   std::cout<<"grad_b_2_CPU["<<i<<"]"<<grad_b_2_CPU[i]<<std::endl;
      // }
      // diff = norm(grad_b_2_GPU - grad_b_2_CPU);
      // std::cout<<"norm of the difference grad_b_2="<<diff<<std::endl;
      // std::cout<<"first element of grad_b_2_CPU"<<grad_b_2_CPU(0, 0)<<std::endl;
      // std::cout<<"first element of grad_b_2_GPU"<<grad_b_2_GPU(0, 0)<<std::endl;
      // std::cout<<"second element of grad_b_2_CPU"<<grad_b_2_CPU(1, 0)<<std::endl;
      // std::cout<<"second element of grad_b_2_GPU"<<grad_b_2_GPU(1, 0)<<std::endl;



      myMULT_ABT(dgrad_b_2,  da_1, dgrad_W_2, row_grad_W_2, col_grad_W_2, col_y_hat);
      myMULT_ATB(dW_2, dgrad_b_2, dgrad_a_1, row_grad_a_1, col_grad_a_1, row_W_2);
      myCOMPGRADZ1(dgrad_a_1, da_1, dgrad_z_1, row_grad_z_1, col_grad_z_1);
      myMULT_ABT(dgrad_z_1,  dX, dgrad_W_1, row_grad_W_2, col_grad_W_2, col_y_hat);
      std::cout<<"end test back propagation"<<std::endl;

      //Test back propagation
      //Compare results in GPU and CPU
      //copy from GPU to CPU

      std::cout<<"start test back propagation"<<std::endl;
      cudaMemcpy(grad_W_1, dgrad_W_1, sizeof(nn_real) * col_grad_W_1 * row_grad_W_1, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> grad_W_1_GPU = arma::Mat<nn_real>(grad_W_1, row_grad_W_1, col_grad_W_1);
      arma::Mat<nn_real> grad_W_1_CPU = bpgrads.dW[0];
      std::cout<<"row of grad_W_1_CPU"<<grad_W_1_CPU.n_rows;
      std::cout<<"col of grad_W_1_CPU"<<grad_W_1_CPU.n_cols;
      std::cout<<std::endl;
      std::cout<<"row of grad_W_1_GPU"<<grad_W_1_GPU.n_rows;
      std::cout<<"col of grad_W_1_GPU"<<grad_W_1_GPU.n_cols;
      std::cout<<std::endl;
      std::cout<<"batch_size="<<batch_size<<std::endl;
      std::cout<<"mini_batch_size="<<mini_batch_size_alloc<<std::endl;
      std::cout<<"end test back propagation"<<std::endl;
      diff = norm(grad_W_1_GPU - grad_W_1_CPU);
      std::cout<<"norm of the difference grad_W_1="<<diff<<std::endl;
      std::cout<<"first element of grad_W_1_CPU"<<grad_W_1_CPU(0, 0)<<std::endl;
      std::cout<<"first element of grad_W_1_GPU"<<grad_W_1_GPU(0, 0)<<std::endl;

      cudaMemcpy(grad_W_2, dgrad_W_2, sizeof(nn_real) * col_grad_W_2 * row_grad_W_2, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> grad_W_2_GPU = arma::Mat<nn_real>(grad_W_2, row_grad_W_2, col_grad_W_2);
      arma::Mat<nn_real> grad_W_2_CPU = bpgrads.dW[1];
      diff = norm(grad_W_2_GPU - grad_W_2_CPU);
      std::cout<<"norm of the difference grad_W_2="<<diff<<std::endl;
      std::cout<<"first element of grad_W_2_CPU"<<grad_W_2_CPU(0, 0)<<std::endl;
      std::cout<<"first element of grad_W_2_GPU"<<grad_W_2_GPU(0, 0)<<std::endl;

      cudaMemcpy(grad_b_1, dgrad_b_1, sizeof(nn_real) * col_grad_b_1 * row_grad_b_1, cudaMemcpyDeviceToHost);
      arma::Mat<nn_real> grad_b_1_GPU = arma::Mat<nn_real>(grad_b_1, row_grad_b_1, col_grad_b_1);
      arma::Mat<nn_real> grad_b_1_CPU = bpgrads.db[0];
      diff = norm(grad_b_1_GPU - grad_b_1_CPU);
      std::cout<<"norm of the difference grad_b_1="<<diff<<std::endl;
      std::cout<<"first element of grad_W_2_CPU"<<grad_b_1_CPU(0, 0)<<std::endl;
      std::cout<<"first element of grad_W_2_GPU"<<grad_b_1_GPU(0, 0)<<std::endl;

      // cudaMemcpy(grad_b_2, dgrad_b_2, sizeof(nn_real) * col_grad_b_2 * row_grad_b_2, cudaMemcpyDeviceToHost);
      // arma::Mat<nn_real> grad_b_2_GPU = arma::Mat<nn_real>(grad_b_2, row_grad_b_2, col_grad_b_2);
      // arma::Mat<nn_real> grad_b_2_CPU = bpgrads.db[1];
      // diff = norm(grad_b_2_GPU - grad_b_2_CPU);
      // std::cout<<"norm of the difference grad_b_2="<<diff<<std::endl;
      // std::cout<<"first element of grad_b_2_CPU"<<grad_b_2_CPU(0, 0)<<std::endl;
      // std::cout<<"first element of grad_b_2_GPU"<<grad_b_2_GPU(0, 0)<<std::endl;






      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      
      int update_W_1 = myGRADIENT_UPDATE(dgrad_W_1, dW_1, learning_rate, row_W_1, col_W_1);
      int update_W_2 = myGRADIENT_UPDATE(dgrad_W_2, dW_2, learning_rate, row_W_2, col_W_2);

      int update_b_1 = myGRADIENT_UPDATE(dgrad_b_1, db_1, learning_rate, len_b_1, col_X);
      int update_b_2 = myGRADIENT_UPDATE(dgrad_b_2, db_2, learning_rate, len_b_2, col_X);

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
        // Copy data back to the CPU ??? weights, b
        // cudamemcpy relationship between space in GPU and CPU

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */

        save_gpu_error(nn, iter, error_file);
      }

      //add
      batch_start = last_col;
      iter++;
    }
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
}

//add
int compareResults(nn_real *myC, nn_real *refC, int NI, int NJ)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(myC, NI, NJ, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(refC, NI, NJ, false);

  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > 1e-4)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My CPU and GPU output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "CPU and GPU data matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}
