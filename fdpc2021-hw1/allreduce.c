/*
  Foundations of Parallel and Distributed Computing, Fall 2021.
  Instructor: Prof. Chao Yang @ Peking University.
  Date: 1/11/2021
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BUFFER_SIZE (4 * 1024 * 1024)
#define ABS(x) (((x) > 0) ? (x) : -(x))
#define EPS 1e-5

double get_walltime() {
#if 1
  return MPI_Wtime();
#else
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
#endif
}

void initialize(int rank, float* data, int n) {
  int i = 0;
  srand(rank);
  for (i = 0; i < n; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

int result_check(float* a, float* b, int n) {
  int i = 0;
  for (i = 0; i < n; ++i) {
    if (ABS(a[i] - b[i]) > EPS) {
      return 0;
    }
  }
  return 1;
}

int main(int argc, char* argv[]) {
  int rank, comm_size;

  float data[BUFFER_SIZE];
  float base_output[BUFFER_SIZE];
  double time0, time1;
  double impl_time = 0;
  int correct_count = 0, correct = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // initialization
  initialize(rank, data, BUFFER_SIZE);

  // ground true results
  MPI_Allreduce(data, base_output, BUFFER_SIZE, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  time0 = get_walltime();
  /* write your codes here */
  time1 = get_walltime() - time0;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&time1, &impl_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  // check correctness and report results
  correct = result_check(base_output, data, BUFFER_SIZE);
  MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();

  if (!correct) {
    printf("Wrong answer on rank %d.\n", rank);
  }
  if (rank == 0 && correct_count == comm_size) {
    printf("Buffer size: %d, comm size: %d\n", BUFFER_SIZE, comm_size);
    printf("Correct results.\n");
    printf("Your implementation wall time:%f\n", impl_time);
  }

  return 0;
}
