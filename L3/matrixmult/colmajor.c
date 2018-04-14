#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

float ScalarSSE(float *m1, float *m2);
void randomFill(float * matrix, int dim);
float dotProduct(int dim, float * A, int row, float * B, int col);
void seqMult(int dim, float * A, float * B, float * C);
void * func(void *);
void printArray(const char * name, int dim, float * M);
void printColMajor(const char * name, int dim, float * M);
void *regionMult(void * taskptr);

int dim = 2000;

struct parallel_task {

  int id;
  int num_tasks;
  float * matrixA;
  float * matrixB;
  float * matrixC;

};

int main() {
  pthread_t * threads;
  int  iret1, iret2;
  int numthreads = 8;
  struct parallel_task* tasks;

  float * matrixA;
  float * matrixB;
  float * matrixC;
  double start, end;

  struct timeval st, et;

  printf("dim is %d\n", dim);

  tasks = calloc(sizeof(struct parallel_task), numthreads);

  matrixA = calloc(sizeof(float),dim*dim);
  randomFill(matrixA, dim);
  
  //make matrix B an identity matrix
  matrixB = calloc(sizeof(float),dim*dim);
//  for (int i = 0; i < dim; i++)
//    matrixB[i*dim + i] = 1;

  randomFill(matrixB, dim);
  matrixC = calloc(sizeof(float),dim*dim);

//  printArray("A", dim, matrixA);
//  printArray("B", dim, matrixB);
//  printArray("Result", dim, matrixC);

gettimeofday(&st, NULL);
#ifdef SEQ
  printf("Sequential\n");
  seqMult(dim, matrixA, matrixB, matrixC);
#else
  printf("Parallel\n");
  threads = calloc(numthreads, sizeof(pthread_t));

  pthread_mutex_t mutex;
  pthread_mutex_init(&mutex, NULL);

  for (int i = 0; i < numthreads; i++) {
    tasks[i].matrixA = matrixA;
    tasks[i].matrixB = matrixB;
    tasks[i].matrixC = matrixC;
    tasks[i].id = i;
    tasks[i].num_tasks = numthreads;
    iret1 = pthread_create( &threads[i], NULL, regionMult, (void *)&tasks[i]);
  }

  for (int i = 0; i < numthreads; i++) {
    pthread_join( threads[i], NULL);
  }

  pthread_mutex_destroy(&mutex);
#endif

//  printArray("A", dim, matrixA);
//  printColMajor("B", dim, matrixB);
//  printArray("Result", dim, matrixC);

  gettimeofday(&et, NULL);

  start =st.tv_sec + ((double)st.tv_usec)/1000000;
  end = et.tv_sec + (double)et.tv_usec/1000000;

  printf("start: %lf\nend: %lf\ndiff = %lf\n", start, end, end-start);

}

float dotProductSSE(int dim, float * A, int row, float * B, int col) {
//float ScalarSSE(float *m1, float *m2) {

  float prod = 0.0, tmp;
  int i;
  __m128 X, Y, Z;

  for(i=0; i<dim; i+=4) {
    X = _mm_load_ps(&A[i]);
    Y = _mm_load_ps(&B[i]);
    X = _mm_mul_ps(X, Y);

    X = _mm_hadd_ps(X, X);
    X = _mm_hadd_ps(X, X);
    _mm_store_ss(&tmp, X);
    prod += tmp;
  }

  return prod;
}

float dotProduct(int dim, float * A, int row, float * B, int col) {

  float sum = 0;

  for (int i = 0; i < dim; i++) {
    sum += A[row*dim + i]*B[col*dim + i];
  }
  return sum;
}

void seqMult(int dim, float * A, float * B, float * C) {

  for (int row = 0; row < dim; row++) {
    for (int col = 0; col < dim; col++) {
      //C[row*dim + col] = dotProduct(dim, A, row, B, col);
      C[row*dim + col] = dotProduct(dim, A, row, B, col);
    }
  }
}

void *regionMult(void * taskptr) {

  int start, stop;
  float chunk;
  struct parallel_task * task;
  task = (struct parallel_task *)taskptr;
  chunk = (float)dim/task->num_tasks;
  start = round(chunk*task->id);
  stop = round(chunk*(task->id + 1));

  printf("Thread %d: computing %d to %d\n", task->id, start, stop);
  for (int row = start; row < stop; row++) {
    for (int col = 0; col < dim; col++) {
      task->matrixC[row*dim + col] = dotProductSSE(dim, task->matrixA, row, task->matrixB, col);
    }
  }
}

void randomFill(float * matrix, int dim) {

  srandom(time(NULL));

  for (int i = 0; i < dim*dim; i++) {
    matrix[i]= ((float)random()/INT_MAX)*10;
  }
}

void printColMajor(const char * name, int dim, float * M) {

  printf("Matrix %s:\n", name);
  for (int row = 0; row < dim; row++) {
    for (int col = 0; col < dim; col++) {
     printf("%5.2f ",M[col*dim + row]);
    }
    printf("\n");
  }
}

void printArray(const char * name, int dim, float * M) {

  printf("Matrix %s:\n", name);
  for (int row = 0; row < dim; row++) {
    for (int col = 0; col < dim; col++) {
     printf("%5.2f ",M[row*dim + col]);
    }
    printf("\n");
  }
}
