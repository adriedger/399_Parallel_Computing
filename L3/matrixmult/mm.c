#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>

void randomFill(int * matrix, int dim);
int dotProduct(int dim, int * A, int row, int * B, int col);
void seqMult(int dim, int * A, int * B, int * C);
void printArray(const char * name, int dim, int * M);

int dim = 2000;

struct parallel_task {

  int id;
  int num_tasks;
  int * matrixA;
  int * matrixB;
  int * matrixC;

};

int main() {
  pthread_t * threads;
  int  iret1, iret2;
  int numthreads = 8;
  struct parallel_task* tasks;

  int * matrixA;
  int * matrixB;
  int * matrixC;
  double start, end;

  struct timeval st, et;

  printf("dim is %d\n", dim);

  tasks = calloc(sizeof(struct parallel_task), numthreads);

  matrixA = calloc(sizeof(int),dim*dim);
  randomFill(matrixA, dim);
  
  matrixB = calloc(sizeof(int),dim*dim);

// make matrix B an identity matrix
  for (int i = 0; i < dim; i++)
    matrixB[i*dim + i] = 1;

//  randomFill(matrixB, dim);
  matrixC = calloc(sizeof(int),dim*dim);

//  printArray("A", dim, matrixA);
//  printArray("B", dim, matrixB);
//  printArray("Result", dim, matrixC);

  gettimeofday(&st, NULL);
  printf("Sequential\n");
  seqMult(dim, matrixA, matrixB, matrixC);

  gettimeofday(&et, NULL);

  start =st.tv_sec + ((double)st.tv_usec)/1000000;
  end = et.tv_sec + (double)et.tv_usec/1000000;

  printf("start: %lf\nend: %lf\ndiff = %lf\n", start, end, end-start);
}


int dotProduct(int dim, int * A, int row, int * B, int col) {

  int sum = 0;

  // multiply 'row' of A by 'column' of B
  for (int i = 0; i < dim; i++) {
    sum += A[row*dim + i]*B[i*dim + col];
  }
  return sum;
}

void seqMult(int dim, int * A, int * B, int * C) {

  for (int row = 0; row < dim; row++) {
    for (int col = 0; col < dim; col++) {
      C[row*dim + col] = dotProduct(dim, A, row, B, col);
    }
  }
}

void randomFill(int * matrix, int dim) {

  srandom(time(NULL));

  for (int i = 0; i < dim*dim; i++) {
    matrix[i]= round(((float)random()/INT_MAX)*10);
  }
}

void printArray(const char * name, int dim, int * M) {

  printf("Matrix %s:\n", name);
  for (int row = 0; row < dim; row++) {
    for (int col = 0; col < dim; col++) {
     printf("%4d ",M[row*dim + col]);
    }
    printf("\n");
  }
}
