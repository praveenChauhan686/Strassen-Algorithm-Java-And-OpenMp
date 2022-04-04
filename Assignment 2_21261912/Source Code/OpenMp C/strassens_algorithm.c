
#include <stdio.h>
#include <omp.h> //OpenMP library
#include <time.h>
#include <stdlib.h>
//#include <libcpuid.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

int sizeOfTheMatrix = 500; // Size of the matrix.
int threadcount = 8; // Thread count of OpenMP
int threshold = 128; // Lower limit for Strassen algorithm

// defining all the 4 matrices matrices
double **MatrixA, **MatrixB, **MatrixC, **resultantMatrix;

void strassenMatrixMultiplicationComputation( double**, double**, double**, int, int, int, int, int, int, int);
void genericMultiplication(double**, double**, double**, int, int, int, int, int, int, int);

// The matrix should be square and divisible by two.
void additionOfMatrices( double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond) {
  
  // resMatrix = firstMatrix + secondMatrix
  // defines the matrix's rows and columns, as well as the matrix's size.
  int i, j;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, rowFirst, columnFirst, rowSecond, columnSecond, sizeOfTheMatrix) private(i, j) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = 0; i < sizeOfTheMatrix; i++) {
      for (j = 0; j < sizeOfTheMatrix; j++) {
        resMatrix[i][j] = firstMatrix[i + rowFirst][j + columnFirst] + secondMatrix[i + rowSecond][j + columnSecond];
      }
    }
  }
}

void additionOfFirstMatrix(double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond, int row3, int col3) {
  // resMatrix = firstMatrix + secondMatrix
  // defines the matrix's rows and columns, as well as the matrix's size.
  int i, j;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, rowFirst, columnFirst, rowSecond, columnSecond, row3, col3, sizeOfTheMatrix) private(i, j) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = row3; i < sizeOfTheMatrix + row3; i++) {
      for (j = row3; j < sizeOfTheMatrix + row3; j++) {
        resMatrix[i][j] = firstMatrix[i - row3 + rowFirst][j - col3 + columnFirst] + secondMatrix[i - row3 + rowSecond][j - col3 + columnSecond];
      }
    }
  }
}

void additionOfSecondMatrix(double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond) {
  // resMatrix = firstMatrix + secondMatrix
  // defines the rows and columns of the matrix and we define the sizeOfTheMatrix of the matrix
  int i, j;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, rowFirst, columnFirst, rowSecond, columnSecond, sizeOfTheMatrix) private (i, j) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = rowSecond; i < sizeOfTheMatrix + rowSecond; i++) {
      for (j = columnSecond; j < sizeOfTheMatrix + columnSecond; j++) {
        resMatrix[i][j] = firstMatrix[i - rowSecond + rowFirst][j - columnSecond + columnFirst] + secondMatrix[i - rowSecond][j - columnSecond];
      }
    }
  }
}

void subtractionOfMatrices(double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond) {
  // resMatrix = firstMatrix - secondMatrix
  // defines the rows and columns of the matrix and we define the sizeOfTheMatrix of the matrix
  int i, j;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, rowFirst, columnFirst, rowSecond, columnSecond, sizeOfTheMatrix) private(i , j) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = 0; i < sizeOfTheMatrix; i++) {
      for (j = 0; j < sizeOfTheMatrix; j++) {
        resMatrix[i][j] = firstMatrix[i + rowFirst][j + columnFirst] - secondMatrix[i + rowSecond][j + columnSecond];
      }
    }
  }
}

void subtractionOfFirstMatrix(double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond, int row3, int col3) {
  // resMatrix = firstMatrix - secondMatrix
  // defines the rows and columns of the matrix and we defines the sizeOfTheMatrix of the matrix
  int i, j;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, rowFirst, columnFirst, rowSecond, columnSecond, row3, col3, sizeOfTheMatrix) private(i, j) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = row3; i < sizeOfTheMatrix + row3; i++) {
      for (j = col3; j < sizeOfTheMatrix + col3; j++) {
        resMatrix[i][j] = firstMatrix[i - row3 + rowFirst][j - col3 + columnFirst] - secondMatrix[i - row3 + rowSecond][j - col3 + columnSecond];
      }
    }
  }
}

void subtractionOfSecondMatrix(double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond) {
  // resMatrix = firstMatrix - secondMatrix
  // defines the rows and columns of the matrix and we define the size of the matrix
  int i, j;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, rowFirst, columnFirst, rowSecond, columnSecond, sizeOfTheMatrix) private(i, j) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = rowSecond; i < sizeOfTheMatrix + rowSecond; i++) {
      for (j = columnSecond; j < sizeOfTheMatrix; j++) {
        resMatrix[i][j] = firstMatrix[i - rowSecond + rowFirst][j - columnSecond + columnFirst] - secondMatrix[i - rowSecond][j - columnSecond];
      }
    }
  }
}

// Multiply the two matrices: resMatrix = firstMatrix * secondMatrix
void genericMultiplication(double **firstMatrix, double **secondMatrix, double **resMatrix, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond, int row3, int col3) {
  int i, j, k;
  #pragma omp parallel shared(firstMatrix, secondMatrix, resMatrix, sizeOfTheMatrix, rowFirst, columnFirst, rowSecond, columnSecond,row3, col3) private(i, j, k) num_threads(threadcount)
  {
    #pragma omp for schedule(static)
    for (i = row3; i < sizeOfTheMatrix + row3; i++) {
      for (j = col3; j < sizeOfTheMatrix; j++) {
        resMatrix[i][j] = 0.0;
        for (k = 0; k < sizeOfTheMatrix; k++) {
          resMatrix[i][j] += firstMatrix[i - row3 + rowFirst][k + columnFirst] * secondMatrix[k + rowSecond][j - col3 + columnSecond];
        }
      }
    }
  }
}

// Strassen algorithm for matrix multiplication
void strassenMatrixMultiplicationComputation(double **a, double **b, double **c, int sizeOfTheMatrix, int rowFirst, int columnFirst, int rowSecond, int columnSecond, int row3, int col3) {
  double **memOne, **memTwo;
  int  newsize = sizeOfTheMatrix / 2;
  int i;

  if (sizeOfTheMatrix >= threshold) { // compares the sizeOfTheMatrix of the matrix with the threshold value
    // memory allocation
    memOne = (double**) malloc(sizeof(double*)*newsize);
    memTwo = (double**) malloc(sizeof(double*)*newsize);

    for (i = 0; i < newsize; i++) {
      memOne[i] = (double*) malloc(sizeof(double) * newsize);
      memTwo[i] = (double*) malloc(sizeof(double) * newsize);
    }

    // calculate M1 for the strassen alogorithm
    additionOfMatrices(a, a, memOne, newsize, 0, 0, newsize, newsize);
    additionOfMatrices(b, b, memTwo, newsize, 0, 0, newsize, newsize);
    strassenMatrixMultiplicationComputation(memOne, memOne, c, newsize, 0, 0, 0 , 0, newsize, 0);

    // calculate M2 for the strassen alogorithm
    additionOfFirstMatrix(c, c, c, newsize, 0, 0, newsize, 0, 0, 0);
    additionOfFirstMatrix(c, c, c, newsize, newsize, 0, newsize, newsize, newsize, newsize);
    additionOfMatrices(a, a, memOne, newsize, newsize, 0, newsize, newsize);
    strassenMatrixMultiplicationComputation(memOne, b, c, newsize, 0, 0, 0, 0, newsize, 0);

    // calculate M3 for the strassen alogorithm
    subtractionOfMatrices(b, b, memTwo, newsize, 0, newsize, newsize, newsize);
    strassenMatrixMultiplicationComputation(a, memTwo, c, newsize, 0, 0, 0, 0, 0, newsize);

    // calculate M4 for the strassen alogorithm
    subtractionOfFirstMatrix(c, c, c, newsize, newsize, newsize, newsize, 0, newsize, newsize);
    additionOfFirstMatrix(c, c, c, newsize, newsize, newsize, 0, newsize, newsize, newsize);
    subtractionOfMatrices(b, b, memTwo, newsize, newsize, 0, 0, 0);
    strassenMatrixMultiplicationComputation(a, memTwo, memOne, newsize, newsize, newsize, 0, 0, 0, 0);

    // calculate M5 for the strassen alogorithm
    additionOfSecondMatrix(c, memOne, c, newsize, 0, 0, 0, 0);
    additionOfSecondMatrix(c, memOne, c, newsize, newsize, 0, newsize, 0);
    additionOfMatrices(a, a, memOne, newsize, 0, 0, 0, newsize);
    strassenMatrixMultiplicationComputation(memOne, b, memTwo, newsize, 0, 0, newsize, newsize, 0, 0);

    subtractionOfSecondMatrix(c, memTwo, c, newsize, 0, 0, 0, 0);
    additionOfSecondMatrix(c, memTwo, c, newsize, 0, newsize, 0, newsize);

    // calculate M6 for the strassen alogorithm
    subtractionOfMatrices(a, a, memOne, newsize, newsize, 0, 0, 0);
    additionOfMatrices(b, b, memTwo, newsize, 0, 0, 0, newsize);
    strassenMatrixMultiplicationComputation(memOne, memTwo, c, newsize, 0, 0, 0, 0, newsize, newsize);

    // calculate M7 for the strassen alogorithm
    subtractionOfMatrices(a, a, memOne, newsize, 0, newsize, newsize, newsize);
    additionOfMatrices(b, b, memTwo, newsize, newsize, 0, newsize, newsize);
    strassenMatrixMultiplicationComputation(memOne, memTwo, c, newsize, 0, 0, 0, 0, 0, 0);

    // memory deallocation
    free(memOne);
    free(memTwo);
  }
  else {
    genericMultiplication(a, b, c, sizeOfTheMatrix, rowFirst, columnFirst, rowSecond, columnSecond, row3, col3);
  }
}

// main method for the program
int main(int argc, char *argv[]) {

  printf("Enter the matrix's size.:\n");
  scanf("%d", &sizeOfTheMatrix);

  printf("Enter the number of threads you want to use.:\n");
  scanf("%d", &threadcount);

  printf("Enter the Strassen implementation's bottom limit.:\n");
  scanf("%d", &threshold);


  double begintime = 0.0;
  double endtime = 0.0;
  double totaltime = 0.0;
  int i, j, k;

  MatrixA = (double**) malloc(sizeof(double*)*sizeOfTheMatrix);
  for (i = 0; i < sizeOfTheMatrix; i++) {
    MatrixA[i] = (double*) malloc(sizeof(double) * sizeOfTheMatrix);
  }

  MatrixB = (double**) malloc(sizeof(double*)*sizeOfTheMatrix);
  for (i = 0; i < sizeOfTheMatrix; i++) {
    MatrixB[i] = (double*) malloc(sizeof(double) * sizeOfTheMatrix);
  }

  MatrixC = (double**) malloc(sizeof(double*)*sizeOfTheMatrix);
  for (i = 0; i < sizeOfTheMatrix; i++) {
    MatrixC[i] = (double*) malloc(sizeof(double) * sizeOfTheMatrix);
  }

  resultantMatrix = (double**) malloc(sizeof(double*)*sizeOfTheMatrix);
  for (i = 0; i < sizeOfTheMatrix; i++) {
    resultantMatrix[i] = (double*) malloc(sizeof(double) * sizeOfTheMatrix);
  }

  for (i = 0; i < sizeOfTheMatrix; i++) {
    for (j = 0; j < sizeOfTheMatrix; j++) {
      MatrixA[i][j] = (i + j) * 1.0;
      MatrixB[i][j] = (i + j) * 1.0;
      MatrixC[i][j] = 0;
      resultantMatrix[i][j] = 0;
    }
  }

 if(sizeOfTheMatrix>=threshold){
  printf("The multiplication of the Strassen Matrix begins here.\n");
  begintime = omp_get_wtime();
  strassenMatrixMultiplicationComputation(MatrixC, MatrixB, MatrixC, sizeOfTheMatrix, 0, 0, 0, 0, 0, 0);
  endtime = omp_get_wtime();
  totaltime = endtime - begintime;
  printf("Strassen matrix multiplication completed\n");
  printf("Strassen Time taken = %0.3f \n", totaltime);
  printf("Number of threads used = %d\n", threadcount);
  }
  else{
  printf("The generic multiplication Matrix begins here.\n");
  begintime = omp_get_wtime();
  genericMultiplication(MatrixC, MatrixB, MatrixC, sizeOfTheMatrix, 0, 0, 0, 0, 0, 0);
  endtime = omp_get_wtime();
  totaltime = endtime - begintime;
  printf("Generic matrix multiplication completed\n");
  printf("Generic Time taken = %0.3f \n", totaltime);
  printf("Number of threads used = %d\n", threadcount);
	  
  }
  
}