#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define BLOCK 16
#define ITERATIONS 10


__device__
void getCellStatusPar(int* val, int matrix[BLOCK][BLOCK], int row, int column) {
        *val = row >= 0 && row < BLOCK && column >= 0 && column < BLOCK ?
                        matrix[row][column] : 0;
}

__global__
void parallelGOL(int* matrix, int m, int n) {
        __shared__ int block[BLOCK][BLOCK];
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int result, it, pos = i * n + j, val = 0;
        int coordY = threadIdx.y, coordX = threadIdx.x;

        if (pos < m * n) {
                for(it = 0; it < ITERATIONS; ++it){
                        block[coordY][coordX] = matrix[pos];//copying to shared memory
                        __syncthreads();

                        getCellStatusPar(&val, block, coordY - 1, coordX - 1); //UP LEFT
                        result = val;

                        getCellStatusPar(&val, block, coordY, coordX - 1); //LEFT
                        result += val;
                        getCellStatusPar(&val, block, coordY + 1, coordX - 1); //BOTTOM LEFT
                        result += val;

                        getCellStatusPar(&val, block, coordY - 1, coordX); //UP
                        result += val;
                        getCellStatusPar(&val, block, coordY + 1, coordX); //DOWN
                        result += val;

                        getCellStatusPar(&val, block, coordY - 1, coordX + 1); //UP RIGHT
                        result += val;
                        getCellStatusPar(&val, block, coordY, coordX + 1); //RIGHT
                        result += val;
                        getCellStatusPar(&val, block, coordY + 1, coordX + 1); //BOTTOM RIGHT
                        result += val;


                        if (result == 2 && matrix[pos] == 1)
                                matrix[pos] = 1;
                        else if (result == 3)
                                matrix[pos] = 1;
                        else
                                matrix[pos] = 0;
                        __syncthreads(); //synchronization step after writing and before updating "block"
                }
        }
}

int main(int argc, char *argv[]) {
        if (argc < 3) {
                printf("No enough arguments.");
                return -1;
        }

        srand (time(NULL));

        int m = atoi(argv[1]);//number of blocks in Y axis!!!!
        int n = atoi(argv[2]);//number of blocks in YXaxis!!!!

        int* matrix = (int*) malloc(BLOCK * BLOCK * m * n * sizeof(int));

        int realM = m * BLOCK;
        int realN = n * BLOCK;

        int i;
        for (i = 0; i < realM * realN; ++i)
                matrix[i] = rand() % 2; //0 = dead, 1 = alive

        //starting parallel execution
        clock_t t = clock();
        int *d_matrix;


        //allocating memory
        cudaMalloc(&d_matrix, realN * realM * sizeof(int));

        //copying memory
        cudaMemcpy(d_matrix, matrix, realN * realM * sizeof(int), cudaMemcpyHostToDevice);

        //dimensions
        dim3 threadblock(BLOCK, BLOCK); //16 * 16 = 256 in total
        dim3 grid(1 + realN / threadblock.x, 1 + realM / threadblock.y);

        //executing the function
        parallelGOL<<<grid, threadblock>>>(d_matrix, realM, realN);
        cudaDeviceSynchronize();


        //once the function has been called I copy the result in matrix
        cudaMemcpy(matrix, d_matrix, realN * realM * sizeof(int), cudaMemcpyDeviceToHost);

        double parallelExecutionTime = ((double) (clock() - t))
                        / ((double) (CLOCKS_PER_SEC));

        printf("%d;%f\n", realM, parallelExecutionTime);

        //free resources
        cudaFree(d_matrix);
        free(matrix);

        return 0;
}
