// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

// #define NAIVE
#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
extern __shared__ _FTYPE_ sharedmem[];
//You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    // Indexing the matrix
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int bm_x = blockDim.x, bm_y = blockDim.y;


    int I_matrix = by * bm_y * TILESCALE_M + ty;
    int J_matrix = bx * bm_x * TILESCALE_N + tx;

    _FTYPE_  * As = &sharedmem[0];
    _FTYPE_  * Bs = &As[TILEDIM_M * TILEDIM_K];  // This is the shared memory used to index the 
    register _FTYPE_ _c[TILESCALE_M][TILESCALE_N] = {0}; // TILESCALE_M = N/TILEDIM_M memory accesses

    #pragma unroll
    for (int kk = 0; kk < N; kk += TILEDIM_K) {  // Itereate through K dimension. 
        // Copy submatrix form A and B to shared Memory
        // Copy A
 
        // Performance would increase if TILEDIM_K is a multiple of TILESCALE_M and TILESCALE_N, in this case take out the if statement. 
        for (int k_block = 0; k_block < (TILEDIM_K + bm_x -1) / bm_x; k_block ++) { // 
            for (int m_block = 0; m_block < TILESCALE_M; m_block ++) {  // Tile dim have to be devisiable by tile scale, or doesn't make sense. 
                if ((kk + threadIdx.x + k_block * blockDim.x < N) && (I_matrix + m_block * blockDim.y < N)){  // To the right padding and to the bottom padding
                    As[(threadIdx.y + m_block * blockDim.y) * TILEDIM_K + k_block * blockDim.x + threadIdx.x] = A[(I_matrix + m_block * blockDim.y) * N + kk + k_block * blockDim.x + threadIdx.x];  // A[Horizontal_Strip + K_Block + m_idx*N_matrix + k_idx]
                } else {
                    As[(threadIdx.y + m_block * blockDim.y) * TILEDIM_K + k_block * blockDim.x + threadIdx.x] = 0;
                }    
            }
        }
        
        __syncthreads();

        // Copy B
        for (int k_block = 0; k_block < (TILEDIM_K + bm_y - 1 ) / bm_y; k_block ++) { // 
            for (int n_block = 0; n_block < TILESCALE_N; n_block ++) {  // Tile dim have to be devisiable by tile scale, or doesn't make sense. 
            if ((J_matrix + n_block * blockDim.x < N) && (kk + k_block * blockDim.y + threadIdx.y < N)) {  // To the bottom padding and to the right padding
                Bs[(threadIdx.y + k_block * blockDim.y) * TILEDIM_N + n_block * blockDim.x + threadIdx.x] = B[(kk + blockDim.y * k_block + threadIdx.y)*N + J_matrix + blockDim.x * n_block];  // B[row * N_matrix + J + k_idx * N_matrix + n_idx]
            } else {
                Bs[(threadIdx.y + k_block * blockDim.y) * TILEDIM_N + n_block * blockDim.x + threadIdx.x] = 0;
            }
        }
        }       
        __syncthreads();

        // Compute C
        for (int k = 0; k < TILEDIM_K; k++){
            for (int j = 0; j < TILESCALE_N; j++) {
                for (int i = 0; i < TILESCALE_M; i++){
                    _c[i][j] += As[(ty + i * bm_y) * TILEDIM_K + k] * Bs[k * TILEDIM_N + tx + j * bm_x];
                }
            }
        }
        __syncthreads();


        // Each thread process 4 elements

        // Calculate Tile C
        // Accumulate Tile C
    }
    
    __syncthreads();
    // Store C back to the global memory. 
    for (int j = 0; j < TILESCALE_N; j++) {
        for (int i = 0; i < TILESCALE_M; i++){
            if ((I_matrix + i * bm_y) < N && (J_matrix + j * bm_x) < N){
                C[(I_matrix + i * bm_y) * N + J_matrix + j * bm_x] = _c[i][j]; 
            }
        }
    }

}
#endif