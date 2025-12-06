// CUDA implementation of attention mechanism
// 1. Take embed of dim size input.
// 2. Apply positional encoding -> (PE(pos, 2i)= sin(pos/10000^(2i/EMBED_DIM))), PE(pos, 2i+1)= cos(pos/10000^(2i/EMBED_DIM)))
// 3. Apply multi-head attention.
// 3.1. Calculate query, key and value -> (qi = Xi * Wq, ...)
// 3.2. Calculate attention scores -> (ai = qi * kj^T)
// 3.3. Normalize and apply softmax to the attention scores -> (ai = softmax(ai/sqrt(EMBED_DIM)))
// 3.4. Calculated weighted values -> (zi = ai * Vi^T)
// 4. Concatenate the result z from each head and multiply by weight of MHA block -> (Zi = Wo * [zi, ...])

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define EMBED_DIM 512
#define QKV_DIM 64
#define RAND_MAX 1024

__shared__ float Wq[EMBED_DIM][QKV_DIM], Wk[EMBED_DIM][QKV_DIM], Wv[EMBED_DIM][QKV_DIM];

__global__ void matMul(float *A, float *B, float *C, int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) // Return if index of the thread is out of bounds
        return;
    float tx[TILE_WIDTH];
    float ty[TILE_WIDTH];
    for (int a = 0; a < m * k / TILE_WIDTH; a++) {
        tx[threadIdx.x] = A[a * TILE_WIDTH + threadIdx.x]; // Load data from global memory to tile in shared memory
        ty[threadIdx.y] = B[a * TILE_WIDTH + threadIdx.y];
        __syncthreads();
        float Pvalue = 0;
        for (int l = 0; l < TILE_WIDTH; l++) {
            Pvalue += tx[l] * ty[l]; // Accumulate the product of elements in the tile
        }
        __syncthreads();
        C[i * TILE_WIDTH + threadIdx.y] = Pvalue; // Assign the result to the output matrix
    }
}

__global__ void positional_encoding(float *input, float *output, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= EMBED_DIM || j >= seq_len) return;
    output[i * seq_len + j] = sin(j / powf(10000, 2.0 * i / EMBED_DIM)); // Even position
    output[i * seq_len + j + 1] = cos(j / powf(10000, 2.0 * i / EMBED_DIM)); // Odd position
}

__global__ void transpose(float *A, float *B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= cols || idy >= rows) return;
    B[idx * rows + idy] = A[idy * cols + idx];
}

__global__ void normReduceSum(float *A, float *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    A[idx] /= sqrtf(size);
    atomicAdd(&B, A[idx]);
}

__global__ void softmax(float *A, float *B, float sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = A[idx] / sum;
}

// __global__ void attention_kernel(float *Q, float *K, float *V, float *output, int seq_len);

// __global__ void mha(float *input, float *output, int pos, int seq_len, int num_heads);

int main() {
    int const seq_len = 10;
    float input[EMBED_DIM][seq_len];

    float h_Wq[EMBED_DIM][QKV_DIM], h_Wk[EMBED_DIM][QKV_DIM], h_Wv[EMBED_DIM][QKV_DIM];
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < QKV_DIM; j++) {
            h_Wq[i][j] = rand() / (float)RAND_MAX;
            h_Wk[i][j] = rand() / (float)RAND_MAX;
            h_Wv[i][j] = rand() / (float)RAND_MAX;
        }
    }

    float *d_input, *d_encoded, *d_Q, *d_K, *d_V, *d_A, *d_z, *d_Z;

    cudaMalloc(&d_input, EMBED_DIM * seq_len * sizeof(float));
    cudaMalloc(&d_encoded, EMBED_DIM * seq_len * sizeof(float));
    cudaMalloc(&d_Q, EMBED_DIM * seq_len * sizeof(float));
    cudaMalloc(&d_K, EMBED_DIM * seq_len * sizeof(float));
    cudaMalloc(&d_V, EMBED_DIM * seq_len * sizeof(float));
    cudaMalloc(&d_A, seq_len * seq_len * sizeof(float));
    cudaMalloc(&d_z, EMBED_DIM * seq_len * sizeof(float));
    cudaMalloc(&d_Z, EMBED_DIM * seq_len * sizeof(float));

    cudaMemcpyToSymbol(Wq, h_Wq, EMBED_DIM * QKV_DIM * sizeof(float));
    cudaMemcpyToSymbol(Wk, h_Wk, EMBED_DIM * QKV_DIM * sizeof(float));
    cudaMemcpyToSymbol(Wv, h_Wv, EMBED_DIM * QKV_DIM * sizeof(float));
    cudaMemcpy(d_input, input, EMBED_DIM * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    positional_encoding<<<EMBED_DIM, seq_len>>>(d_input, d_encoded, seq_len);
    cudaDeviceSynchronize();
    matMul<<<EMBED_DIM, seq_len>>>(d_encoded, (float *) Wq, d_Q, EMBED_DIM, seq_len, QKV_DIM);
    matMul<<<EMBED_DIM, seq_len>>>(d_encoded, (float *) Wk, d_K, EMBED_DIM, seq_len, QKV_DIM);
    matMul<<<EMBED_DIM, seq_len>>>(d_encoded, (float *) Wv, d_V, EMBED_DIM, seq_len, QKV_DIM);
    cudaDeviceSynchronize();


    return 0;
}
