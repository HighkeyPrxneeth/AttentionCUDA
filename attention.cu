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
#include <cuda_runtime.h>
#include <math.h>

#define TILE_WIDTH 16
#define EMBED_DIM 512
#define QKV_DIM 64
#define SEQ_LEN 64

// 1. Tiled Matrix Multiplication Kernel (C = A * B)
__global__ void matMul(float *A, float *B, float *C, int M, int N, int K_dim) {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Iterate over the K dimension in tiles
    for (int ph = 0; ph < (K_dim + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {

        // Load Tile A
        if (row < M && (ph * TILE_WIDTH + tx) < K_dim)
            As[ty][tx] = A[row * K_dim + ph * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0f;

        // Load Tile B
        if ((ph * TILE_WIDTH + ty) < K_dim && col < N)
            Bs[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Dot Product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}

// 2. Transpose Kernel
__global__ void transpose(float *A, float *B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        B[x * rows + y] = A[y * cols + x];
    }
}

// 3. Positional Encoding
__global__ void positional_encoding(float *input, float *output, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Embed Dim
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Seq Len

    if (i >= EMBED_DIM || j >= seq_len) return;

    float div_term = expf((2 * (i / 2.0) * -logf(10000.0f)) / EMBED_DIM);
    float val = input[j * EMBED_DIM + i];

    if (i % 2 == 0)
        output[j * EMBED_DIM + i] = val + sinf(j * div_term);
    else
        output[j * EMBED_DIM + i] = val + cosf(j * div_term);
}

// 4. Softmax Kernel (Row-wise)
__global__ void softmax_kernel(float *input, float *output, int rows, int cols, float scale) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    // 1. Find Max for Numerical Stability (prevent exp infinity)
    float max_val = -1e30f; // Minimal float
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = input[row * cols + c] * scale;
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Reduction for Max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float global_max = sdata[0];
    __syncthreads();

    // 2. Calculate Exponentials and Sum
    float sum_val = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        sum_val += expf(input[row * cols + c] * scale - global_max);
    }
    sdata[tid] = sum_val;
    __syncthreads();

    // Reduction for Sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float global_sum = sdata[0];

    // 3. Normalize
    for (int c = tid; c < cols; c += blockDim.x) {
        output[row * cols + c] = expf(input[row * cols + c] * scale - global_max) / global_sum;
    }
}

int main() {
    // A. Setup
    int const seq_len = SEQ_LEN;
    size_t input_size = EMBED_DIM * seq_len * sizeof(float);
    size_t qkv_size   = EMBED_DIM * QKV_DIM * sizeof(float);
    size_t proj_size  = seq_len * QKV_DIM * sizeof(float);
    size_t score_size = seq_len * seq_len * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_Wq = (float*)malloc(qkv_size);
    float *h_Wk = (float*)malloc(qkv_size);
    float *h_Wv = (float*)malloc(qkv_size);
    float *h_Z  = (float*)malloc(proj_size);

    // B. Initialization
    for (int i = 0; i < EMBED_DIM * seq_len; i++)
        h_input[i] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < EMBED_DIM * QKV_DIM; i++) {
        h_Wq[i] = (float)rand() / (float)RAND_MAX;
        h_Wk[i] = (float)rand() / (float)RAND_MAX;
        h_Wv[i] = (float)rand() / (float)RAND_MAX;
    }

    // C. GPU Allocations
    float *d_input, *d_encoded, *d_Wq, *d_Wk, *d_Wv;
    float *d_Q, *d_K, *d_V, *d_Kt, *d_Scores, *d_Attn, *d_Z;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_encoded, input_size);
    cudaMalloc(&d_Wq, qkv_size);
    cudaMalloc(&d_Wk, qkv_size);
    cudaMalloc(&d_Wv, qkv_size);
    cudaMalloc(&d_Q, proj_size);
    cudaMalloc(&d_K, proj_size);
    cudaMalloc(&d_V, proj_size);
    cudaMalloc(&d_Kt, proj_size);
    cudaMalloc(&d_Scores, score_size);
    cudaMalloc(&d_Attn, score_size);
    cudaMalloc(&d_Z, proj_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv, qkv_size, cudaMemcpyHostToDevice);

    // D. Grid Configs
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridProj((QKV_DIM + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);
    dim3 gridScores((seq_len + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);
    dim3 blockPE(16, 16);
    dim3 gridPE((EMBED_DIM + blockPE.x - 1) / blockPE.x, (seq_len + blockPE.y - 1) / blockPE.y);
    dim3 gridTrans((seq_len + blockDim.x - 1) / blockDim.x, (QKV_DIM + blockDim.y - 1) / blockDim.y);

    // 1. Positional Encoding
    positional_encoding<<<gridPE, blockPE>>>(d_input, d_encoded, seq_len);

    // 2. Q, K, V Projections
    matMul<<<gridProj, blockDim>>>(d_encoded, d_Wq, d_Q, seq_len, QKV_DIM, EMBED_DIM);
    matMul<<<gridProj, blockDim>>>(d_encoded, d_Wk, d_K, seq_len, QKV_DIM, EMBED_DIM);
    matMul<<<gridProj, blockDim>>>(d_encoded, d_Wv, d_V, seq_len, QKV_DIM, EMBED_DIM);

    // 3. Transpose K
    transpose<<<gridTrans, blockDim>>>(d_K, d_Kt, seq_len, QKV_DIM);

    // 4. Attention Scores (Q * K^T)
    matMul<<<gridScores, blockDim>>>(d_Q, d_Kt, d_Scores, seq_len, seq_len, QKV_DIM);

    // 5. Softmax (Masking and Scaling)
    int threadsPerBlock = 256;
    float scale = 1.0f / sqrtf((float)QKV_DIM);
    softmax_kernel<<<seq_len, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_Scores, d_Attn, seq_len, seq_len, scale);

    // 6. Weighted Sum (Scores * V)
    matMul<<<gridProj, blockDim>>>(d_Attn, d_V, d_Z, seq_len, QKV_DIM, seq_len);

    cudaMemcpy(h_Z, d_Z, proj_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("Attention Calculation Complete.\n");
    printf("Sample Output:\n");
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < QKV_DIM; j++) {
            printf("%.2f ", h_Z[i * QKV_DIM + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input); cudaFree(d_encoded);
    cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Kt);
    cudaFree(d_Scores); cudaFree(d_Attn); cudaFree(d_Z);
    free(h_input); free(h_Wq); free(h_Wk); free(h_Wv); free(h_Z);

    return 0;
}
