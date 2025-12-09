# Attention Mechanism Implementation in CUDA

This repository has a very minimal and basic implementation of the attention mechanism using CUDA.
I wrote this code to exercise my HPC skills and understanding.
This code is not intended for production use and is provided as-is.

## Overview

The code is divided into 4 parts:
1. Setup and initialization of weights and input embeddings. In my implementation, I am not taking input from the user, but rather assigning random values. All values are ranging from 0 to 1 (`(float)rand() / (float)RAND_MAX`).

2. Here, I apply positional encoding to the input embeddings. This is done using the `positional_encoding` global kernel which is launched from the host. It uses the traditional formula for positional encoding, i.e., 
$PE_{pos, 2i} = \sin\left(\frac{j}{10000^{2i/d}}\right)$
$PE_{pos, 2i+1} = \cos\left(\frac{j}{10000^{2i/d}}\right)$

3. Now starts the actual attention mechanism. I have randomly assigned weights for $W_{q}$, $W_{k}$, and $W_{v}$ matrices that produce Q, K and V on multiplication with encoded input. The multiplication is taken care of by `matMul` global kernel which uses tile-based matrix multiplication strategy. This is launched from the host to calculate the Q, K and V values for each token in the input sequence. Now for each token, we calculate the attention scores using the formula: $a_{i} = {Q_{i} \cdot K_j^T}$. Now the attention scores are normalized and passed through the softmax function to obtain the attention weights. It used the following formula: $softmax(a_{i}) = \frac{e^{a_{i}}}{\sum_{j=1}^{n} e^{a_{j}}}$. For softmax function, we need the sum of the exponentiated scores. This is calculated using `normReduceSum` global kernel which uses parallel reduction strategy to calculate the sum of the exponentiated scores. The sum along witht the attention scores in passed to `softmax` kernel which calculates the softmax values for each token in the input sequence. The final result $z_{i}$ for the ith attention head is obtained by multiplying the softmax values with the V matrix and summing them up, i.e., $z_{i} = {A_{i} \cdot V_j}$.

4. At last, the results from each head is concatenated and multiplied with the output projection matrix $W_{o}$ to obtain the final output. However, this is not implemented because my implementation uses only 1 attention head.

## Usage

The repostory can be cloned using the following command:

```bash
git clone https://github.com/HighkeyPrxneeth/AttentionCUDA.git
```

Navigate into the repository directory using the following command:

```bash
cd AttentionCUDA
```

Compile the code using the following command:

```bash
nvcc -o attention attention.cu
```

Run the code using the following command:

```bash
./attention
```

If you're using windows, you can use the following command:

```bash
attention.exe
```
