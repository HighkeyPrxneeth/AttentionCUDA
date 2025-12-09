# Attention Mechanism Implementation in CUDA

This repository has a very minimal and basic implementation of the attention mechanism using CUDA.
I wrote this code to exercise my HPC skills and understanding.
This code is not intended for production use and is provided as-is.

## Overview

The code is divided into 4 parts:
1. Setup and initialization of weights and input embeddings. In my implementation, I am not taking input from the user, but rather assigning random values. All values are ranging from 0 to 1 (`(float)rand() / (float)RAND_MAX`).
2. Here, I apply positional encoding to the input embeddings. This is done using the `positional_encoding` global kernel which is launched from the host. It uses the traditional formula for positional encoding, i.e., 
$PE_{i,j} = \sin\left(\frac{j}{10000^{2i/d}}\right)$
