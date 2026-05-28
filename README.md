# Performance Comparison of Parallel Image Processing (Gaussian Blur)

This repository contains the source code and performance analysis for a Parallel and Distributed Algorithms university project. It implements a 5x5 Gaussian Blur filter using different parallelization strategies to compare their efficiency against a sequential baseline.

## Authors
* **Cipu Marinel** 
* **Bancu Alexandru-Georgian**

## Parallelization Strategies
* **Sequential:** Baseline C++ implementation processing pixels one-by-one.
* **OpenMP:** Shared-memory parallelization utilizing CPU threads.
* **MPI:** Distributed-memory parallelization simulating a network cluster.
* **Hybrid:** Combines MPI for memory distribution and OpenMP for local thread-level processing.

## Prerequisites
This project was built and tested on an **Apple Silicon M4 (10-Core CPU)**. To compile and run it locally, you will need:
* macOS with Apple Clang (`g++`)
* Homebrew
* OpenMP (`brew install libomp`)
* OpenMPI (`brew install open-mpi`)

## How to Run
An automated bash script (`benchmark.sh`) is included to easily compile all implementations and run the full suite of benchmarks (Strong Scaling, Weak Scaling, and Batch Processing).

1. Make the script executable:
   ```bash
   chmod +x benchmark.sh