#!/bin/bash

# 1. CONFIGURATION & COMPILATION
OMP_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
OUT="out.png"

# Choosing 5 images to prove what's required:
IMG_1080="images/1080_1.png"                           # For Weak Scaling
IMG_2K_BATCH=("images/2k_1.png" "images/2k_2.png" "images/2k_3.png") # For Batch / Weak Scaling
IMG_4K="images/4k_1.jpg"                               # For Strong Scaling / Speedup

echo "Compiling"
g++ sequential.cpp -o sequential
g++ $OMP_FLAGS openmp.cpp -o openmp
mpic++ mpi.cpp -o mpi
mpic++ $OMP_FLAGS hibrid.cpp -o hibrid
echo "Done."

# 2. SPEEDUP, STRONG SCALING & OVERHEAD 
# Proves: Speedup vs Sequential, Strong Scaling (descending time), and MPI Overhead.
echo -e "\nTEST 1: STRONG SCALING & OVERHEAD (4K Image)"
echo "Sequential Baseline:"
./sequential $IMG_4K $OUT

for cores in 2 4 8; do
    echo "Testing with $cores Cores:"
    OMP_NUM_THREADS=$cores ./openmp $IMG_4K $OUT
    mpirun -np $cores ./mpi $IMG_4K $OUT
done

echo "Hybrid (2 procs x 4 threads)"
OMP_NUM_THREADS=4 mpirun -np 2 ./hibrid $IMG_4K $OUT


# 3. WEAK SCALING 
# Proves: Execution time stays flat when workload and hardware scale together.
echo -e "\nTEST 2: WEAK SCALING"
echo "1 Core   + 1080p:"
mpirun -np 1 ./mpi $IMG_1080 $OUT
echo "2 Cores  + 2K:"
mpirun -np 2 ./mpi ${IMG_2K_BATCH[0]} $OUT
echo "4 Cores  + 4K:"
mpirun -np 4 ./mpi $IMG_4K $OUT


# 4. BATCH TEST
# Proves: Efficiency based on the number of images.
echo -e "\nTEST 3: BATCH PROCESSING (3 x 2K Images)"
echo "Processing 3 images Sequentially:"
for img in "${IMG_2K_BATCH[@]}"; do
    ./sequential $img $OUT
done

echo "Processing 3 images with OpenMP (8 Threads):"
for img in "${IMG_2K_BATCH[@]}"; do
    OMP_NUM_THREADS=8 ./openmp $img $OUT
done

echo -e "\nTests complete!"