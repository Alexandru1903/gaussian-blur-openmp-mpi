#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <omp.h> // both librarieis are needed for this implementation

// STB headers
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// gaussian blur 5x5 kernel
const float kernel[5][5] = {
    {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
    {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
    {6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
    {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
    {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}};

void blur(unsigned char *in, unsigned char *local_out, int width, int height, int y_start, int y_end)
{
// the OpenMP directive added to the MPI logic
#pragma omp parallel for schedule(dynamic)
    // go through every row of the image (y)
    for (int y = y_start; y < y_end; y++)
    {
        // go through every column/pixel in that row (x)
        for (int x = 0; x < width; x++)
        {
            // go through each color channel (R, G, B)
            for (int c = 0; c < 3; c++)
            {
                float sum = 0.0f;

                // convolve the kernel with the neighboring pixels
                for (int ky = -2; ky <= 2; ky++)
                {
                    for (int kx = -2; kx <= 2; kx++)
                    {
                        // clamp to edges so we don't read memory outside the image
                        int ny = std::min(std::max(y + ky, 0), height - 1);
                        int nx = std::min(std::max(x + kx, 0), width - 1);

                        // multiply the neighbor's color by the kernel weight and add to total
                        sum += in[(ny * width + nx) * 3 + c] * kernel[ky + 2][kx + 2];
                    }
                }
                // write to (y - y_start) because this is a local sub-array for the worker
                local_out[((y - y_start) * width + x) * 3 + c] = static_cast<unsigned char>(sum);
            }
        }
    }
}

int main(int argc, char **argv)
{
    // initializing mpi environment and getting rank/size info
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 0, height = 0, channels = 3;
    unsigned char *img_in = nullptr;

    // sync time measurement across all processes
    double start_time = MPI_Wtime();

    // rank 0 handles file loading
    if (rank == 0)
    {
        img_in = stbi_load(argv[1], &width, &height, nullptr, 3);
    }

    // broadcast dimensions to all workers so they can allocate memory
    int dims[2] = {width, height};
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    width = dims[0];
    height = dims[1];

    // non-root workers allocate memory for the image after receiving dimensions
    if (rank != 0)
        img_in = new unsigned char[width * height * 3];
    MPI_Bcast(img_in, width * height * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // divide the image rows equally among the mpi processes (last process takes any remaining rows if height is not perfectly divisible by size)
    int rows_per_proc = height / size;
    int y_start = rank * rows_per_proc;
    int y_end = (rank == size - 1) ? height : y_start + rows_per_proc;
    int local_rows = y_end - y_start;

    // allocate memory only for this worker's chunk
    unsigned char *local_out = new unsigned char[local_rows * width * 3];

    // the hibrid function where mpi divides the work and openmp parallelizes the convolution within each worker's chunk
    blur(img_in, local_out, width, height, y_start, y_end);

    // gather the blurred chunks back to the root process
    std::vector<int> counts(size), displs(size);
    unsigned char *img_out = nullptr;

    if (rank == 0)
    {
        img_out = new unsigned char[width * height * 3];
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int r_start = i * (height / size);
            int r_end = (i == size - 1) ? height : r_start + (height / size);
            counts[i] = (r_end - r_start) * width * 3; // number of bytes for this worker's chunk
            displs[i] = offset;                        // byte offset in the final image where this worker's chunk should be placed
            offset += counts[i];
        }
    }

    // gather the blurred chunks back to the root process
    MPI_Gatherv(local_out, local_rows * width * 3, MPI_UNSIGNED_CHAR,
                img_out, counts.data(), displs.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // rank 0 writes the output image and prints the time
    if (rank == 0)
    {
        double time = MPI_Wtime() - start_time;
        std::cout << "Hybrid time: " << time << " seconds" << std::endl;
        stbi_write_png(argv[2], width, height, 3, img_out, width * 3);
        delete[] img_out;
    }

    // cleanup
    delete[] local_out;
    if (rank != 0)
        delete[] img_in;
    else
        stbi_image_free(img_in);

    MPI_Finalize();
    return 0;
}