#include <iostream>
#include <algorithm>
#include <omp.h> // Required for OpenMP functions and pragmas

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

void blur(unsigned char *in, unsigned char *out, int width, int height, int y_start, int y_end)
{
// compiler automatically divides this loop among CPU cores
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
                // save the calculated blurred pixel
                out[(y * width + x) * 3 + c] = static_cast<unsigned char>(sum);
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./openmp <input.jpg> <output.png>" << std::endl;
        return 1;
    }

    int width, height, channels;
    // load the image (force 3 channels for RGB) and allocate memory for the output
    unsigned char *img_in = stbi_load(argv[1], &width, &height, &channels, 3);
    unsigned char *img_out = new unsigned char[width * height * 3];

    // OpenMP precision timer
    double start = omp_get_wtime();

    // call function to blur entire image (from row 0 to height)
    blur(img_in, img_out, width, height, 0, height);

    // calculate and print the elapsed time
    double time = omp_get_wtime() - start;
    std::cout << "OpenMP time: " << time << " seconds" << std::endl;

    // output and cleanup (freeing the memory)
    stbi_write_png(argv[2], width, height, 3, img_out, width * 3);
    stbi_image_free(img_in);
    delete[] img_out;

    return 0;
}