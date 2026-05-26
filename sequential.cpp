#include <iostream>
#include <chrono> // for precision
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// 5x5 gaussian blur kernel (a weight matrix)
const float kernel[5][5] = {
    {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
    {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
    {6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
    {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
    {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}};

int main(int argc, char **argv)
{
    int width, height, channels;
    // load the image and force it to have 3 color channels (RGB) to simplify math
    unsigned char *img_in = stbi_load(argv[1], &width, &height, &channels, 3);
    // allocate memory for the final blurred image
    unsigned char *img_out = new unsigned char[width * height * 3];

    // start the stopwatch
    auto start = std::chrono::high_resolution_clock::now();

    // loop 1: go through every row of the image (y)
    for (int y = 0; y < height; y++)
    {
        // loop 2: go through every column/pixel in that row (x)
        for (int x = 0; x < width; x++)
        {
            // loop 3: go through the red, green, and blue channels (c)
            for (int c = 0; c < 3; c++)
            {
                float sum = 0.0f;

                // apply the 3x3 matrix to the current pixel and its 8 neighbors
                for (int ky = -2; ky <= 2; ky++)
                {
                    for (int kx = -2; kx <= 2; kx++)
                    {
                        // clamp to edges so we don't read memory outside the image
                        int ny = std::min(std::max(y + ky, 0), height - 1);
                        int nx = std::min(std::max(x + kx, 0), width - 1);

                        // multiply the neighbor's color by the kernel weight and add to total
                        sum += img_in[(ny * width + nx) * 3 + c] * kernel[ky + 2][kx + 2];
                    }
                }
                // save the calculated blurred pixel
                img_out[(y * width + x) * 3 + c] = static_cast<unsigned char>(sum);
            }
        }
    }

    // stop the stopwatch and print the time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = stop - start;
    std::cout << "Sequential Time: " << time.count() << " seconds\n";

    // save the new image to the disk
    stbi_write_png(argv[2], width, height, 3, img_out, width * 3);

    // free the memory to prevent leaks
    stbi_image_free(img_in);
    delete[] img_out;

    return 0;
}