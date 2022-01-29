#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel( float lowerX, float lowerY, int width, float stepX, float stepY, 
int maxIterations, int *output_m) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + tx * stepX;
    float y = lowerY + ty * stepY;

    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < maxIterations; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    
    output_m[ty * width+ tx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);

    int *host_m;
    host_m = (int * ) malloc (size);
    int *output_m;
    cudaMalloc(&output_m, size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, resX, stepX, stepY, maxIterations, output_m);

    cudaMemcpy(host_m, output_m, size, cudaMemcpyDeviceToHost);
    cudaFree(output_m);
    
    memcpy(img, host_m, size);
    free(host_m);
}
