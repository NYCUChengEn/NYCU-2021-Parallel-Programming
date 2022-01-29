#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel( float lowerX, float lowerY, int width, int height, float stepX, float stepY, 
    int maxIterations, int *decive_m, size_t pitch) {
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
    
    int *pt = (int * )((char*)decive_m + ty * pitch) + tx;
    *pt = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    size_t pitch;

    int *host_m;
    cudaHostAlloc(&host_m, size, cudaHostAllocPortable);
    int *decive_m;
    cudaMallocPitch((void**)&decive_m, &pitch, resX*sizeof(int), resY); 


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, resX, resY, stepX, stepY, maxIterations, 
        decive_m, pitch);

    cudaMemcpy2D(host_m, sizeof(int) * resX, decive_m, pitch, sizeof(int) * resX, resY, cudaMemcpyDeviceToHost);
    cudaFree(decive_m);
    
    memcpy(img, host_m, size);
    cudaFreeHost(host_m);
}