#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel( float lowerX, float lowerY, int width, int height, float stepX, float stepY, 
    int maxIterations, int *decive_m, size_t pitch, int n) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int tx = (blockIdx.x * blockDim.x + threadIdx.x) * n;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // float x = lowerX + tx * stepX;
    float y = lowerY + ty * stepY;

    // float z_re = x;
    float z_im, z_re;
    int *pt = (int * )((char*)decive_m + ty * pitch) + tx;

    for(int j=0; j<n; ++j){

        int i;
        float x = lowerX + (tx+j) * stepX;
        z_im = y;
        z_re = x;
        for (i = 0; i < maxIterations; ++i)
        {

            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
        }
        
        
        pt[j] = i;
    }
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

    int group = 4;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(resX / threadsPerBlock.x / group, resY / threadsPerBlock.y);
    // printf("numBlocks.x: %d, numBlocks.y: %d, n:%d\n", numBlocks.x, numBlocks.y, group);

    
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, resX, resY, stepX, stepY, maxIterations, 
        decive_m, pitch, group);

    cudaMemcpy2D(host_m, sizeof(int) * resX, decive_m, pitch, sizeof(int) * resX, resY, cudaMemcpyDeviceToHost);
    cudaFree(decive_m);
    
    memcpy(img, host_m, size);
    cudaFreeHost(host_m);
}


