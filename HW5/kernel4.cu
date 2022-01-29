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

    float z_re = x, z_im = y, z_tmp;
    int i, tmp = maxIterations;
    for (i = 0;  i < maxIterations ; ++i)
    {
        
        if( z_re * z_re + z_im * z_im > 4.f ){
           
            tmp = i; 
            break;
        }

        z_tmp = z_re * z_re - z_im * z_im + x;
        z_im = y + 2.f * z_re * z_im;
        z_re = z_tmp;
    }

    int *pt = (int * )((char*)decive_m + ty * pitch) + tx;
    *pt = tmp;

}

__global__ void SpemandelKernel( float lowerX, float lowerY, int width, int height, float stepX, float stepY, 
    int maxIterations, int *decive_m, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    

    int flag = (ty&1) * 2 - 1 ;
    ty = ((height-1)*(ty&1) -(ty/2) ) * flag ;

    float x = lowerX + tx * stepX;
    float y = lowerY + ty * stepY;

    float z_re = x, z_im = y, z_tmp;
    int i, tmp = maxIterations;
    
    for (i = 0; i < maxIterations; ++i)
    {

        if( (z_re * z_re + z_im * z_im > 4.f) ){
            tmp = i; 
            break;
        }

        z_tmp = z_re * z_re - z_im * z_im + x;
        z_im = y + 2.f * z_re * z_im;
        z_re = z_tmp;
    }

    int *pt = (int * )((char*)decive_m + ty * pitch) + tx;
    // int *pt = (int * )((char*)decive_m + ty * pitch);
    *pt = tmp;
}


// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    size_t pitch;

    // int *host_m;
    // cudaHostAlloc(&host_m, size, cudaHostAllocWriteCombined);
    int *decive_m;

    cudaMallocPitch((void**)&decive_m, &pitch, resX*sizeof(int), resY); 

    // (1600, 1200) = 400
  
    if( upperX==(1.00) && lowerX==(-2.00) && upperY==(1.00) && lowerY==(-1.00) ){
        dim3 threadsPerBlock(2, 16);  //4, 8  2, 16
        dim3 numBlocks(resX  / threadsPerBlock.x, resY  / threadsPerBlock.y );
        SpemandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, resX, resY, stepX, stepY, maxIterations, 
            decive_m, pitch);
    }
    else{
        dim3 threadsPerBlock(4, 8);  //4, 8  2, 16
        dim3 numBlocks(resX  / threadsPerBlock.x, resY  / threadsPerBlock.y );
        mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, resX, resY, stepX, stepY, maxIterations, 
            decive_m, pitch);
    }
        
    cudaDeviceSynchronize();


    // dim3 threadsPerBlock2(32, 1);  //4, 8  2, 16
    // dim3 numBlocks2(resX  / threadsPerBlock.x, );

    // mandelKernel<<< numBlocks2, threadsPerBlock2 >>>(lowerX, lowerY, resX, resY, stepX, stepY, maxIterations, 
    //     decive_m, pitch);
    // cudaDeviceSynchronize();


    cudaMemcpy2D(img, sizeof(int) * resX, decive_m, pitch, sizeof(int) * resX, resY, cudaMemcpyDeviceToHost);
    cudaFree(decive_m);
    // memcpy(img, host_m, size);
    // cudaFreeHost(host_m);


}


