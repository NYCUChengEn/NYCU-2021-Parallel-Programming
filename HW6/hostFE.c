#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#include <time.h>
#define BLOCKSIZE 8
/* refernece: kernel error message                                 */
/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{

    cl_int cirErrNum;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    // Creat kernel
    cl_int errcode_ret;
    cl_kernel conv = clCreateKernel(*program, "convolution", &errcode_ret);
    // CHECK(errcode_ret, "clCreateKernel");

    // Create command queue
    cl_command_queue myqueue;
    myqueue = clCreateCommandQueue(*context, *device, 0, &cirErrNum);
    // CHECK(cirErrNum, "clCreateCommandQueue");

    // Memory allocate
    cl_mem device_filter, device_image, device_output;
    device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize, filter, &cirErrNum);
    device_image = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize, inputImage, &cirErrNum);
    device_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, imageSize, outputImage, &cirErrNum);

    // cl_mem device_filter, device_image, device_output;
    // device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &cirErrNum);
    // device_image = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, &cirErrNum);
    // device_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &cirErrNum);

    // // Copy data to device
    // cirErrNum = clEnqueueWriteBuffer(myqueue, device_filter, CL_TRUE, 0, filterSize, (void *)filter, 0, NULL, NULL);
    // cirErrNum = clEnqueueWriteBuffer(myqueue, device_image, CL_TRUE, 0, imageSize, (void *)inputImage, 0, NULL, NULL);

    // Set Arguments
    clSetKernelArg(conv, 0, sizeof(cl_mem), (void *)&device_image);
    clSetKernelArg(conv, 1, sizeof(cl_mem), (void *)&device_filter);
    clSetKernelArg(conv, 2, sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(conv, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(conv, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(conv, 5, sizeof(cl_mem), (void *)&device_output);

    // Set local and gobal workgroup sizes
    size_t localws[2] = {BLOCKSIZE, BLOCKSIZE}; // 400 800
    size_t globalws[2] = {imageWidth, imageHeight};

    // Execute kernel
    cirErrNum = clEnqueueNDRangeKernel(myqueue, conv, 2, 0, globalws, localws, 0, NULL, NULL);
    CHECK(cirErrNum, "clEnqueueNDRangeKernel");

    // Copy data from device back to host
    cirErrNum = clEnqueueReadBuffer(myqueue, device_output, CL_TRUE, 0, imageSize, (void *)outputImage, NULL, NULL, NULL);
    // CHECK(cirErrNum, "clEnqueueReadBuffer");

    // release object
    // cirErrNum = clReleaseCommandQueue(myqueue);
    // cirErrNum = clReleaseMemObject(device_filter);
    // cirErrNum = clReleaseMemObject(device_image);
    // cirErrNum = clReleaseMemObject(device_output);
    // cirErrNum = clReleaseKernel(conv);
}
