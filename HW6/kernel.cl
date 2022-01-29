#define BLOCKSIZE 8
#define MAX_HALFFILTER 3
__kernel void convolution(__global __read_only float *inputImage,
                          __constant float *filter, int filterWidth,
                          int imageHeight, int imageWidth,
                          __global __write_only float *outputImage) {

  int halffilterSize = filterWidth / 2;

  int i = get_global_id(1); // row of global
  int j = get_global_id(0);
  float sum = 0;
  int k, l;

  for (k = -halffilterSize; k <= halffilterSize; ++k) {

    if (!(i + k >= 0 && i + k < imageHeight))
      continue;

    for (l = -halffilterSize; l <= halffilterSize; ++l) {

      float filter_value =
          filter[(k + halffilterSize) * filterWidth + l + halffilterSize];

      if (!(j + l >= 0 && j + l < imageWidth) || (filter_value == 0))
        continue;

      sum += inputImage[(i + k) * imageWidth + j + l] * filter_value;
    }
  }

  outputImage[i * imageWidth + j] = sum;
}

// __kernel void convolution(__global __read_only float *inputImage,
//                           __constant float *filter, int filterWidth,
//                           int imageHeight, int imageWidth,
//                           __global __write_only float *outputImage) {

// __local float localImage[BLOCKSIZE + 2 * MAX_HALFFILTER]
//                         [BLOCKSIZE + 2 * MAX_HALFFILTER];

// int halffilterSize = filterWidth / 2;

// int row = get_local_id(1);
// int col = get_local_id(0);

// int i = get_global_id(1); // row of global
// int j = get_global_id(0);

// if ((i - halffilterSize) >= 0 && (j - halffilterSize) >= 0) {

//   localImage[row][col] =
//       inputImage[(i - halffilterSize) * imageWidth + (j - halffilterSize)];

// } else {

//   localImage[row][col] = 0;
// }

// // printf("global: %d %d, local: %d, %d here\n", i, j, row, col);

// if (row < 2 * halffilterSize) {

//   if ((i - halffilterSize + BLOCKSIZE) >= 0 && (j - halffilterSize) >= 0 &&
//       (i - halffilterSize + BLOCKSIZE) < imageHeight &&
//       (j - halffilterSize) < imageWidth)
//     localImage[row + BLOCKSIZE][col] =
//         inputImage[(i - halffilterSize + BLOCKSIZE) * imageWidth +
//                    (j - halffilterSize)];
//   else
//     localImage[row + BLOCKSIZE][col] = 0;
// }

// if (col < 2 * halffilterSize) {

//   if ((i - halffilterSize) >= 0 && (j - halffilterSize + BLOCKSIZE) >= 0 &&
//       (i - halffilterSize) < imageHeight &&
//       (j - halffilterSize + BLOCKSIZE) < imageWidth)
//     localImage[row][col + BLOCKSIZE] =
//         inputImage[(i - halffilterSize) * imageWidth +
//                    (j - halffilterSize + BLOCKSIZE)];
//   else
//     localImage[row][col + BLOCKSIZE] = 0;
// }

// if (row < 2 * halffilterSize && col < 2 * halffilterSize) {
//   if ((i - halffilterSize + BLOCKSIZE) >= 0 &&
//       (j - halffilterSize + BLOCKSIZE) >= 0 &&
//       (i - halffilterSize + BLOCKSIZE) < imageHeight &&
//       (j - halffilterSize + BLOCKSIZE) < imageWidth) {
//     // printf("here!!\n");
//     localImage[row + BLOCKSIZE][col + BLOCKSIZE] =
//         inputImage[(i - halffilterSize + BLOCKSIZE) * imageWidth +
//                    (j - halffilterSize + BLOCKSIZE)];
//   } else {
//     localImage[row + BLOCKSIZE][col + BLOCKSIZE] = 0;
//   }
// }

// barrier(CLK_LOCAL_MEM_FENCE);
// float sum = 0;
// int k, l;

// for (k = -halffilterSize; k <= halffilterSize; ++k) {

//   if (!(i + k >= 0 && i + k < imageHeight))
//     continue;

//   for (l = -halffilterSize; l <= halffilterSize; ++l) {

//     if (!(j + l >= 0 && j + l < imageWidth))
//       continue;

//     float filter_value =
//         filter[(k + halffilterSize) * filterWidth + l + halffilterSize];

//     sum += localImage[row + k + halffilterSize][col + l + halffilterSize] *
//            filter_value;
//   }
// }

// outputImage[i * imageWidth + j] = sum;

// }
