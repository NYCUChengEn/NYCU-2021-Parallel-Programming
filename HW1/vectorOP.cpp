#include "PPintrin.h"
// compile command g++ main.cpp serialOP.cpp vectorOP.cpp PPintrin.cpp logger.cpp -O3 -o myexp
// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x, sum, ZeroFloat, Nine;
  __pp_vec_int y, OneInt, ZeroInt;
  __pp_vec_float result;
  __pp_mask maskSelect, maskIsZero, maskIsNotZero, maskIfOverNine;

  ZeroFloat = _pp_vset_float(0.f);
  ZeroInt = _pp_vset_int(0);
  OneInt = _pp_vset_int(1);
  Nine = _pp_vset_float(9.999999);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Going to compute x^y
    
    //check if index out od range
    if(i+VECTOR_WIDTH > N) 
      maskSelect = _pp_init_ones(N-i);
    else
      maskSelect = _pp_init_ones();

    // All zeros
    maskIsNotZero = _pp_init_ones(0);
    maskIfOverNine = _pp_init_ones(0);
    //Load x, y according maskSelect
    _pp_vload_float(x, values + i, maskSelect);
    _pp_vload_int(y, exponents + i, maskSelect);
    sum = _pp_vset_float(1.0);

    // check if exist x[i] == 0
    _pp_veq_float(maskIsZero, x, ZeroFloat, maskSelect);
    _pp_vset_float(sum, 0, maskIsZero);

    // check if exist y[i] == 0
    _pp_vgt_int(maskIsNotZero, y, ZeroInt, maskSelect); 
    
    // Compute
    while(_pp_cntbits(maskIsNotZero) != 0){
      
      _pp_vmult_float(sum, sum, x, maskIsNotZero); // if y != 0, sum = sum * x 

      _pp_vsub_int(y, y, OneInt, maskIsNotZero); // if y != 0, sum = sum * x 

      _pp_vgt_int(maskIsNotZero, y, ZeroInt, maskSelect); // check if exist y[i] == 0
    }

    // check if value bigger than 9.999999
    
    _pp_vgt_float(maskIfOverNine, sum, Nine, maskSelect);
    _pp_vset_float(sum, 9.999999, maskIfOverNine);

    // store value
    _pp_vstore_float(output + i, sum, maskSelect);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float x;
  __pp_mask maskAll, maskFirst;

  float sum, *pt = new float[N];
  sum = 0.0;
  maskAll = _pp_init_ones();
  maskFirst = _pp_init_ones(1);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int cnt = VECTOR_WIDTH;
    // All ones
    

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);

    while(cnt != 1){
      
      cnt = cnt >> 1;

      _pp_hadd_float(x, x);

      _pp_interleave_float(x, x);

    }    

    _pp_vstore_float(pt, x, maskAll);
    sum += pt[0];
  }

  return sum;
}