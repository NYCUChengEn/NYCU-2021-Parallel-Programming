# include <stdio.h>
# include <pthread.h>
# include <nmmintrin.h>
# include <time.h>
# include <atomic>
# include "./SIMDxorshift/include/simdxorshift128plus.h"
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
// gcc toss.c ./SIMDxorshift/src/xorshift128plus.c ./SIMDxorshift/src/simdxorshift128plus.c -o pi.out -fPIC -std=c99 -O3 -mavx2  -march=native -Wall -Wextra -pedantic -Wshadow -pthread -Iinclude -flto


typedef long long int lli;
__m256 MAX, ONES;
// std :: atomic<lli> sum(0);
lli sum = 0;

void* Monte_Carlo(void* number_of_tosses)
{
    lli NOT = (lli)number_of_tosses;
    lli number_in_circle = 0;
    pthread_t ID = pthread_self();
    __m256i rx, ry;
    __m256 x, y, distance_squared, mask;
    unsigned int hits;

    avx_xorshift128plus_key_t Key;
    avx_xorshift128plus_init(12 ,45, &Key);

    for ( lli toss = 0; toss < NOT; toss+=8) {
        
        // Generate random variable between [-1,1]
        rx = avx_xorshift128plus(&Key);
        ry = avx_xorshift128plus(&Key);

        x = _mm256_cvtepi32_ps(rx); // convert to float to do floating point div
        y = _mm256_cvtepi32_ps(ry);

        x = _mm256_div_ps(x, MAX);
        y = _mm256_div_ps(y, MAX);

        // Compute its distance to orgin
        distance_squared = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));

        mask = _mm256_cmp_ps(distance_squared, ONES, _CMP_LE_OQ);

		hits = _mm256_movemask_ps(mask);
        number_in_circle += _mm_popcnt_u32(hits);
    
    }
    // Add the total hit to sum
    pthread_mutex_lock(&mutex);
    sum += number_in_circle;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main(int argc, char *argv[])
{
    int NUM_THREAD = atoi(argv[1]);
    void *ret;
    lli NUM_TOSS = atoll(argv[2]);
    pthread_t * pthread;
    pthread = new pthread_t[NUM_THREAD];
    
    lli each_tosses = NUM_TOSS/NUM_THREAD;

    MAX = _mm256_set1_ps(INT32_MAX);
    ONES = _mm256_set1_ps(1.0f);

    // Thread create
    for(int i=0; i<NUM_THREAD-1; ++i)
        pthread_create(&pthread[i], NULL, Monte_Carlo, (void *)each_tosses);
    pthread_create(&pthread[NUM_THREAD-1], NULL, Monte_Carlo, (void *)(each_tosses + NUM_TOSS%NUM_THREAD)); //Deal with remain number

    //Wait for all thread ending 
    for(int i=0; i<NUM_THREAD; ++i)
        pthread_join(pthread[i], &ret);
    

    printf("%lf\n", (sum*4) / (double)NUM_TOSS);
}