#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

long long int Monte_Carlo(long long int each_toss, unsigned int seed){

    long long int cnt = 0;
    double x, y;

    for(long long int i = 0; i < each_toss; ++i){
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if ( x*x + y*y <= 1)
            cnt++;
    }

    return cnt;
}

bool fnz (long long int *collect_cnt, int world_size, long long int *sum_of_cnt)
{
    int cnt = 0;

    for(int i=1; i<world_size; ++i){

        if(collect_cnt[i] != 0){
            cnt++;
            if(collect_cnt[i] != -1){
                // printf("rank %d: %lld\n",i, collect_cnt[i]);
                *sum_of_cnt += collect_cnt[i];
                collect_cnt[i] = -1;
            }        
        }
    }
    return (cnt == world_size-1);
}


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the cnt
    long long int cnt, sum_of_cnt = 0;

    if( world_rank == 0)
        cnt = Monte_Carlo( tosses / world_size + tosses % world_size, world_rank);
    else
        cnt = Monte_Carlo( tosses / world_size, world_rank);
    

    if (world_rank == 0)
    {
        // Master

       // Use MPI to allocate memory for the target window
       long long int *collect_cnt;
       MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &collect_cnt);

       // Init
       for (int i = 0; i < world_size; i++){
          collect_cnt[i] = 0;
       }

       // Create a window. Set the displacement unit to sizeof(int) to simplify
       // the addressing at the originator processes
       MPI_Win_create(collect_cnt, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL,
          MPI_COMM_WORLD, &win);

       bool ready = 0;
       while (!ready)
       {
          // Without the lock/unlock schedule stays forever filled with 0s
          MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
          ready = fnz(collect_cnt, world_size, &sum_of_cnt);
          MPI_Win_unlock(0, win);
       }

       // rank 0
       sum_of_cnt += cnt;
       // Free the allocated memory
       MPI_Free_mem(collect_cnt);
    }
    else
    {
       // Workers
       MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

       // Register with the master
       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
       MPI_Put(&cnt, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
       MPI_Win_unlock(0, win);

       
    //    printf("Worker %d done\n", world_rank);
    }

    // Release the window
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = sum_of_cnt * 4. / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}