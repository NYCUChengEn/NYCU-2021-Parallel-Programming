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

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the cnt
    long long int cnt, *collect_cnt;

    if( world_rank == 0)
        cnt = Monte_Carlo( tosses / world_size + tosses % world_size, world_rank);
    else
        cnt = Monte_Carlo( tosses / world_size, world_rank);

    // TODO: use MPI_Gather
    if(world_rank == 0)
        collect_cnt = (long long int *) malloc(world_size * sizeof(long long int));

    // Remember to setup the right size
    MPI_Gather(&cnt, 1, MPI_LONG_LONG, collect_cnt, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        for(int i=1; i<world_size; ++i){
            cnt += collect_cnt[i];
        }
        pi_result = cnt * 4. / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---

        free(collect_cnt);
    }
    
    MPI_Finalize();
    return 0;
}
