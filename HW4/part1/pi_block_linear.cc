#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Set up rand seed
    long long int cnt = 0;
    double x, y;
    static unsigned int seed = world_rank;

    if (world_rank > 0)
    {
        // TODO: handle workers
        long long int each_toss = tosses / world_size;
        for(long long int i = 0; i < each_toss; ++i){
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if ( x*x + y*y <= 1)
                cnt++;
        }
        MPI_Send(&cnt, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        
    }
    else if (world_rank == 0)
    {
        // TODO: master
        long long int each_toss = tosses / world_size + tosses % world_size;
        for(long long int i = 0; i < each_toss; ++i){
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if ( x*x + y*y <= 1)
                cnt++;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        long long int each_cnt;
        for (int i = 1; i < world_size; ++i)
        {
            MPI_Recv(&each_cnt, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt += each_cnt;
        }
        pi_result = cnt * 4. / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
