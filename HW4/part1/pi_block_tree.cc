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

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Print off a hello world message
    // printf("Hello world from processor %s, rank %d out of %d processors\n",
    //      processor_name, world_rank, world_size);
    // TODO: binary tree redunction
    static unsigned int seed = world_rank;
    double x, y;
    long long int cnt = 0;
    long long int each_toss = tosses / world_size;
    if ( world_rank == 0 )
        each_toss += tosses % world_size;

    for(long long int i = 0; i < each_toss; ++i){
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if ( x*x + y*y <= 1)
            cnt++;
    }
    
    
    if (world_rank == 0)
    {
        // TODO: PI result
        long long int other_cnt;
        int size = world_size;
        while(size!=1){
            MPI_Recv(&other_cnt, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt += other_cnt;
            size = size / 2;
        }
        
        pi_result = cnt * 4. / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    else{
        int currnt_rank = world_rank, iter = 0;
        while(!( (world_rank>>iter) & 1)){
            // printf("Hello world from processor %s, rank %d out of %d processors\n",
            // processor_name, world_rank, world_size);
            long long int other_cnt;
            MPI_Recv(&other_cnt, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt += other_cnt;
            iter ++ ;
        }
        // printf("Hello world from processor %s, rank %d send to %d, 1>>iter = %d\n",
        //     processor_name, world_rank, world_rank-(1<<iter), 1<<iter);
        MPI_Send(&cnt, 1, MPI_LONG_LONG, world_rank-(1<<iter), 0, MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;
}
