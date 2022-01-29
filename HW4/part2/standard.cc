#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "main.cc"
#define MASTTER 0
// #define VERBOSE 1
struct SizeInfo{
    int n, m, l;
};

void load_matrix(int **A, int **B, int **C, SizeInfo &size_info);

void stardard_method(const int *A, const int *B, SizeInfo &size_info, int world_rank, int world_size);

void matrix_multiplication();

void print_matrix(int *A, int n, int m);

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{

#ifdef VERBOSE
        double start_time = MPI_Wtime();
#endif

    int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char *input_buf, *p;

    int buf[3];
    if (world_rank == MASTTER){
        // scanf("%d %d %d", buf, buf+1, buf+2);  
        input_buf = (char *)malloc(sizeof(char) * (1<<29));
        p = input_buf;
        int size = fread(input_buf, sizeof(char), 1<<29, stdin);

        for(int i=0; i<3; ++i){

            buf[i] = 0;
            while( *p != ' ' && *p != '\n'){
                buf[i] = buf[i] * 10 + *p - '0';
                ++p;
            }
            ++p;
        }

    }

    MPI_Bcast(buf, 3, MPI_INT, 0,  MPI_COMM_WORLD);
    (*n_ptr) = buf[0];
    (*m_ptr) = buf[1];
    (*l_ptr) = buf[2];

    // MPI_Bcast(n_ptr, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    // MPI_Bcast(m_ptr, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    // MPI_Bcast(l_ptr, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    // printf("rank: %d Finish Bcast\n", world_rank);

#ifdef VERBOSE
        double end_time = MPI_Wtime();
        printf("MPI Bcast n,m,l time: %lf Seconds in rank %d\n", end_time - start_time, world_rank);
#endif
    // Input & Construct array

    int avgrow = (*n_ptr) / world_size;

#ifdef VERBOSE
        start_time = MPI_Wtime();
#endif

    if (world_rank == MASTTER){

        (*a_mat_ptr) = (int *) calloc((*n_ptr) * (*m_ptr), sizeof(int)); 
        (*b_mat_ptr) = (int *) calloc((*m_ptr) * (*l_ptr), sizeof(int));     


        for(int cnt =0; cnt < (*n_ptr) * (*m_ptr); ++p)
            if( *p == ' '){
                ++cnt;
            }
            else if( '0' <= *p && *p <= '9'){
                (*a_mat_ptr)[cnt] = (*a_mat_ptr)[cnt] * 10 + *p  - '0';
            }

        for(int cnt =0; cnt < (*m_ptr) * (*l_ptr); ++p)
            if( *p == ' '){
                ++cnt;
            }
            else if( '0' <= *p && *p <= '9'){
                (*b_mat_ptr)[cnt] = (*b_mat_ptr)[cnt] * 10 + *p  - '0';
            }

        free(input_buf);
        // (*a_mat_ptr) = (int *) malloc((*n_ptr) * (*m_ptr) * sizeof(int));
        // (*b_mat_ptr) = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));

        // for(int i=0; i<(*n_ptr); ++i){
        //     int offset = i * (*m_ptr);
        //     for(int j=0; j<(*m_ptr); ++j){    
        //         scanf("%d", &(*a_mat_ptr)[offset + j]);
        //     }
        // }
            
                
        // for(int i=0; i<(*m_ptr); ++i){
        //     int offset = i * (*l_ptr);
        //     for(int j=0; j<(*l_ptr); ++j){    
        //         scanf("%d", &(*b_mat_ptr)[offset + j]);
        //     }
        // }

    }
    else{
        
        (*a_mat_ptr) = (int *) malloc( avgrow * (*m_ptr) * sizeof(int));
        (*b_mat_ptr) = (int *) malloc((*m_ptr) * (*l_ptr) * sizeof(int));
        
    }

    if(world_rank == MASTTER) {

        int count = avgrow * (*m_ptr);
        for(int i=1; i<world_size; ++i)
            MPI_Send(&(*a_mat_ptr)[(i-1) * avgrow * (*m_ptr)], count, MPI_INT, i, 0, MPI_COMM_WORLD);
        
        MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0,  MPI_COMM_WORLD);
    }
    else{

        int count = avgrow * (*m_ptr);
        MPI_Recv(*a_mat_ptr, count, MPI_INT, MASTTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0,  MPI_COMM_WORLD);
    }

#ifdef VERBOSE
        end_time = MPI_Wtime();
        printf("MPI input time: %lf Seconds in rank %d\n", end_time - start_time, world_rank);
#endif

}

void destruct_matrices(int *A, int *B){

    // int world_rank;
	// MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    free(A);
    free(B);
    
}
void matrix_multiply(const int n, const int m, const int l,const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    SizeInfo size_info;
    size_info.n = n;
    size_info.m = m;
    size_info.l = l;

    stardard_method(a_mat, b_mat, size_info, world_rank, world_size);

}


void print_matrix(int *A, int n, int m){

    for(int i=0; i<n; ++i){
        int offset = i * m;
        for(int j=0; j<m; ++j)
            printf("%d ", A[offset+j]);
        printf("\n");
    }
}

void stardard_method(const int *A, const int *B, SizeInfo &size_info, int world_rank, int world_size)
{
    int *C;
    int averow = size_info.n / world_size;

    if(world_rank == MASTTER){
        
        C = (int *) calloc(size_info.n * size_info.l, sizeof(int)); 
        
        // Calculate its part

#ifdef VERBOSE
        double start_time = MPI_Wtime();
#endif


        int offset_A, offset_C;
        for(int i= averow * (world_size-1); i<size_info.n; ++i){
            offset_A = i * size_info.m;
            for(int j=0; j<size_info.l; ++j){

                offset_C = i * size_info.l + j;

                for(int k=0; k<size_info.m; ++k){
                    // printf("C: %d, A: %d B: %d\n", offset_C, offset_A + k, k*size_info.l + j);
                    C[offset_C] += A[offset_A + k] * B[k*size_info.l + j];
                }
            }
        }
        // printf("Finsh calculate\n");
        MPI_Request  *reqs = (MPI_Request *) malloc((world_size-1) * sizeof(MPI_Request));
        MPI_Status  *statuses = (MPI_Status *) malloc((world_size-1) * sizeof(MPI_Status));


#ifdef VERBOSE
        double end_time = MPI_Wtime();
        printf("MPI calculate time: %lf Seconds in rank %d\n", end_time - start_time, world_rank);
#endif


#ifdef VERBOSE
        start_time = MPI_Wtime();
#endif
        for(int i=1; i<world_size; ++i){
            // MPI_Irecv(&C[(i-1) * averow * size_info.l], averow * size_info.l, MPI_INT, i, 0, MPI_COMM_WORLD, &reqs[i-1] );
            MPI_Recv(&C[(i-1) * averow * size_info.l], averow * size_info.l, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
            
        // MPI_Waitall(world_size-1, reqs, statuses);

#ifdef VERBOSE
        end_time = MPI_Wtime();
        printf("MPI Irecv time: %lf Seconds\n", end_time - start_time);
#endif

        ////////////////////////////////////////////////////////////////////
        //////////////////// Show the result ///////////////////////////////
        ////////////////////////////////////////////////////////////////////

        print_matrix(C, size_info.n, size_info.l);
    
        free(C);
        free(reqs);
        free(statuses);
    }
    else{ 

        ////////////////////////////////////////////////////////////////////
        ////////////////////  part of worker ///////////////////////////////
        ////////////////////////////////////////////////////////////////////

        C = (int *) calloc(averow * size_info.l, sizeof(int));

        MPI_Request  *reqs = (MPI_Request *) malloc(2 * sizeof(MPI_Request));
        MPI_Status  *statuses = (MPI_Status *) malloc(2 * sizeof(MPI_Status));

        // Calculate its part

#ifdef VERBOSE
        double start_time = MPI_Wtime();
#endif

        int offset_A, offset_C;
        for(int i= 0; i<averow; ++i){
            offset_A = (i) * size_info.m;
            for(int j=0; j<size_info.l; ++j){

                offset_C = (i) * size_info.l + j;

                for(int k=0; k<size_info.m; ++k){
                    // printf("C[%d]: %d, A[%d]: %d B[%d]: %d\n", offset_C, C[offset_C], offset_A + k,buf_A[offset_A + k], k*size_info.l + j, buf_B[k*size_info.l + j]);
                    C[offset_C] += A[offset_A + k] * B[k*size_info.l + j];
                }
            }
        }

#ifdef VERBOSE
        double end_time = MPI_Wtime();
        printf("MPI calculate time: %lf Seconds in rank %d\n", end_time - start_time, world_rank);
#endif
        MPI_Request req;
        MPI_Status status;
        int cnt = averow * size_info.l;
        // printf("Result of C in Rand %d\n", world_rank);
        // print_matrix(C, averow, size_info.l);
        // MPI_Isend(C, cnt, MPI_INT, MASTTER, 0, MPI_COMM_WORLD, &req);
        // MPI_Wait(&req, &status);
        MPI_Send(C, cnt, MPI_INT, MASTTER, 0, MPI_COMM_WORLD);

        free(C);
        free(reqs);
        free(statuses);
    }

    
    

}