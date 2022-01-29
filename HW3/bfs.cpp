#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
// #define VERBOSE 1
void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *mf,
    bool *bitmap,
    bool *new_bitmap)
{
    int sum = 0;
    
    #pragma omp parallel for reduction(+:sum)  schedule(static, 1)
    for (int i = 0; i < frontier->count; i++)
    {
        int node = frontier->vertices[i];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                        ? g->num_edges
                        : g->outgoing_starts[node + 1];
        // attempt to add all neighbors to the new frontier
        
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            if(distances[outgoing] == NOT_VISITED_MARKER){
                new_bitmap[outgoing] = 1;
                if( __sync_bool_compare_and_swap( distances+outgoing, NOT_VISITED_MARKER, distances[node] + 1) ){   
                    int index = __sync_fetch_and_add(&(new_frontier->count), 1);
                    new_frontier->vertices[index] = outgoing;
                    sum += outgoing_size(g, outgoing); 
                    
                }
            }
            
        }
    }
    

    *mf = sum;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes); // Number of vertices in the graph
    vertex_set_init(&list2, graph->num_nodes); // Number of vertices in the graph

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *bitmap = new bool [graph->num_nodes];
    bitmap = (bool *)calloc( graph->num_nodes, sizeof( bool ) );
    bitmap[0] = 1;

    int mf = 0;
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;  // id: 0
    sol->distances[ROOT_NODE_ID] = 0; // id: 0 = root = distance 0

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier); // vertex_set -> count = 0

        top_down_step(graph, frontier, new_frontier, sol->distances, &mf, bitmap, bitmap);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free (bitmap);
}

void bottom_up_step(Graph g,
    int *distances,
    int prev_distance,
    int &cnt,
    int *mf,
    bool *bitmap,
    bool *new_bitmap)
{
    // int node = frontier->vertices[0];
    // printf("node:id %d\n", frontier->vertices[0]);
    int sum = 0;
    int curr_distance = prev_distance + 1;
    cnt = 0;
    #pragma omp parallel for reduction(+ : cnt, sum) schedule(static, 1)
    for(int i=0; i<g->num_nodes; ++i){

        if(distances[i] == NOT_VISITED_MARKER){ 
            
            // Go through i's neighbor
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];
            
            for(int neighbor = start_edge; neighbor < end_edge; neighbor++){
                
                int incoming = g->incoming_edges[neighbor];
                if(bitmap[incoming]){ 
                    // Since whenn bitmap[incoming]) == 1, it must be the frontier in this situation, 
                    // If it is a previous frontier, it would occur -><-
                    distances[i] = prev_distance + 1;
                    new_bitmap[i] = 1;
                    sum += outgoing_size(g, i);
                    cnt += 1;
                    break;
                }

            }
        }  
    }
    *mf = sum;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    

    bool *bitmap, *new_bitmap;
    bitmap = (bool *)calloc( graph->num_nodes, sizeof( bool ) );
    new_bitmap = (bool *)calloc( graph->num_nodes, sizeof( bool ) );
    bitmap[0] = 1;
    new_bitmap[0] = 1;
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0; // id: 0 = root = distance 0
    int pre_distance = 0, cnt = 1, mf = 0;

    
    // cnt means # of frontier
    while (cnt != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        // vertex_set_clear(new_frontier); // new_frontier -> count = 0

        bottom_up_step(graph, sol->distances, pre_distance++, cnt, &mf, bitmap, new_bitmap);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", cnt, end_time - start_time);
#endif

        bool *tmp;
        tmp = bitmap;
        bitmap = new_bitmap;
        new_bitmap = tmp;
        
    }
    free( bitmap ) ;
    free( new_bitmap ) ;
}

void bfs_hybrid(Graph graph, solution *sol)
{
    int mf = 0, nf, mu = graph->num_edges, alpha = 14, beta = 24, pre_cnt, pre_distance = 0;

    // Initialization
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes); // Number of vertices in the graph
    vertex_set_init(&list2, graph->num_nodes); // Number of vertices in the graph

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *bitmap, *new_bitmap, mode = true, ifgrowing = true;
    bitmap = (bool *)calloc( graph->num_nodes, sizeof( bool ) );
    new_bitmap = (bool *)calloc( graph->num_nodes, sizeof( bool ) );
    bitmap[0] = 1;
    new_bitmap[0] = 1;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;  // id: 0
    sol->distances[ROOT_NODE_ID] = 0;

    // Start iteration
    while(frontier->count != 0){
        
        pre_cnt = frontier->count;
        nf = frontier->count;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        // printf("mf:%d, (1.*mu/alpha):%lf, nf:%d, (n/beta):%lf\n", mf, 1.*mu/alpha, nf, 1.*graph->num_edges/beta);
        if(mode & mf > 1.*mu/alpha & ifgrowing){
            // printf("to bottom up\n");
            mode = false;  // turn to bottom up
        }
        else if(!mode & nf < 1.*graph->num_edges/beta & !ifgrowing){ 
            // printf("to top down\n");
            mode = true;  // turn to top down
            frontier->count = 0;
            
            for(int i=0; i<graph->num_nodes; ++i){
                if(sol->distances[i] == pre_distance){
                    frontier->vertices[frontier->count] = i;
                    frontier->count = frontier->count + 1;
                }
                    
            }
            
        }
        // printf("mf:%d, (1.*mu/alpha):%lf, nf:%d, (n/beta):%lf, frontier=%-10d, mode:%d\n", mf, 1.*mu/alpha, nf, 1.*graph->num_edges/beta, frontier->count, mode);

        mu -= mf;
        mf = 0;

        // printf("mf:%d, (1.*mu/alpha):%lf, nf:%d, (n/beta):%lf, frontier=%-10d, mode:%d\n", mf, 1.*mu/alpha, nf, 1.*graph->num_edges/beta, frontier->count, mode);
        if(mode){
            
            vertex_set_clear(new_frontier);
            top_down_step(graph, frontier, new_frontier, sol->distances, &mf, bitmap, new_bitmap);
            vertex_set *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
        else{
            bottom_up_step(graph, sol->distances, pre_distance, frontier->count, &mf, bitmap, new_bitmap);
        }
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("mode: %d, frontier=%-10d %.4f sec\n", mode, frontier->count, end_time - start_time);
#endif
        pre_distance++;
        ifgrowing = (pre_cnt < frontier->count);

        bool *tmp;
        tmp = bitmap;
        bitmap = new_bitmap;
        new_bitmap = tmp;
    }

    free( bitmap );
    free( new_bitmap );
}
