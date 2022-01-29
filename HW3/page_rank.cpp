#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//


void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  bool *mask;
  double *score_new, *weight;

  // Memory allocate
  score_new = new double [numNodes];
  weight = new double [numNodes];
  mask = new bool [numNodes]; 

  // Initialization
  memset(mask, 1, numNodes); // true means it have outgoing edge

  #pragma omp parallel for
  for(int i=0; i<numNodes; ++i){

    solution[i] = equal_prob;
    int outgoing_num = outgoing_size(g, i);
    
    if(outgoing_num != 0)
      weight[i] = 1./outgoing_num;
    else
      mask[i] = 0; 
    
  }

  bool converged = true;
  double cst = (1.0-damping) / numNodes, no_outgoing_sum, global_diff;
  // printf("finish\n");
  // Start to iteration
  while(converged){
    
    // Initialize value
    global_diff = 0.0;
    no_outgoing_sum = 0.0;

    // Sum over all nodes v in graph with no outgoing edges
  #pragma omp parallel for reduction(+: no_outgoing_sum)
    for(int j=0; j<numNodes; ++j)
      if(!mask[j])
        no_outgoing_sum += solution[j];
    no_outgoing_sum *= equal_prob;  

  #pragma omp parallel for reduction(+: global_diff)
    for(int i=0; i<numNodes; ++i){
      
      double sum = 0;
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      for (const Vertex* v=start; v!=end; v++)
        sum += weight[*v] * solution[*v];
      
      sum = (sum + no_outgoing_sum) * damping + cst;
      score_new[i] = sum;
      global_diff += fabs(sum - solution[i]);
    }

    // update the solution
    memcpy(solution, score_new, numNodes*sizeof(double));

    // check if it converged
    converged = (global_diff > convergence);
  }

  // Release the memory
  delete [] weight;
  delete [] score_new;
  delete [] mask;
  
}
/*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     solution[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { solution[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * solution[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - solution[vi]) };
       converged = (global_diff < convergence)
     }

   */