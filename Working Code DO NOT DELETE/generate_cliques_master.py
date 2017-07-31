from setup import *
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import random

def choose(k,n):
    try: 
        return int(math.factorial(n)/math.factorial(k)/math.factorial(n-k))
    except:
        return 0

num_verts = 6
ramsey = np.array([2,4])
num_clrs = len(ramsey)
choose_table = np.array([[choose(k,n) for n in range(num_verts+1)] for k in range(ramsey.max()+1)])
num_edges = choose_table[2,num_verts]
num_verts_per_clique = ramsey
num_edges_per_clique = np.array([choose_table[2,ram] for ram in ramsey])
num_cliques = np.array([choose_table[ram,num_verts] for ram in ramsey])
num_cliques_cum = num_cliques.cumsum()

edge_idx_to_pair = np.array(list(it.combinations(range(num_verts),2)))
edge_pair_to_idx = np.zeros([num_verts,num_verts])
for (idx,e) in enumerate(edge_idx_to_pair):
    edge_pair_to_idx[tuple(e)] = idx
edge_pair_to_idx += edge_pair_to_idx.T
np.fill_diagonal(edge_pair_to_idx,num_edges) 


def mtogpu(arr, dtype=None):
    arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)    
    arr_gpu = gpuarray.to_gpu(arr.ravel())
    return arr, arr_gpu

ramsey, ramsey_gpu = mtogpu(ramsey,'uint16')
choose_table, choose_table_gpu = mtogpu(choose_table,'uint32')
num_cliques_cum, num_cliques_cum_gpu = mtogpu(num_cliques_cum,'uint64')
edge_pair_to_idx, edge_pair_to_idx_gpu = mtogpu(edge_pair_to_idx,'uint16')
edge_idx_to_pair, edge_idx_to_pair_gpu = mtogpu(edge_idx_to_pair,'uint16')

#Optionally, return the clique list to python.  TURN ME OFF for large jobs or I will eat all your brains (RAM).
clique_list = np.ones([num_cliques_cum[-1],ramsey.max()]) * num_verts
clique_list, clique_list_gpu = mtogpu(clique_list,'uint16')

print(num_cliques_cum_gpu)


kernel_code ="""
#include <stdio.h>

extern "C"
    #include <stdio.h>
    
    
        __device__ void edge_idx_to_pair_fcn(ushort *edge_idx_to_pair, ushort idx, ushort *v, ushort *w)
    {
        ushort start = 2*idx;
        *v = edge_idx_to_pair[start];
        *w = edge_idx_to_pair[start+1];    
    }



    __device__ void get_assignment(uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx, ushort *clr, uint *clique_idx, ushort *n_verts, ushort *verts, ushort *n_edges, ushort *edges)
    {
        ushort i, j, k, row, col, start;
        uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

        //First, determine which color we are.
        (*clr) = 0;
        while(num_cliques_cum[(*clr)] <= block_idx){(*clr)++;}
        (*n_verts) = ramsey[(*clr)];
        (*n_edges) = choose_table[2*(NUM_VERTS+1)+(*n_verts)];


        //Now, determine which clique_idx in that color we are.
        uint N = block_idx;
        if((*clr) > 0)
        {
            N -= num_cliques_cum[(*clr)-1];
        }
        (*clique_idx) = N;        


        //Now,  determine our list of vertices.
        for(i = 0; i < (*n_verts); i++)
        {
            row = (*n_verts) - i;
            start = row*(NUM_VERTS+1);
            col = 0;
            while(choose_table[start+col] <= N){
                col++;
            }
            col--;
            verts[row-1] = col;
            N -= choose_table[start+col];
        }
        printf("block_idx = %5u, color = %1u, n_verts = %2u, n_edges = %3u, clique_idx = %5u\\n", block_idx, (*clr), (*n_verts), (*n_edges), (*clique_idx));


        //Now,  determine our list of edges.
        k = 0;
        ushort v, w;
        for(i = 0; i < (*n_verts); i++)
        {
            for(j = i+1; j < (*n_verts); j++)
            {
                v = verts[i];
                w = verts[j];
                edges[k] = edge_pair_to_idx[v*NUM_VERTS+w];
                if(block_idx == PRINT_ID)
                {
                    ushort r,s;
                    edge_idx_to_pair_fcn(edge_idx_to_pair, edges[k], &r, &s);
                    printf("edge (%u,%u) -> edge_idx %u -> (%u,%u)\\n",v,w,edges[k],r,s);                    
                }
                k++;
            }
        }
    }
    
    __global__ void run(uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx, ushort *clique_list)
    {
        uint i, start;
        int block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;        
        
        __shared__ ushort verts[MAX_RAMSEY];
        for(i = 0; i < MAX_RAMSEY; i++){verts[i] = NUM_VERTS;}
        
        __shared__ ushort edges[MAX_EDGES];
        for(i = 0; i < MAX_EDGES; i++){edges[i] = choose_table[2*(NUM_VERTS+1)+NUM_VERTS];}
        
        ushort clr, n_verts, n_edges;
        uint clique_idx;
        get_assignment(choose_table, ramsey, num_cliques_cum, edge_idx_to_pair, edge_pair_to_idx, &clr, &clique_idx, &n_verts, verts, &n_edges, edges);        
        
        //Everything beyone this point is optional
        //printf("block_idx = %5u, color = %1u, n_verts = %2u, n_edges = %3u, clique_idx = %5u\\n", block_idx, clr, n_verts, n_edges, clique_idx);
        
        //print from specified block
        if(block_idx == PRINT_ID)
        {
            for(i = 0; i < MAX_RAMSEY; i++)
            {
                printf("%u ",verts[i]);
            }
            printf("\\n");
            
            ushort v, w;
            for(i = 0; i < MAX_EDGES; i++)
            {
                edge_idx_to_pair_fcn(edge_idx_to_pair, edges[i], &v, &w);
                printf("edge_idx %u -> (%u,%u)", edges[i], v, w);
                    printf("\\n");
            }
        }        

        //Optionally, return the clique list to python.  TURN ME OFF for large jobs or I will eat all your brains (RAM).
        start = block_idx*MAX_RAMSEY;
        for(i = 0; i < n_verts; i++)
        {
            clique_list[start+i] = verts[i];
        }
    }
"""

block_dimensions=(1,1,1)
grid_dimensions=(int(num_cliques_cum[-1]),1,1)

regex_fixing_dictionary = {"NUM_VERTS":num_verts, "MAX_RAMSEY":ramsey.max(), "MAX_EDGES":num_edges_per_clique.max(),
                          "PRINT_ID":23}
for key, val in regex_fixing_dictionary.items():
    kernel_code = kernel_code.replace(str(key),str(val))

mod = SourceModule(kernel_code)

run = mod.get_function("run")
run(choose_table_gpu, ramsey_gpu, num_cliques_cum_gpu, edge_idx_to_pair_gpu, edge_pair_to_idx_gpu, clique_list_gpu, block=block_dimensions, grid=grid_dimensions, shared=0)

clique_list = clique_list_gpu.get().reshape(clique_list.shape)
print()
print(clique_list)