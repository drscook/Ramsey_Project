#import the packages
from __future__ import division
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools
import random
import timeit
import math
import numpy as np
import pandas as pd
rng = np.random.RandomState(42)
#######################################################################################################
#Look into checking some random number of edges and compute probability of no problems
#######################################################################################################
#See how it takes to check all 10-cliques
#######################################################################################################

#This generates a Ramsey Graph where "1" is a red edge and "0" is a blue edge
def Ramsey_graph(nodes = 2):
    df = pd.DataFrame(rng.randint(0, 2, (nodes+1, nodes+1)))
    for i in range(df.shape[0]):
        for j in range(i,df.shape[1]):
            df.iloc[j,i] = 0
    return (df)

def sumtorial(unsigned_int):
    return(np.sum([i for i in range(unsigned_int+1)]))

#f = """import random
#import pandas as pd
#import numpy as np
#import random
#rng = np.random.RandomState(42)
#data = [np.bool(random.getrandbits(1)) for i in range(780)]
#edges = list(rng.randint(0,780, (45)))"""
#
#g = """import random
#import pandas as pd
#import numpy as np
#import random
#rng = np.random.RandomState(42)
#data = [np.bool(random.getrandbits(1)) for i in range(780)]
#edges = list(rng.randint(0,780, (45)))
#[data[i] for i in edges]"""
#print timeit.timeit(g, number = 85000) - timeit.timeit(f, number = 85000) = .0776 seconds
#Based off of this, it will take ~780 seconds to divide up all of the values to the appropiate arrays in Python

def Ramsey_vector(nodes = 2):
    return(pd.Series([np.bool(random.getrandbits(1)) for i in range(sumtorial(nodes))]))

########################################################################################
def sub_graph(data_frame, node_vector):
    return(pd.Series(dict(zip([(node_vector[i],node_vector[j]) for i in range(len(node_vector))
                            for j in range(i+1,len(node_vector))],
                            [data_frame.iloc[i,j] for i in node_vector for j in node_vector]))))

def sub_vector(data_vector, edge_vector):
    return([data_vector[i] for i in edge_vector])

gpu_Ramsey = """
    __global__ void Ramsey(unsigned char *Big_Vector, int length_Big_Vector,unsigned int length_sub_graph, unsigned char *sub_graph, int *problems)
    {
        int id = threadIdx.x + ((blockIdx.x)*(blockDim.x));
        if(id==0)
        {
            unsigned char *subgraph = sub_graph[0];
            if(Big_Vector[subgraph[0]] == 1)
            {
                for(int i = 1; i < length_sub_graph; i++)
                {
                    if(Big_Vector[subgraph[i]] == 0)
                    {
                        break;
                    }
                    else if(i == length_sub_graph - 1 && Big_Vector[subgraph[i]] == 1)
                    {
                        atomicAdd(&problems[0], 1);
                    }
                }
            }
            else if(Big_Vector[subgraph[0]] == 0)
            {
                for(int i = 1; i < length_sub_graph; i++)
                {
                    if(Big_Vector[subgraph[i]] == 1)
                    {
                        break;
                    }
                    else if(i == length_sub_graph - 1 && Big_Vector[subgraph[i]] == 0)
                    {
                        atomicAdd(&problems[0], 1);
                    }
                }
            }
        }
    }
    """

def Ramsey_checker(Big_Vector, sub_graph):
    mod = SourceModule(gpu_Ramsey)
    Ramsey = mod.get_function("Ramsey")
    Big_Vector = np.array(Big_Vector).astype(np.uint8)
    sub_graph = np.array(sub_graph).astype(np.uint8)
    Big_Vector_GPU=gpuarray.to_gpu(Big_Vector)
    sub_graph_GPU=gpuarray.to_gpu(sub_graph)
    length_Big_Vector = np.int16(len(Big_Vector))
    length_sub_graph = np.uint8(3)#np.uint8(len(sub_graph))
    problems = np.array([0,0]).astype(np.int32)
    problems_GPU = gpuarray.to_gpu(problems)
    blocks = int(length_Big_Vector/1024 + 1)
    Ramsey(Big_Vector_GPU, length_Big_Vector, length_sub_graph, sub_graph_GPU,  \
        problems_GPU, block=(1024, 1, 1), grid=(blocks,1,1))
    problems = problems_GPU.get()
    return(problems)#[x for x in B]

data = [random.getrandbits(1) for i in range(780)]
edges = [[1,1,1], [1,0,1]]#list(rng.randint(0,780,(45)))
#test = sub_vector(data, edges)
#print test
print Ramsey_checker(data, edges)
########################################################################################
gpu_triangle_checker = """
    __global__ triangle_checker(unsigned char * Big_Vector, int length_Big_Vector)
    {
        int id = threadIdx.x + blockDim.x*dimBlock.x;

        /*//THE "length_Big_Vector" PART MUST BE HARD CODED BEFORE HAND DON'T FORGET!!!!!!!!!
        /__shared__ shared_Big[length_Big_Vector];
        for(int i = 0; i < length_Big_Vector; i++)
        {
            shared_Big[id] = Big_Vector[id];
        }*/

    }
"""


########################################################################################

#MH_sub_graph(data_frame, node_vector, size)


#This should count up the number of problems, i.e. a red triangle, in a given graph
def problem_counter(data_frame, red = 3, blue = 4):
    dim = data_frame.shape[0]
    nodes = [[i,j] for i in range(dim) for j in range(i+1,dim+1)]
    problems = 0
    reds = 0
    #if np.sum([])
    triangles = 0
    for i in range(1,dim):
        for j in range(i,dim):
            for k in range(j,dim):
                if data_frame.iloc[i,j]+data_frame.iloc[i,k]+data_frame.iloc[j,k] == 3:
                    triangles+=1
    return("number of red triangles are ", triangles,"; number of blue probems is ", problems)



#This is the Metropolis-Hastings algorithm
#Choose starting state.  Store current goodness.
#while steps < n{
    #Randomly choose neighboring state.
    #Test goodness of neighboring state.
    #Random variable within some function f(current goodness, future goodness) with range [0,1]
    #If going to new state
        #Store goodness.  Save state.
    #steps < n.

###################################################
def MH(start, steps, neighbor, goodness, moveprob):
    #  object starting state   |         |
    #         integer steps to be taken for M-H algorithm.
    #                function returning a neighbor of current state
    #                          function for determining goodness.
    #                                    function which takes goodnesses and returns probabilities.

    current = start.copy()#Remove any .copy()
    current_goodness = goodness(current)
    best_goodness = current_goodness
    better_hops = 0
    worse_hops = 0
    stays = 0
    for i in range(steps):
        possible = neighbor(current)
        possible_goodness = goodness(possible)#Combine gen_neighbor and compute comparative goodness function
                                              #Pass on stuff that needed for goodness calculation
        if best_goodness < possible_goodness:
            best_state = possible.copy()
            best_goodness = possible_goodness
        if random.random() < moveprob(current_goodness, possible_goodness):#Flip this two if's
            if current_goodness < possible_goodness :
                better_hops += 1
            else:
                worse_hops += 1
            current = possible.copy()
            current_goodness = possible_goodness
        else:
            stays += 1
    return((best_state, best_goodness, better_hops, worse_hops, stays))

#######################################################################

def MH_swarm(starts, steps, neighbor, goodness, moveprob):
    #        As in MH, but with multiple starts
    walkers = [MH(start, steps, neighbor, goodness, moveprob) for start in starts]
    return sorted(walkers, key = lambda x: x[1], reverse = True)
