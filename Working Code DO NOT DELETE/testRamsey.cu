//To compile nvcc testRamsey.cu -o Ramsey -lcudart -run
#define nodes 40
#define N 780
#define triangles 9880
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>

short *graph_CPU, *sub_CPU, *bill_CPU;int *problems_CPU;
short *graph_GPU, *sub_GPU, *bill_GPU;int *problems_GPU;
dim3 dimBlock; //This variable will hold the Dimensions of your block

void AllocateMemory()
{
	//Allocate Device (GPU) Memory, & allocates the value of the specific pointer/array
	cudaMalloc(&graph_GPU,N*sizeof(short));
	cudaMalloc(&sub_GPU,triangles*3*sizeof(short));
	cudaMalloc(&bill_GPU,triangles*3*sizeof(short));
	cudaMalloc(&problems_GPU,2*sizeof(int));
	//Allocate Host (CPU) Memory
	graph_CPU = (short*)malloc(N*sizeof(short));
	sub_CPU = (short*)malloc(triangles*3*sizeof(short));
	bill_CPU = (short*)malloc(triangles*3*sizeof(short));
	problems_CPU = (int*)malloc(2*sizeof(int));
}

void Innitialize()
{
	int i;
  srand(time(NULL));
	for(i = 0; i < N; i++)
	{
		graph_CPU[i] = (short)(rand()%2);
	}
  for(i = 0; i< triangles; i++)//This forces each 3 spots correspond to different nodes
  {
		sub_CPU[3*i] = (short)(rand()%nodes);
		do {
			sub_CPU[3*i+1] = (short)(rand()%nodes);
		} while(sub_CPU[3*i+1]==sub_CPU[3*i]);
		do {
			sub_CPU[3*i+2] = (short)(rand()%nodes);
		} while(sub_CPU[3*i+2]==sub_CPU[3*i]||sub_CPU[3*i+2]==sub_CPU[3*i+1]);

	}
	for(i = 0; i<triangles; i++)//This links node 1 with node 2 and etc.
	{
		bill_CPU[3*i] = sub_CPU[3*i+1];
		bill_CPU[3*i+1] = sub_CPU[3*i+2];
		bill_CPU[3*i+2] = sub_CPU[3*i];
	}
	problems_CPU[0] = 0;
	problems_CPU[1] = 0;
}
//clear up space
void CleanUp(short *graph_CPU,short *sub_CPU, short *bill_CPU, int *problems_CPU, int *problems_GPU, short *graph_GPU,short *sub_GPU,short *bill_GPU)
{
	free(graph_CPU);free(problems_CPU);free(sub_CPU);free(bill_CPU);
	cudaFree(graph_GPU); cudaFree(sub_GPU);cudaFree(bill_GPU);cudaFree(problems_GPU);
}

__global__ void converter(short *edges, short *bill, int n)//This converts the tuple formed by zip(sub,bill) to the corresponding points on the big graph vector
{
	int i;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id < n)
	{																			//Let's say you were examing the edge going with (7,9). The first row takes up 39 spots, the second takes up 38 and etc. So, add up 39 + 38+...+ 34
		if(edges[id] < bill[id]) 						//Now we're at the spot of the big graph vector that corresonds with node 7. But, the edge connecting to node 9 is only two spots down, so just add
		{																		//the difference between the two.
			bill[id] -= edges[id];
			for(i = 0; i < (edges[id]-1); i++)
			{
				bill[id] += (nodes - 1) - i;
			}
		}
		else
		{
			int mabel = bill[id];
			edges[id] -= bill[id];
			for(i = 0; i < (bill[id]-1); i++)
			{
				edges[id] += (nodes -1) - i;
			}
			bill[id] = edges[id];
			edges[id] = mabel;
		}
	}
}
//n is the number of triangles to check, graph is the big graph, edges is the array of which edges to check
__global__ void Triangle_checking(short *graph, short *edges,int *problems, int n)
{
 	  int id = threadIdx.x;//We operating under the assumption that each block checks multiple subgraphs determined by one edges vector
		int k = 0;
		do//This do-while loop lets one block repeat multiple times for the edges array, allowing each thread to check multiple subgraphs
		{
    	if(id + blockDim.x*k< n)
    	{
				int i = id*3;
				/*if(graph[edges[i]]+graph[edges[i+1]]+graph[edges[i+2]]== 3)
				{
					atomicAdd(&problems[0], 1);
				}*/
				if(graph[edges[i]] ==1)//if the first edge is red, check others
				{
					for(int j = i+1; j < i+3; j++)
					{
						if(graph[edges[j]] == 0)//if any are blue, stop
						{
							break;
						}
						else if(j== i+2)//if I've hit the end with out breaking, it must be a red triangle and so add something to problem
						{
							atomicAdd(&problems[0], (int)1);
						}
					}
    		}
			}
			k++;
		}while(k*blockDim.x<n);//This keep the do-while loop running while there is still subgraphs to check
}

int main()
{
	int i;
	timeval start, end;

	AllocateMemory();

	Innitialize();

	/*for(i = 0; i < sub; i++)
	{
		printf("graph[%d] = %d  bill[%d] = %d\n", i, sub_CPU[i], i, bill_CPU[i]);
	}*/

	cudaMemcpyAsync(graph_GPU, graph_CPU, N*sizeof(short), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(sub_GPU, sub_CPU, triangles*3*sizeof(short), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(bill_GPU, bill_CPU, triangles*3*sizeof(short), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(problems_GPU, problems_CPU, 2*sizeof(int), cudaMemcpyHostToDevice);

	gettimeofday(&start, NULL);
	converter<<<1,1024>>>(sub_GPU, bill_GPU, triangles*3);
	Triangle_checking<<<1,1024>>>(graph_GPU, bill_GPU, problems_GPU,triangles);
	gettimeofday(&end, NULL);

	cudaMemcpyAsync(sub_CPU, sub_GPU, triangles*3*sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(bill_CPU, bill_GPU, triangles*3*sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(problems_CPU, problems_GPU, 2*sizeof(int), cudaMemcpyDeviceToHost);

	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

	printf("Time in milliseconds= %.15f the number of problems were %d\n", (time/1000.0), problems_CPU[0]);
	/*
	for(i = 0; i < triangles*3; i++)
	{
		printf("graph[%d] = %d\n ", bill_CPU[i], graph_CPU[bill_CPU[i]]);//sub[%d] = %d bill[%d] = %d\n", i, graph_CPU[bill_CPU[i]], i, sub_CPU[i], i, bill_CPU[i]);
	}*/

	CleanUp(graph_CPU,sub_CPU,bill_CPU,problems_CPU,problems_GPU,graph_GPU,sub_GPU,bill_GPU);

	return(0);
}
