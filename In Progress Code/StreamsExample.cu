/*What this code does:
	-Generates 245,760 random numbers between 0 and 100 inclusive, using 72 for the random seed, stores them into an array.
	-Uses 3 streams to add 8,192 at a time each. 
	-Copies the sum of each chunk of 8,192 numbers to the CPU.
	-Sum each chunk's sum on the CPU.
To compile: nvcc StreamsExample.cu -run
*/
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define rnd( x ) (x*(float)rand())/((float)RAND_MAX)
#define BIGN 245760
#define N 8192

dim3 dimBlock;
int blocks = (N-1)/(1024) + 1;

dim3 dimGrid;

float *rand_num_CPU, *rand_sum_CPU;
float *rand_num_GPU, *rand_sum_GPU;

void AllocateMemory()
{	
	rand_num_CPU = (float*)malloc(sizeof(float)*BIGN);//cudaHostAlloc((void**)&rand_num_CPU,sizeof(int)*BIGN,cudaHostAllocDefault);
	rand_sum_CPU = (float*)malloc(sizeof(float)*30);//cudaHostAlloc((void**)&rand_sum_CPU,sizeof(int)*30,cudaHostAllocDefault);
	cudaMalloc(&rand_sum_GPU, 30*sizeof(float));
	cudaMalloc(&rand_num_GPU, sizeof(float)*N);
}

int Innitialize()
{
	int i;
	dimBlock.x = 1024;
	dimGrid.x = blocks;
	srand(72);
	for(i = 0; i<BIGN; i++)
	{
		rand_num_CPU[i] = (float)(rnd(100));
	}
	for(i = 0; i<30; i++)
	{
		rand_sum_CPU[i] = 0;
	}
	return(1);
}

__global__ void Add_em_up(float *rand_num_GPU, float *rand_sum_GPU,int stream_number, int loop_number, int n)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if(id < n)
	{
		atomicAdd(&rand_sum_GPU[3*loop_number + stream_number-1], rand_num_GPU[id]);	
	}
}

void Cleanup(float *rand_num_CPU, float *rand_sum_CPU, float *rand_num_GPU, float *rand_sum_GPU)
{
	free(rand_num_CPU); free(rand_sum_CPU);
	cudaFree(rand_num_GPU); cudaFree(rand_sum_GPU);
}

void finalSum(float *rand_sum_CPU,int n)
{
	for(int i = 1; i<n; i++)
	{
		rand_sum_CPU[0] += rand_sum_CPU[i];
	}
}

int main()
{
	timeval start, end;
	int i;
	gettimeofday(&start, NULL);
	
	AllocateMemory();
	Innitialize();
	//creating the three different streams
	cudaStream_t stream_one, stream_two, stream_three;
	cudaStreamCreate(&stream_one);
	cudaStreamCreate(&stream_two);
	cudaStreamCreate(&stream_three);
	//makes sure that streams will be useful
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	
	if(!prop.deviceOverlap)
	{
		printf("Device will not handle overlaps, so no speed up from streams");
		return(1);
	}
	//each stream is summing up 8192 numbers of the total and there is three streams running
	cudaMemcpyAsync(rand_sum_GPU, rand_sum_CPU, 30*sizeof(float), cudaMemcpyHostToDevice);
	for(i=0; i<10; i++)
	{
		cudaMemcpyAsync(rand_num_GPU, rand_num_CPU + i*N, N*sizeof(float), cudaMemcpyHostToDevice,stream_one);
		Add_em_up<<<dimGrid,dimBlock, 0,stream_one>>>(rand_num_GPU, rand_sum_GPU,1, i,N);
		cudaMemcpyAsync(rand_num_GPU, rand_num_CPU + N*(i+1), N*sizeof(float), cudaMemcpyHostToDevice,stream_two);
		Add_em_up<<<dimGrid,dimBlock, 0,stream_two>>>(rand_num_GPU, rand_sum_GPU, 2,i,N);
		cudaMemcpyAsync(rand_num_GPU, rand_num_CPU + N*(i+2), N*sizeof(float), cudaMemcpyHostToDevice,stream_three);
		Add_em_up<<<dimGrid,dimBlock, 0,stream_three>>>(rand_num_GPU, rand_sum_GPU, 3,i,N);	
	}
	cudaMemcpyAsync(rand_sum_CPU, rand_sum_GPU, 30*sizeof(float), cudaMemcpyDeviceToHost);
	
	for(i=0; i<30; i++)
	{
		printf("final sum number %d is %f\n", i, rand_sum_CPU[i]);
	}
	finalSum(rand_sum_CPU, 30);
	printf("The final sum is %f\n", rand_sum_CPU[0]);
	
	gettimeofday(&end, NULL);
	float time = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
	
	printf("time in milliseconds: %.15f\n", time/1000);
	
	Cleanup(rand_num_CPU, rand_sum_CPU, rand_num_GPU, rand_sum_GPU);
}
/*
final sum number 0 is 411865.031250
final sum number 1 is 412250.656250
final sum number 2 is 406504.093750
final sum number 3 is 412250.156250
final sum number 4 is 406504.187500
final sum number 5 is 411834.062500
final sum number 6 is 406504.312500
final sum number 7 is 411833.906250
final sum number 8 is 409671.375000
final sum number 9 is 411833.406250
final sum number 10 is 409672.156250
final sum number 11 is 408217.968750
final sum number 12 is 409671.718750
final sum number 13 is 408217.875000
final sum number 14 is 407320.781250
final sum number 15 is 408217.875000
final sum number 16 is 407321.718750
final sum number 17 is 407714.343750
final sum number 18 is 407322.125000
final sum number 19 is 403816.593750
final sum number 20 is 409465.093750
final sum number 21 is 403815.781250
final sum number 22 is 410225.875000
final sum number 23 is 408284.968750
final sum number 24 is 410226.156250
final sum number 25 is 407310.875000
final sum number 26 is 406670.156250
final sum number 27 is 407311.406250
final sum number 28 is 406670.531250
final sum number 29 is 414002.281250
The final sum is 12262527.000000
time in milliseconds: 239.712997436523438
*/
