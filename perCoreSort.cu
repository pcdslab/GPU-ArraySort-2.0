#include<iostream>
#include<vector>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<ctime>
#include<algorithm>
#include<utility>
#include <curand.h>
#include <curand_kernel.h>
#include<random>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

#define elesPerBucket 20
#define sampleRate 10
#define totArrays 100000
#define maxSize 2000L
#define minSize 1000
#define SAMPLED (sampleRate*maxSize)/100
#define maxBuckets (maxSize/elesPerBucket)

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//swap function for Insertion sort
template <class type>
 __device__ void  swapD (type &a, type &b)

{
    /* &a and &b are reference variables */
    type temp;
        temp=a;
	a=b;
        b=temp;
}
//insertion sort
template <class type>
 __device__ void insertionSort(type *input, int begin, int end){
int i, j; //,tmp;
 for (i = begin+1; i < end; i++) {
 j = i;
 while (j > begin && input[j - 1] > input[j]) {
 swapD(input[j], input[j-1]);
 j--;
 }//end of while loop
}
}


//data generation
template <typename mType>
struct dataArrays{
	vector<mType> dataList;
	int *prefixArray;
};


template <typename type> 
dataArrays<type> dataGen (int numOfArrays, int maxArraySize, int minArraySize){
	
   dataArrays<int> data;
   data.prefixArray = new int[numOfArrays+1]; //exclusive prefix scan
   const int range_from = 0;
   const unsigned int range_to = 5000;//2147483647; //2^31 - 1
   random_device rand_dev;
   mt19937 generator(rand_dev());
   uniform_int_distribution<int> distr(range_from, range_to);
   int prefixSum = 0;
   srand(time(0));
	for( int i = 0; i < numOfArrays; i++){
	
		int size = rand()%(maxArraySize-minArraySize + 1) + minArraySize;
		data.prefixArray[i] = prefixSum;
		for(int j = prefixSum; j < prefixSum + size; j++){
			data.dataList.push_back(distr(generator));
		}
		prefixSum += size;
	}
	
	data.prefixArray[numOfArrays] = prefixSum;
	return data;
}

__global__ void sortKernel(int *d_inputData, int *d_prefixSum){
	
	if(blockIdx.x < totArrays){
		insertionSort(d_inputData, d_prefixSum[blockIdx.x], d_prefixSum[blockIdx.x+1]);
		
	}
	
	
}

int main(){
	//generate data
   dataArrays<int> newData = (dataGen<int>(totArrays,maxSize,minSize));
   int *prefixSum = newData.prefixArray;
   int *d_prefixSum, *d_numOfBuckets, *d_inputData, *d_sample, *d_splitters, *d_bucketSizes;
    //generating events
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   int *h_totalBuckets = new int[1];
   size_t size_heap, size_stack;
    //setting stack size limit
   cudaDeviceSetLimit(cudaLimitStackSize,10240);
   cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
   cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
   
   cudaMalloc((void**) &d_prefixSum, (totArrays+1)*sizeof(int));
   
     cudaEventRecord(start);
   cudaMemcpy(d_prefixSum, prefixSum, sizeof(int)*(1+totArrays), cudaMemcpyHostToDevice);
   thrust::device_vector<int> inData (newData.dataList.size());
   thrust::copy(newData.dataList.begin(), newData.dataList.end(), inData.begin());
   
   d_inputData = thrust::raw_pointer_cast(&inData[0]);
   
   sortKernel<<<totArrays,1>>>(d_inputData, d_prefixSum);
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
   int *h_bucketedData = new int[newData.prefixArray[totArrays]];
   checkCuda(cudaMemcpy(h_bucketedData, d_inputData, (newData.prefixArray[totArrays])*sizeof(int), cudaMemcpyDeviceToHost));
   
    cout<<"printing sorted array:";
    for(int i = newData.prefixArray[totArrays-2]; i < newData.prefixArray[totArrays-1]; i++)
	   cout<< i<<":"<<h_bucketedData[i]<<endl;
    cout<<"Time:"<<milliseconds;
   
   }