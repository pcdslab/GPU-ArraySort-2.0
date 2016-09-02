/*
Copyright (C) Muaaz Gul Awan and Fahad Saeed  
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/


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



__device__ void getMinMax(int input[], int beginPtr, int endPtr, int *ret){
          int min = input[beginPtr];
          int max = 0;
        // int *ret = new int[2];
          for(int i = beginPtr; i < endPtr; i++){
              if(min > input[i])
                  min = input[i];
              if (max < input[i])
                  max = input[i];     
            }

     ret[0] = min;
     ret[1] = max;
//return ret;

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

//kernel for obtaining num of buckets for each array
__global__ void getNumOfBuckets(int *prefixSumArray, int *numOfBucketsArray){
	int id = blockIdx.x; // * blockDim.x + threadIdx.x;
	
	if(id < totArrays)
		numOfBucketsArray[id] = (prefixSumArray[id+1] - prefixSumArray[id])/elesPerBucket;
}

template <typename mType>
__device__ void getSplitters (mType *data, mType *splittersArray, int sample[], int beginPtr, int endPtr, int arraySize, int *prefixBucketsArray){
           __shared__ mType mySamples[SAMPLED];
		   
            //int *ret = new int[2];
			//int arraySize = endPtr - beginPtr;
			// calculating samples for this array
			int numOfSamples = ((float)sampleRate/100)*(arraySize);
			//calculating the number of buckets for this array
			int numOfBuckets = (blockIdx.x == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[blockIdx.x] - prefixBucketsArray[blockIdx.x-1]);
			
            for(int i = 0; i < numOfSamples; i++)
	           mySamples[i] = data[beginPtr+sample[i]];

	        insertionSort(mySamples, 0, numOfSamples);
	 
	        //calculate splitter index for this array 
            int splitterIndex = ((blockIdx.x == 0)? 1 : (prefixBucketsArray[blockIdx.x-1]+1))+1; //the other plus one is for leaving space for smallest splitter(added later)
            int splittersSize=0;
	        for(int i = (numOfSamples)/(numOfBuckets); splittersSize < numOfBuckets-1; i +=(numOfSamples)/(numOfBuckets)){
                 splittersArray[splitterIndex] = mySamples[i];
                 splitterIndex++;
                 splittersSize++;
             }
            //getMinMax(data, beginPtr, endPtr, ret);
			int bits = 8*sizeof(mType);
            mType min = -(1 << (bits-1));
            mType max = (1 << (bits - 1)) - 1;//int max =  (1 << (bits-1)) – 1;
            splittersArray[((blockIdx.x == 0)? 0 : (prefixBucketsArray[blockIdx.x-1]+1))] = min;//ret[0]-2;//to accodmodate the smallest
            splittersArray[((blockIdx.x == 0)? prefixBucketsArray[0] : (prefixBucketsArray[blockIdx.x]))] = max;//ret[1]+2;
      
            //delete [] ret;
}

//kernel for obtaining splitters
template <typename mType>
__global__ void splitterKer(mType *data, mType *splittersArray, int *prefixSizeArray, int *prefixBucketsArray){
          if(blockIdx.x < totArrays){
             int id = blockIdx.x;
	         __shared__ int sampleSh[SAMPLED];
			 int arraySize = prefixSizeArray[id+1] - prefixSizeArray[id];
			// calculating samples for this array
			int numOfSamples = ((float)sampleRate/100)*(arraySize);
            //int *h_sample = new int[SAMPLED];
            int max = arraySize;
            int  sam = numOfSamples;
            int stride = max/sam;
	        int sampleVal = 0;
            for( int i = 0; i < numOfSamples; i++)
            {
               sampleSh[i] = sampleVal;
               sampleVal += stride; 
            }
			 
	        //for(int i = 0; i < numOfSamples; i++)
	           // sampleSh[i] = mySample[i];

	        getSplitters(data, splittersArray, sampleSh, prefixSizeArray[id], prefixSizeArray[id+1], prefixSizeArray[id+1] - prefixSizeArray[id], prefixBucketsArray);

           }
        }

		
template <typename mType>
__device__ void getBuckets(mType *input, mType *splitters, int beginPtr, int endPtr, int *bucketsSize, mType *myInput, int *prefixBucketsArray){
     volatile int numOfBuckets = (blockIdx.x == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[blockIdx.x] - prefixBucketsArray[blockIdx.x-1]);
        
	if(blockIdx.x < totArrays && threadIdx.x < numOfBuckets){
	  
	  int id = threadIdx.x;
	  int sizeOffset = (blockIdx.x == 0) ? (0+threadIdx.x) : (prefixBucketsArray[blockIdx.x-1] + threadIdx.x);  //blockIdx.x*BUCKETS+threadIdx.x;
	  int sizeOffsetBlock = (blockIdx.x == 0) ? (0) : (prefixBucketsArray[blockIdx.x-1]);
      int bucketSizeOff = sizeOffset+1;
      mType myBucket[maxSize]; //make it shared as well
     // int bucketIndexOffset;
      int indexSum=0;
      bucketsSize[bucketSizeOff] = 0;

     for(int i = 0; i < (endPtr - beginPtr); i++){
         if(myInput[i] > splitters[id] && myInput[i] <= splitters[id+1]){
         myBucket[bucketsSize[bucketSizeOff]] = myInput[i];
         bucketsSize[bucketSizeOff]++;

        }
     }
       
    __syncthreads();
     
         //prefix sum for bucket sizes of current array
     for(int j = 0; j < threadIdx.x; j++)
        indexSum += bucketsSize[sizeOffsetBlock+j+1];

         //writing back current buckt back to the input memory
	 for(int i = 0; i < bucketsSize[bucketSizeOff]; i++)
             input[indexSum+beginPtr+i] = myBucket[i];
	}
      

}
		
//kernel for obtaining buckets
template <typename mType>
__global__ void bucketKernel(mType *data, mType *splittersArray, int *prefixSizeArray, int *prefixBucketsArray, int *bucketSizes){
    
	int numOfBuckets = (blockIdx.x == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[blockIdx.x] - prefixBucketsArray[blockIdx.x-1]);
        
	if(blockIdx.x < totArrays && threadIdx.x < numOfBuckets){
        bucketSizes[0] = 0;
		int bid = blockIdx.x;
        int tid = threadIdx.x;
		int arraySize = prefixSizeArray[blockIdx.x+1] - prefixSizeArray[blockIdx.x];
	    int leftOvers = arraySize%numOfBuckets;
        int jmpFac = arraySize/numOfBuckets;
        int gArrayStart = prefixSizeArray[blockIdx.x] + tid*jmpFac;
        int gArrayEnd = (tid==(numOfBuckets-1))?(gArrayStart + jmpFac+leftOvers):(gArrayStart + jmpFac);
        int lArrayStart = tid*jmpFac;
        __shared__ int myInput [maxSize];

        int arrBegin = prefixSizeArray[bid];
        int arrEnd = prefixSizeArray[bid+1];
		    
        int splitterIndexSt = ((blockIdx.x == 0)? 0 : (prefixBucketsArray[blockIdx.x-1]+1));//blockIdx.x*(BUCKETS+1);
        int splitterIndexEd = splitterIndexSt + numOfBuckets+1;
        __shared__ mType splitters[maxBuckets+2];
//copy my array in shared memory in parallel
        for(int i=lArrayStart,j=gArrayStart;j<gArrayEnd;i++,j++){
            myInput[i] = data[j];
         }
        __syncthreads(); 
        int j = 0;
        for(int i = splitterIndexSt; i < splitterIndexEd; i++){
           splitters[j] = splittersArray[i];
           j++;
        }
       
	    getBuckets(data, splitters, arrBegin, arrEnd, bucketSizes, myInput, prefixBucketsArray);

	}
}		
		
		
		
//sorting kernel	
template <typename mType>	
__global__ void sortBuckets(mType *buckets, int *bucketsSize, int *prefixBucketsArray, int *prefixSizeArray){
	int numOfBuckets = (blockIdx.x == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[blockIdx.x] - prefixBucketsArray[blockIdx.x-1]);
     
	
       if(blockIdx.x < totArrays && threadIdx.x < numOfBuckets){
		int sizeOffset = (blockIdx.x == 0) ? (0+threadIdx.x) : (prefixBucketsArray[blockIdx.x-1] + threadIdx.x); 
        int sizeOffsetBlock = (blockIdx.x == 0) ? (0) : (prefixBucketsArray[blockIdx.x-1]);
       // int bid = blockIdx.x;
        int tid = threadIdx.x;
		int arraySize = prefixSizeArray[blockIdx.x+1] - prefixSizeArray[blockIdx.x];
	    int leftOvers = arraySize%numOfBuckets;
        int jmpFac = arraySize/numOfBuckets;
        int gArrayStart = prefixSizeArray[blockIdx.x] + tid*jmpFac;
        int gArrayEnd = (tid==(numOfBuckets-1))?(gArrayStart + jmpFac+leftOvers):(gArrayStart + jmpFac);
        int lArrayStart = tid*jmpFac;
        //int lArrayEnd = (tid==(BUCKETS-1))?(lArrayStart + jmpFac+leftOvers):(lArrayStart + jmpFac);

        __shared__ mType myArray [maxSize];
        int indexSum = 0;
    

          for(int i=lArrayStart,j=gArrayStart;j<gArrayEnd;i++,j++){
                 myArray[i] = buckets[j];
           
        }
        __syncthreads();
        //prefix sum for bucket sizes of current array
        
     	  for(int j = 0; j < threadIdx.x; j++)
              indexSum += bucketsSize[sizeOffsetBlock+j+1];

 
          insertionSort(myArray, indexSum,indexSum + bucketsSize[sizeOffset+1]);
          __syncthreads();


           for(int i=lArrayStart,j=gArrayStart;j<gArrayEnd;i++,j++){
                 buckets[j] = myArray[i];
           
        }
     __syncthreads(); 
}


}

template <typename mType>		
void gpuArraySort(dataArrays<mType> newData, int *prefixSum, int flag ){
	
	int *d_prefixSum, *d_numOfBuckets;
	if(flag == 1){
		int *d_prefixSumDo;
		cudaMalloc((void**) &d_prefixSumDo, (totArrays+1)*sizeof(int));
		cudaMemcpy(d_prefixSumDo, prefixSum, sizeof(int)*(1+totArrays), cudaMemcpyHostToDevice);
   
        //casting device ptr to thrust dev_ptr
        thrust::device_ptr<int> prefixDo = thrust::device_pointer_cast(d_prefixSumDo);
        //performing prefixSum using thrust
        thrust::exclusive_scan(prefixDo, prefixDo + totArrays, prefixDo);
		prefixDo[totArrays] = prefixDo[totArrays-1] + prefixSum[totArrays-1];
		d_prefixSum = d_prefixSumDo;
	}
	
    mType *d_inputData, *d_splitters, *d_bucketSizes;
    //creating events
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   int *h_totalBuckets = new int[1];
   size_t size_heap, size_stack;
    //setting stack size limit
   cudaDeviceSetLimit(cudaLimitStackSize,10240);
   cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
   cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
   
   
   // allocating device memory for prefixSum, num Of buckets, splitters, bucketSizes
   cudaMalloc((void**) &d_prefixSum, (totArrays+1)*sizeof(int));
   cudaMalloc((void**) &d_numOfBuckets, (totArrays)*sizeof(int));
   cudaEventRecord(start);
   //copying prefixSums to Device
   cudaMemcpy(d_prefixSum, prefixSum, sizeof(int)*(1+totArrays), cudaMemcpyHostToDevice);
   
   //clculating buckets on GPU
   getNumOfBuckets<<<totArrays,1>>>(d_prefixSum, d_numOfBuckets);
   
   //casting device ptr to thrust dev_ptr
   thrust::device_ptr<int> prefixNumBuckets = thrust::device_pointer_cast(d_numOfBuckets);
   //performing prefixSum using thrust
   thrust::inclusive_scan(prefixNumBuckets, prefixNumBuckets + totArrays, prefixNumBuckets);
   //copying total number of buckets back
   checkCuda(cudaMemcpy(h_totalBuckets, d_numOfBuckets+(totArrays-1), sizeof(int), cudaMemcpyDeviceToHost));
   //allocating device memory for splitters
   cudaMalloc((void**) &d_splitters, (totArrays+h_totalBuckets[0])*sizeof(mType));
   
   cudaMalloc((void**) &d_bucketSizes, (1+h_totalBuckets[0])*sizeof(int));
   //allocating device memory for inputData
    thrust::device_vector<mType> inData (newData.dataList.size());
    thrust::copy(newData.dataList.begin(), newData.dataList.end(), inData.begin());
   
    d_inputData = thrust::raw_pointer_cast(&inData[0]);
   
    cout<< "**** Generating Splitters ****" << endl;
     
    splitterKer<<<totArrays, 1>>>(d_inputData, d_splitters, d_prefixSum, d_numOfBuckets);
    
	cout<< "**** Splitters Generated****" << endl;
	
	cout<< "**** Generating Buckets ****" << endl;
	
    bucketKernel<<<totArrays, maxBuckets>>>(d_inputData, d_splitters, d_prefixSum, d_numOfBuckets, d_bucketSizes);
   
    cout<< "**** Buckets Generated ****" << endl;
	
	cout<< "**** Sorting Buckets ****" << endl;
    sortBuckets<<<totArrays, maxBuckets>>>(d_inputData, d_bucketSizes,d_numOfBuckets, d_prefixSum);
     
	cout<< "**** Writing Back ****" << endl; 
	
    cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    mType *h_bucketedData = new mType[newData.prefixArray[totArrays]];
    checkCuda(cudaMemcpy(h_bucketedData, d_inputData, (newData.prefixArray[totArrays])*sizeof(mType), cudaMemcpyDeviceToHost));
   
   // cout<<"printing bucketed array:";
  //  for(int i = newData.prefixArray[totArrays-2]; i < newData.prefixArray[totArrays-1]; i++)
	//   cout<< i<<":"<<h_bucketedData[i]<<endl;
   cout<< "**** Arrays Sorted, Time Taken : "<< milliseconds<<"****" << endl;
    
	
} 
		
int main(){
   //generate data
   dataArrays<int> newData = (dataGen<int>(totArrays,maxSize,minSize));
   int *prefixSum = newData.prefixArray;
   //calling GPU-ArraySort
   gpuArraySort<int>(newData, prefixSum, 0 );
	
	
}
