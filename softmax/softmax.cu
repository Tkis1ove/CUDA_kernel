#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void softmax(float *input, float *output) {
    
    int tx = threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedata[BLOCK_SIZE];
    sharedata[tx] = input[x];
    __syncthreads();

    for(int i=1;i<blockDim.x;i*=2){
        int index=i*2*tx;
        if(index<blockDim.x){
            sharedata[index]+=sharedata[index+i];
        }
        __syncthreads();
    }
    if(tx==0){
        output[blockIdx.x]=sharedata[0];
    }
}