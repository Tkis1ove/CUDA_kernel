/*
浮点数相加会有误差出现，当n比较大时，就会出错错误，可以将float修改为double
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_N 1048576 //1024 * 1024
#define BLOCK_SIZE 1024
#define CEIL(a, b) ((a + b - 1)/b)

//volatile修饰的变量会强制编译器每次访问从内存中读取最新值，而非使用寄存器中的缓存值
__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce0(float* input, float* output, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    if(idx >= n) return;

    __shared__ float sharedata[BLOCK_SIZE];

    sharedata[tx] =  idx < n ? input[idx] : 0.0f;

    for(int offset = 1; offset < blockDim.x; offset *= 2){
        __syncthreads();
        if(tx % (2 * offset) == 0){
            sharedata[tx] += sharedata[tx + offset]; //存在warp divergent
        }
    }

    if(tx == 0) output[blockIdx.x] = sharedata[tx]; 
}

__global__ void reduce1(float* input, float* output, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    if(idx >= n) return;

    __shared__ float sharedata[BLOCK_SIZE];

    sharedata[tx] =  idx < n ? input[idx] : 0.0f;

    for(int offset = 1; offset < blockDim.x; offset *= 2){
        __syncthreads();
        int index = 2 * offset * tx; //只会在最后32个数据继续规约时存在warp divergent,但此时还有bank conflict，比如在第一次迭代中，0号线程访问0，1地址，16号线程访问32，33地址，此时0和32产生冲突
        if(index < blockDim.x){
            sharedata[index] += sharedata[index + offset];
        }
    }

    if(tx == 0) output[blockIdx.x] = sharedata[tx]; 
}

__global__ void reduce2(float* input, float* output, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    if(idx >= n) return;

    __shared__ float sharedata[BLOCK_SIZE];

    sharedata[tx] =  idx < n ? input[idx] : 0.0f;

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        __syncthreads(); 
        if(tx < offset){
            sharedata[tx] += sharedata[tx + offset];//解决了bank conflict，但是造成了大量的线程浪费，想办法把他们利用起来
        }
    }

    if(tx == 0) output[blockIdx.x] = sharedata[tx]; 
}

__global__ void reduce3(float* input, float* output, int n){
    int idx = blockDim.x * 2 * blockIdx.x + threadIdx.x;//每个线程块多管一些数据
    int tx = threadIdx.x;

    if(idx >= n) return;

    __shared__ float sharedata[BLOCK_SIZE];

    sharedata[tx] =  idx < n ? input[idx] + input[idx + blockDim.x] : 0.0f;//让每个线程至少做一次加法

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        __syncthreads(); 
        if(tx < offset){
            sharedata[tx] += sharedata[tx + offset];//当只有warp0在干活时，还在执行同步操作，造成浪费
        }
    }

    if(tx == 0) output[blockIdx.x] = sharedata[tx]; 
}

__global__ void reduce4(float* input, float* output, int n){//暴力循环展开后，调整block大小，最后用上shuffle指令，基本达到最优？？？
    int idx = blockDim.x * 2 * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    if(idx >= n) return;

    __shared__ float sharedata[BLOCK_SIZE];

    sharedata[tx] =  idx < n ? input[idx] + input[idx + blockDim.x] : 0.0f;
    for(int offset = blockDim.x >> 1; offset > 32; offset >>= 1){
        __syncthreads(); 
        if(tx < offset){
            sharedata[tx] += sharedata[tx + offset];
        }
    }

    __syncthreads();
    if(tx < 32) warpReduce(sharedata, tx);
    if(tx == 0) output[blockIdx.x] = sharedata[tx]; 
}

__global__ void reduce_sum(float* input, float* output, int n){
    int tx = threadIdx.x;

    __shared__ float sharedata[BLOCK_SIZE];

    sharedata[tx] = tx < n ? input[tx] : 0.0f; 

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        __syncthreads();
        if(tx < offset) sharedata[tx] += sharedata[tx + offset];
    }

    if(tx == 0) *output = sharedata[0];
}

int main(int argc, char** argv){

    int n = atoi(argv[1]);

    float* nums = (float*)malloc(sizeof(float) * n);
    float sum = 0;
    float d_sum = 0;

    float* device_input, *device_output, *device_sum;
    cudaMalloc((void**)&device_input, sizeof(float) * n);
    cudaMalloc((void**)&device_output, sizeof(float) * BLOCK_SIZE);
    cudaMalloc((void**)&device_sum, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i = 0; i < n; i++){
        nums[i] = (float)i;
    }

    cudaMemcpy(device_input, nums, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(CEIL(n, BLOCK_SIZE * 2));

    cudaEventRecord(start,0);
    float time_elapsed1=0.0;

    for(int i = 0; i < n; i++){
        sum += nums[i];
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed1,start,stop);
    printf("CPU time: %f us\n", time_elapsed1*1000);

    cudaEventRecord(start,0);
    float time_elapsed2=0.0;

    reduce4<<<grid, block>>>(device_input, device_output, n);
    cudaDeviceSynchronize();
    reduce_sum<<<1, block>>>(device_output, device_sum, BLOCK_SIZE);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed2,start,stop);
    printf("GPU time: %f us\n", time_elapsed2*1000);

    cudaMemcpy(&d_sum, device_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if(sum == d_sum) printf("CPU = %f, GPU = %f, You are right!\n",sum,d_sum);
    else printf("CPU = %f, GPU = %f, You are wrong.\n",sum,d_sum);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_sum);

    free(nums);

    return 0;
}