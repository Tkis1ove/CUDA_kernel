#include <cuda_runtime.h>
#include <stdio.h>
#include <random>

#define BLOCK_SIZE 256
#define CEIL(a,b) ((a + b - 1) / b) //向上取整



__global__ void max_kernel(float* input, float* max, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float shareInput[BLOCK_SIZE];
    shareInput[tx] = idx < n ? input[idx] : 0.0f;

    for(int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1){
        if(idx < offset)
        shareInput[tx] = shareInput[tx] > shareInput[tx + offset] ? shareInput[tx] : shareInput[tx + offset];
        __syncthreads();
    }
    
    float temp = input[idx];
    if(tx == 0)input[idx] = shareInput[0]; 
    __syncthreads();

    for(int offset = gridDim.x*BLOCK_SIZE >> 1; offset > 0; offset >>= 1){
        if()
        if(idx < offset && tx == 0)
        input[idx] = input[idx] > input[idx + offset] ? input[idx] : input[idx + offset];
        __syncthreads();
    }

    if(idx == 0){
        *max = input[0];
        printf("max = %lf\n",*max);
    }

    //if(tx == 0) input[idx] = temp;`    
}

int main() {

    int n = 1000;
    float* input = (float*)malloc(n * sizeof(float));
    float* output = (float*)malloc(n * sizeof(float));
    float* max = (float*)malloc(1 * sizeof(float));

    float* device_input, *device_output, *device_max, *device_sum;
    cudaMalloc(&device_input, n * sizeof(float));
    cudaMalloc(&device_output, n * sizeof(float));
    cudaMalloc(&device_max, sizeof(float));
    cudaMalloc(&device_sum, sizeof(float));

    //数据初始化
    for(int i = 0; i < n; i++){
        //input[i] = (float)rand() % 100;
        input[i] = i;
        output[i] = 0.0f; 
    }

    //CPU数据搬运到GPU
    cudaMemcpy(device_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, output, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(CEIL(n, BLOCK_SIZE));
    max_kernel<<<grid, block>>>(device_input, device_max, n);
    cudaMemcpy(max, device_max, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    free(input);
    free(output);

    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
