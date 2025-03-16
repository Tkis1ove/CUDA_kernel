#include <cuda_runtime.h>
#include <stdio.h>
#include <random>

#define BLOCK_SIZE 256
#define CEIL(a,b) (a + b - 1) / b //向上取整

int main() {

    int n = 1000000;
    float* input = (float*)malloc(n * sizeof(float));
    float* output = (float*)malloc(n * sizeof(float));

    float* device_input, *device_output, *device_max, *device_sum;
    cudaMalloc((void**)&device_input, n * sizeof(float));
    cudaMalloc((void**)&device_output, n * sizeof(float));

    //数据初始化
    for(int i = 0; i < n; i++){
        input[i] = (float)rand() % 100;
        output[i] = 0.0f; 
    }

    //CPU数据搬运到GPU
    cudaMemcpy((void**)&device_input, (void**)&input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)&device_output, (void**)&output, n * sizeof(float, cudaMemcpyHostToDevice));
    
}
