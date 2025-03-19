#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 2
#define CEIL(a, b) ((a + b - 1) / b)

__global__ void transpose0(float* input, float* output, int m, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx >= n || idy >= m) return;

    output[idx * m + idy] = input[idy * n + idx];
}

__global__ void transpose2(float* input, float* output, int m, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if(idx >= n || idy >= m) return;

    __shared__ float sharedata[BLOCK_SIZE][BLOCK_SIZE + 1];
    sharedata[ty][tx] = input[idy * n + idx];
    __syncthreads();

    output[idx * m + idy] = sharedata[ty][tx]; 
}

__global__ void transpose1(float* input, float* output, int m, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx >= n || idy >= m) return;

    output[idx * m + idy] = __ldg(&input[idy * n + idx]);
}

int main(int argc, char** argv){

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);

    float* input = (float*)malloc(sizeof(float) * m * n);
    float* output = (float*)malloc(sizeof(float) * m * n);
    float* d_output = (float*)malloc(sizeof(float) * m * n);

    float* device_input, *device_output;
    cudaMalloc((void**)&device_input, sizeof(float) * m * n);
    cudaMalloc((void**)&device_output, sizeof(float) * m * n);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            input[i * n + j] = i * n + j;
            output[i * n + j] = 0.0f;
        }
    }

    cudaMemcpy(device_input, input, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, output, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    //CPU
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            output[j * m + i] = input[i * n + j];
        }
    }

    //GPU
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(CEIL(m, BLOCK_SIZE), CEIL(n, BLOCK_SIZE));
    transpose2<<<grid, block>>>(device_input, device_output, m, n);

    cudaMemcpy(d_output, device_output, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%lf ",d_output[i * m + j]);
        }
        printf("\n");
    }

    cudaFree(device_input);
    cudaFree(device_output);

    free(input);
    free(output);
    free(d_output);

    return 0;
}