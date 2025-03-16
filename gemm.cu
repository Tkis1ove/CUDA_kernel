__global__ void gemm(float* A, float* B, float* C, int m, int n, int k){
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    int row = tx / n;
    int col = tx % n;

    for(int i = 0; i < k; i++){
        if(row < m && col < n){
            sum += A[row * k + i] * B[col + n * i];
        }
    }

    C[tx] = sum;
}