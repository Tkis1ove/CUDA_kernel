#include <stdio.h>
#include "verfiy.h"
#include "conv2d.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

int main(int argc, char**argv)
{
    //从命令行读入参数
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);
    int h = atoi(argv[3]);
    int w = atoi(argv[4]);
    int k = atoi(argv[5]);
    int r = atoi(argv[6]);
    int s = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int p = atoi(argv[10]);
    int q = atoi(argv[11]);

    int outh = (h - r + 2*p)/u + 1;
    int outw = (w - s + 2*q)/v + 1;

    half *pIn       = (half*)malloc(n*c*h*w*sizeof(half));           //原始数据
    half *pWeight   = (half*)malloc(k*c*r*s*sizeof(half));           //卷积核
    half *pOut      = (half*)malloc(n*k*outh*outw*sizeof(half));     //存储正确的计算结果
    half *pOut_host = (half*)malloc(n*k*outh*outw*sizeof(half));     //存储你的的计算结果

    half *pIn_device,*pWeight_device,*pOut_device,*pIn_ori,*pWeight_ori,*pOut_ori;
    cudaMalloc((void**)&pIn_device, n*c*h*w*sizeof(half));
    cudaMalloc((void**)&pWeight_device, k*c*r*s*sizeof(half));
    cudaMalloc((void**)&pOut_device, n*k*outh*outw*sizeof(half));
    cudaMalloc((void**)&pIn_ori, n*c*h*w*sizeof(half));
    cudaMalloc((void**)&pWeight_ori, k*c*r*s*sizeof(half));
    cudaMalloc((void**)&pOut_ori, n*k*outh*outw*sizeof(half));
    
    for(int i = 0; i < n*c*h*w; i++)
    {
        pIn[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < k*c*r*s; i++)
    {
        pWeight[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < n*k*outh*outw; i++)
    {
        pOut[i] = 0.0;
        pOut_host[i] = 0.0;
    }
              
    cudaMemcpy(pIn_device, pIn, n*c*h*w*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pWeight_device, pWeight, k*c*r*s*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pOut_device, pOut, n*k*outh*outw*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pIn_ori, pIn, n*c*h*w*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pWeight_ori, pWeight, k*c*r*s*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pOut_ori, pOut, n*k*outh*outw*sizeof(half), cudaMemcpyHostToDevice);

    /********************step 1*****************************/
    problem_t problem;
    int paramSize;
    kernelInfo_t kernelInfo;

    problem.in        = pIn_device;        
    problem.weight    = pWeight_device;
    problem.out       = pOut_device;             
    problem.n         = n;                             
    problem.c         = c;                             
    problem.h         = h;                             
    problem.w         = w;                             
    problem.k         = k;                             
    problem.r         = r;                             
    problem.s         = s;                             
    problem.u         = u;                             
    problem.v         = v;                             
    problem.p         = p;                             
    problem.q         = q;                               

    /********************************** step 2****************************/
    getParamsize(&problem, &paramSize);
    printf("paramsize:%d\n", paramSize);
    void* param = malloc(paramSize);
    
    getkernelInfo(&problem, &kernelInfo, param);

    dim3 groups(kernelInfo.blockx, kernelInfo.blocky, kernelInfo.blockz);
    dim3 threads(kernelInfo.threadx, kernelInfo.thready, kernelInfo.threadz);
    int ldsSize = kernelInfo.dynmicLdsSize;
        
    /*******************************warm up and get result************************************/
    cudaLaunchKernel(kernelInfo.kernelPtr,groups,threads,(void**)&param,ldsSize);

    cudaMemcpy(pOut_host, pOut_device,  n*k*outh*outw*sizeof(half), cudaMemcpyDeviceToHost); 

    /*******************************cost time test************************************/
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    float time_elapsed=0.0;
    
    int iternum = 100;
    for(int i=0; i<iternum; i++)
    {
        cudaLaunchKernel(kernelInfo.kernelPtr,groups,threads,(void**)&param,ldsSize); 
    }
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);

    printf("time: %f us\n", time_elapsed*1000/iternum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  

    /*******************************verify************************************/
    printf("start verfiy\n");

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((outh * outw + 15) / 16, (k + 15) / 16, n);
    Conv2dGpu<<<numBlocks, threadsPerBlock>>>(pIn_ori, pWeight_ori, pOut_ori, n, c, h, w, k, r, s, u, v, p, q);
    cudaMemcpy(pOut, pOut_ori,  n*k*outh*outw*sizeof(half), cudaMemcpyDeviceToHost); 

    int error=0;
    for(int i=0;i<n*k*outh*outw;i++)
    {
        float your_result = __half2float(pOut_host[i]);
        float right_result = __half2float(pOut[i]);
        if(isnan(right_result)|isinf(right_result)) 
        {
            printf("right result is nan or inf! It is impossible!");
            break;
        }

        if(isnan(your_result)|isinf(your_result)) 
        {
            printf("your result is nan or inf!");
            break;
        }
        if((fabs(your_result - right_result))/ your_result > 0.01)
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, (float)pOut_host[i], (float)pOut[i]);
            error++;
            break;
        }        
    }

    printf("finish,error:%d\n",error);

    cudaFree(pIn_device);
    cudaFree(pWeight_device);
    cudaFree(pOut_device);
    free(param);
    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_host);

    return 0;
}