#include "conv2d.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/*选手自定义的kernel入参结构体*/
typedef struct mykernelParamType
{
    half*   pin;                            //输入数据地址
    half*   pweight;                        //权值数据地址
    half*   pout;                           //输出数据地址
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
    unsigned int      revs0;                          //预留                          
    unsigned int      revs1;                          //预留
    unsigned int      revs2;                          //预留
    unsigned int      revs3;                          //预留
    unsigned int      revs4;                          //预留
    unsigned int      revs5;                          //预留
    unsigned int      revs6;                          //预留
    unsigned int      revs7;                          //预留
}mykernelParamType;                          

__global__ void Gemm(int m, int n, int k, half* A, half* B, half* C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < m && col < n){
        half sum = 0.0;
        for(int i = 0; i < k; i++){
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

/*选手自己实现的kernel*/
extern "C" __global__ void myKernelConv2dGpu(mykernelParamType param) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if(x >= param.Oh*param.Ow || y >= param.k || z >= param.n)
    {
        return;
    }

    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/param.Ow;
    int posOw = x%param.Ow;

    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;

    float sum = 0.0;

    int inOffset = z*param.c*param.h*param.w + posh_ori*param.w + posw_ori;
    int weiOffset = y*param.c*param.r*param.s;
    int inChannelOffset = param.h*param.w;
    int weightChannelOffset = param.r*param.s;

    for(int i = 0; i < param.r; i++)
    {
        for(int j = 0; j < param.s; j++)
        {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;            

            if(posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h)
            {
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for(int channel = 0; channel<param.c; channel++)
                {
                    sum += __half2float(param.pin[inOffsetTmp + i*param.w + j] * param.pweight[weiOffsetTmp + i*param.s + j]);
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }               
            }
        }
    }   

    //计算输出偏移
    int outOffset = z*param.k*param.Oh*param.Ow + y*param.Oh*param.Ow + x;
    param.pout[outOffset] = __float2half(sum);
}

extern "C" __global__ void myKernelConv2dGpu1(mykernelParamType param) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if(x >= param.Oh*param.Ow || y >= param.k || z >= param.n)
    {
        return;
    }

    //当前线程处理的数据点在oh、ow上的坐标
    int posOh = x/param.Ow;
    int posOw = x%param.Ow;

    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;

    float sum = 0.0;

    int inOffset = z*param.c*param.h*param.w + posh_ori*param.w + posw_ori;
    int weiOffset = y*param.c*param.r*param.s;
    int inChannelOffset = param.h*param.w;
    int weightChannelOffset = param.r*param.s;

    //for(int i = 0; i < param.r; i++)
    //{
    //    for(int j = 0; j < param.s; j++)
    //    {
    //        int posh_real = posh_ori + i;
    //        int posw_real = posw_ori + j;            
//
    //        if(posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h)
    //        {
    //            int inOffsetTmp = inOffset;
    //            int weiOffsetTmp = weiOffset;
    //            for(int channel = 0; channel<param.c; channel++)
    //            {
    //                sum += __half2float(param.pin[inOffsetTmp + i*param.w + j] * param.pweight[weiOffsetTmp + i*param.s + j]);
    //                inOffsetTmp += inChannelOffset;
    //                weiOffsetTmp += weightChannelOffset;
    //            }               
    //        }
    //    }
    //}
    
    __shared__ half sharedInput[16][16];  // 使用固定大小，确保足够大
    __shared__ half sharedWeight[16][16];  // 使用固定大小，确保足够大

    for(int i = 0; i < param.r * param.s * param.c; i += 16){
        
        sharedInput[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        int weiOffsetTmp = i + threadIdx.x;
        sharedWeight[threadIdx.y][threadIdx.x] = param.pweight[weiOffset + weiOffsetTmp];

        int curC = (i + threadIdx.y) / (param.r * param.s);             // channel offset
        int curR = ((i + threadIdx.y) % (param.r * param.s)) / param.s; // kernel r offset
        int curS = ((i + threadIdx.y) % (param.r * param.s)) % param.s; // kernel s offset
        int curH = posh_ori + curR;                            // input h
        int curW = posw_ori + curS;                            // input w
        int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
        if(curH>=0 && curW>=0 && curW<param.w && curH<param.h){
            sharedInput[threadIdx.y][threadIdx.x] = param.pin[inOffset + inOffsetTmp];
        }

        __syncthreads();
  
#pragma unroll
        for (int subcrs = 0; subcrs < 16; ++subcrs)
        {
            sum += __half2float(sharedWeight[threadIdx.y][subcrs] * sharedInput[subcrs][threadIdx.x]);
        }

        __syncthreads();
    }

    //计算输出偏移
    int outOffset = z*param.k*param.Oh*param.Ow + y*param.Oh*param.Ow + x;
    param.pout[outOffset] = __float2half(sum);
}

/*返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);

    return 0;
}

/*返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param)
{
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    kernelInfo->blockx   = (outh*outw + 15)/16;                    //blockx  number
    kernelInfo->blocky   = (k+15)/16;                    //blocky  number
    kernelInfo->blockz   = n;                    //blockz  number
    kernelInfo->threadx  = 16;                   //threadx number per block
    kernelInfo->thready  = 16;                   //thready number per block
    kernelInfo->threadz  = 1;                   //threadz number per block
    kernelInfo->dynmicLdsSize = 0;
    kernelInfo->kernelPtr= (void*)myKernelConv2dGpu1;                 //kernel ptr

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->n = n;                              //batch szie             
    pArgs->c = c;                              //channel number          
    pArgs->h = h;                              //数据高                  
    pArgs->w = w;                              //数据宽                  
    pArgs->k = k;                              //卷积核数量              
    pArgs->r = r;                              //卷积核高               
    pArgs->s = s;                              //卷积核宽                
    pArgs->u = u;                              //卷积在高方向上的步长     
    pArgs->v = v;                              //卷积在宽方向上的步长     
    pArgs->p = p;                              //卷积在高方向上的补边     
    pArgs->q = q;                              //卷积在宽方向上的补边     
    pArgs->Oh = outh;
    pArgs->Ow = outw;       

    return 0;
}