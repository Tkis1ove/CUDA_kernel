#include <cuda_fp16.h>

void conv2dcpu(half* pin, half* pwei, half* pout, int n, int c, int h, int w, int k, int r, int s, int u, int v,  int p, int q)
{
    int oh = (h + 2*p - r)/u + 1;
    int ow = (w + 2*q - s)/v + 1;
    
    for(int nNum = 0; nNum < n; nNum++)
    {
        for(int kNum = 0; kNum< k; kNum++)
        {
            for(int i=0; i<oh; i++)
            {
                for(int j = 0; j< ow; j++)
                { 
                    double sum = 0.0;
                    int posh = i*u - p;
                    int posw = j*v - q;

                    for(int cNum = 0; cNum < c; cNum++)
                    {                       
                        for(int khNum = 0; khNum < r; khNum++)
                        {
                            for(int kwNum = 0; kwNum < s; kwNum++)
                            {
                                int posh_ori = posh + khNum;
                                int posw_ori = posw + kwNum;
                                if(posw_ori >= 0 && posh_ori >= 0 && posw_ori < w  && posh_ori < h)
                                {
                                    sum += (double)(pin[nNum*c*h*w + cNum*(w*h)+ posh_ori*w + posw_ori] * pwei[kNum*r*s*c + cNum*r*s + khNum*s + kwNum]);
                                }
                            }                       
                        }
                    }

                    pout[nNum*k*oh*ow + kNum*oh*ow + i*ow + j] = (half)sum;
                }
            }
        }
    }
}

__global__ void Conv2dGpu(half* pin, half* pwei, half* pout, int n, int c, int h, int w, int k, int r, int s, int u, int v, int p, int q) 
{
    int oh = (h + 2*p - r) / u + 1;
    int ow = (w + 2*q - s) / v + 1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= oh * ow || y >= k || z >= n)
    {
        return;
    }

    // 当前线程处理的数据点在 oh、ow 上的坐标
    int posOh = x / ow;
    int posOw = x % ow;

    int posh_ori = posOh * u - p;
    int posw_ori = posOw * v - q;

    float sum = 0.0;

    int inOffset = z * c * h * w + posh_ori * w + posw_ori;
    int weiOffset = y * c * r * s;
    int inChannelOffset = h * w;
    int weightChannelOffset = r * s;

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < s; j++)
        {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;

            if (posh_real >= 0 && posw_real >= 0 && posw_real < w && posh_real < h)
            {
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for (int channel = 0; channel < c; channel++)
                {
                    sum += __half2float(pin[inOffsetTmp + i * w + j] * pwei[weiOffsetTmp + i * s + j]);
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }
            }
        }
    }

    // 计算输出偏移
    int outOffset = z * k * oh * ow + y * oh * ow + x;
    pout[outOffset] = __float2half(sum);
}

