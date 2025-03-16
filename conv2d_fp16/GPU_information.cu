#include <cuda_runtime.h>
#include <stdio.h>

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device Number: %d\n", device);
    printf("  Device name: %s\n", prop.name);
    printf("  Core Clock Rate (KHz): %d\n", prop.clockRate);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Total global memory: %lu bytes\n", prop.totalGlobalMem);
    printf("  Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Concurrent kernels: %d\n", prop.concurrentKernels);
    printf("  ECC enabled: %d\n", prop.ECCEnabled);
    printf("  Total constant memory: %lu bytes\n", prop.totalConstMem);
    printf("  Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max pitch: %lu bytes\n", prop.memPitch);
    printf("  Texture alignment: %lu bytes\n", prop.textureAlignment);
    printf("  Device overlap: %d\n", prop.deviceOverlap);
    printf("  Multi-GPU board: %d\n", prop.isMultiGpuBoard);
    printf("  Multi-GPU board group ID: %d\n", prop.multiGpuBoardGroupID);
    printf("  Unified addressing: %d\n", prop.unifiedAddressing);
    printf("  Can map host memory: %d\n", prop.canMapHostMemory);
    printf("  Compute mode: %d\n", prop.computeMode);
    printf("  Concurrent copy and execution: %d\n", prop.asyncEngineCount);
    printf("  PCI bus ID: %d\n", prop.pciBusID);
    printf("  PCI device ID: %d\n", prop.pciDeviceID);
    printf("  PCI domain ID: %d\n", prop.pciDomainID);
    printf("  L2 cache size: %d bytes\n", prop.l2CacheSize);
    printf("  Max surface 1D: %d\n", prop.maxSurface1D);
    printf("  Max surface 2D: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
    printf("  Max surface 3D: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
    printf("  Max surface cubemap: %d\n", prop.maxSurfaceCubemap);
    printf("  Max texture 1D: %d\n", prop.maxTexture1D);
    printf("  Max texture 2D: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("  Max texture 3D: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("  Max texture cubemap: %d\n", prop.maxTextureCubemap);
    printf("  Max texture 1D layered: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
    printf("  Max texture 2D layered: (%d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1]);
    printf("  Max texture 2D mipmapped: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
    printf("  Max texture cubemap layered: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        printDeviceProperties(device);
    }

    return 0;
}