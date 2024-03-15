#include <cuda.h>
#include <iostream>

using namespace std;

int main( )
{
    int dev = 0;

    cudaDeviceProp prop;
    cudaError_t    err;

    err = cudaGetDeviceProperties( &prop, dev );
    if( err != cudaSuccess )
    {
        cerr << "Failed to read CUDA device properties from device " << dev << endl
             << cudaGetErrorString(cudaGetLastError()) << endl;
        return -1;
    }

    /* Many more properties are here:
     * https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
     */


    cout << "Device name: " << prop.name << endl;
    cout << "    CUDA Compute Capability: " << prop.major << "." << prop.minor << endl
         << "        Integrated GPU: " << (prop.integrated ? "yes" : "no") << endl
         << "    Memory:" << endl
         << "        Mapping host memory into the GPU          : " << (prop.canMapHostMemory ? "yes" : "no") << endl
         << "        Copy memory while a kernel runs           : " << (prop.deviceOverlap ? "yes" : "no") << endl
         // << "        Can register host memory        : " << (prop.hostRegisterSupported ? "yes" : "no") << endl
         // << "        Can register host memory (reda) : " << (prop.hostRegisterReadOnlySupported ? "yes" : "no") << endl
         << "        Access registered mem with host pointer   : " << (prop.canUseHostPointerForRegisteredMem ? "yes" : "no") << endl
         << "        GPU can access mgmd mem coherent with CPU : " << (prop.concurrentManagedAccess ? "yes" : "no") << endl
         << "        CPU can access mgmd mem directly          : " << (prop.directManagedMemAccessFromHost ? "yes" : "no") << endl
         << "        Unified memory support :" << endl
         << "          Unified addressing   : " << prop.unifiedAddressing << endl
         << "          All values must be 1 for full CUDA unified memory support" << endl
         << "            prop.pageableMemoryAccess                   : " << prop.pageableMemoryAccess << endl
         << "            prop.hostNativeAtomicSupported              : " << prop.hostNativeAtomicSupported << endl
         << "            prop.pageableMemoryAccessUsesHostPageTables : " << prop.pageableMemoryAccessUsesHostPageTables << endl
         << "            prop.directManagedMemAccessFromHost         : " << prop.directManagedMemAccessFromHost << endl
         << "          Must be 1 for full CUDA managed memory support" << endl
         << "            prop.pageableMemoryAccess : " << prop.pageableMemoryAccess << endl
         << "          Must be 1 for unified addressing by no concurrent access" << endl
         << "            prop.managedMemory : " << prop.managedMemory << endl
         << "    Threads:" << endl
         << "        Number of threads in a warp : " << prop.warpSize << endl
         << "        Max threads in one block    : " << prop.maxThreadsPerBlock << endl
         << "        Max threads per dim         : (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl
         << "        Number of multiprocessors   : " << prop.multiProcessorCount << endl
         << "        Concurrently active kernels : " << (prop.concurrentKernels ? "yes" : "no") << endl
         << endl;
}

