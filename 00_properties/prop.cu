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
        cerr << "Failed to read CUDA device properties from device " << dev << endl;
        return -1;
    }

    cout << "Device name: " << prop.name << endl;
    cout << "    CUDA Compute Capability: " << prop.major << "." << prop.minor << endl
         << "        Integrated GPU: " << (prop.integrated ? "yes" : "no") << endl
         << "    Memory:" << endl
         << "        Mapping host memory into the GPU: " << (prop.canMapHostMemory ? "yes" : "no") << endl
         << "        Copy memory while a kernel runs:  " << (prop.deviceOverlap ? "yes" : "no") << endl
         << "    Threads:" << endl
         << "        Number of threads in a warp : " << prop.warpSize << endl
         << "        Max threads in one block    : " << prop.maxThreadsPerBlock << endl
         << "        Max threads per dim         : (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl
         << "        Number of multiprocessors   : " << prop.multiProcessorCount << endl
         << "        Concurrently active kernels : " << (prop.concurrentKernels ? "yes" : "no") << endl
         << endl;
}

