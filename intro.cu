#include <iostream>
#include <math.h>

// function to add the elements of two arrays
//CUDA Kernel function to add the elements of two arrays on the GPU
__global__
__global__
void add(int n, float *x, float *y, double *w, double *z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        y[i] = x[i] * y[i];
        z[i] = w[i] * z[i];
  }
}

int main(void)
{
    int N = 1<<20; // 1M elements

    float *x;
    float *y;
    double *w;
    double *z;

    //Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&w, N*sizeof(double));
    cudaMallocManaged(&z, N*sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 0.01f * (float)i;
        y[i] = 1.01f * (float)(N-i);

        w[i] = 0.01 * (double)i;
        z[i] = 1.01 * (double)(N-i);
    }

    // Run kernel on 1M elements on the CPU
    int blockSize = 256;
    int numBlocks = (N+blockSize)/blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y, w, z);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    double maxError = 0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-z[i]));
    std::cout << "Max error: " << maxError << std::endl;
    
    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(w);
    cudaFree(z);

    return 0;
}