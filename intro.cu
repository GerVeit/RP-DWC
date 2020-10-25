#include <iostream>
#include <math.h>

float maxError = 0;

// function to add the elements of two arrays
//// Wait for GPU to finish before accessing on host
__global__
__global__
void compare(int n, float *x, float *y, double *w, double *z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        y[i] = x[i] * y[i];
        z[i] = w[i] * z[i];

        y[i] =  __fdividef(y[i], float(z[i]));
    }
}

int main(void)
{
    double Nr = 1<<20; 

    float *x;
    float *y;
    double *w;
    double *z;

    //Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&x, Nr*sizeof(float));
    cudaMallocManaged(&y, Nr*sizeof(float));
    cudaMallocManaged(&w, Nr*sizeof(double));
    cudaMallocManaged(&z, Nr*sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < Nr; i++) {

        w[i] = 0.97 * (double)i;
        z[i] = 1.65 * (Nr-i);

        x[i] = (float)w[i];
        y[i] = (float)z[i];
    }

    // Run kernel on 1M elements on the CPU
    int blockSize = 256;
    int numBlocks = (Nr+blockSize)/blockSize;
    compare<<<numBlocks, blockSize>>>(Nr, x, y, w, z);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    float rel=0;
    float abs=0;
    for(int i=0; i < Nr; i++){
        //std::cout << y[i] << std::endl;
        if(y[i] > rel)
            rel = y[i];
        //if(absolute[i] > abs)
        //    abs = absolute[i];
    }
    
    std::cout << "Max relative error: " << rel << std::endl;
    //std::cout << "Max absolute error: " << abs << std::endl;
    
    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(w);
    cudaFree(z);

    return 0;
}