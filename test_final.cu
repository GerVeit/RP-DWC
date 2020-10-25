#include <iostream>
#include <math.h>
#include <random>

float maxError = 0;


__global__
void vet_mul(float* dst_float, double* x, double* y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xif = float(xi);
    float yif = float(yi);
    dst_float[i] = xif * yif;
}


__global__
void vet_mul(double* dst_double, double* x, double* y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double xi = x[i];
    double yi = y[i];
    dst_double[i] = xi * yi + float(i);
}


__global__ 
void compare(float* dst, float* x, double* y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dst[index] = x[index] / float(y[index]);
}


int main(void)
{
    double Nr = 1<<20; 
    f();
    double* dst_double;
    float *dst_float;
    double *x;
    double *y;

    //Allocate Unified Memory - accessible from CPU or GPU
    cudaMallocManaged(&dst_double, Nr*sizeof(double));
    cudaMallocManaged(&dst_float, Nr*sizeof(float));
    cudaMallocManaged(&x, Nr*sizeof(double));
    cudaMallocManaged(&y, Nr*sizeof(double));
    
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<double> dist(-10, 10);

    // initialize x and y arrays on the host
    for (int i = 0; i < Nr; i++) {
        x[i] = dist(e2);
        y[i] = dist(e2);
    }

    // Run kernel on 1M elements on the CPU
    int blockSize = 256;
    int numBlocks = Nr / blockSize;
    vet_mul<<<numBlocks, blockSize>>>(dst_float, x, y);
    vet_mul<<<numBlocks, blockSize>>>(dst_double, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    float *dst;
    cudaMallocManaged(&dst, Nr*sizeof(double));
    compare<<<numBlocks, blockSize>>>(dst, dst_float, dst_double);
    cudaDeviceSynchronize();


    float rel=0;

    for(int i=0; i < Nr; i++){
        //std::cout << y[i] << std::endl;
        if(dst[i] > rel)
            rel = dst[i];
        //if(absolute[i] > abs)
        //    abs = absolute[i];
    }
    
    std::cout << "Max relative error: " << rel << std::endl;
    //std::cout << "Max absolute error: " << abs << std::endl;
    
    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(dst_double);
    cudaFree(dst_float);
    cudaFree(dst);

    return 0;
}
