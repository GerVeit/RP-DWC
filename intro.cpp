#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y, double *w, double *z)
{
    for (int i = 0; i < n; i++){
        y[i] = x[i] * y[i];
        z[i] = w[i] * z[i];  
    }
}

int main(void)
{
    int N = 1<<20; // 1M elements

    float *x = new float[N];
    float *y = new float[N];
    double *w = new double[N];
    double *z = new double[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.01f * (float)i;
        y[i] = 1.99f * (float)(N-i);

        w[i] = 1.01 * (double)i;
        z[i] = 1.99 * (double)(N-i);
    }

    // Run kernel on 1M elements on the CPU
    add(N, x, y, w, z);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i]-z[i]));  
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}