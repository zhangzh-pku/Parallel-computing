#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

#define N 512
#define MAXITER 100
#define RTOL 1e-6
#define PI 3.14159265358979323846
#define u_size (N + 2) * (N + 2) * (N + 2)
#define b_size N * N * N

//init of the variable
void init_sol(double *__restrict__ b, double *__restrict__ u_exact, double *__restrict__ u)
{
    double a = N / 4.;
    double h = 1. / (N + 1);
#pragma omp parallel for
    for (int i = 0; i < N + 2; i++)
    #pragma omp parallel for
        for (int j = 0; j < N + 2; j++)
        #pragma omp parallel for
            for (int k = 0; k < N + 2; k++)
            {
                u_exact[i * (N + 2) * (N + 2) + j * (N + 2) + k] =
                    sin(a * PI * i * h) * sin(a * PI * j * h) * sin(a * PI * k * h);
                u[i * (N + 2) * (N + 2) + j * (N + 2) + k] = 0.;
            }
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    #pragma omp parallel for
        for (int j = 0; j < N; j++)
        #pragma omp parallel for
            for (int k = 0; k < N; k++)
            {
                b[i * N * N + j * N + k] =
                    3. * a * a * PI * PI *
                    sin(a * PI * (i + 1) * h) * sin(a * PI * (j + 1) * h) * sin(a * PI * (k + 1) * h) * h * h;
            }
}

__global__ void jacobi(double* u,double* b,double* t)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] =
                (b[i * N * N + j * N + k] + 
                t[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] + 
                t[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1] + 
                t[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0] + 
                t[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2] + 
                t[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1] + 
                t[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                ) / 6.0;
}

__global__ void mycopy(double* u,double* b,double* n,double* t)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    
        t[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] =
            u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1];
           
            double r = b[i * N * N + j * N + k] +
            u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] +
            u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1] +
            u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0] +
            u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2] +
            u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1] +
            u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] -
            6.0 * u[(i + 1) * ((N + 2) * (N + 2)) + (j + 1) * (N + 2) + (k + 1)];
        n[i * N * N + j * N + k] = r * r;
        
}
//怎么reduce呢
__global__ void r_norm(double* n)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    int idx = i * N * N + j * N + k;
    int pos = 2;
    int all = b_size;
    while(all >= pos)
    {
        if((idx + 1) % pos == 0)
        {
            n[idx] += n[idx - pos/2];
        }
        pos *= 2;
    }
}

double error(double *__restrict__ u, double *__restrict__ u_exact)
{
    double tmp = 0;
#pragma omp parallel for reduction(+ \
                                   : tmp)
    for (int i = 0; i < N; i++)
    #pragma omp parallel for reduction(+ \
        : tmp)
        for (int j = 0; j < N; j++)
        #pragma omp parallel for reduction(+ \
            : tmp)
            for (int k = 0; k < N; k++)
            {
                tmp += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] - u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    double tmp2 = 0;
#pragma omp parallel for reduction(+ \
                                   : tmp2)
    for (int i = 0; i < N; i++)
    #pragma omp parallel for reduction(+ \
        : tmp2)
        for (int j = 0; j < N; j++)
        #pragma omp parallel for reduction(+ \
            : tmp2)
            for (int k = 0; k < N; k++)
            {
                tmp2 += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    return pow(tmp, 0.5) / pow(tmp2, 0.5);
}

double residual_norm(double *__restrict__ u, double *__restrict__ b)
{
    double norm2 = 0;
    #pragma omp parallel for reduction(+ \
        : norm2)
    for (int i = 0; i < N; i++)
    {
        #pragma omp parallel for reduction(+ \
            : norm2)
        for (int j = 0; j < N; j++)
        {
            #pragma omp parallel for reduction(+ \
                : norm2)
            for (int k = 0; k < N; k++)
            {
                double r = b[i * N * N + j * N + k] +
                           u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] +
                           u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1] +
                           u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0] +
                           u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2] +
                           u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1] +
                           u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] -
                           6.0 * u[(i + 1) * ((N + 2) * (N + 2)) + (j + 1) * (N + 2) + (k + 1)];
                norm2 += r * r;
            }
        }
    }
    return sqrt(norm2);
}

__device__
double myresidual_norm(double *d_u, double *d_b, int i, int j, int k)
{
    double r = d_b[i * N * N + j * N + k] +
                + d_u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1]
                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0]
                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2]
                + d_u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1]
                + d_u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                - 6.0 * d_u[(i + 1) * ((N + 2) * (N + 2)) + (j + 1) * (N + 2) + (k + 1)];
    return r*r;
}

__global__ void binary_sum1D_kernel(double *a, int size, int stride, double *d_u, double *d_b)
{
    long long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i * stride + stride / 2 < size)
        if (stride > 2)
            a[i*stride] += a[i*stride + stride/2];
        else
        {
            int x = 2*i / (N*N), y = 2*i / N - N*x, z = 2*i % N;
            a[2*i] = myresidual_norm(d_u, d_b, x, y, z) + myresidual_norm(d_u, d_b, x, y, z+1);
        }
}

double residual_norm_cuda(double *d_u, double *d_b, double *d_res)
{
    int d;

    for (int i=2; i<=b_size; i*=2)
    {
        if (i<=b_size/32)
            d = (b_size/32)/i;
        else d = 1;
        binary_sum1D_kernel<<<d, 32>>>(d_res, b_size, i, d_u, d_b);
    }
    double res_norm;
    cudaMemcpy(&res_norm, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    return sqrt(res_norm);
}

int main(void)
{   
    double *u = (double *)malloc(u_size * sizeof(double));
    double *u_exact = (double *)malloc(u_size * sizeof(double));
    double *b = (double *)malloc(b_size * sizeof(double));
    double *d_u;
    double *d_tem;
    //d_tem表示上一次的迭代值
    double *d_b;
    double *d_n;
    //d_n方便规约norm
    cudaMalloc((void**)&d_u, u_size * sizeof(double));
    cudaMalloc((void**)&d_tem, u_size * sizeof(double));
    cudaMalloc((void**)&d_b, b_size * sizeof(double));
    cudaMalloc((void**)&d_n, b_size * sizeof(double));
    init_sol(b, u_exact, u);cudaMemcpy(d_tem, u, u_size * sizeof(double), cudaMemcpyHostToDevice);
    double* _nor=(double*)malloc(b_size * sizeof(double));
    cudaMemcpy(d_b, b, b_size * sizeof(double), cudaMemcpyHostToDevice);

    int tsteps = MAXITER;
    double norm0=residual_norm(u,b);
    double norm = 100.;
    dim3 grid_dim(N,1,1);
    dim3 block_dim(N,N,1);
    //printf("time\n");
    clock_t start, finish;
    start = clock();
    for (int k = 0; k < MAXITER; k++)
    {
        jacobi<<<block_dim,grid_dim>>>(d_u, d_b, d_tem);   
        mycopy<<<block_dim,grid_dim>>>(d_u, d_b, d_n, d_tem);
        //r_norm<<<block_dim,grid_dim>>>(d_u);
        norm = residual_norm_cuda(d_u,d_b,d_n);
        
        //printf("###%g，%g\n",norm,norm0);
        if(norm < RTOL * norm0)
        {
            printf("###%g，%g\n",norm,norm0);
            printf("Iteration %d, normr/normr0=%g\n", k + 1, norm / norm0);
            tsteps = k + 1;
            printf("Converged with %d iterations.\n", tsteps);
            break;
        }
    }
    finish = clock();
    cudaMemcpy(u, d_u, u_size * sizeof(double), cudaMemcpyDeviceToHost);
    double _time = (float)(finish - start) / CLOCKS_PER_SEC; 
    printf("time: %g\n", _time);

    printf("Residual norm: %g\n", norm);

    double final_normr = residual_norm(u, b); // Please ensure that this residual_norm is exact.
    printf("Final residual norm: %g\n", final_normr);
    printf("|r_n|/|r_0| = %g\n", final_normr / norm0);

    long long residual_norm_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + (N * N * N)) * tsteps;
    long long gs_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + 2 * (N * N * N)) * tsteps;

    long long total_bytes = residual_norm_bytes + gs_bytes;
    double bandwidth = total_bytes / _time;

    printf("total bandwidth: %g GB/s\n", bandwidth / (double)(1 << 30));

    double relative_err = error(u, u_exact);
    printf("relative error: %g\n", relative_err);

    free(u);
    free(u_exact);
    free(b);
    return 0;
}