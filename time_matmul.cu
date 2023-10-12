#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define DIM_M 1024
#define DIM_N 64
#define DIM_K 1024

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size_A = DIM_M * DIM_K * sizeof(float);
    int size_B = DIM_K * DIM_N * sizeof(float);
    int size_C = DIM_M * DIM_N * sizeof(float);

    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    for(int i = 0; i < DIM_M * DIM_K; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for(int i = 0; i < DIM_K * DIM_N; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM_N, DIM_M, DIM_K, &alpha, d_B, DIM_N, d_A, DIM_K, &beta, d_C, DIM_N);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration_ms = end - start;
    printf("Matrix multiplication time: %f ms\n", duration_ms.count());

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
