%%writefile add_vector_task.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_e));                                       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void addVector(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       size_t N)
{
    // grid-stride loop for any N
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * (size_t)gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 50'000'000ULL; // 50M by default
    size_t bytes = N * sizeof(float);
    printf("Vector addition N=%zu (%.2f MB per array)\n", N, bytes/1e6);

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s, SM %d.%d, globalMem=%.1f GB, memClock=%.0f MHz, memBusWidth=%d-bit\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0*1024.0*1024.0),
           prop.memoryClockRate/1000.0, prop.memoryBusWidth);

    float *hA, *hB, *hC;
    CUDA_CHECK(cudaMallocHost(&hA, bytes));
    CUDA_CHECK(cudaMallocHost(&hB, bytes));
    CUDA_CHECK(cudaMallocHost(&hC, bytes));

    for (size_t i = 0; i < N; ++i) {
        hA[i] = 1.0f;             // simple pattern
        hB[i] = (float)(i % 100); // varies to avoid trivial constant folding
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    float* hRef = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i) hRef[i] = hA[i] + hB[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t eStart, eAfterH2D, eAfterKernel, eStop;
    CUDA_CHECK(cudaEventCreate(&eStart));
    CUDA_CHECK(cudaEventCreate(&eAfterH2D));
    CUDA_CHECK(cudaEventCreate(&eAfterKernel));
    CUDA_CHECK(cudaEventCreate(&eStop));

    int block = 256;
    int smCount = prop.multiProcessorCount;
    // 4 blocks per SM is a decent start; tweak if you like
    int grid = std::min<int>((int)((N + block - 1) / block), smCount * 4);

    // --- Time Host2Device + Kernel + Device2Host end-to-end ---
    CUDA_CHECK(cudaEventRecord(eStart));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(eAfterH2D));

    addVector<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());   // check kernel launch
    CUDA_CHECK(cudaEventRecord(eAfterKernel));

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(eStop));
    CUDA_CHECK(cudaEventSynchronize(eStop));

    float h2d_ms=0, kern_ms=0, d2h_ms=0, total_ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, eStart, eStop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,  eStart, eAfterH2D));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, eAfterH2D, eAfterKernel));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,  eAfterKernel, eStop));

    double max_abs_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        max_abs_err = std::max(max_abs_err, (double)std::abs(hRef[i] - hC[i]));
    }
    printf("Verification: max |CPU-GPU| = %.6g\n", max_abs_err);

    // Report
    printf("\nCPU (single-thread)              : %8.3f ms\n", cpu_ms);
    printf("GPU Host2Device                    : %8.3f ms\n", h2d_ms);
    printf("GPU Kernel                         : %8.3f ms\n", kern_ms);
    printf("GPU Device2Host                    : %8.3f ms\n", d2h_ms);
    printf("GPU Total (H2D + Kernel + D2H)     : %8.3f ms\n", total_ms);

    double bytes_moved = 3.0 * bytes; // A,B to device + C back
    double gb_moved = bytes_moved / 1e9;
    double gbps_e2e = gb_moved / (total_ms / 1e3);
    double gbps_kernel_bound = (3.0 * bytes) / (kern_ms / 1e3) / 1e9; // if mem-limited in device
    printf("\nApprox bandwidth (E2E including PCIe): %.2f GB/s\n", gbps_e2e);
    printf("Approx device bandwidth (kernel-only) : %.2f GB/s (rough upper bound)\n", gbps_kernel_bound);

    // Cleanup
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
    free(hRef);
    CUDA_CHECK(cudaEventDestroy(eStart));
    CUDA_CHECK(cudaEventDestroy(eAfterH2D));
    CUDA_CHECK(cudaEventDestroy(eAfterKernel));
    CUDA_CHECK(cudaEventDestroy(eStop));
    return 0;
}
