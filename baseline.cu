
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define ITERATIONS 1000

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void raw_baseline_kernel(
    const uint32_t* __restrict__ packed_weight,
    const uint32_t* __restrict__ packed_zero,
    const half* __restrict__ scale_tile,
    half* __restrict__ global_sink)
{
    extern __shared__ half shmem[];
    alignas(16) half decoded_vals[8];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    #pragma unroll 1
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                // RAW
                const float w = static_cast<float>((packed_weight[j] >> (4 * i)) & 0xF);
                const float z = static_cast<float>((packed_zero[j]   >> (4 * i)) & 0xF);
                const float s = __half2float(scale_tile[j * 8 + i]);
                decoded_vals[i] = __float2half_rn((w - z) * s);
            }

            // 线性存储，无 Bank Conflict
            int shmem_idx = threadIdx.x * 8;
            *(uint4*)(&shmem[shmem_idx]) = *(uint4*)(decoded_vals);
        }
    }

    if (decoded_vals[0] == (half)999.0f) {
        global_sink[tid] = decoded_vals[0];
    }
}

int main() {
    uint32_t *d_weight, *d_zero;
    half *d_scale, *d_global_sink;

    int threads_per_block = 512;
    int blocks_per_grid = 10;
    int total_threads = threads_per_block * blocks_per_grid;

    CHECK_CUDA(cudaMalloc(&d_weight, 4 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_zero, 4 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_scale, 32 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_global_sink, total_threads * sizeof(half)));

    CHECK_CUDA(cudaMemset(d_weight, 0xAB, 4 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_zero, 0x01, 4 * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_scale, 0x00, 32 * sizeof(half)));

    dim3 block(threads_per_block);
    dim3 grid(blocks_per_grid);
    size_t shmem_size = 32 * 1024;

    printf("Launching Baseline Kernel: %d blocks, %d iterations...\n", grid.x, ITERATIONS);

    raw_baseline_kernel<<<grid, block, shmem_size>>>(d_weight, d_zero, d_scale, d_global_sink);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Done.\n");

    cudaFree(d_weight);
    cudaFree(d_zero);
    cudaFree(d_scale);
    cudaFree(d_global_sink);

    return 0;
}
