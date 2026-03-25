

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

__device__ __forceinline__ void awq_path_unit(int i, uint32_t pw, uint32_t pz, half s_in, half &out) {
    const float w = static_cast<float>((pw >> (4 * i)) & 0xF);
    const float z = static_cast<float>((pz >> (4 * i)) & 0xF);
    const float s = __half2float(s_in);
    out = __float2half_rn((w - z) * s);
}

__global__ void case2_clustered_kernel(
    const uint32_t* __restrict__ packed_weight,
    const uint32_t* __restrict__ packed_zero,
    const half* __restrict__ scale_tile,
    half* __restrict__ global_sink) 
{
    extern __shared__ half shmem[];
    alignas(16) half decoded_vals[8];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    
    // 核心逻辑：同一个 SP (warp_id % 4) 里的 4 个 Warp 跑相同的 i 路径
    // 导致 SP 内部 4 个 Warp 同时卡在相同的 RAW 依赖上
    int i_base = ((warp_id / 4) % 4) * 2; 

    #pragma unroll 1
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // 执行 2 路计算以模拟原版工作量
        awq_path_unit(i_base, packed_weight[0], packed_zero[0], scale_tile[i_base], decoded_vals[0]);
        awq_path_unit(i_base + 1, packed_weight[0], packed_zero[0], scale_tile[i_base + 1], decoded_vals[1]);
        
        int shmem_idx = threadIdx.x * 8;
        *(uint2*)(&shmem[shmem_idx]) = *(uint2*)(decoded_vals);
    }

    if (__half2float(decoded_vals[0]) == 999.0f) {
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

    printf("Launching Case 2 (Clustered): %d blocks, %d iterations...\n", grid.x, ITERATIONS);
    case2_clustered_kernel<<<grid, block, shmem_size>>>(d_weight, d_zero, d_scale, d_global_sink);

    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Done.\n");

    cudaFree(d_weight); cudaFree(d_zero); cudaFree(d_scale); cudaFree(d_global_sink);
    return 0;
}
