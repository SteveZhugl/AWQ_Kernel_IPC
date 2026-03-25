

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

__global__ void manual_reorder_kernel(
    const uint32_t* __restrict__ packed_weight,
    const uint32_t* __restrict__ packed_zero,
    const half* __restrict__ scale_tile,
    half* __restrict__ global_sink) 
{
    extern __shared__ half shmem[];
    
    // 使用寄存器数组存储中间值，强制解开依赖链
    float w_vals[8];
    float z_vals[8];
    float s_vals[8];
    half decoded_vals[8];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    #pragma unroll 1
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        
        // --- Phase 1: 集中发射提取指令 (Hide Shift Latency) ---
        // 这里的目标是让 SASS 出现连续的 SHF 和 LOP3
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            w_vals[i] = static_cast<float>((packed_weight[0] >> (4 * i)) & 0xF);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            z_vals[i] = static_cast<float>((packed_zero[0] >> (4 * i)) & 0xF);
        }

        // --- Phase 2: 集中发射加载指令 (Hide Memory Latency) ---
        // 对应 SASS 里的 LDG.E.U16.CONSTANT 连续发射
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_vals[i] = __half2float(scale_tile[i]);
        }

        // --- Phase 3: 集中执行计算 (EX Benefit) ---
        // 此时第一组 w_vals[0] 可能已经 Ready 了
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            decoded_vals[i] = __float2half_rn((w_vals[i] - z_vals[i]) * s_vals[i]);
        }

        // 存储到 Shared Memory
        int shmem_idx = threadIdx.x * 8;
        *(uint4*)(&shmem[shmem_idx]) = *(uint4*)(decoded_vals);
    }

    if (__half2float(decoded_vals[0]) == 999.0f) {
        global_sink[tid] = decoded_vals[0];
    }
}

int main() {
    uint32_t *d_weight, *d_zero;
    half *d_scale, *d_global_sink;

    int threads_per_block = 512; // 使用 4 个 Warp 填满 SP
    int blocks_per_grid = 10;    // 保持 SM 压力适中
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
    size_t shmem_size = threads_per_block * 8 * sizeof(half);

    printf("Launching Manual Reordered Kernel: %d blocks, %d threads...\n", grid.x, threads_per_block);

    manual_reorder_kernel<<<grid, block, shmem_size>>>(d_weight, d_zero, d_scale, d_global_sink);

    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Done.\n");

    cudaFree(d_weight); cudaFree(d_zero); cudaFree(d_scale); cudaFree(d_global_sink);
    return 0;
}
