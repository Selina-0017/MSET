#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK_BOOL(call)                                                   \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                    \
            return false;                                                       \
        }                                                                       \
    } while (0)

constexpr int HEADS = 128;
constexpr int D_NOPE = 512;
constexpr int D_ROPE = 64;
constexpr int SAFE_SEQ = 29120;
constexpr int BUG_SEQ = 29130;
constexpr int THREADS = 256;
constexpr int TILE_ROWS = 32;
constexpr int OUTPUT_LANES = 16;

__host__ __device__ constexpr int alignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__global__ void safeMlaConcatKernel(const uint16_t *rope, unsigned long long *output, int rows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= rows) {
        return;
    }

    int base = idx * D_ROPE;
    unsigned long long accum = 0ULL;
    for (int i = 0; i < 4; ++i) {
        accum += static_cast<unsigned long long>(rope[base + i]);
    }
    atomicAdd(&output[idx % OUTPUT_LANES], accum);
}

__global__ void mlaThresholdOverflowKernel(const uint16_t *rope, unsigned long long *output, int rows,
                                           int paddedRows) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= paddedRows) {
        return;
    }

    int base = idx * D_ROPE;
    unsigned long long accum = 0ULL;
    for (int i = 0; i < 4; ++i) {
        accum += static_cast<unsigned long long>(rope[base + i]);
    }
    atomicAdd(&output[idx % OUTPUT_LANES], accum);
}

void fillRope(uint16_t *buffer, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        buffer[i] = static_cast<uint16_t>((i % 251) + 1);
    }
}

uint64_t hashOutput(const unsigned long long *buffer, int n) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        hash ^= static_cast<uint64_t>(buffer[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool runCase(const char *name) {
    int seqLen = 0;
    int paddedRows = 0;
    bool bugCase = false;

    if (strcmp(name, "safe_mla_concat_read") == 0) {
        seqLen = SAFE_SEQ;
        paddedRows = SAFE_SEQ;
        bugCase = false;
    } else if (strcmp(name, "mla_concat_threshold_oob_read") == 0) {
        seqLen = BUG_SEQ;
        paddedRows = alignUp(BUG_SEQ, TILE_ROWS);
        bugCase = true;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    const size_t ropeCount = static_cast<size_t>(seqLen) * D_ROPE;
    uint16_t *hostRope = static_cast<uint16_t *>(std::malloc(sizeof(uint16_t) * ropeCount));
    unsigned long long *hostOutput =
        static_cast<unsigned long long *>(std::malloc(sizeof(unsigned long long) * OUTPUT_LANES));
    if (hostRope == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostRope);
        std::free(hostOutput);
        return false;
    }

    fillRope(hostRope, ropeCount);
    std::memset(hostOutput, 0, sizeof(unsigned long long) * OUTPUT_LANES);

    uint16_t *deviceRope = nullptr;
    unsigned long long *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceRope, sizeof(uint16_t) * ropeCount));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(unsigned long long) * OUTPUT_LANES));
    CUDA_CHECK_BOOL(cudaMemcpy(deviceRope, hostRope, sizeof(uint16_t) * ropeCount, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(unsigned long long) * OUTPUT_LANES));

    const int launchRows = bugCase ? paddedRows : seqLen;
    const int blocks = (launchRows + THREADS - 1) / THREADS;
    if (bugCase) {
        mlaThresholdOverflowKernel<<<blocks, THREADS>>>(deviceRope, deviceOutput, seqLen, paddedRows);
    } else {
        safeMlaConcatKernel<<<blocks, THREADS>>>(deviceRope, deviceOutput, seqLen);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceRope));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostRope);
        std::free(hostOutput);
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "sync error: %s\n", cudaGetErrorString(syncErr));
    }

    cudaError_t copyErr = cudaMemcpy(hostOutput, deviceOutput, sizeof(unsigned long long) * OUTPUT_LANES,
                                     cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
    }

    std::printf("case=%s seq_len=%d padded_rows=%d d_nope=%d d_rope=%d heads=%d output_hash=%llu "
                "sync_status=%s copy_status=%s\n",
                name, seqLen, paddedRows, D_NOPE, D_ROPE, HEADS,
                static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_LANES)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceRope));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostRope);
    std::free(hostOutput);
    return syncErr == cudaSuccess && copyErr == cudaSuccess;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <case>\n", argv[0]);
        return 1;
    }

    bool ok = runCase(argv[1]);
    return ok ? 0 : 3;
}
