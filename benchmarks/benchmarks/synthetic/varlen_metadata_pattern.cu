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

constexpr int ACTUAL_TOKENS = 256;
constexpr int TOKEN_STRIDE = 4096;
constexpr int SAFE_SEQUENCE_COUNT = 1;
constexpr int BUG_SEQUENCE_COUNT = 3;
constexpr int SAFE_CU_SEQLENS[SAFE_SEQUENCE_COUNT + 1] = {0, 256};
constexpr int BUG_CU_SEQLENS[BUG_SEQUENCE_COUNT + 1] = {0, 256, 512, 768};
constexpr int OUTPUT_LANES = 4;

__global__ void safeVarlenMetadataKernel(const uint16_t *q, const int *cuSeqlens,
                                         unsigned long long *output) {
    int sequenceId = blockIdx.x;
    if (threadIdx.x != 0 || sequenceId >= SAFE_SEQUENCE_COUNT) {
        return;
    }

    int lastToken = cuSeqlens[sequenceId + 1] - 1;
    int base = lastToken * TOKEN_STRIDE;
    unsigned long long accum = 0ULL;
    for (int i = 0; i < 4; ++i) {
        accum += static_cast<unsigned long long>(q[base + i]);
    }
    atomicAdd(&output[sequenceId], accum);
}

__global__ void mismatchedVarlenMetadataKernel(const uint16_t *q, const int *cuSeqlens,
                                               unsigned long long *output, int sequenceCount) {
    int sequenceId = blockIdx.x;
    if (threadIdx.x != 0 || sequenceId >= sequenceCount) {
        return;
    }

    int lastToken = cuSeqlens[sequenceId + 1] - 1;
    int base = lastToken * TOKEN_STRIDE;
    unsigned long long accum = 0ULL;
    for (int i = 0; i < 4; ++i) {
        accum += static_cast<unsigned long long>(q[base + i]);
    }
    atomicAdd(&output[sequenceId], accum);
}

void fillQ(uint16_t *buffer, size_t count) {
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
    bool bugCase = false;
    int sequenceCount = 0;

    if (strcmp(name, "safe_varlen_metadata_read") == 0) {
        bugCase = false;
        sequenceCount = SAFE_SEQUENCE_COUNT;
    } else if (strcmp(name, "varlen_cu_seqlens_mismatch_read") == 0) {
        bugCase = true;
        sequenceCount = BUG_SEQUENCE_COUNT;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    const size_t qCount = static_cast<size_t>(ACTUAL_TOKENS) * TOKEN_STRIDE;
    uint16_t *hostQ = static_cast<uint16_t *>(std::malloc(sizeof(uint16_t) * qCount));
    unsigned long long *hostOutput =
        static_cast<unsigned long long *>(std::malloc(sizeof(unsigned long long) * OUTPUT_LANES));
    if (hostQ == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostQ);
        std::free(hostOutput);
        return false;
    }

    fillQ(hostQ, qCount);
    std::memset(hostOutput, 0, sizeof(unsigned long long) * OUTPUT_LANES);

    uint16_t *deviceQ = nullptr;
    int *deviceCuSeqlens = nullptr;
    unsigned long long *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceQ, sizeof(uint16_t) * qCount));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(unsigned long long) * OUTPUT_LANES));
    if (bugCase) {
        CUDA_CHECK_BOOL(cudaMalloc(&deviceCuSeqlens, sizeof(int) * (BUG_SEQUENCE_COUNT + 1)));
        CUDA_CHECK_BOOL(cudaMemcpy(deviceCuSeqlens, BUG_CU_SEQLENS, sizeof(int) * (BUG_SEQUENCE_COUNT + 1),
                                   cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK_BOOL(cudaMalloc(&deviceCuSeqlens, sizeof(int) * (SAFE_SEQUENCE_COUNT + 1)));
        CUDA_CHECK_BOOL(cudaMemcpy(deviceCuSeqlens, SAFE_CU_SEQLENS, sizeof(int) * (SAFE_SEQUENCE_COUNT + 1),
                                   cudaMemcpyHostToDevice));
    }
    CUDA_CHECK_BOOL(cudaMemcpy(deviceQ, hostQ, sizeof(uint16_t) * qCount, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(unsigned long long) * OUTPUT_LANES));

    if (bugCase) {
        mismatchedVarlenMetadataKernel<<<sequenceCount, 1>>>(deviceQ, deviceCuSeqlens, deviceOutput,
                                                             sequenceCount);
    } else {
        safeVarlenMetadataKernel<<<sequenceCount, 1>>>(deviceQ, deviceCuSeqlens, deviceOutput);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceQ));
        CUDA_CHECK_BOOL(cudaFree(deviceCuSeqlens));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostQ);
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

    int claimedTokens = bugCase ? BUG_CU_SEQLENS[BUG_SEQUENCE_COUNT] : SAFE_CU_SEQLENS[SAFE_SEQUENCE_COUNT];
    std::printf("case=%s actual_tokens=%d claimed_tokens=%d token_stride=%d sequence_count=%d output_hash=%llu "
                "sync_status=%s copy_status=%s\n",
                name, ACTUAL_TOKENS, claimedTokens, TOKEN_STRIDE, sequenceCount,
                static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_LANES)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceQ));
    CUDA_CHECK_BOOL(cudaFree(deviceCuSeqlens));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostQ);
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
