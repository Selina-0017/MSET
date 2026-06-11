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

constexpr int Q_HEADS = 32;
constexpr int HEAD_DIM = 256;
constexpr int STRIDE_QBS = Q_HEADS * HEAD_DIM;
constexpr int PREPAD_ELEMENTS = 4096;
constexpr int SAFE_VISIBLE_TOKENS = 2;
constexpr int TOTAL_ELEMENTS = PREPAD_ELEMENTS + SAFE_VISIBLE_TOKENS * STRIDE_QBS + 64;
constexpr int SAFE_QO_INDPTR = 1;
constexpr int BUG_QO_INDPTR = 524287;
constexpr int OUTPUT_LANES = 8;

__host__ __device__ int32_t wrapToInt32(uint64_t value) {
    uint32_t low = static_cast<uint32_t>(value & 0xffffffffULL);
    int64_t signedValue = static_cast<int64_t>(low);
    if ((low & 0x80000000U) != 0U) {
        signedValue -= (1LL << 32);
    }
    return static_cast<int32_t>(signedValue);
}

__global__ void safeQoIndptrKernel(const uint16_t *visibleQ, unsigned long long *output, int tokenBase) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int64_t offset = static_cast<int64_t>(tokenBase) * STRIDE_QBS;
    unsigned long long accum = 0ULL;
    for (int i = 0; i < 4; ++i) {
        accum += static_cast<unsigned long long>(visibleQ[offset + i]);
    }
    output[0] = accum;
}

__global__ void overflowQoIndptrKernel(const uint16_t *visibleQ, unsigned long long *output, int tokenBase) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    uint64_t fullProduct = static_cast<uint64_t>(static_cast<uint32_t>(tokenBase)) *
                           static_cast<uint64_t>(STRIDE_QBS);
    int32_t wrappedOffset = wrapToInt32(fullProduct);
    unsigned long long accum = 0ULL;
    for (int i = 0; i < 4; ++i) {
        accum += static_cast<unsigned long long>(visibleQ[wrappedOffset + i]);
    }
    output[0] = accum;
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
    int tokenBase = 0;
    bool bugCase = false;

    if (strcmp(name, "safe_qo_indptr_read") == 0) {
        tokenBase = SAFE_QO_INDPTR;
        bugCase = false;
    } else if (strcmp(name, "qo_indptr_int32_overflow_read") == 0) {
        tokenBase = BUG_QO_INDPTR;
        bugCase = true;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    uint16_t *hostQ = static_cast<uint16_t *>(std::malloc(sizeof(uint16_t) * TOTAL_ELEMENTS));
    unsigned long long *hostOutput =
        static_cast<unsigned long long *>(std::malloc(sizeof(unsigned long long) * OUTPUT_LANES));
    if (hostQ == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostQ);
        std::free(hostOutput);
        return false;
    }

    fillQ(hostQ, TOTAL_ELEMENTS);
    std::memset(hostOutput, 0, sizeof(unsigned long long) * OUTPUT_LANES);

    uint16_t *deviceQ = nullptr;
    unsigned long long *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceQ, sizeof(uint16_t) * TOTAL_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(unsigned long long) * OUTPUT_LANES));
    CUDA_CHECK_BOOL(cudaMemcpy(deviceQ, hostQ, sizeof(uint16_t) * TOTAL_ELEMENTS, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(unsigned long long) * OUTPUT_LANES));

    const uint16_t *visibleQ = deviceQ + PREPAD_ELEMENTS;
    if (bugCase) {
        overflowQoIndptrKernel<<<1, 1>>>(visibleQ, deviceOutput, tokenBase);
    } else {
        safeQoIndptrKernel<<<1, 1>>>(visibleQ, deviceOutput, tokenBase);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceQ));
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

    uint64_t fullProduct = static_cast<uint64_t>(static_cast<uint32_t>(tokenBase)) *
                           static_cast<uint64_t>(STRIDE_QBS);
    int32_t wrappedOffset = wrapToInt32(fullProduct);
    std::printf("case=%s token_base=%d stride_qbs=%d visible_shift=%d wrapped_offset=%d output_hash=%llu "
                "sync_status=%s copy_status=%s\n",
                name, tokenBase, STRIDE_QBS, PREPAD_ELEMENTS, wrappedOffset,
                static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_LANES)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceQ));
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
