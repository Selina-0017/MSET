#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK_BOOL(call)                                                   \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err));                    \
            return false;                                                       \
        }                                                                       \
    } while (0)

constexpr int ALLOCATION_BYTES = 80;
constexpr int LOGICAL_BYTES = 16;
constexpr int SAFE_OFFSET_BYTES = 8;
constexpr int BUG_OFFSET_BYTES = 16;
constexpr int OUTPUT_ELEMENTS = 8;

__global__ void chunkCatReadKernel(const unsigned char *input, unsigned long long *output, int offsetBytes) {
    int idx = threadIdx.x;
    if (idx == 0) {
        const unsigned long long *typedInput =
            reinterpret_cast<const unsigned long long *>(input + offsetBytes);
        output[0] = typedInput[0];
    }
}

void fillInput(unsigned char *buffer, int n) {
    for (int i = 0; i < n; ++i) {
        buffer[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);
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
    unsigned char *hostInput = static_cast<unsigned char *>(std::malloc(ALLOCATION_BYTES));
    unsigned long long *hostOutput =
        static_cast<unsigned long long *>(std::malloc(sizeof(unsigned long long) * OUTPUT_ELEMENTS));
    if (hostInput == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    fillInput(hostInput, ALLOCATION_BYTES);
    std::memset(hostOutput, 0, sizeof(unsigned long long) * OUTPUT_ELEMENTS);

    unsigned char *deviceInput = nullptr;
    unsigned long long *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceInput, ALLOCATION_BYTES));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(unsigned long long) * OUTPUT_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMemcpy(deviceInput, hostInput, ALLOCATION_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(unsigned long long) * OUTPUT_ELEMENTS));

    int offsetBytes = 0;
    if (strcmp(name, "safe_chunk_cat_read") == 0) {
        offsetBytes = SAFE_OFFSET_BYTES;
    } else if (strcmp(name, "logical_chunk_cat_oob_read") == 0) {
        offsetBytes = BUG_OFFSET_BYTES;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        CUDA_CHECK_BOOL(cudaFree(deviceInput));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    chunkCatReadKernel<<<1, 1>>>(deviceInput, deviceOutput, offsetBytes);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceInput));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "sync error: %s\n", cudaGetErrorString(syncErr));
    }

    cudaError_t copyErr = cudaMemcpy(hostOutput, deviceOutput, sizeof(unsigned long long) * OUTPUT_ELEMENTS,
                                    cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
    }

    std::printf("case=%s logical_bytes=%d read_offset=%d output_hash=%llu sync_status=%s copy_status=%s\n",
                name, LOGICAL_BYTES, offsetBytes,
                static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_ELEMENTS)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceInput));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostInput);
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
