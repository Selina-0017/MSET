#include <cuda_fp16.h>
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

constexpr int KV_HALF_ELEMENTS = 16;
constexpr int SAFE_INDEX = KV_HALF_ELEMENTS - 1;
constexpr int BUG_INDEX = KV_HALF_ELEMENTS;
constexpr int OUTPUT_ELEMENTS = 8;

__global__ void convertF16BoundaryKernel(const __half *input, float *output, int index) {
    int idx = threadIdx.x;
    if (idx == 0) {
        output[0] = __half2float(input[index]);
    }
}

void fillInput(__half *buffer, int n) {
    for (int i = 0; i < n; ++i) {
        buffer[i] = __float2half(static_cast<float>(i) + 0.5f);
    }
}

uint64_t hashOutput(const float *buffer, int n) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &buffer[i], sizeof(bits));
        hash ^= static_cast<uint64_t>(bits);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool runCase(const char *name) {
    __half *hostInput = static_cast<__half *>(std::malloc(sizeof(__half) * KV_HALF_ELEMENTS));
    float *hostOutput = static_cast<float *>(std::malloc(sizeof(float) * OUTPUT_ELEMENTS));
    if (hostInput == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    fillInput(hostInput, KV_HALF_ELEMENTS);
    std::memset(hostOutput, 0, sizeof(float) * OUTPUT_ELEMENTS);

    __half *deviceInput = nullptr;
    float *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceInput, sizeof(__half) * KV_HALF_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(float) * OUTPUT_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMemcpy(deviceInput, hostInput, sizeof(__half) * KV_HALF_ELEMENTS, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(float) * OUTPUT_ELEMENTS));

    int readIndex = 0;
    if (strcmp(name, "safe_llama_f16_read") == 0) {
        readIndex = SAFE_INDEX;
    } else if (strcmp(name, "llama_f16_padding_oob_read") == 0) {
        readIndex = BUG_INDEX;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        CUDA_CHECK_BOOL(cudaFree(deviceInput));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    convertF16BoundaryKernel<<<1, 1>>>(deviceInput, deviceOutput, readIndex);

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

    cudaError_t copyErr = cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * OUTPUT_ELEMENTS, cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
    }

    std::printf("case=%s half_elements=%d read_index=%d output_hash=%llu sync_status=%s copy_status=%s\n",
                name, KV_HALF_ELEMENTS, readIndex,
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
