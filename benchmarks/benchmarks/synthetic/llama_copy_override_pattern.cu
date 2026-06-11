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
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                    \
            return false;                                                       \
        }                                                                       \
    } while (0)

constexpr int HALF_ELEMENTS = 16;
constexpr int SAFE_HALF_OFFSET = 12;
constexpr int BUG_HALF_OFFSET = 16;

__global__ void safeOverrideCopyKernel(const __half *source, unsigned long long *output, int halfOffset) {
    unsigned long long packed =
        *reinterpret_cast<const unsigned long long *>(source + halfOffset);
    output[0] = packed;
}

__global__ void overrideOnePastEndKernel(const __half *source, unsigned long long *output, int halfOffset) {
    unsigned long long packed =
        *reinterpret_cast<const unsigned long long *>(source + halfOffset);
    output[0] = packed;
}

void fillSource(__half *buffer, int count) {
    for (int i = 0; i < count; ++i) {
        buffer[i] = __float2half(1.0f + static_cast<float>(i) * 0.5f);
    }
}

bool runCase(const char *name) {
    int halfOffset = 0;
    bool bugCase = false;

    if (strcmp(name, "safe_llama_override_copy") == 0) {
        halfOffset = SAFE_HALF_OFFSET;
        bugCase = false;
    } else if (strcmp(name, "llama_override_one_past_end_read") == 0) {
        halfOffset = BUG_HALF_OFFSET;
        bugCase = true;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    __half *hostSource = static_cast<__half *>(std::malloc(sizeof(__half) * HALF_ELEMENTS));
    unsigned long long *hostOutput =
        static_cast<unsigned long long *>(std::malloc(sizeof(unsigned long long)));
    if (hostSource == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostSource);
        std::free(hostOutput);
        return false;
    }

    fillSource(hostSource, HALF_ELEMENTS);
    hostOutput[0] = 0ULL;

    __half *deviceSource = nullptr;
    unsigned long long *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceSource, sizeof(__half) * HALF_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(unsigned long long)));
    CUDA_CHECK_BOOL(
        cudaMemcpy(deviceSource, hostSource, sizeof(__half) * HALF_ELEMENTS, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(unsigned long long)));

    if (bugCase) {
        overrideOnePastEndKernel<<<1, 1>>>(deviceSource, deviceOutput, halfOffset);
    } else {
        safeOverrideCopyKernel<<<1, 1>>>(deviceSource, deviceOutput, halfOffset);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceSource));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostSource);
        std::free(hostOutput);
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "sync error: %s\n", cudaGetErrorString(syncErr));
    }

    cudaError_t copyErr =
        cudaMemcpy(hostOutput, deviceOutput, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
    }

    std::printf("case=%s half_elements=%d byte_size=%zu half_offset=%d output_hash=%llu "
                "sync_status=%s copy_status=%s\n",
                name, HALF_ELEMENTS, sizeof(__half) * HALF_ELEMENTS, halfOffset,
                static_cast<unsigned long long>(hostOutput[0]), cudaGetErrorString(syncErr),
                cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceSource));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostSource);
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
