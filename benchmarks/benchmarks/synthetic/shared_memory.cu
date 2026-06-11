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

constexpr int OUTPUT_ELEMENTS = 128;
constexpr int SHARED_ELEMENTS = 64;
constexpr int THREADS = 128;
constexpr int BLOCKS = 1;

__global__ void sharedSafeCopyKernel(const int *input, int *output) {
    __shared__ int scratch[SHARED_ELEMENTS];
    int idx = threadIdx.x;
    if (idx < SHARED_ELEMENTS) {
        scratch[idx] = input[idx];
    }
    __syncthreads();
    if (idx < SHARED_ELEMENTS) {
        output[idx] = scratch[idx];
    }
}

__global__ void sharedOobReadByOneKernel(const int *input, int *output) {
    __shared__ int scratch[SHARED_ELEMENTS];
    int idx = threadIdx.x;
    if (idx < SHARED_ELEMENTS) {
        scratch[idx] = input[idx];
    }
    __syncthreads();
    if (idx < SHARED_ELEMENTS) {
        int src = idx == SHARED_ELEMENTS - 1 ? SHARED_ELEMENTS : idx;
        output[idx] = scratch[src];
    }
}

__global__ void sharedOobWriteByOneKernel(const int *input, int *output) {
    __shared__ int scratch[SHARED_ELEMENTS];
    int idx = threadIdx.x;
    if (idx < SHARED_ELEMENTS) {
        scratch[idx] = input[idx];
    }
    __syncthreads();
    if (idx < SHARED_ELEMENTS) {
        int dst = idx == SHARED_ELEMENTS - 1 ? SHARED_ELEMENTS : idx;
        scratch[dst] = input[idx];
    }
    __syncthreads();
    if (idx < SHARED_ELEMENTS) {
        output[idx] = scratch[idx];
    }
}

__global__ void sharedNegativeIndexReadKernel(const int *input, int *output) {
    __shared__ int scratch[SHARED_ELEMENTS];
    int idx = threadIdx.x;
    if (idx < SHARED_ELEMENTS) {
        scratch[idx] = input[idx];
    }
    __syncthreads();
    if (idx == 0) {
        output[0] = scratch[-1];
    } else if (idx < SHARED_ELEMENTS) {
        output[idx] = scratch[idx];
    }
}

__global__ void sharedGuardMismatchWriteKernel(const int *input, int *output) {
    __shared__ int scratch[SHARED_ELEMENTS];
    int idx = threadIdx.x;
    if (idx <= SHARED_ELEMENTS) {
        scratch[idx] = input[idx % OUTPUT_ELEMENTS];
    }
    __syncthreads();
    if (idx < SHARED_ELEMENTS) {
        output[idx] = scratch[idx];
    }
}

void fillHostInput(int *buffer, int n) {
    for (int i = 0; i < n; ++i) {
        buffer[i] = i * 19 + 5;
    }
}

uint64_t hashOutput(const int *buffer, int n) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        hash ^= static_cast<uint64_t>(buffer[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool runCase(const char *name) {
    int *hostInput = static_cast<int *>(std::malloc(sizeof(int) * OUTPUT_ELEMENTS));
    int *hostOutput = static_cast<int *>(std::malloc(sizeof(int) * OUTPUT_ELEMENTS));
    if (hostInput == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    fillHostInput(hostInput, OUTPUT_ELEMENTS);
    std::memset(hostOutput, 0, sizeof(int) * OUTPUT_ELEMENTS);

    int *deviceInput = nullptr;
    int *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceInput, sizeof(int) * OUTPUT_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(int) * OUTPUT_ELEMENTS));
    CUDA_CHECK_BOOL(cudaMemcpy(deviceInput, hostInput, sizeof(int) * OUTPUT_ELEMENTS, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(int) * OUTPUT_ELEMENTS));

    if (strcmp(name, "shared_safe_copy") == 0) {
        sharedSafeCopyKernel<<<BLOCKS, THREADS>>>(deviceInput, deviceOutput);
    } else if (strcmp(name, "shared_oob_read_by_one") == 0) {
        sharedOobReadByOneKernel<<<BLOCKS, THREADS>>>(deviceInput, deviceOutput);
    } else if (strcmp(name, "shared_oob_write_by_one") == 0) {
        sharedOobWriteByOneKernel<<<BLOCKS, THREADS>>>(deviceInput, deviceOutput);
    } else if (strcmp(name, "shared_negative_index_read") == 0) {
        sharedNegativeIndexReadKernel<<<BLOCKS, THREADS>>>(deviceInput, deviceOutput);
    } else if (strcmp(name, "shared_guard_mismatch_write") == 0) {
        sharedGuardMismatchWriteKernel<<<BLOCKS, THREADS>>>(deviceInput, deviceOutput);
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        CUDA_CHECK_BOOL(cudaFree(deviceInput));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

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

    CUDA_CHECK_BOOL(cudaMemcpy(hostOutput, deviceOutput, sizeof(int) * OUTPUT_ELEMENTS, cudaMemcpyDeviceToHost));
    std::printf("case=%s output_hash=%llu sync_status=%s\n", name,
                static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_ELEMENTS)),
                cudaGetErrorString(syncErr));

    CUDA_CHECK_BOOL(cudaFree(deviceInput));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostInput);
    std::free(hostOutput);
    return syncErr == cudaSuccess;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <case>\n", argv[0]);
        return 1;
    }

    bool ok = runCase(argv[1]);
    return ok ? 0 : 3;
}
