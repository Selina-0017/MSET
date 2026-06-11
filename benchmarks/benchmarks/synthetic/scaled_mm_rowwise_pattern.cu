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

constexpr int TILE_ROWS = 256;
constexpr int SAFE_ROWS = 512;
constexpr int BUG_ROWS = 257;
constexpr int OUTPUT_ELEMENTS = 8;

__host__ __device__ constexpr int alignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__global__ void safeRowwiseScaleKernel(const unsigned int *scale, unsigned long long *output, int rows) {
    int idx = threadIdx.x;
    if (idx >= rows) {
        return;
    }

    unsigned int value = scale[idx];
    atomicAdd(&output[idx % OUTPUT_ELEMENTS], static_cast<unsigned long long>(value));
}

__global__ void rowwiseScaleTailKernel(const unsigned int *scale, unsigned long long *output, int rows,
                                       int paddedRows) {
    int idx = threadIdx.x;
    if (idx >= paddedRows) {
        return;
    }

    unsigned int value = scale[idx];
    atomicAdd(&output[idx % OUTPUT_ELEMENTS], static_cast<unsigned long long>(value));
}

void fillScale(unsigned int *buffer, int rows) {
    for (int i = 0; i < rows; ++i) {
        buffer[i] = static_cast<unsigned int>(1000 + i * 3);
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
    int rows = 0;
    int launchRows = 0;
    bool bugCase = false;

    if (strcmp(name, "safe_rowwise_scale_read") == 0) {
        rows = SAFE_ROWS;
        launchRows = SAFE_ROWS;
        bugCase = false;
    } else if (strcmp(name, "rowwise_scale_tail_oob_read") == 0) {
        rows = BUG_ROWS;
        launchRows = alignUp(BUG_ROWS, TILE_ROWS);
        bugCase = true;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    unsigned int *hostScale = static_cast<unsigned int *>(std::malloc(sizeof(unsigned int) * rows));
    unsigned long long *hostOutput =
        static_cast<unsigned long long *>(std::malloc(sizeof(unsigned long long) * OUTPUT_ELEMENTS));
    if (hostScale == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostScale);
        std::free(hostOutput);
        return false;
    }

    fillScale(hostScale, rows);
    std::memset(hostOutput, 0, sizeof(unsigned long long) * OUTPUT_ELEMENTS);

    unsigned int *deviceScale = nullptr;
    unsigned long long *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceScale, sizeof(unsigned int) * rows));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(unsigned long long) * OUTPUT_ELEMENTS));
    CUDA_CHECK_BOOL(
        cudaMemcpy(deviceScale, hostScale, sizeof(unsigned int) * rows, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(unsigned long long) * OUTPUT_ELEMENTS));

    if (bugCase) {
        rowwiseScaleTailKernel<<<1, launchRows>>>(deviceScale, deviceOutput, rows, launchRows);
    } else {
        safeRowwiseScaleKernel<<<1, launchRows>>>(deviceScale, deviceOutput, rows);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceScale));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostScale);
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

    std::printf("case=%s rows=%d padded_rows=%d tile_rows=%d output_hash=%llu sync_status=%s copy_status=%s\n",
                name, rows, launchRows, TILE_ROWS,
                static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_ELEMENTS)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceScale));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostScale);
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
